"""In-sandbox backtest harness. Runs with no network, read-only root, and a
read-only copy of the price lake.

The generated strategy code supplies ONE function:

    def signal(df: pd.DataFrame) -> pd.Series   # target weight per bar, [-1, 1]

Everything else — lagging the signal, filling at the next open, charging costs,
computing metrics — is owned by this harness, on purpose. A model asked to
"write a backtest" reliably introduces look-ahead by acting on the same bar's
close; it cannot do that here, because it never touches the execution loop.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Defaults are the container's mount points; the subprocess fallback overrides
# them with environment variables rather than rewriting this file's source.
JOB = Path(os.environ.get("BT_JOB", "/job/job.json"))
OUT = Path(os.environ.get("BT_OUT", "/out/result.json"))
LAKE = Path(os.environ.get("BT_LAKE", "/lake"))

TRADING_DAYS = 252


def load_prices(symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    path = LAKE / f"symbol={symbol}" / f"{symbol}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{symbol} is not in the price lake")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df = df.sort_index()
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    if len(df) < 60:
        raise ValueError(f"{symbol}: only {len(df)} bars in range, need >= 60")
    return df


def compile_strategy(code: str):
    """Execute the generated module in a bare namespace and hand back `signal`."""
    ns: dict = {"pd": pd, "np": np}
    exec(compile(code, "<strategy>", "exec"), ns)  # noqa: S102 - that is the job
    fn = ns.get("signal")
    if not callable(fn):
        raise ValueError("generated code defines no callable named `signal`")
    return fn


def evaluate(df: pd.DataFrame, fn, *, cost_bps: float, slippage_bps: float) -> dict:
    raw = fn(df.copy())
    if not isinstance(raw, pd.Series):
        raw = pd.Series(raw, index=df.index)
    weight = raw.reindex(df.index).astype(float).fillna(0.0).clip(-1.0, 1.0)

    # Decide on bar t's close, trade at bar t+1's open, hold to t+1's close.
    # The single shift below is what makes the result honest.
    held = weight.shift(1).fillna(0.0)

    open_, close = df["open"].astype(float), df["close"].astype(float)
    prev_close = close.shift(1)
    overnight = (open_ / prev_close - 1.0).fillna(0.0)
    intraday = (close / open_ - 1.0).fillna(0.0)

    # Attribution matters more than it looks. The gap from bar t-1's close to
    # bar t's open happens BEFORE this bar's fill, so it belongs to whatever was
    # already on the book -- `held.shift(1)`, not `held`. Charging it to `held`
    # hands every position the gap that followed its own entry signal, which is
    # free money for exactly the close-to-close mean-reversion and breakout
    # strategies a model likes to write. It is look-ahead wearing a costume:
    # the equity curve stays smooth and only the Sharpe is wrong.
    carried = held.shift(1).fillna(0.0)
    gross = carried * overnight + held * intraday

    turnover = (held - carried).abs()
    cost = turnover * (cost_bps + slippage_bps) / 10_000.0
    net = gross - cost

    equity = (1.0 + net).cumprod()
    gross_equity = (1.0 + gross).cumprod()
    return {
        "metrics": metrics(net, equity, turnover, held),
        "gross_metrics": metrics(gross, gross_equity, turnover * 0, held),
        "equity_curve": [
            {"date": d.strftime("%Y-%m-%d"), "equity": round(float(v), 6)}
            for d, v in equity.items()
        ],
        "bars": int(len(df)),
        "start": df.index[0].strftime("%Y-%m-%d"),
        "end": df.index[-1].strftime("%Y-%m-%d"),
    }


def metrics(returns: pd.Series, equity: pd.Series, turnover: pd.Series,
            held: pd.Series) -> dict:
    n = len(returns)
    if n == 0 or equity.empty:
        return {}
    years = n / TRADING_DAYS
    total = float(equity.iloc[-1])
    cagr = total ** (1 / years) - 1 if years > 0 and total > 0 else float("nan")
    std = float(returns.std())
    sharpe = float(returns.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else float("nan")
    downside = returns[returns < 0].std()
    sortino = (float(returns.mean() / downside * np.sqrt(TRADING_DAYS))
               if downside and downside > 0 else float("nan"))
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())
    trough = dd.idxmin() if len(dd) else None
    active = returns[held != 0]
    return {
        "total_return": round(total - 1.0, 6),
        "cagr": _r(cagr),
        "sharpe": _r(sharpe),
        "sortino": _r(sortino),
        "max_drawdown": round(max_dd, 6),
        "max_drawdown_date": trough.strftime("%Y-%m-%d") if trough is not None else None,
        "calmar": _r(cagr / abs(max_dd)) if max_dd < 0 and cagr == cagr else None,
        "volatility": _r(std * np.sqrt(TRADING_DAYS)),
        "hit_rate": _r(float((active > 0).mean())) if len(active) else None,
        "exposure": _r(float((held != 0).mean())),
        "turnover_annualised": _r(float(turnover.sum() / (n / TRADING_DAYS))) if n else None,
        "trades": int((held.diff().fillna(held) != 0).sum()),
    }


def _r(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f or f in (float("inf"), float("-inf")) else round(f, 6)


def main() -> int:
    job = json.loads(JOB.read_text())
    result = {"ok": False, "symbols": {}, "error": None}
    try:
        fn = compile_strategy(job["code"])
    except Exception:
        result["error"] = traceback.format_exc(limit=6)
        result["stage"] = "compile"
        OUT.write_text(json.dumps(result))
        return 1

    for symbol in job["symbols"]:
        try:
            df = load_prices(symbol, job.get("start"), job.get("end"))
            result["symbols"][symbol] = evaluate(
                df, fn,
                cost_bps=float(job.get("cost_bps", 5)),
                slippage_bps=float(job.get("slippage_bps", 5)),
            )
        except Exception:
            result["symbols"][symbol] = {"error": traceback.format_exc(limit=6)}

    graded = [v for v in result["symbols"].values() if "metrics" in v]
    result["ok"] = bool(graded)
    if not graded:
        result["stage"] = "evaluate"
        first = next(iter(result["symbols"].values()), {})
        result["error"] = first.get("error", "no symbol produced a result")
    OUT.write_text(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
