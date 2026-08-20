"""Backtest agent: the harness's execution semantics, the sandbox, the repair loop."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents_work.agents import backtest
from agents_work.agents.backtest import (BacktestJob, build_brief, execute, extract_code,
                                         run_in_subprocess, sparkline, static_check)
from agents_work.llm import FakeLLM

ROOT = Path(__file__).resolve().parent.parent


def _runner():
    spec = importlib.util.spec_from_file_location("bt_runner", ROOT / "sandbox" / "runner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _runner()


def alternating_gap_frame(n: int = 200, gap: float = 0.01) -> pd.DataFrame:
    """Every move happens overnight, and the moves alternate sign.

    open == close on every bar, so intraday return is exactly zero and the whole
    return series lives in the gaps. That makes the attribution question — who
    owns the gap — the only thing the arithmetic can be measuring.
    """
    rets = np.array([gap if i % 2 == 0 else -gap for i in range(n)])
    close = 100 * np.cumprod(1 + rets)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                         "adj_close": close, "volume": 1_000_000}, index=idx)


def mean_reversion(df):
    """Yesterday went up, so fade it. Knowable at the close; no peeking."""
    return -np.sign(df["close"].pct_change()).fillna(0.0)


# -- B1: overnight attribution ----------------------------------------------

@pytest.mark.benchmark
def test_overnight_gap_belongs_to_the_book_not_the_entry():
    """B1 (regression). On alternating-gap data a fader's P&L is deterministic:
    if the gap is credited to the position that entered *at that morning's open*,
    the strategy prints +1% a day forever; credited to the position that actually
    carried the risk overnight it prints -1% a day. The old harness printed the
    first number, and nothing about the equity curve looked wrong."""
    df = alternating_gap_frame()
    res = runner.evaluate(df, mean_reversion, cost_bps=0, slippage_bps=0)
    fixed_total = res["metrics"]["total_return"]

    held = (-np.sign(df["close"].pct_change()).fillna(0.0)).shift(1).fillna(0.0)
    overnight = (df["open"] / df["close"].shift(1) - 1.0).fillna(0.0)
    intraday = (df["close"] / df["open"] - 1.0).fillna(0.0)
    old_total = float((1 + held * (overnight + intraday)).cumprod().iloc[-1]) - 1.0

    assert old_total > 0.5, "fixture no longer reproduces the old free-money result"
    assert fixed_total < 0, f"gap still being credited to the entry (total {fixed_total})"
    assert fixed_total < old_total


@pytest.mark.benchmark
def test_signal_cannot_act_on_its_own_bar():
    """B2. `sign(today's return)` is legitimate information at today's close.
    A harness that fills on the same bar turns it into a Sharpe above 10."""
    rng = np.random.default_rng(11)
    n = 1500
    rets = rng.normal(0, 0.01, n)
    close = 100 * np.cumprod(1 + rets)
    idx = pd.bdate_range("2019-01-01", periods=n)
    df = pd.DataFrame({"open": close * (1 + rng.normal(0, 0.0005, n)), "high": close,
                       "low": close, "close": close, "adj_close": close,
                       "volume": 1}, index=idx)

    res = runner.evaluate(df, lambda d: np.sign(d["close"].pct_change()).fillna(0.0),
                          cost_bps=0, slippage_bps=0)
    assert abs(res["metrics"]["sharpe"]) < 1.0

    same_bar = np.sign(df["close"].pct_change()).fillna(0.0) * df["close"].pct_change().fillna(0)
    cheat_sharpe = same_bar.mean() / same_bar.std() * np.sqrt(252)
    assert cheat_sharpe > 10, "fixture no longer distinguishes the two conventions"


# -- B3/B4: costs and metrics -----------------------------------------------

@pytest.mark.benchmark
def test_costs_are_charged_exactly_once_per_unit_of_turnover():
    """B3."""
    df = alternating_gap_frame(120)
    free = runner.evaluate(df, mean_reversion, cost_bps=0, slippage_bps=0)
    charged = runner.evaluate(df, mean_reversion, cost_bps=5, slippage_bps=5)
    turnover = charged["metrics"]["turnover_annualised"] * (charged["bars"] / 252)
    expected_drag = turnover * 10 / 10_000
    gross_curve = np.array([p["equity"] for p in free["equity_curve"]])
    net_curve = np.array([p["equity"] for p in charged["equity_curve"]])
    assert net_curve[-1] < gross_curve[-1]
    assert expected_drag > 0
    assert charged["metrics"]["sharpe"] < free["metrics"]["sharpe"]


def test_cost_arithmetic_matches_the_definition():
    df = alternating_gap_frame(60)
    res = runner.evaluate(df, mean_reversion, cost_bps=7, slippage_bps=3)
    weight = mean_reversion(df).clip(-1, 1)
    held = weight.shift(1).fillna(0.0)
    carried = held.shift(1).fillna(0.0)
    gross = carried * (df["open"] / df["close"].shift(1) - 1).fillna(0) + \
        held * (df["close"] / df["open"] - 1).fillna(0)
    net = gross - (held - carried).abs() * 10 / 10_000
    assert float((1 + net).cumprod().iloc[-1]) - 1 == pytest.approx(
        res["metrics"]["total_return"], abs=1e-6)  # the metric is rounded to 6dp


@pytest.mark.benchmark
def test_metrics_match_closed_form_for_buy_and_hold():
    """B4. Buy and hold must reproduce the instrument's own statistics."""
    rng = np.random.default_rng(3)
    n = 900
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    # No overnight gaps: each bar opens where the last one closed, so the whole
    # move is intraday and the harness's return decomposition is exact rather
    # than the usual additive approximation.
    open_ = np.concatenate([[close[0]], close[:-1]])
    idx = pd.bdate_range("2021-01-01", periods=n)
    df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close),
                       "low": np.minimum(open_, close), "close": close,
                       "adj_close": close, "volume": 1}, index=idx)
    res = runner.evaluate(df, lambda d: pd.Series(1.0, index=d.index),
                          cost_bps=0, slippage_bps=0)
    m = res["metrics"]

    rets = pd.Series(close, index=idx).pct_change().fillna(0.0)
    rets.iloc[0] = 0.0  # bar 0 is unfunded: the first fill is at bar 1's open
    equity = (1 + rets).cumprod()
    # every metric is stored rounded to six decimals
    assert m["total_return"] == pytest.approx(float(equity.iloc[-1]) - 1, abs=1e-6)
    assert m["max_drawdown"] == pytest.approx(float((equity / equity.cummax() - 1).min()),
                                              abs=1e-6)
    assert m["sharpe"] == pytest.approx(
        float(rets.mean() / rets.std() * np.sqrt(252)), abs=1e-6)
    years = n / 252
    assert m["cagr"] == pytest.approx(float(equity.iloc[-1]) ** (1 / years) - 1, abs=1e-6)
    assert m["exposure"] == pytest.approx(1 - 1 / n, rel=1e-6)


def test_nan_signal_is_treated_as_flat():
    df = alternating_gap_frame(80)
    res = runner.evaluate(df, lambda d: pd.Series(np.nan, index=d.index),
                          cost_bps=5, slippage_bps=5)
    assert res["metrics"]["total_return"] == pytest.approx(0.0, abs=1e-12)
    assert res["metrics"]["trades"] == 0


def test_weights_are_clipped_to_the_declared_range():
    df = alternating_gap_frame(80)
    res = runner.evaluate(df, lambda d: pd.Series(50.0, index=d.index),
                          cost_bps=0, slippage_bps=0)
    unlevered = runner.evaluate(df, lambda d: pd.Series(1.0, index=d.index),
                                cost_bps=0, slippage_bps=0)
    assert res["metrics"]["total_return"] == pytest.approx(
        unlevered["metrics"]["total_return"])


def test_scalar_signal_is_broadcast():
    df = alternating_gap_frame(80)
    res = runner.evaluate(df, lambda d: 1.0, cost_bps=0, slippage_bps=0)
    assert res["metrics"] is not None and res["bars"] == 80


# -- code handling -----------------------------------------------------------

def test_extract_code_pulls_the_fenced_function():
    text = "Here you go:\n```python\ndef signal(df):\n    return df['close'] * 0\n```\nDone."
    assert extract_code(text).startswith("def signal")


def test_extract_code_rejects_prose_without_a_function():
    with pytest.raises(ValueError, match="no `def signal`"):
        extract_code("I cannot help with that.")


def test_static_check_flags_banned_constructs_and_syntax():
    problems = static_check("import os\ndef signal(df):\n    return (")
    assert any("banned construct" in p for p in problems)
    assert any("syntax error" in p for p in problems)


def test_static_check_passes_ordinary_strategy_code():
    assert static_check("def signal(df):\n    return df['close'].rolling(20).mean() * 0") == []


def test_compile_strategy_requires_a_callable_named_signal():
    with pytest.raises(ValueError, match="no callable named"):
        runner.compile_strategy("signal = 3")


# -- sandbox -----------------------------------------------------------------

docker_missing = not backtest.docker_available()


@pytest.mark.benchmark
@pytest.mark.skipif(docker_missing, reason="sandbox image not built")
def test_sandbox_blocks_network_egress(lake):
    """B5. The container is the security boundary; the static check is a courtesy."""
    job = BacktestJob(idea="probe", symbols=["SPY"], code=(
        "def signal(df):\n"
        "    import socket\n"
        "    socket.create_connection(('1.1.1.1', 80), timeout=5)\n"
        "    return df['close'] * 0\n"))
    result = execute(job, lake, timeout=90)
    assert not result["ok"]
    assert result["isolation"] == "docker"


@pytest.mark.benchmark
@pytest.mark.skipif(docker_missing, reason="sandbox image not built")
def test_sandbox_root_filesystem_is_read_only(lake):
    """B5."""
    job = BacktestJob(idea="probe", symbols=["SPY"], code=(
        "def signal(df):\n"
        "    open('/etc/passwd', 'w').write('x')\n"
        "    return df['close'] * 0\n"))
    result = execute(job, lake, timeout=90)
    assert not result["ok"]


@pytest.mark.skipif(docker_missing, reason="sandbox image not built")
def test_sandbox_produces_metrics_for_a_real_strategy(lake):
    job = BacktestJob(idea="20/50 crossover", symbols=["SPY"], code=(
        "def signal(df):\n"
        "    fast = df['close'].rolling(20).mean()\n"
        "    slow = df['close'].rolling(50).mean()\n"
        "    return (fast > slow).astype(float)\n"))
    result = execute(job, lake, timeout=120)
    assert result["ok"], result.get("error")
    assert result["isolation"] == "docker"
    m = result["symbols"]["SPY"]["metrics"]
    assert set(m) >= {"sharpe", "max_drawdown", "cagr", "trades"}
    assert -1 <= m["max_drawdown"] <= 0


@pytest.mark.benchmark
@pytest.mark.skipif(shutil.which("python3") is None, reason="no interpreter")
def test_subprocess_fallback_is_labelled_as_weaker(lake):
    """B6. A weaker sandbox is fine; a weaker sandbox that looks like the strong
    one is not."""
    job = BacktestJob(idea="flat", symbols=["SPY"],
                      code="def signal(df):\n    return df['close'] * 0\n")
    result = run_in_subprocess(job, lake, timeout=120)
    assert result["isolation"] == "subprocess"
    brief = build_brief(job, result, degradations=[])
    assert any("not a container" in d for d in brief.degradations)
    assert brief.extra_meta["isolation"] == "subprocess"


def test_missing_symbol_is_an_error_on_that_symbol_only(lake):
    job = BacktestJob(idea="flat", symbols=["SPY", "NOSUCH"],
                      code="def signal(df):\n    return df['close'] * 0\n")
    result = run_in_subprocess(job, lake, timeout=120)
    assert result["ok"]
    assert "metrics" in result["symbols"]["SPY"]
    assert "error" in result["symbols"]["NOSUCH"]


# -- agent entry point -------------------------------------------------------

GOOD = "```python\ndef signal(df):\n    return (df['close'] > df['close'].rolling(20).mean()).astype(float)\n```"
BAD = "```python\ndef signal(df):\n    return df['nope'] * 1\n```"


@pytest.mark.benchmark
def test_repair_loop_retries_then_gives_up_honestly(ctx, cfg):
    """B7."""
    ctx.llm = FakeLLM(cfg, [BAD, BAD, BAD])
    res = backtest.run(ctx, "buy the dip", symbols=["SPY"], max_repairs=2,
                       timeout=120, commit=False)
    assert not res.ok
    assert res.error and "unknown" not in res.error
    assert len(ctx.llm.prompts) == 3          # first attempt plus two repairs


def test_repair_loop_recovers_from_a_first_bad_attempt(ctx, cfg):
    ctx.llm = FakeLLM(cfg, [BAD, GOOD])
    res = backtest.run(ctx, "trend", symbols=["SPY"], max_repairs=2,
                       timeout=120, commit=False)
    assert res.ok, res.error
    assert res.brief is not None
    assert "Repair history" in [s.heading for s in res.brief.sections]


def test_backtest_without_a_lake_fails_loudly(ctx, cfg, tmp_path):
    from dataclasses import replace
    ctx.cfg = replace(cfg, lake_dir=tmp_path / "gone")
    res = backtest.run(ctx, "anything", symbols=["SPY"], commit=False)
    assert not res.ok
    assert "no price lake" in res.error


def test_supplied_code_skips_the_model_entirely(ctx):
    res = backtest.run(ctx, "flat", symbols=["SPY"], timeout=120, commit=False,
                       code="def signal(df):\n    return df['close'] * 0\n")
    assert res.ok, res.error
    assert ctx.llm.prompts == []


# -- rendering ---------------------------------------------------------------

def test_sparkline_is_valid_inline_svg_with_a_label():
    curve = [{"date": "2020-01-01", "equity": 1.0}, {"date": "2020-01-02", "equity": 1.2},
             {"date": "2020-01-03", "equity": 0.9}]
    svg = sparkline(curve)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert 'role="img"' in svg and "aria-label" in svg
    assert "polyline" in svg


def test_sparkline_declines_to_draw_a_single_point():
    assert sparkline([{"date": "2020-01-01", "equity": 1.0}]) == ""
