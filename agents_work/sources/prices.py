"""Prices and history.

Two backends on purpose. The Parquet lake (built by ~/market-lab) is local,
fast, offline, and slightly stale; yfinance is live and flaky. History reads
the lake first and only falls back to the network for what the lake lacks —
which is what makes the backtest sandbox able to run with no network at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]

# Symbol -> display label for the pre-open dashboard.
FUTURES = {
    "ES=F": "S&P 500 futures",
    "NQ=F": "Nasdaq 100 futures",
    "YM=F": "Dow futures",
    "RTY=F": "Russell 2000 futures",
}
# Quoted as a percentage *level*, not a price. A 4.66 -> 4.70 move is four basis
# points, but naive percent-change arithmetic calls it +0.86%, and a reader (or a
# model) seeing "+0.86%" next to a yield reasonably hears 86bp.
YIELDS = {"^TNX", "^TYX", "^FVX", "^IRX"}

MACRO = {
    "^VIX": "VIX",
    "^TNX": "US 10y yield",
    "CL=F": "WTI crude",
    "GC=F": "Gold",
    "DX-Y.NYB": "Dollar index",
}


@dataclass
class Quote:
    symbol: str
    last: float | None = None
    prev_close: float | None = None
    as_of: str = ""
    label: str = ""
    error: str = ""

    @property
    def change_pct(self) -> float | None:
        if self.last is None or not self.prev_close:
            return None
        return (self.last / self.prev_close - 1.0) * 100.0

    @property
    def ok(self) -> bool:
        return self.last is not None


def _import_yf():
    try:
        import yfinance as yf  # noqa: PLC0415 - optional at import time by design
        return yf
    except ImportError:  # pragma: no cover - exercised by the no-network path
        log.warning("yfinance not installed; live prices unavailable")
        return None


class PriceSource:
    def __init__(self, lake_dir: Path | None = None, *, cache_dir: Path | None = None,
                 allow_network: bool = True) -> None:
        self.lake_dir = Path(lake_dir) if lake_dir else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.allow_network = allow_network
        self.notes: list[str] = []

    # -- history ---------------------------------------------------------
    def lake_path(self, symbol: str) -> Path | None:
        if not self.lake_dir:
            return None
        p = self.lake_dir / f"symbol={symbol.upper()}" / f"{symbol.upper()}.parquet"
        return p if p.is_file() else None

    def history(self, symbol: str, *, start: str | None = None, end: str | None = None,
                min_rows: int = 30) -> pd.DataFrame:
        """Daily OHLCV. Empty DataFrame (with the right columns) if unavailable."""
        symbol = symbol.upper()
        df = self._from_lake(symbol)
        if df is not None and len(df) >= min_rows:
            out = self._slice(df, start, end)
            if len(out) >= min_rows or not self.allow_network:
                return out
        live = self._from_network(symbol, start, end)
        if live is not None and not live.empty:
            return self._slice(live, start, end)
        if df is not None:
            return self._slice(df, start, end)
        self.notes.append(f"no price history available for {symbol}")
        return pd.DataFrame(columns=COLUMNS)

    def _from_lake(self, symbol: str) -> pd.DataFrame | None:
        path = self.lake_path(symbol)
        if not path:
            return None
        try:
            df = pd.read_parquet(path)
        except Exception as e:  # noqa: BLE001 - a corrupt shard must not kill a run
            log.warning("unreadable lake shard %s: %s", path, e)
            self.notes.append(f"lake shard for {symbol} is unreadable; using network")
            return None
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        return df[[c for c in COLUMNS if c in df.columns]].sort_index()

    def _from_network(self, symbol: str, start: str | None, end: str | None) -> pd.DataFrame | None:
        if not self.allow_network:
            return None
        yf = _import_yf()
        if yf is None:
            return None
        try:
            raw = yf.Ticker(symbol).history(
                start=start, end=end,
                period=None if start else "5y",
                auto_adjust=False, raise_errors=False,
            )
        except Exception as e:  # noqa: BLE001 - yfinance raises a wide variety
            log.warning("yfinance history failed for %s: %s", symbol, e)
            self.notes.append(f"live price fetch failed for {symbol} ({type(e).__name__})")
            return None
        if raw is None or raw.empty:
            return None
        raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                  "Close": "close", "Adj Close": "adj_close",
                                  "Volume": "volume"})
        if "adj_close" not in raw:
            raw["adj_close"] = raw["close"]
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        return raw[[c for c in COLUMNS if c in raw.columns]].sort_index()

    @staticmethod
    def _slice(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
        out = df
        if start:
            out = out[out.index >= pd.Timestamp(start)]
        if end:
            out = out[out.index <= pd.Timestamp(end)]
        return out

    # -- quotes ----------------------------------------------------------
    def quotes(self, symbols: dict[str, str] | list[str]) -> list[Quote]:
        labels = symbols if isinstance(symbols, dict) else {s: s for s in symbols}
        out: list[Quote] = []
        yf = _import_yf() if self.allow_network else None
        for sym, label in labels.items():
            q = Quote(symbol=sym, label=label)
            if yf is not None:
                try:
                    t = yf.Ticker(sym)
                    fi = t.fast_info
                    q.last = _f(fi.get("lastPrice"))
                    q.prev_close = _f(fi.get("previousClose"))
                    q.as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                except Exception as e:  # noqa: BLE001
                    q.error = f"{type(e).__name__}"
                    log.warning("quote failed for %s: %s", sym, e)
            if not q.ok:
                # Last resort: the newest two closes we have anywhere.
                hist = self.history(sym, min_rows=2)
                if len(hist) >= 2:
                    q.last = float(hist["close"].iloc[-1])
                    q.prev_close = float(hist["close"].iloc[-2])
                    q.as_of = f"{hist.index[-1].date()} close (stale)"
                    q.error = q.error or "live quote unavailable; showing last stored close"
            if not q.ok and not q.error:
                q.error = "no data"
            out.append(q)
        return out

    def movers(self, symbols: list[str], *, top: int = 5) -> list[Quote]:
        qs = [q for q in self.quotes(symbols) if q.change_pct is not None]
        qs.sort(key=lambda q: abs(q.change_pct), reverse=True)
        return qs[:top]


def _f(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None
