"""Fixtures. Everything here is offline and deterministic.

Nothing in the suite touches the network, the real data directory, or the real
notes repo. Sources are served from `FakeFetcher`; the LLM is `FakeLLM`; the
price lake is synthesised into tmp_path. That is not test hygiene for its own
sake — these agents are *made* of external dependencies, so a suite that needed
them would only run on a good day.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents_work.agents.base import Context
from agents_work.config import Config
from agents_work.gitsink import NotesRepo
from agents_work.llm import FakeLLM
from agents_work.netcache import Fetcher, Response

FIXTURES = Path(__file__).parent / "fixtures"


class FakeFetcher(Fetcher):
    """Serves canned bodies by URL substring; unknown URLs return None.

    Returning None rather than raising is the same contract the real Fetcher
    offers when a source is down, so "source missing" is exercised by default
    in every test that does not explicitly register a route.
    """

    def __init__(self, routes: dict[str, object] | None = None, cache_dir: Path | None = None):
        super().__init__(cache_dir or Path("/nonexistent"), ttl=0, offline=False)
        self.routes: dict[str, object] = dict(routes or {})
        self.requested: list[str] = []

    def route(self, fragment: str, body, status: int = 200) -> "FakeFetcher":
        self.routes[fragment] = (body, status)
        return self

    def fetch(self, url, *, headers=None, ttl=None, params=None):
        if params:
            import httpx
            url = str(httpx.URL(url).copy_merge_params(params))
        self.requested.append(url)
        for fragment, value in self.routes.items():
            if fragment in url:
                body, status = value if isinstance(value, tuple) else (value, 200)
                if body is None:
                    self.stats["error"] += 1
                    return None
                text = body if isinstance(body, str) else json.dumps(body)
                self.stats["hit"] += 1
                return Response(url, status, text, from_cache=True)
        self.stats["error"] += 1
        return None


@pytest.fixture
def lake(tmp_path) -> Path:
    """A synthetic Parquet lake shaped exactly like market-lab's."""
    root = tmp_path / "lake"
    rng = np.random.default_rng(7)
    for symbol in ("SPY", "QQQ"):
        n = 800
        idx = pd.bdate_range("2020-01-01", periods=n)
        steps = rng.normal(0.0004, 0.011, n)
        close = 100 * np.exp(np.cumsum(steps))
        open_ = close * (1 + rng.normal(0, 0.002, n))
        df = pd.DataFrame({
            "open": open_, "high": np.maximum(open_, close) * 1.004,
            "low": np.minimum(open_, close) * 0.996, "close": close,
            "adj_close": close, "volume": rng.integers(1e6, 9e6, n),
        }, index=idx)
        d = root / f"symbol={symbol}"
        d.mkdir(parents=True)
        df.to_parquet(d / f"{symbol}.parquet")
    return root


@pytest.fixture
def vault(tmp_path) -> Path:
    """A small notes vault with known content and known dates."""
    root = tmp_path / "vault"
    root.mkdir()
    today = date(2026, 8, 20)
    for name, days_ago, body in _VAULT_NOTES:
        d = today - timedelta(days=days_ago)
        (root / name).write_text(
            f"---\ntitle: {name}\ndate: {d.isoformat()}\n---\n\n{body}\n")
    return root


_VAULT_NOTES = [
    ("2026-08-14-nvda.md", 6, "# NVDA\n\nConcluded the datacenter backlog is the whole "
     "thesis. Gross margin held at 75% despite the Blackwell ramp. I am comfortable "
     "holding through the print."),
    ("2026-07-02-nvda-doubts.md", 49, "# NVDA revisited\n\nEarlier I thought the backlog "
     "was durable. Now the hyperscaler capex commentary reads softer and I trimmed."),
    ("2026-08-18-rates.md", 2, "# Rates\n\nThe ten year broke 4.5% and the curve "
     "disinverted. Duration is finally paying."),
    ("2026-08-01-energy.md", 19, "# Energy\n\nRefining margins compressed. Crack spreads "
     "are the tell, not headline crude."),
    ("2026-06-15-banks.md", 66, "# Banks\n\nNet interest margin peaked. Credit costs are "
     "the next leg and reserves look thin."),
    ("2026-08-10-portfolio.md", 10, "# Portfolio review\n\nRebalanced out of momentum into "
     "quality. Cash is 12% and I want it lower before year end."),
    ("2026-05-20-backtest-notes.md", 92, "# Backtest hygiene\n\nLook-ahead creeps in via "
     "the overnight gap. Fill at the next open and charge slippage or the Sharpe lies."),
    ("2026-08-05-semis.md", 15, "# Semis\n\nMemory pricing turned. Samsung and Hynix "
     "guidance implies a real cycle, not a blip."),
    ("2026-07-20-internships.md", 31, "# Internship search\n\nJane Street and Optiver both "
     "open in September. Prioritise the quant trading track over generalist SWE."),
    ("2026-08-19-fed.md", 1, "# Fed\n\nPowell signalled patience. Two cuts priced for next "
     "year, down from four."),
]

# question -> the file that answers it. The retrieval benchmark scores against this.
GOLD = [
    ("what did I conclude about NVDA last month", "2026-08-14-nvda.md"),
    ("did I change my mind on the datacenter backlog", "2026-07-02-nvda-doubts.md"),
    ("where is the ten year yield", "2026-08-18-rates.md"),
    ("what is the tell for refining", "2026-08-01-energy.md"),
    ("are bank reserves adequate", "2026-06-15-banks.md"),
    ("how much cash am I holding", "2026-08-10-portfolio.md"),
    ("how does look-ahead sneak into a backtest", "2026-05-20-backtest-notes.md"),
    ("is the memory cycle real", "2026-08-05-semis.md"),
    ("which firms should I prioritise applying to", "2026-07-20-internships.md"),
    ("how many cuts are priced", "2026-08-19-fed.md"),
]


@pytest.fixture
def cfg(tmp_path, lake, vault) -> Config:
    data = tmp_path / "data"
    return Config(
        root=tmp_path, data_dir=data,
        llm_base_url="https://llm.invalid", llm_api_key=None,
        write_model="test-write", fast_model="test-fast",
        git_remote=None, sec_user_agent="agents_work-tests",
        lake_dir=lake, watchlist=["SPY", "QQQ"],
        vault_roots=[vault], port=8110, http_cache_ttl=0,
    )


@pytest.fixture
def fetcher(tmp_path) -> FakeFetcher:
    return FakeFetcher(cache_dir=tmp_path / "cache")


@pytest.fixture
def llm(cfg) -> FakeLLM:
    return FakeLLM(cfg)


@pytest.fixture
def ctx(cfg, fetcher, llm) -> Context:
    c = Context(cfg, llm=llm, fetcher=fetcher,
                notes=NotesRepo(cfg.notes_repo, remote=None))
    yield c
    c.close()


@pytest.fixture
def offline_ctx(cfg, tmp_path) -> Context:
    """No network, no LLM — the degraded path X1 exercises."""
    c = Context(cfg, llm=FakeLLM(cfg, available=False),
                fetcher=FakeFetcher(cache_dir=tmp_path / "cache"),
                notes=NotesRepo(cfg.notes_repo, remote=None), offline=True)
    yield c
    c.close()
