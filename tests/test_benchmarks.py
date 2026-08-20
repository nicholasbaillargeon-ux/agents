"""Cross-cutting gates from BENCHMARKS.md — the X and P series.

These do not test one agent, they test the property every agent has to share:
that a missing dependency produces a smaller answer instead of a wrong one, and
that you can always tell which happened.
"""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

from agents_work.agents import analyst, backtest, briefing, research, scout
from agents_work.agents.base import Context
from agents_work.gitsink import NotesRepo
from agents_work.llm import FakeLLM
from agents_work.store import recent

FLAT = "def signal(df):\n    return (df['close'] > df['close'].rolling(30).mean()).astype(float)\n"


def _run_all(ctx):
    """Every agent, once, in the same degraded conditions."""
    return {
        "research": research.run(ctx, "SPY", commit=False),
        "backtest": backtest.run(ctx, "trend following", symbols=["SPY"],
                                 code=FLAT, timeout=180, commit=False),
        "briefing": briefing.run(ctx, commit=False),
        "scout": scout.run(ctx, use_llm=False, commit=False),
        "analyst": analyst.run(ctx, "what did I conclude about NVDA", reindex=True),
    }


# -- X1 ----------------------------------------------------------------------

@pytest.mark.benchmark
def test_every_agent_completes_with_no_network_and_no_model(offline_ctx):
    """X1. Nothing raises, and every agent still writes a document."""
    results = _run_all(offline_ctx)
    assert set(results) == {"research", "backtest", "briefing", "scout", "analyst"}
    for name, res in results.items():
        assert res.ok, f"{name} failed: {res.error}"
        assert res.artifact and res.artifact.is_file(), f"{name} wrote no artifact"
        assert res.artifact.read_text().strip(), f"{name} wrote an empty artifact"


@pytest.mark.benchmark
def test_a_backtest_with_no_model_fails_by_name(offline_ctx):
    """X1's exception: with no model there is no strategy code, and saying so is
    the only honest outcome."""
    res = backtest.run(offline_ctx, "buy the dip", symbols=["SPY"], commit=False)
    assert not res.ok
    assert "could not generate strategy code" in res.error


# -- X2 ----------------------------------------------------------------------

@pytest.mark.benchmark
def test_a_degraded_run_says_so_everywhere(offline_ctx):
    """X2. The frontmatter flag, the banner and the run log must agree."""
    results = _run_all(offline_ctx)
    for name, res in results.items():
        text = res.artifact.read_text()
        assert res.degradations, f"{name} claimed a clean run while offline"
        assert "degraded: true" in text, f"{name} frontmatter does not admit it"
        assert "Ran degraded" in text, f"{name} has no banner"

    logged = {row["agent"]: row for row in recent(offline_ctx.db, limit=50)}
    for name in results:
        assert logged[name]["degradations"], f"{name} logged no degradations"


@pytest.mark.benchmark
def test_a_healthy_run_does_not_cry_wolf(ctx, fetcher, monkeypatch):
    """X2, the other half: a warning that is always on is not a warning."""
    from agents_work.sources.edgar import Edgar
    monkeypatch.setattr(Edgar, "company_facts", lambda self, cik: {})
    ctx.llm.default_response = "## Answer\n\nSomething."
    idx = analyst.Index(ctx.cfg.data_dir / "analyst.db")
    idx.build(ctx.cfg.vault_roots)
    answer = analyst.ask(ctx, "what did I conclude about NVDA", index=idx)
    idx.close()
    assert answer.degradations == []


# -- X3 ----------------------------------------------------------------------

@pytest.mark.benchmark
def test_one_invocation_writes_exactly_one_run_row(offline_ctx):
    """X3."""
    before = len(recent(offline_ctx.db, limit=500))
    _run_all(offline_ctx)
    rows = recent(offline_ctx.db, limit=500)
    assert len(rows) - before == 5
    assert sorted(r["agent"] for r in rows[:5]) == [
        "analyst", "backtest", "briefing", "research", "scout"]


@pytest.mark.benchmark
def test_a_failed_run_is_logged_too(offline_ctx):
    """X3."""
    before = len(recent(offline_ctx.db, limit=500))
    res = backtest.run(offline_ctx, "impossible", symbols=["SPY"], commit=False)
    assert not res.ok
    rows = recent(offline_ctx.db, limit=500)
    assert len(rows) - before == 1
    assert rows[0]["ok"] is False and rows[0]["error"]


# -- X4 ----------------------------------------------------------------------

SECRET = "sk-ant-DO-NOT-LEAK-0123456789"


@pytest.mark.benchmark
def test_no_artifact_or_log_row_contains_the_credential(cfg, tmp_path):
    """X4."""
    keyed = replace(cfg, llm_api_key=SECRET)
    from conftest import FakeFetcher
    ctx = Context(keyed, llm=FakeLLM(keyed), fetcher=FakeFetcher(cache_dir=tmp_path / "c"),
                  notes=NotesRepo(keyed.notes_repo), offline=True)
    try:
        results = _run_all(ctx)
        for name, res in results.items():
            assert SECRET not in res.artifact.read_text(), f"{name} leaked the key"
            assert SECRET not in (res.summary or "")
            assert SECRET not in str(res.degradations)
        for row in recent(ctx.db, limit=50):
            assert SECRET not in str(dict(row))
    finally:
        ctx.close()


@pytest.mark.benchmark
def test_the_health_endpoint_does_not_expose_the_credential(cfg):
    """X4."""
    from fastapi.testclient import TestClient

    from agents_work.web.app import create_app
    client = TestClient(create_app(replace(cfg, llm_api_key=SECRET)))
    assert SECRET not in client.get("/api/health").text
    assert SECRET not in client.get("/").text


# -- X5 ----------------------------------------------------------------------

@pytest.mark.benchmark
def test_briefs_attribute_their_sources_or_say_why_not(offline_ctx):
    """X5."""
    for name, res in _run_all(offline_ctx).items():
        text = res.artifact.read_text()
        assert "## Sources" in text or res.degradations, (
            f"{name} cited nothing and admitted nothing")


# -- P3 / P4 -----------------------------------------------------------------

@pytest.mark.benchmark
def test_brief_render_latency(offline_ctx):
    """P3: a full brief renders in under 25 ms."""
    res = research.run(offline_ctx, "SPY", commit=False)
    brief = res.brief
    timings = []
    for _ in range(20):
        t0 = time.perf_counter()
        brief.render()
        timings.append((time.perf_counter() - t0) * 1000)
    assert min(timings) < 25, f"fastest render {min(timings):.1f} ms"


@pytest.mark.benchmark
def test_edgar_requests_per_ticker(fetcher):
    """P4: <= 3 fetches for a cold profile plus fundamentals.

    Before companyfacts this was nine — one per XBRL concept — against a
    published 10 requests/second limit shared with everything else on the box.
    """
    from agents_work.sources.edgar import Edgar
    from edgar_facts import nvda_facts

    fetcher.route("company_tickers.json",
                  {"0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"}})
    fetcher.route("submissions/CIK", {"filings": {"recent": {}}})
    fetcher.route("companyfacts", {"facts": nvda_facts()})

    edgar = Edgar(fetcher)
    profile = edgar.profile("NVDA")
    edgar.fundamentals(profile.cik)
    assert len(fetcher.requested) <= 3, fetcher.requested


@pytest.mark.benchmark
def test_company_facts_are_fetched_once_per_company(fetcher):
    """P4."""
    from agents_work.sources.edgar import Edgar
    from edgar_facts import nvda_facts

    fetcher.route("companyfacts", {"facts": nvda_facts()})
    edgar = Edgar(fetcher)
    edgar.fundamentals(1045810)
    edgar.fundamentals(1045810)
    assert sum("companyfacts" in u for u in fetcher.requested) == 1
