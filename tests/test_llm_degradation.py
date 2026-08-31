"""An LLM failure must reach the reader.

The bug these cover: for weeks every model call 404'd on a dead model name and
every brief still said `degraded: false`. The agents caught the error, logged a
warning, and shipped a document with its analysis sections missing and nothing
on the page saying so. Green systemd unit, green dashboard, hollow brief.

So these assert the *reporting*, not the recovery. Degrading is already correct
behaviour; degrading silently is the defect.
"""

from __future__ import annotations

import pytest

from agents_work.agents import briefing, scout
from test_scout import REGISTRY, wire
from agents_work.llm import FakeLLM, LLMUnavailable


def _dead(cfg, reason="model 'test-fast' not served by https://llm.invalid: 404"):
    """An LLM that is configured and reachable but fails every call.

    Distinct from `available=False`, which the briefs already reported. The
    outage that went unnoticed looked healthy right up to the call.
    """
    return FakeLLM(cfg, responses=[LLMUnavailable(reason)] * 50)


class TestLLMRecordsItsOwnFailures:
    def test_complete_records_before_raising(self, cfg):
        llm = _dead(cfg)
        with pytest.raises(LLMUnavailable):
            llm.complete("hi")
        assert llm.failures, "the exception was raised but left no trace"
        assert llm.degradations and "LLM call failed" in llm.degradations[0]

    def test_json_records_even_though_it_swallows(self, cfg):
        """`json()` returns a default instead of raising — the silent path."""
        llm = _dead(cfg)
        assert llm.json("rank these", default=[]) == []
        assert llm.failures, "json() swallowed the failure without recording it"

    def test_unparseable_output_counts_as_a_failure(self, cfg):
        """A call that succeeds and returns garbage loses the section too."""
        llm = FakeLLM(cfg, responses=["not json at all, just prose"])
        assert llm.json("rank these", default=[]) == []
        assert any("unparseable" in f for f in llm.failures)

    def test_failures_are_deduplicated(self, cfg):
        llm = _dead(cfg)
        for _ in range(5):
            llm.json("x", default=[])
        assert len(llm.failures) == 1

    def test_healthy_llm_reports_nothing(self, cfg):
        llm = FakeLLM(cfg)
        llm.complete("hi")
        assert llm.failures == []
        assert llm.degradations == []


class TestBriefingSurfacesIt:
    def test_dead_llm_marks_the_brief_degraded(self, ctx, cfg):
        """Isolate the LLM cause: a briefing degrades for unrelated reasons too
        (thin fixtures, weekends), so asserting the list is merely non-empty
        passes against the very bug this file exists to catch."""
        healthy, _ = briefing.build_brief(ctx)
        ctx.llm = _dead(cfg)
        dead, _ = briefing.build_brief(ctx)
        new = set(dead.degradations) - set(healthy.degradations)
        assert new, "the outage added no degradation the healthy run lacked"
        assert "degraded: true" in dead.frontmatter()

    def test_the_reader_is_told_the_lede_is_missing(self, ctx, cfg):
        ctx.llm = _dead(cfg)
        brief, _ = briefing.build_brief(ctx)
        assert any("lede" in d for d in brief.degradations)
        assert "Ran degraded." in brief.render()

    def test_no_overnight_section_is_silently_dropped(self, ctx, cfg):
        ctx.llm = _dead(cfg)
        brief, _ = briefing.build_brief(ctx)
        assert not any(s.heading == "Overnight" for s in brief.sections)
        # The tables are assembled from fetched data and must survive.
        assert any(s.heading == "Futures" for s in brief.sections)

    def test_healthy_run_is_not_marked_degraded_by_this(self, ctx):
        brief, _ = briefing.build_brief(ctx)
        assert not any("lede" in d or "LLM call failed" in d
                       for d in brief.degradations)


class TestScoutSurfacesIt:
    def test_ranking_outage_is_reported(self, ctx, cfg, fetcher):
        """`rank` swallows failures and returns a short dict; say so.

        Needs a registry that actually yields postings — with nothing to rank
        there is no model call to fail, and the test would pass vacuously.
        """
        wire(fetcher)
        ctx.llm = _dead(cfg)
        brief, data = scout.build_brief(ctx, registry=REGISTRY, use_llm=True)
        assert data["new"], "fixture produced nothing to rank"
        assert any("unranked" in d for d in brief.degradations), brief.degradations
        assert "degraded: true" in brief.frontmatter()

    def test_a_healthy_ranking_run_is_not_flagged(self, ctx, fetcher):
        wire(fetcher)
        ctx.llm.default_response = (
            '[{"url": "https://jobs.ashbyhq.com/aico/1", "verdict": "apply", "why": "fit"}]')
        brief, _ = scout.build_brief(ctx, registry=REGISTRY, use_llm=True)
        assert not any("LLM call failed" in d for d in brief.degradations)


class TestFinalizeIsWired:
    """The safety net: even an agent that forgets to degrade cannot ship clean."""

    def test_finalize_folds_llm_failures_into_the_brief(self, ctx, cfg):
        from agents_work.agents.base import AgentResult, finalize
        from agents_work.brief import Brief

        ctx.llm = _dead(cfg)
        ctx.llm.json("x", default=[])          # a failure nobody handled
        brief, res = Brief(title="t", agent="a"), AgentResult(agent="a")
        assert not brief.degradations
        finalize(ctx, brief, res)
        assert brief.degradations, "finalize did not surface the recorded failure"
        assert res.degradations == brief.degradations

    def test_finalize_is_a_noop_when_the_llm_is_healthy(self, ctx):
        from agents_work.agents.base import AgentResult, finalize
        from agents_work.brief import Brief

        brief, res = Brief(title="t", agent="a"), AgentResult(agent="a")
        finalize(ctx, brief, res)
        assert brief.degradations == []
