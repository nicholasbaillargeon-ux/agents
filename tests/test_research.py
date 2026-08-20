"""Research agent: the document, the grounding check, the commit."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from edgar_facts import NVDA_CORRECT_TTM, nvda_facts

from agents_work.agents import research
from agents_work.agents.research import EXPECTED_SECTIONS as EXPECTED
from agents_work.agents.base import Context
from agents_work.gitsink import NotesRepo
from agents_work.grounding import ungrounded
from agents_work.llm import FakeLLM

RSS = """<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>SPY rallies on soft inflation print</title>
<link>https://news.example/1</link><source>Example Wire</source>
<pubDate>{now}</pubDate></item>
<item><title>Analysts lift targets</title><link>https://news.example/2</link>
<source>Example Wire</source><pubDate>{now}</pubDate></item>
</channel></rss>"""

SECTIONS = """## Thesis

Revenue of $253.49B compounding at 70.7% is the whole case, and it holds only if
datacenter demand outruns the cycle.

## Risks

- A single customer concentration unwinds
- Margins revert toward 45.0% as competition lands

## Valuation context

At 20.7 times sales the multiple already assumes the growth persists.
"""


@pytest.fixture
def wired(cfg, fetcher, llm):
    """A context whose every external source is a fixture."""
    fetcher.route("company_tickers.json",
                  {"0": {"cik_str": 1045810, "ticker": "SPY", "title": "NVIDIA CORP"}})
    fetcher.route("submissions/CIK", {
        "sicDescription": "Semiconductors", "exchanges": ["Nasdaq"],
        "filings": {"recent": {
            "form": ["10-Q", "8-K"],
            "filingDate": ["2026-05-20", "2026-08-17"],
            "reportDate": ["2026-04-26", "2026-08-17"],
            "accessionNumber": ["0001045810-26-000052", "0001045810-26-000069"],
            "primaryDocument": ["nvda-20260426.htm", "nvda-20260817.htm"],
            "primaryDocDescription": ["10-Q", "8-K"]}}})
    fetcher.route("news.google.com",
                  RSS.format(now=datetime.now(timezone.utc).strftime(
                      "%a, %d %b %Y %H:%M:%S GMT")))
    fetcher.route("index.json", {"directory": {"item": []}})
    ctx = Context(cfg, llm=llm, fetcher=fetcher,
                  notes=NotesRepo(cfg.notes_repo), offline=True)
    # companyfacts is large; inject it rather than serialising a fixture file
    from agents_work.sources.edgar import Edgar
    original = Edgar.company_facts
    Edgar.company_facts = lambda self, cik: nvda_facts()
    yield ctx
    Edgar.company_facts = original
    ctx.close()


# -- structure ---------------------------------------------------------------

@pytest.mark.benchmark
def test_brief_has_the_sections_a_reader_expects(wired):
    """R7."""
    wired.llm.default_response = SECTIONS
    brief, data = research.build_brief(wired, "SPY")
    headings = [s.heading for s in brief.sections]
    assert "Snapshot" in headings
    assert "Recent filings" in headings
    assert "Headlines" in headings
    assert "Thesis" in headings and "Risks" in headings
    assert brief.sources


def test_snapshot_carries_the_corrected_fundamentals(wired):
    brief, data = research.build_brief(wired, "SPY")
    text = brief.render()
    assert "$253.49B" in text
    assert data["ps"] and 5 < data["ps"] < 60      # sane, not the 483 the bug gave
    assert "How the TTM was assembled" in text


def test_frontmatter_is_parseable_and_complete(wired):
    brief, _ = research.build_brief(wired, "SPY")
    fm = brief.frontmatter().splitlines()
    keys = {line.split(":")[0] for line in fm if ":" in line}
    assert {"title", "agent", "target", "date", "generated_at", "cik", "degraded"} <= keys


def test_dossier_names_the_revenue_tag(wired):
    wired.llm.default_response = SECTIONS
    research.build_brief(wired, "SPY")
    assert "Revenues" in wired.llm.prompts[0]
    assert "TTM built from quarters" in wired.llm.prompts[0]


# -- grounding ---------------------------------------------------------------

@pytest.mark.benchmark
def test_fabricated_figures_are_flagged(wired):
    """R6. 45.0% appears nowhere in the dossier."""
    wired.llm.default_response = SECTIONS
    brief, _ = research.build_brief(wired, "SPY")
    text = brief.render()
    assert "Unverified figures" in text
    assert "45.0%" in text.split("Unverified figures")[1]


@pytest.mark.benchmark
def test_grounded_figures_are_not_flagged(wired):
    """R6."""
    wired.llm.default_response = (
        "## Thesis\n\nRevenue of $253.49B is the case.\n\n"
        "## Risks\n\n- Concentration\n\n## Valuation context\n\nRich.\n")
    brief, _ = research.build_brief(wired, "SPY")
    assert "Unverified figures" not in brief.render()


def test_grounding_tolerates_rounding_and_units():
    dossier = "REVENUE TTM: $253,490,000,000 · net margin 0.63 · income 11,729"
    assert ungrounded("Revenue $253.5B, margin 63%, income $11,729M.", dossier) == []


def test_grounding_is_silent_without_facts():
    assert ungrounded("Revenue was $500B.", "") == []


# -- degradation -------------------------------------------------------------

def test_missing_model_leaves_the_data_sections_intact(cfg, wired):
    wired.llm = FakeLLM(cfg, available=False)
    brief, _ = research.build_brief(wired, "SPY")
    text = brief.render()
    assert "$253.49B" in text
    assert "Snapshot" in [s.heading for s in brief.sections]
    assert any("analysis sections omitted" in d for d in brief.degradations)
    assert "the language model was unavailable" in text


def test_model_prose_without_headings_is_kept(wired):
    wired.llm.default_response = "Just some prose with no headings at all."
    brief, _ = research.build_brief(wired, "SPY")
    assert "Analysis" in [s.heading for s in brief.sections]
    assert "Just some prose" in brief.render()


def test_unknown_ticker_degrades_rather_than_crashes(cfg, fetcher, llm):
    fetcher.route("company_tickers.json", {})
    ctx = Context(cfg, llm=llm, fetcher=fetcher, notes=NotesRepo(cfg.notes_repo),
                  offline=True)
    res = research.run(ctx, "NOTREAL", commit=False)
    ctx.close()
    assert res.ok                       # a thin brief is still a brief
    assert any("not in the SEC ticker file" in d for d in res.degradations)


# -- persistence -------------------------------------------------------------

def test_run_commits_to_the_notes_repo(wired, cfg):
    res = research.run(wired, "SPY", commit=True)
    assert res.ok
    assert res.data["commit"]["committed"]
    assert (cfg.notes_repo / "briefs" / res.artifact.name).is_file()
    log = NotesRepo(cfg.notes_repo).log()
    assert any("SPY: research brief" in c["subject"] for c in log)


def test_identical_content_does_not_create_an_empty_commit(wired, cfg):
    repo = NotesRepo(cfg.notes_repo)
    first = repo.commit_file("briefs/x.md", "same", "first")
    second = repo.commit_file("briefs/x.md", "same", "second")
    assert first.committed and not second.committed
    assert "unchanged" in second.message


def test_a_git_failure_never_loses_the_brief(wired, monkeypatch, cfg):
    from agents_work import gitsink
    monkeypatch.setattr(gitsink.NotesRepo, "commit_file",
                        lambda *a, **k: (_ for _ in ()).throw(gitsink.GitError("no repo")))
    res = research.run(wired, "SPY", commit=True)
    assert res.ok
    assert res.artifact.is_file()
    assert any("could not commit" in d for d in res.degradations)


def test_run_watchlist_returns_one_result_per_ticker(wired):
    results = research.run_watchlist(wired, ["SPY", "SPY"], commit=False)
    assert len(results) == 2
    assert all(r.ok for r in results)


# -- grounding precision (R6) ------------------------------------------------

@pytest.mark.benchmark
@pytest.mark.parametrize("prose,dossier,label", [
    ("sits 0.9% off its 52-week high", "PERFORMANCE: 52w_high_gap -0.9",
     "direction lives in the words, not the sign"),
    ("the Nasdaq-100 tracks the largest names", "HEADLINES: none retrieved",
     "a hyphen glued to a word is not a minus sign"),
    ("a $10,000-to-$66,000 decade return", "HEADLINE: $10,000 to $66,000 over a decade",
     "a range is two numbers, not a negative"),
    ("the S&P 500 rallied and 22 stocks hit highs", "PRICE: last 762.60",
     "bare integers are index names and counts"),
    ("revenue fell 6.8% year on year", "REVENUE growth -6.8%", "a fall stated as positive"),
])
def test_grounding_does_not_cry_wolf(prose, dossier, label):
    """R6: a warning that fires on a correct brief is one nobody reads."""
    assert ungrounded(prose, dossier) == [], label


@pytest.mark.benchmark
@pytest.mark.parametrize("prose,dossier,expected", [
    ("gross margin was 74.2%", "REVENUE: $253.49B, net margin 63.0%", ["74.2%"]),
    ("a $500B backlog", "REVENUE TTM: $253.49B", ["$500B"]),
    ("the 10y sits at 4.7%", "PRICE: last 82.34, 1y +1.4%", ["4.7%"]),
    ("margins compressed 240bp", "NET MARGIN: 63.0%", ["240bp"]),
])
def test_grounding_still_catches_what_matters(prose, dossier, expected):
    """R6: the loosening must not cost the catches."""
    assert ungrounded(prose, dossier) == expected


def test_a_figure_only_in_a_headline_is_grounded():
    dossier = ("PRICE: last 710.93\nHEADLINES (most recent first):\n"
               "  - [2h ago] Nasdaq slides as WMT drops 6.4% on guidance (Reuters)")
    assert ungrounded("WMT fell 6.4%, dragging the index.", dossier) == []


@pytest.mark.benchmark
def test_every_performance_figure_the_model_sees_is_on_the_page(wired):
    """R8. The dossier carried the 52-week-high gap and the Snapshot did not, so
    a brief could cite "4.7% below its 52-week high" with nothing on the page to
    check it against. A figure the reader cannot verify is worse than one that
    was never offered."""
    wired.llm.default_response = SECTIONS
    brief, d = research.build_brief(wired, "SPY")
    dossier = wired.llm.prompts[0]
    snapshot = next(s.body for s in brief.sections if s.heading == "Snapshot")

    assert d["perf"], "fixture produced no performance figures to check"
    for key, value in d["perf"].items():
        assert f"{value:+.1f}" in snapshot or f"{value:.1f}" in snapshot, (
            f"{key} ({value:.1f}) is in the dossier but not the Snapshot table")
        assert f"{value:.1f}" in dossier or f"{value:+.1f}" in dossier


@pytest.mark.benchmark
@pytest.mark.parametrize("prose,dossier,expected,why", [
    ("70%+ revenue growth", "REVENUE growth YoY +70.7%", [], "a true lower bound"),
    ("about $250B of revenue", "REVENUE TTM: $253.49B", [], "the same number said out loud"),
    ("roughly 63% net margin", "NET MARGIN: 63.0%", [], "exact, stated coarsely"),
    ("gross margin was 74.2%", "NET MARGIN: 63.0%", ["74.2%"], "ten bands away"),
    ("a $500B backlog", "REVENUE TTM: $253.49B", ["$500B"], "sixty bands away"),
    ("the 10y sits at 4.7%", "PRICE: last 82.34, 1y +1.4%", ["4.7%"], "imported figure"),
])
def test_prose_restates_rather_than_transcribes(prose, dossier, expected, why):
    """R6: a brief that rounds for readability is not a brief that invented a
    number, and the check has to tell those apart to be worth reading."""
    assert ungrounded(prose, dossier) == expected, why


@pytest.mark.benchmark
def test_a_brief_missing_a_section_says_so(wired):
    """R9 (regression). A real MSFT brief shipped with Thesis only — no Risks, no
    Valuation context — and `degraded: false`. It looked finished."""
    wired.llm.default_response = "## Thesis\n\nThe case is the datacenter backlog."
    brief, _ = research.build_brief(wired, "SPY")
    headings = [s.heading for s in brief.sections]
    assert "Risks" not in headings
    assert any("analysis incomplete" in d for d in brief.degradations)
    assert "Risks, Valuation context" in " ".join(brief.degradations)
    assert "degraded: true" in brief.render()


@pytest.mark.benchmark
def test_a_truncated_reply_is_named_as_truncated(wired):
    """R9: the two causes need different fixes, so the brief distinguishes them."""
    wired.llm.default_response = "## Thesis\n\nThe case is the backlog."
    wired.llm.last_stop_reason = "max_tokens"
    brief, _ = research.build_brief(wired, "SPY")
    assert any("cut off at the token limit" in d for d in brief.degradations)


def test_a_complete_brief_is_not_flagged(wired):
    wired.llm.default_response = SECTIONS
    brief, _ = research.build_brief(wired, "SPY")
    assert not any("analysis incomplete" in d for d in brief.degradations)


def test_unheaded_prose_is_not_treated_as_missing_sections(wired):
    """The model ignoring headings entirely is already handled by keeping the
    prose under 'Analysis'; it must not also be reported as three losses."""
    wired.llm.default_response = "Just prose, no headings at all."
    brief, _ = research.build_brief(wired, "SPY")
    assert "Analysis" in [s.heading for s in brief.sections]
    assert not any("analysis incomplete" in d for d in brief.degradations)


@pytest.mark.benchmark
def test_a_truncated_reply_is_retried_once_with_more_room(wired, cfg):
    """R9. The endpoint's thinking block counts against max_tokens and its length
    varies run to run, so the same ticker fits one morning and not the next."""
    from agents_work.agents.research import SECTION_TOKENS

    class Truncating(FakeLLM):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.budgets = []

        def complete(self, prompt, **kw):
            self.budgets.append(kw.get("max_tokens"))
            self.last_stop_reason = "max_tokens" if len(self.budgets) == 1 else "end_turn"
            return ("## Thesis\n\nCut off." if len(self.budgets) == 1 else SECTIONS)

    wired.llm = Truncating(cfg)
    brief, _ = research.build_brief(wired, "SPY")
    assert wired.llm.budgets == [SECTION_TOKENS, SECTION_TOKENS * 2]
    assert [s.heading for s in brief.sections if s.heading in EXPECTED] == list(EXPECTED)
    assert not any("analysis incomplete" in d for d in brief.degradations)


@pytest.mark.benchmark
def test_a_second_truncation_is_reported_not_retried_forever(wired, cfg):
    """R9: one retry, then say so."""
    class AlwaysTruncating(FakeLLM):
        def complete(self, prompt, **kw):
            self.prompts.append(prompt)
            self.last_stop_reason = "max_tokens"
            return "## Thesis\n\nStill cut off."

    wired.llm = AlwaysTruncating(cfg)
    brief, _ = research.build_brief(wired, "SPY")
    assert len(wired.llm.prompts) == 2
    assert any("cut off at the token limit" in d for d in brief.degradations)
