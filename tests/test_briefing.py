"""Market open briefing: readable in thirty seconds, honest when the tape is dark."""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from datetime import date

import pytest

from agents_work.agents import briefing
from agents_work.agents.briefing import build_brief, earnings_today
from agents_work.sources.news import Headline
from agents_work.sources.prices import FUTURES, MACRO, PriceSource, Quote

THURSDAY = date(2026, 8, 20)
SATURDAY = date(2026, 8, 22)


def fake_yfinance(calendars: dict) -> types.ModuleType:
    mod = types.ModuleType("yfinance")

    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        @property
        def calendar(self):
            entry = calendars.get(self.symbol)
            if isinstance(entry, Exception):
                raise entry
            return entry

    mod.Ticker = Ticker
    return mod


# -- M1: everything dead -----------------------------------------------------

@pytest.mark.benchmark
def test_briefing_renders_with_every_source_dead(offline_ctx, cfg, tmp_path):
    """M1. A brief that fails to render is a brief nobody reads at 8am."""
    offline_ctx.cfg = replace(cfg, lake_dir=tmp_path / "no-lake")
    brief, data = build_brief(offline_ctx, today=THURSDAY)
    text = brief.render()

    for heading in ("Futures", "Macro", "Reporting today"):
        assert heading in text
    assert "n/a" in text
    assert brief.degradations
    assert "degraded: true" in text
    assert all(not q.ok for q in data["futures"])


def test_briefing_uses_stored_closes_when_quotes_are_unavailable(offline_ctx):
    brief, data = build_brief(offline_ctx, watchlist=["SPY", "QQQ"], today=THURSDAY)
    movers = {q.symbol: q for q in data["movers"]}
    assert movers["SPY"].last is not None
    assert "stale" in movers["SPY"].as_of
    assert "last stored close" in movers["SPY"].error


# -- M2: non-trading days ----------------------------------------------------

@pytest.mark.benchmark
def test_weekend_briefing_is_labelled(offline_ctx):
    """M2."""
    brief, _ = build_brief(offline_ctx, today=SATURDAY)
    assert any("closed today" in d for d in brief.degradations)
    assert "Saturday 22 August 2026" in brief.title


def test_weekday_briefing_is_not_labelled_closed(offline_ctx):
    brief, _ = build_brief(offline_ctx, today=THURSDAY)
    assert not any("closed today" in d for d in brief.degradations)


# -- M3: earnings calendar ---------------------------------------------------

@pytest.mark.benchmark
def test_only_todays_earnings_are_listed(monkeypatch):
    """M3."""
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance({
        "AAPL": {"Earnings Date": [THURSDAY]},
        "MSFT": {"Earnings Date": [date(2026, 8, 21)]},
        "NVDA": {"Earnings Date": []},
    }))
    rows, note = earnings_today(["AAPL", "MSFT", "NVDA"], today=THURSDAY)
    assert [r[0] for r in rows] == ["AAPL"]
    assert note is None


def test_a_broken_symbol_does_not_kill_the_calendar(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance({
        "AAPL": {"Earnings Date": [THURSDAY]},
        "BAD": RuntimeError("delisted"),
    }))
    rows, note = earnings_today(["AAPL", "BAD"], today=THURSDAY)
    assert [r[0] for r in rows] == ["AAPL"]
    assert "unavailable for 1/2" in note


def test_estimated_dates_are_marked(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance({
        "AAPL": {"Earnings Date": [THURSDAY, date(2026, 8, 25)]}}))
    rows, _ = earnings_today(["AAPL"], today=THURSDAY)
    assert rows[0][2] == "estimated"


def test_offline_skips_the_calendar_rather_than_hanging(offline_ctx):
    brief, data = build_brief(offline_ctx, today=THURSDAY)
    assert data["earnings"] == []
    assert any("earnings calendar not checked" in d for d in brief.degradations)


# -- M4: movers --------------------------------------------------------------

@pytest.mark.benchmark
def test_movers_rank_by_absolute_move(monkeypatch):
    """M4: a 4% fall is more interesting than a 1% rise."""
    quotes = [Quote("A", last=101, prev_close=100), Quote("B", last=96, prev_close=100),
              Quote("C", last=100.5, prev_close=100), Quote("D", last=None)]
    monkeypatch.setattr(PriceSource, "quotes", lambda self, syms: quotes)
    ranked = PriceSource().movers(["A", "B", "C", "D"], top=3)
    assert [q.symbol for q in ranked] == ["B", "A", "C"]


# -- structure ---------------------------------------------------------------

def test_brief_covers_every_futures_and_macro_instrument(offline_ctx):
    _, data = build_brief(offline_ctx, today=THURSDAY)
    assert {q.symbol for q in data["futures"]} == set(FUTURES)
    assert {q.symbol for q in data["macro"]} == set(MACRO)


def test_lede_is_written_when_the_model_is_available(offline_ctx, ctx, cfg):
    ctx.offline = True
    ctx.llm.default_response = "Futures are flat. Nothing overnight moved the tape."
    brief, _ = build_brief(ctx, today=THURSDAY)
    assert "Overnight" in [s.heading for s in brief.sections]
    assert "Futures are flat" in brief.render()


def test_lede_is_skipped_without_a_model(offline_ctx):
    brief, _ = build_brief(offline_ctx, today=THURSDAY)
    assert "Overnight" not in [s.heading for s in brief.sections]


def test_run_writes_records_and_summarises(offline_ctx):
    res = briefing.run(offline_ctx, today=THURSDAY, commit=False)
    assert res.ok
    assert res.artifact.is_file()
    assert "futures priced" in res.summary


def test_failure_is_recorded_not_raised(offline_ctx, monkeypatch):
    monkeypatch.setattr(briefing, "build_brief",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    res = briefing.run(offline_ctx, commit=False)
    assert not res.ok
    assert "boom" in res.error


# -- M5: units ---------------------------------------------------------------

@pytest.mark.benchmark
def test_yields_are_reported_in_basis_points(offline_ctx):
    """M5 (regression). Given "US 10y yield: +0.92%" a model opened the brief
    with "yields spiked 92 basis points overnight". The move was four."""
    from agents_work.agents.briefing import change_label
    q = Quote("^TNX", last=4.70, prev_close=4.66, label="US 10y yield")
    assert change_label(q) == "▲ +4bp"
    assert "%" not in change_label(q)


@pytest.mark.benchmark
def test_non_yield_instruments_stay_in_percent():
    """M5."""
    from agents_work.agents.briefing import change_label
    assert change_label(Quote("SPY", last=101, prev_close=100)) == "▲ +1.00%"
    assert change_label(Quote("^VIX", last=110, prev_close=100)) == "▲ +10.00%"


def test_change_label_handles_a_dead_quote():
    from agents_work.agents.briefing import change_label
    assert change_label(Quote("^TNX", last=None)) == "n/a"
    assert change_label(Quote("SPY", last=None)) == "n/a"


TAPE = {
    "futures": [Quote("ES=F", last=7667.75, prev_close=7738.9, label="S&P 500 futures")],
    "macro": [Quote("^TNX", last=4.70, prev_close=4.66, label="US 10y yield"),
              Quote("^VIX", last=16.01, prev_close=14.89, label="VIX")],
    "movers": [],
    "headlines": [Headline("Wall Street sinks as bond yields rise, Walmart results "
                           "disappoint", "https://e/1", "Reuters", None),
                  Headline("Crypto Stocks Surge as MSTR Gains Over 10%",
                           "https://e/2", "Wire", None)],
}


# -- M6: the lede is checked against its own inputs --------------------------

@pytest.mark.benchmark
def test_a_lede_figure_absent_from_tape_and_headlines_is_flagged(ctx):
    """M6. The tables are assembled from fetched data and cannot be wrong about a
    number. The lede is the only model-written part of this brief, so it is the
    only part that can be."""
    ctx.llm.default_response = (
        "Equities fell with the VIX up 7.5%. Breadth was poor, with 68.4% of the "
        "S&P declining and the 10y at 5.90%.")
    text, flagged = briefing._lede(ctx, TAPE)
    assert "7.5%" in text
    assert flagged == ["68.4%", "5.90%"]


@pytest.mark.benchmark
def test_the_caveat_lands_under_the_lede(ctx, offline_ctx, monkeypatch):
    """M6: an unverified figure has to be visible next to the sentence using it."""
    monkeypatch.setattr(briefing, "_lede",
                        lambda c, d: ("Breadth was poor, 68.4% declining.", ["68.4%"]))
    brief, _ = build_brief(offline_ctx, today=THURSDAY)
    text = brief.render()
    assert "Not found in this morning's tape or headlines: `68.4%`" in text
    assert text.index("68.4% declining") < text.index("Not found in this morning")
    assert text.index("Not found in this morning") < text.index("## Futures")
    assert brief.extra_meta["ungrounded_figures"] == 1


@pytest.mark.benchmark
def test_a_lede_grounded_in_the_headlines_is_not_flagged(ctx):
    """M6. Attributing a selloff to Walmart is the lede doing its job when a
    Reuters headline says so; flagging it would train the reader to ignore the
    warning."""
    ctx.llm.default_response = (
        "Equities are selling off, with the VIX up 7.5% and yields 4bp higher, on "
        "disappointing Walmart results. Crypto is the outlier, MSTR up 10%.")
    _, flagged = briefing._lede(ctx, TAPE)
    assert flagged == [], f"false positives: {flagged}"


@pytest.mark.benchmark
def test_the_lede_check_uses_the_headlines_not_just_the_tape(ctx):
    """M6: a figure that appears only in a headline is grounded."""
    ctx.llm.default_response = "MSTR gained 10.0% overnight."
    _, flagged = briefing._lede(ctx, TAPE)
    assert flagged == []
    ctx.llm.default_response = "MSTR gained 33.7% overnight."
    _, flagged = briefing._lede(ctx, TAPE)
    assert flagged == ["33.7%"]


def test_no_model_means_no_lede_and_no_caveat(offline_ctx):
    text, flagged = briefing._lede(offline_ctx, TAPE)
    assert text == "" and flagged == []
    brief, _ = build_brief(offline_ctx, today=THURSDAY)
    assert "Not found in this morning's tape" not in brief.render()
    assert "ungrounded_figures" not in brief.extra_meta


def test_a_clean_lede_adds_no_metadata_noise(offline_ctx, monkeypatch):
    monkeypatch.setattr(briefing, "_lede",
                        lambda c, d: ("Futures are lower. Nothing explains it.", []))
    brief, _ = build_brief(offline_ctx, today=THURSDAY)
    assert "ungrounded_figures" not in brief.extra_meta
    assert "Not found in this morning" not in brief.render()


def test_the_lede_prompt_carries_levels_and_units(ctx):
    """The model gets "now 4.70, +4bp", not a bare percentage it has to guess at."""
    ctx.llm.default_response = "Quiet."
    briefing._lede(ctx, {
        "futures": [Quote("ES=F", last=7660.5, prev_close=7738.6, label="S&P 500 futures")],
        "macro": [Quote("^TNX", last=4.70, prev_close=4.66, label="US 10y yield")],
        "movers": [], "headlines": [],
    })
    prompt = ctx.llm.prompts[0]
    assert "US 10y yield: now 4.70, +4bp on the session" in prompt
    assert "S&P 500 futures: now 7,660.50, -1.01% on the session" in prompt
