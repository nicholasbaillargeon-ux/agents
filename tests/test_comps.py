"""The comps engine: the EV bridge, the multiples, and the blank cells.

The rule these gates protect is that a peer whose filings do not support a
multiple gets an empty cell, never a plugged number. One invented cell in a
comps table is invisible and moves the median everyone reads off it.
"""

from __future__ import annotations

from datetime import date

import pytest

from agents_work.agents import comps
from agents_work.agents.comps import CompRow, median_of
from agents_work.sources.edgar import Fundamentals



def _row(**kw) -> CompRow:
    base = dict(ticker="AAA", price=100.0, shares=1e9, market_cap=100e9,
                enterprise_value=110e9, revenue_ttm=50e9, ebitda_ttm=10e9,
                net_income_ttm=5e9, eps_ttm=5.0)
    base.update(kw)
    return CompRow(**base)


def test_the_multiples_are_the_ratios_they_claim_to_be():
    r = _row()
    assert r.ev_sales == pytest.approx(2.2)
    assert r.ev_ebitda == pytest.approx(11.0)
    assert r.pe == pytest.approx(20.0)
    assert r.ps == pytest.approx(2.0)


@pytest.mark.benchmark
def test_a_negative_ebitda_yields_no_multiple_rather_than_a_negative_one():
    """C4. A company losing money is not trading at -8x. The cell is blank."""
    assert _row(ebitda_ttm=-10e9).ev_ebitda is None
    assert _row(eps_ttm=-2.0).pe is None


def test_a_missing_input_blanks_the_cell_rather_than_zeroing_it():
    r = _row(enterprise_value=None, market_cap=None)
    assert r.ev_sales is None and r.ev_ebitda is None and r.ps is None


@pytest.mark.benchmark
def test_the_median_is_taken_over_the_peers_that_have_the_metric():
    """C3. (regression) A blank is not a zero. Averaging blanks in drags the peer
    median toward a multiple no peer trades at."""
    rows = [_row(ticker="A", ebitda_ttm=10e9),      # 11.0x
            _row(ticker="B", ebitda_ttm=5e9),       # 22.0x
            _row(ticker="C", ebitda_ttm=None)]      # blank
    assert median_of(rows, "ev_ebitda") == pytest.approx(16.5)


def test_a_column_nobody_reports_has_no_median():
    assert median_of([_row(ebitda_ttm=None), _row(ebitda_ttm=None)], "ev_ebitda") is None


def test_the_enterprise_value_bridge_adds_debt_and_nets_cash():
    f = Fundamentals(total_debt=30e9, cash=10e9, short_term_investments=5e9,
                     preferred=1e9, minority_interest=2e9)
    assert f.enterprise_value(100e9) == pytest.approx(100e9 + 30e9 + 1e9 + 2e9 - 15e9)


@pytest.mark.benchmark
def test_unknown_debt_gives_no_enterprise_value_rather_than_a_market_cap():
    """C2. (regression) An EV that silently treats unknown debt as zero is a market
    cap wearing a different label, and it lands in the table looking
    authoritative."""
    assert Fundamentals(total_debt=None, cash=10e9).enterprise_value(100e9) is None


def test_no_market_cap_means_no_enterprise_value():
    assert Fundamentals(total_debt=30e9).enterprise_value(None) is None


def test_ebitda_is_operating_income_plus_da_and_nothing_else():
    assert Fundamentals(operating_income_ttm=80e9,
                        depreciation_amortization_ttm=20e9).ebitda_ttm == 100e9


@pytest.mark.benchmark
@pytest.mark.parametrize("oi,da", [(None, 20e9), (80e9, None), (None, None)])
def test_ebitda_is_never_approximated_from_a_missing_leg(oi, da):
    """C1. A number that is EBITDA-shaped and wrong is worse than a blank cell,
    because the multiple built on it looks perfectly reasonable."""
    assert Fundamentals(operating_income_ttm=oi,
                        depreciation_amortization_ttm=da).ebitda_ttm is None


def test_net_cash_is_negative_when_the_company_is_levered():
    assert Fundamentals(total_debt=50e9, cash=20e9).net_cash == pytest.approx(-30e9)


def test_a_curated_set_resolves_without_a_model(ctx):
    tickers, note = comps.resolve_peers(ctx, "megacap tech")
    assert tickers == comps.PEER_SETS["megacap-tech"]
    assert "curated" in note
    assert ctx.llm.prompts == []      # no model call was needed


@pytest.mark.benchmark
def test_a_model_resolved_ticker_is_checked_against_the_sec_file(ctx, fetcher):
    """C5. A model asked for mid-cap asset managers will happily produce a
    plausible ticker belonging to something else. A wrong constituent is far
    harder to spot in a finished table than a missing one."""
    fetcher.route("company_tickers.json",
                  {"0": {"cik_str": 1, "ticker": "TROW", "title": "T Rowe Price"},
                   "1": {"cik_str": 2, "ticker": "BEN", "title": "Franklin Resources"}})
    ctx.llm.default_response = (
        '{"tickers": ["TROW", "BEN", "NOTREAL"], "note": "mid-cap managers"}')
    tickers, note = comps.resolve_peers(ctx, "mid-cap asset managers")
    assert tickers == ["TROW", "BEN"]
    assert "NOTREAL" in note and "not in the SEC ticker file" in note


def test_the_table_carries_a_median_row_and_renders_blanks_as_dashes():
    body = comps.comps_table([_row(ticker="A"), _row(ticker="B", ebitda_ttm=None)])
    assert "**Median**" in body
    assert "—" in body


@pytest.mark.benchmark
def test_every_median_reports_how_many_peers_it_rests_on():
    """C6 (regression). The advisory set returns one EV/EBITDA out of seven
    names, and a bold "12.6x" in a median row reads like a sector multiple
    rather than the single filer it is."""
    rows = [_row(ticker="A", ebitda_ttm=10e9),
            _row(ticker="B", ebitda_ttm=None),
            _row(ticker="C", ebitda_ttm=None)]
    assert comps.coverage(rows, "ev_ebitda") == 1
    assert comps.coverage(rows, "pe") == 3
    body = comps.comps_table(rows)
    assert "peers with the metric" in body
    assert "_1/3_" in body and "_3/3_" in body


def test_an_empty_peer_set_still_renders():
    assert "No data" in comps.comps_table([])
