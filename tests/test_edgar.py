"""EDGAR fundamentals — the numbers every research brief is built on."""

from __future__ import annotations

from datetime import date

import pytest
from edgar_facts import (B, NVDA_CORRECT_TTM, NVDA_DERIVED_Q4_FY26, NVDA_GAP_JUMPING_TTM,
                         NVDA_PERIODS, STALE_PERIODS, durations, facts, instants, nvda_facts)

from agents_work.sources.edgar import Edgar, Filing, derive_quarters, html_to_text


@pytest.fixture
def edgar(fetcher):
    return Edgar(fetcher)


def _load(edgar, payload, cik=1045810):
    edgar._facts_cache[cik] = payload
    return cik


# -- quarter reconstruction --------------------------------------------------

def test_derive_quarters_keeps_explicit_quarters():
    quarters = derive_quarters(durations(NVDA_PERIODS))
    assert (date(2025, 7, 28), date(2025, 10, 26)) in quarters


def test_derive_quarters_reconstructs_missing_fiscal_q4():
    """The Q4 that only exists as (full year - nine months)."""
    quarters = derive_quarters(durations(NVDA_PERIODS))
    q4 = quarters[(date(2025, 10, 27), date(2026, 1, 25))]
    assert q4 == pytest.approx(NVDA_DERIVED_Q4_FY26)


def test_derive_quarters_ignores_annual_and_ytd_durations():
    quarters = derive_quarters(durations(NVDA_PERIODS))
    assert all(60 <= (e - s).days <= 100 for s, e in quarters)


def test_derive_quarters_prefers_the_latest_restatement():
    points = [{"start": "2025-01-27", "end": "2025-04-27", "val": 1.0, "filed": "2025-05-01"},
              {"start": "2025-01-27", "end": "2025-04-27", "val": 2.0, "filed": "2025-11-01"}]
    assert list(derive_quarters(points).values()) == [2.0]


def test_derive_quarters_survives_malformed_facts():
    points = [{"start": "not-a-date", "end": "2025-04-27", "val": 1.0},
              {"start": "2025-01-27", "end": "2025-04-27", "val": "n/a"},
              {"end": "2025-04-27", "val": 3.0},
              {"start": "2025-01-27", "end": "2025-04-27", "val": 5.0}]
    assert list(derive_quarters(points).values()) == [5.0]


# -- TTM ---------------------------------------------------------------------

@pytest.mark.benchmark
def test_ttm_sums_four_contiguous_quarters_across_a_fiscal_q4_hole():
    """R2 (regression): the gap-jumping answer is 24B too low and looks fine."""
    value, periods = Edgar._ttm(durations(NVDA_PERIODS))
    assert value == pytest.approx(NVDA_CORRECT_TTM)
    assert value != pytest.approx(NVDA_GAP_JUMPING_TTM)
    assert len(periods) == 4
    assert periods[0].endswith("2026-04-26")


@pytest.mark.benchmark
def test_ttm_periods_are_contiguous():
    """R1: no overlap, no holes — the window really is twelve months."""
    _, periods = Edgar._ttm(durations(NVDA_PERIODS))
    bounds = [tuple(date.fromisoformat(x) for x in p.split("..")) for p in periods]
    for newer, older in zip(bounds, bounds[1:]):
        assert 0 <= (newer[0] - older[1]).days <= 5
    span = (bounds[0][1] - bounds[-1][0]).days
    assert 355 <= span <= 375, f"TTM spans {span} days"


@pytest.mark.benchmark
def test_ttm_refuses_an_unfillable_gap():
    """R4: a missing quarter with no YTD to reconstruct it yields None."""
    holed = [p for p in NVDA_PERIODS
             if p[1] not in ("2026-01-25", "2025-10-26")] + [("2025-07-28", "2025-10-26",
                                                              NVDA_PERIODS[3][2])]
    value, periods = Edgar._ttm(durations(holed))
    assert value is None
    assert len(periods) < 4


def test_ttm_never_double_counts_an_overlapping_annual_fact():
    value, _ = Edgar._ttm(durations(NVDA_PERIODS))
    assert value < 2 * NVDA_CORRECT_TTM


def test_ttm_before_gives_the_prior_year_window():
    _, periods = Edgar._ttm(durations(NVDA_PERIODS))
    window_start = date.fromisoformat(periods[-1].split("..")[0])
    older, _ = Edgar._ttm(durations(NVDA_PERIODS), before=window_start)
    assert older is None or older < NVDA_CORRECT_TTM


# -- tag selection -----------------------------------------------------------

@pytest.mark.benchmark
def test_stale_tag_never_wins(edgar):
    """R3 (regression): EDGAR keeps serving tags a filer abandoned years ago."""
    cik = _load(edgar, nvda_facts(with_stale_tag=True))
    f = edgar.fundamentals(cik, today=date(2026, 8, 20))
    assert f.revenue_tag == "Revenues"
    assert f.revenue_ttm == pytest.approx(NVDA_CORRECT_TTM)


def test_only_stale_data_is_reported_as_stale(edgar):
    cik = _load(edgar, facts(
        RevenueFromContractWithCustomerExcludingAssessedTax=durations(STALE_PERIODS)))
    f = edgar.fundamentals(cik, today=date(2026, 8, 20))
    assert f.revenue_ttm is not None          # it is still a real TTM
    assert any("days ago" in n for n in f.notes)


def test_no_revenue_tag_is_admitted(edgar):
    cik = _load(edgar, facts(Assets=instants([("2026-06-30", 10 * B)])))
    f = edgar.fundamentals(cik, today=date(2026, 8, 20))
    assert f.revenue_ttm is None
    assert any("total-revenue tag" in n for n in f.notes)


def test_missing_company_facts_is_a_note_not_a_crash(edgar):
    f = edgar.fundamentals(999999, today=date(2026, 8, 20))
    assert f.revenue_ttm is None
    assert any("company facts unavailable" in n for n in f.notes)


# -- share counts ------------------------------------------------------------

@pytest.mark.benchmark
def test_multi_class_share_count_is_refused(edgar):
    """R5 (regression): one price cannot value two share classes."""
    cik = _load(edgar, facts(dei__EntityCommonStockSharesOutstanding=instants(
        [("2026-05-20", 941_481), ("2026-05-20", 1_300_000_000)])))
    f = edgar.fundamentals(cik, today=date(2026, 8, 20))
    assert f.shares is None
    assert any("share classes" in n for n in f.notes)


@pytest.mark.benchmark
def test_stale_share_count_is_refused(edgar):
    """R5 (regression): Berkshire's cover-page count stops in 2011."""
    cik = _load(edgar, facts(dei__EntityCommonStockSharesOutstanding=instants(
        [("2011-04-29", 941_481)])))
    f = edgar.fundamentals(cik, today=date(2026, 8, 20))
    assert f.shares is None
    assert any("no current share count" in n for n in f.notes)


def test_share_count_falls_through_to_us_gaap(edgar):
    cik = _load(edgar, facts(
        dei__EntityCommonStockSharesOutstanding=instants([("2011-04-29", 941_481)]),
        CommonStockSharesOutstanding=instants([("2026-05-20", 24.2e9)])))
    f = edgar.fundamentals(cik, today=date(2026, 8, 20))
    assert f.shares == pytest.approx(24.2e9)


def test_derived_ratios(edgar):
    cik = _load(edgar, nvda_facts())
    f = edgar.fundamentals(cik, today=date(2026, 8, 20))
    assert f.net_margin == pytest.approx(0.63, abs=1e-6)
    assert f.revenue_growth is None or -1 < f.revenue_growth < 5


# -- profile and documents ---------------------------------------------------

def test_profile_reports_an_unknown_ticker(edgar, fetcher):
    fetcher.route("company_tickers.json", {"0": {"cik_str": 320193, "ticker": "AAPL",
                                                 "title": "Apple Inc."}})
    prof = edgar.profile("NOTATICKER")
    assert not prof.found
    assert any("not in the SEC ticker file" in n for n in prof.notes)


def test_profile_filters_to_interesting_forms(edgar, fetcher):
    fetcher.route("company_tickers.json", {"0": {"cik_str": 1, "ticker": "T", "title": "T Inc"}})
    fetcher.route("submissions/CIK", {
        "sicDescription": "Software", "exchanges": ["Nasdaq"],
        "filings": {"recent": {
            "form": ["10-K", "4", "10-Q"],
            "filingDate": ["2026-02-01", "2026-02-02", "2026-05-01"],
            "reportDate": ["2025-12-31", "", "2026-03-31"],
            "accessionNumber": ["0000001-26-000001", "0000001-26-000002", "0000001-26-000003"],
            "primaryDocument": ["a.htm", "b.htm", "c.htm"],
            "primaryDocDescription": ["10-K", "Form 4", "10-Q"]}}})
    prof = edgar.profile("T")
    assert [f.form for f in prof.filings] == ["10-K", "10-Q"]
    assert prof.exchange == "Nasdaq"


def test_unexpected_submissions_shape_is_survived(edgar, fetcher):
    fetcher.route("company_tickers.json", {"0": {"cik_str": 1, "ticker": "T", "title": "T Inc"}})
    fetcher.route("submissions/CIK", {"filings": {"recent": {"form": ["10-K"]}}})
    prof = edgar.profile("T")
    assert prof.filings == []
    assert any("unexpected submissions shape" in n for n in prof.notes)


def test_filing_url_is_built_from_the_accession():
    f = Filing("10-K", "2026-02-01", "2025-12-31", "0001045810-26-000069", "nvda.htm")
    assert f.url == ("https://www.sec.gov/Archives/edgar/data/1045810/"
                     "000104581026000069/nvda.htm")


def test_html_to_text_strips_scripts_and_entities():
    text = html_to_text("<html><script>bad()</script><p>Revenue &amp; margin</p></html>")
    assert "bad()" not in text
    assert "Revenue & margin" in text
