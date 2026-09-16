"""The enterprise value bridge: debt, cash, EBITDA and the share count.

Every gate here is a number that reached a live comps table and was wrong.
EDGAR serves abandoned tags forever, so "the first tag that returns anything"
is a trap for every line on the balance sheet, not just revenue.
"""

from __future__ import annotations

from datetime import date

import pytest
from edgar_facts import B, NVDA_PERIODS, durations, facts, instants

from agents_work.sources.edgar import Edgar, STALE_INSTANT_DAYS

TODAY = date(2026, 8, 20)
FRESH = "2026-06-30"
STALE = "2015-03-31"


@pytest.fixture
def edgar(fetcher):
    return Edgar(fetcher)


def _load(edgar, payload, cik=1045810):
    edgar._facts_cache[cik] = payload
    return cik


# -- balance-sheet tag selection ---------------------------------------------

@pytest.mark.benchmark
def test_the_most_recent_instant_wins_not_the_first_tag_listed():
    """V1. (regression) Microsoft's combined-debt tag stops in 2015 and JP Morgan's
    cash tag in 2018, and EDGAR still serves both. First-match-wins gave MSFT a
    $31.8B debt balance against a true $40.3B, and JPM a 2018 cash balance —
    each entirely plausible in a comps table and off by a decade."""
    e = Edgar.__new__(Edgar)
    payload = facts(
        CashAndCashEquivalentsAtCarryingValue=instants([(STALE, 278.8 * B)]),
        CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents=instants(
            [(FRESH, 309.8 * B)]))
    value, tag, as_of = e._instant_from(
        payload, ("CashAndCashEquivalentsAtCarryingValue",
                  "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
        today=TODAY)
    assert as_of == FRESH
    assert value == pytest.approx(309.8 * B)


def test_an_instant_older_than_the_staleness_window_is_refused_entirely():
    e = Edgar.__new__(Edgar)
    payload = facts(Assets=instants([(STALE, 100 * B)]))
    value, tag, _ = e._instant_from(payload, ("Assets",), today=TODAY)
    assert value is None and tag == ""


def test_a_dimensioned_instant_is_skipped_rather_than_guessed_at():
    """Two values at one instant means the line is split by class or segment.
    Summing them is as likely to be wrong as right."""
    e = Edgar.__new__(Edgar)
    payload = facts(Assets=instants([(FRESH, 10 * B), (FRESH, 90 * B)]))
    assert e._instant_from(payload, ("Assets",), today=TODAY)[0] is None


# -- total debt ---------------------------------------------------------------

def test_the_freshest_debt_strategy_wins_over_the_first_one_listed():
    e = Edgar.__new__(Edgar)
    payload = facts(
        DebtLongtermAndShorttermCombinedAmount=instants([(STALE, 31.8 * B)]),
        LongTermDebt=instants([(FRESH, 40.3 * B)]))
    total, basis, as_of = e._total_debt(payload, today=TODAY)
    assert total == pytest.approx(40.3 * B)
    assert "LongTermDebt" in basis and as_of == FRESH


@pytest.mark.benchmark
def test_a_subtotal_is_never_added_to_its_own_component():
    """V2. `LongTermDebt` already includes current maturities, so adding
    `LongTermDebtCurrent` double-counts the maturity wall — in the direction
    that makes a levered company look cheap on EV/EBITDA."""
    e = Edgar.__new__(Edgar)
    payload = facts(LongTermDebt=instants([(FRESH, 40.3 * B)]),
                    LongTermDebtNoncurrent=instants([(FRESH, 31.1 * B)]),
                    LongTermDebtCurrent=instants([(FRESH, 9.2 * B)]))
    total, _, _ = e._total_debt(payload, today=TODAY)
    assert total == pytest.approx(40.3 * B)
    assert total != pytest.approx(40.3 * B + 9.2 * B)


@pytest.mark.benchmark
def test_a_stale_component_is_dropped_rather_than_added_as_a_zero():
    """V3. (regression) Apple's `ShortTermBorrowings` is a zero last filed in 2018.
    Adding it asserts the line is empty today, which nobody filed."""
    e = Edgar.__new__(Edgar)
    payload = facts(LongTermDebt=instants([(FRESH, 84.3 * B)]),
                    ShortTermBorrowings=instants([("2018-06-30", 0.0)]))
    total, basis, _ = e._total_debt(payload, today=TODAY)
    assert total == pytest.approx(84.3 * B)
    assert "ShortTermBorrowings" not in basis


def test_short_term_borrowings_from_the_same_balance_sheet_are_added():
    e = Edgar.__new__(Edgar)
    payload = facts(LongTermDebt=instants([(FRESH, 80 * B)]),
                    CommercialPaper=instants([(FRESH, 4.3 * B)]))
    total, basis, _ = e._total_debt(payload, today=TODAY)
    assert total == pytest.approx(84.3 * B)
    assert "CommercialPaper" in basis


def test_a_filer_with_no_matching_debt_tag_gets_no_debt_figure():
    e = Edgar.__new__(Edgar)
    assert e._total_debt(facts(Assets=instants([(FRESH, 1 * B)])), today=TODAY)[0] is None


# -- D&A ----------------------------------------------------------------------

@pytest.mark.benchmark
def test_a_split_depreciation_and_amortisation_filer_still_gets_ebitda():
    """V4. (regression) Microsoft tags no combined D&A line at all — it reports
    `Depreciation` and `AmortizationOfIntangibleAssets` separately — so a flat
    tag list returned nothing and silently cost MSFT its EBITDA."""
    e = Edgar.__new__(Edgar)
    payload = facts(Depreciation=durations(NVDA_PERIODS),
                    AmortizationOfIntangibleAssets=durations(NVDA_PERIODS))
    value, basis = e._ttm_group(payload, (
        ("combined", ("DepreciationDepletionAndAmortization",)),
        ("split", ("Depreciation", "AmortizationOfIntangibleAssets")),
    ))
    assert value is not None
    assert "Depreciation" in basis and "AmortizationOfIntangibleAssets" in basis


def test_a_half_found_split_strategy_says_which_leg_it_actually_has():
    """Alphabet tags `Depreciation` and no intangible amortisation line, so the
    figure is depreciation alone — true, and materially different from D&A for
    a filer that amortises acquired intangibles."""
    e = Edgar.__new__(Edgar)
    payload = facts(Depreciation=durations(NVDA_PERIODS))
    _, basis = e._ttm_group(payload, (
        ("split", ("Depreciation", "AmortizationOfIntangibleAssets")),))
    assert "only" in basis and "AmortizationOfIntangibleAssets" in basis


# -- share count --------------------------------------------------------------

def test_a_per_class_cover_page_falls_through_to_the_diluted_derivation(edgar):
    """(regression) Stopping at the first multi-class tag cost Meta its market
    cap: its cover page is filed per class and it tags no consolidated count,
    but net income over diluted EPS is exactly that number."""
    e = edgar
    cik = _load(e, facts(
        Revenues=durations(NVDA_PERIODS),
        NetIncomeLoss=durations([(s, en, 10 * B) for s, en, _ in NVDA_PERIODS]),
        EarningsPerShareDiluted=durations([(s, en, 4.0) for s, en, _ in NVDA_PERIODS]),
        dei__EntityCommonStockSharesOutstanding=instants(
            [("2026-05-20", 2_000_000_000), ("2026-05-20", 500_000_000)])))
    f = e.fundamentals(cik, today=TODAY)
    assert f.shares == pytest.approx(40 * B / 16.0)
    assert "derived" in f.shares_basis


@pytest.mark.benchmark
def test_classes_that_are_not_economically_equal_refuse_a_single_share_count(edgar):
    """V5. Berkshire's B is 1/1500 of an A. Diluted EPS is stated per A-equivalent,
    so the derived count is three orders of magnitude below the cover page —
    and multiplying it by the B-class price is wrong by exactly that ratio."""
    e = edgar
    cik = _load(e, facts(
        Revenues=durations(NVDA_PERIODS),
        NetIncomeLoss=durations([(s, en, 20 * B) for s, en, _ in NVDA_PERIODS]),
        EarningsPerShareDiluted=durations([(s, en, 14_000.0) for s, en, _ in NVDA_PERIODS]),
        dei__EntityCommonStockSharesOutstanding=instants(
            [("2026-05-20", 941_481), ("2026-05-20", 1_300_000_000)])))
    f = e.fundamentals(cik, today=TODAY)
    assert f.shares is None
    assert any("not economically equal" in n for n in f.notes)
