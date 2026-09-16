"""The Treasury curve and the fed funds strip.

Every test here corresponds to a wrong number that reached a brief, not to a
line of code that looked untested.
"""

from __future__ import annotations

from datetime import date

import pytest

from agents_work.agents import tape
from agents_work.sources import rates
from agents_work.sources.rates import (BP, Curve, FedPath, PolicyExpectation,
                                       _parse_curve_xml, clean_anchor,
                                       fomc_meetings, implied_rate, months_ahead,
                                       zq_symbol)

CURVE_XML = """<?xml version="1.0" encoding="utf-8" standalone="yes" ?>
<feed xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices">
<entry><content><m:properties>
  <d:NEW_DATE m:type="Edm.DateTime">2026-09-14T00:00:00</d:NEW_DATE>
  <d:BC_3MONTH m:type="Edm.Double">4.11</d:BC_3MONTH>
  <d:BC_2YEAR m:type="Edm.Double">4.65</d:BC_2YEAR>
  <d:BC_10YEAR m:type="Edm.Double">4.97</d:BC_10YEAR>
  <d:BC_30YEAR m:type="Edm.Double">5.34</d:BC_30YEAR>
  <d:BC_30YEARDISPLAY m:type="Edm.Double">5.34</d:BC_30YEARDISPLAY>
</m:properties></content></entry>
<entry><content><m:properties>
  <d:NEW_DATE m:type="Edm.DateTime">2026-09-15T00:00:00</d:NEW_DATE>
  <d:BC_3MONTH m:type="Edm.Double">4.11</d:BC_3MONTH>
  <d:BC_2YEAR m:type="Edm.Double">4.67</d:BC_2YEAR>
  <d:BC_10YEAR m:type="Edm.Double">5.00</d:BC_10YEAR>
  <d:BC_30YEAR m:type="Edm.Double">5.36</d:BC_30YEAR>
  <d:BC_30YEARDISPLAY m:type="Edm.Double">5.36</d:BC_30YEARDISPLAY>
</m:properties></content></entry>
</feed>"""


def test_curve_parses_oldest_first_with_every_tenor():
    curves = _parse_curve_xml(CURVE_XML)
    assert [c.as_of for c in curves] == [date(2026, 9, 14), date(2026, 9, 15)]
    assert curves[-1].tenors == {"3M": 4.11, "2Y": 4.67, "10Y": 5.00, "30Y": 5.36}


@pytest.mark.benchmark
def test_the_display_duplicate_does_not_become_a_second_tenor():
    """M8. BC_30YEARDISPLAY repeats BC_30YEAR; an unanchored pattern matches both."""
    curves = _parse_curve_xml(CURVE_XML)
    assert list(curves[-1].tenors).count("30Y") == 1
    assert curves[-1].get("30Y") == 5.36


@pytest.mark.benchmark
def test_spreads_are_basis_points_not_percent():
    """M7. A 4.67 -> 5.00 gap is 33bp. Reported as a percentage change it is 7%,
    which is the same class of error that once opened a brief with 'yields
    spiked 92 basis points' on a 4bp move."""
    curve = _parse_curve_xml(CURVE_XML)[-1]
    assert curve.spread_bp("10Y", "2Y") == pytest.approx(33.0)


def test_a_missing_leg_gives_no_spread_rather_than_zero():
    assert Curve(as_of=date(2026, 9, 15), tenors={"10Y": 5.0}).spread_bp("10Y", "2Y") is None


def test_implied_rate_is_a_hundred_minus_price():
    assert implied_rate(96.125) == pytest.approx(3.875)
    assert implied_rate(None) is None


def test_month_codes_roll_into_the_next_year():
    assert zq_symbol(2026, 12) == "ZQZ26.CBT"
    assert zq_symbol(2027, 1) == "ZQF27.CBT"
    assert months_ahead(date(2026, 11, 1), 3) == [(2026, 11), (2026, 12), (2027, 1)]


def _path() -> FedPath:
    p = FedPath(spot=3.74)
    p.points = [PolicyExpectation(2026, 9, "ZQU26.CBT", 3.74),
                PolicyExpectation(2026, 10, "ZQV26.CBT", 3.88),
                PolicyExpectation(2026, 12, "ZQZ26.CBT", 4.11)]
    return p


def test_the_priced_move_is_reported_in_both_units():
    p = _path()
    assert p.move_bp(2026, 12) == pytest.approx(37.0)
    assert p.steps(2026, 12) == pytest.approx(1.48)


def test_an_unquoted_contract_prices_nothing_rather_than_zero():
    p = _path()
    assert p.at(2027, 3) is None
    assert p.move_bp(2027, 3) is None
    assert p.steps(2027, 3) is None


@pytest.mark.benchmark
def test_a_meeting_inside_the_front_month_is_declared_a_blend():
    """M10. ZQ settles to a monthly *average*, so in a month holding a decision the
    front contract straddles the move and understates every move measured
    against it."""
    note = clean_anchor(_path(), [date(2026, 9, 16)], today=date(2026, 9, 15))
    assert "average" in note and "blend" in note


def test_a_meeting_free_front_month_needs_no_caveat():
    assert clean_anchor(_path(), [date(2026, 10, 28)], today=date(2026, 9, 15)) == ""


FOMC_HTML = """
<h4>2026 FOMC Meetings</h4><div>January 27-28</div><div>September 15-16*</div>
<div>December 8-9</div>
<p>Note: A two-day meeting is scheduled for January 26-27, 2027.</p>
<h4>2027 FOMC Meetings</h4><div>January 26-27</div><div>March 16-17*</div>
<p>Note: A two-day meeting is scheduled for January 25-26, 2028.</p>
"""


def test_meeting_dates_are_the_decision_day_not_the_first_day(fetcher):
    fetcher.route("fomccalendars", FOMC_HTML)
    meetings, notes = fomc_meetings(fetcher, today=date(2026, 9, 1))
    assert notes == []
    assert date(2026, 9, 16) in meetings
    assert date(2026, 9, 15) not in meetings


@pytest.mark.benchmark
def test_a_trailing_note_is_attributed_to_its_own_year_not_the_heading(fetcher):
    """M9. (regression) The page closes each year's table with a note naming a
    meeting in the year after next. Read under the enclosing heading it became
    a phantom meeting twelve months early, and every 'priced by the January
    meeting' line downstream anchored on a date that does not exist."""
    fetcher.route("fomccalendars", FOMC_HTML)
    meetings, _ = fomc_meetings(fetcher, today=date(2026, 1, 1))
    assert date(2028, 1, 26) in meetings
    assert date(2027, 1, 26) not in meetings
    assert date(2027, 1, 27) in meetings


def test_past_meetings_are_dropped(fetcher):
    fetcher.route("fomccalendars", FOMC_HTML)
    meetings, _ = fomc_meetings(fetcher, today=date(2026, 12, 31))
    assert all(m >= date(2026, 12, 31) for m in meetings)


def test_an_unreachable_calendar_degrades_rather_than_raises(fetcher):
    meetings, notes = fomc_meetings(fetcher, today=date(2026, 9, 15))
    assert meetings == []
    assert notes and "unavailable" in notes[0]


@pytest.mark.benchmark
def test_the_push_body_leads_with_levels_and_carries_the_talking_point():
    """M11."""
    data = tape.TapeData()
    data.curve = _parse_curve_xml(CURVE_XML)[-1]
    data.prior = _parse_curve_xml(CURVE_XML)[0]
    data.spreads = [("2s10s", 33.0, 1.0)]
    data.path = _path()
    data.meetings = [date(2026, 12, 9)]
    data.tape_lede = "Yields pushed higher across the curve."
    data.talking_point = "The strip prices hikes, not cuts."
    body = tape.push_body(data, equity_line="**S&P fut** 7,666 -0.21%")
    assert body.index("S&P fut") < body.index("Talking point")
    assert "5.00%" in body and "+33bp" in body
    assert "The strip prices hikes, not cuts." in body


def test_the_push_body_says_so_when_there_is_no_tape():
    assert "No tape available" in tape.push_body(tape.TapeData())
