"""The Treasury curve, and what the futures strip says the Fed will do.

Two keyless sources on purpose. Treasury publishes the par yield curve as an
XML feed with no key and no rate limit worth naming, which makes it a better
primary than FRED — FRED needs a key the deploy would have to carry, and its
DGS2/DGS10 series are this same file with a day's extra latency. FRED stays
wired as an optional enrichment (real yields, breakevens) when a key exists.

Fed expectations come from 30-day fed funds futures rather than a scraped
FedWatch number, because the contract *is* the market's expectation: ZQ settles
to the monthly average effective fed funds rate, so 100 - price is the rate the
strip implies for that month, and the difference between two months is what the
market has priced between them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from ..netcache import Fetcher

log = logging.getLogger(__name__)

CURVE_XML = ("https://home.treasury.gov/resource-center/data-chart-center/"
             "interest-rates/pages/xml?data=daily_treasury_yield_curve"
             "&field_tdr_date_value_month={ym}")
FOMC_CALENDAR = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

# XBRL-style tag -> the name a person uses. Order is the order they print in.
TENORS = (
    ("BC_1MONTH", "1M"), ("BC_3MONTH", "3M"), ("BC_6MONTH", "6M"),
    ("BC_1YEAR", "1Y"), ("BC_2YEAR", "2Y"), ("BC_3YEAR", "3Y"),
    ("BC_5YEAR", "5Y"), ("BC_7YEAR", "7Y"), ("BC_10YEAR", "10Y"),
    ("BC_20YEAR", "20Y"), ("BC_30YEAR", "30Y"),
)

# The spreads a desk actually quotes. (label, long leg, short leg).
SPREADS = (("2s10s", "10Y", "2Y"), ("3m10s", "10Y", "3M"), ("5s30s", "30Y", "5Y"))

# One basis point is 0.01 of a percentage point. Treasury publishes percent.
BP = 100.0


@dataclass
class Curve:
    """One day's par yield curve, in percent."""
    as_of: date
    tenors: dict[str, float] = field(default_factory=dict)

    def get(self, tenor: str) -> float | None:
        return self.tenors.get(tenor)

    def spread_bp(self, long_leg: str, short_leg: str) -> float | None:
        a, b = self.get(long_leg), self.get(short_leg)
        if a is None or b is None:
            return None
        return (a - b) * BP


def _parse_curve_xml(xml_text: str) -> list[Curve]:
    """Curves out of Treasury's Atom feed, oldest first.

    Parsed with regex rather than ElementTree because the feed nests every
    field under two Microsoft dataservices namespaces that change shape between
    the monthly and annual endpoints; the tag names themselves never do.
    """
    out: list[Curve] = []
    for entry in re.split(r"<entry[\s>]", xml_text)[1:]:
        m = re.search(r"<d:NEW_DATE[^>]*>([\d-]{10})", entry)
        if not m:
            continue
        try:
            as_of = date.fromisoformat(m.group(1))
        except ValueError:
            continue
        curve = Curve(as_of=as_of)
        for tag, label in TENORS:
            # `[^>]*>` and not `>` — every field carries an m:type attribute.
            # Anchored with a word boundary so BC_30YEAR does not also match
            # BC_30YEARDISPLAY, which the feed repeats with the same value.
            v = re.search(rf"<d:{tag}\b[^>]*>([^<]*)</d:{tag}>", entry)
            if not v or not v.group(1).strip():
                continue
            try:
                curve.tenors[label] = float(v.group(1))
            except ValueError:
                continue
        if curve.tenors:
            out.append(curve)
    out.sort(key=lambda c: c.as_of)
    return out


class Treasury:
    def __init__(self, fetcher: Fetcher) -> None:
        self.f = fetcher
        self.notes: list[str] = []

    def _month(self, when: date) -> list[Curve]:
        # Today's row appears the evening of the same business day, so a short
        # TTL matters on the current month and not at all on a past one.
        resp = self.f.fetch(CURVE_XML.format(ym=f"{when:%Y%m}"), ttl=1800)
        if not resp or not resp.ok:
            return []
        return _parse_curve_xml(resp.text)

    def recent(self, *, today: date | None = None, minimum: int = 2) -> list[Curve]:
        """The last few daily curves, oldest first.

        Reaches back a month when it has to: on the 1st of a month the current
        file holds a single row, and a "change on the day" computed against
        nothing is the kind of blank that gets rendered as zero.
        """
        today = today or datetime.now(timezone.utc).date()
        curves = self._month(today)
        if len(curves) < minimum:
            prior = self._month(today.replace(day=1) - timedelta(days=1))
            curves = prior + curves
        if not curves:
            self.notes.append("Treasury yield curve feed unavailable; rates section is empty")
        elif len(curves) < minimum:
            self.notes.append("only one Treasury curve published this period; "
                              "day-over-day changes are omitted")
        return curves

    def latest(self, *, today: date | None = None) -> tuple[Curve | None, Curve | None]:
        """(most recent curve, the one before it). Either may be None."""
        curves = self.recent(today=today)
        if not curves:
            return None, None
        if len(curves) == 1:
            return curves[-1], None
        return curves[-1], curves[-2]


# --- Fed funds futures ---------------------------------------------------

# CME month codes. ZQ lists every calendar month.
MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
               7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}
# One 25bp move. The Fed has moved in other sizes and may again; this is the
# unit the number is *expressed* in ("1.4 cuts priced"), not an assumption
# that every move is this size.
STEP = 0.25


def zq_symbol(year: int, month: int) -> str:
    """Yahoo's symbol for the ZQ contract settling on that month's average EFFR."""
    return f"ZQ{MONTH_CODES[month]}{year % 100:02d}.CBT"


def implied_rate(price: float | None) -> float | None:
    """100 minus the price is the rate, in percent. That is the whole contract."""
    if price is None:
        return None
    return 100.0 - float(price)


def months_ahead(start: date, count: int) -> list[tuple[int, int]]:
    out = []
    y, m = start.year, start.month
    for _ in range(count):
        out.append((y, m))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


@dataclass
class PolicyExpectation:
    """What the strip prices for one month."""
    year: int
    month: int
    symbol: str
    rate: float | None = None
    error: str = ""

    @property
    def label(self) -> str:
        return f"{date(self.year, self.month, 1):%b %Y}"


@dataclass
class FedPath:
    spot: float | None = None            # front contract = this month's EFFR
    points: list[PolicyExpectation] = field(default_factory=list)
    meetings: list[date] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def at(self, year: int, month: int) -> float | None:
        for p in self.points:
            if p.year == year and p.month == month and p.rate is not None:
                return p.rate
        return None

    def move_bp(self, year: int, month: int) -> float | None:
        """Basis points of easing (negative) or tightening priced by that month."""
        r = self.at(year, month)
        if r is None or self.spot is None:
            return None
        return (r - self.spot) * BP

    def steps(self, year: int, month: int) -> float | None:
        """The same move expressed in 25bp increments."""
        bp = self.move_bp(year, month)
        return None if bp is None else bp / (STEP * BP)


def _quote_last(symbol: str):
    try:
        import yfinance as yf  # noqa: PLC0415 - optional by design
    except ImportError:
        return None, "yfinance not installed"
    yf_log = logging.getLogger("yfinance")
    level = yf_log.level
    yf_log.setLevel(logging.CRITICAL)
    try:
        fi = yf.Ticker(symbol).fast_info
        price = fi.get("lastPrice")
        return (float(price) if price is not None else None), ""
    except Exception as e:  # noqa: BLE001 - one dead contract must not kill the path
        return None, type(e).__name__
    finally:
        yf_log.setLevel(level)


def fed_path(*, today: date | None = None, horizon: int = 13,
             allow_network: bool = True) -> FedPath:
    """The priced policy path, front month first.

    The front contract is used as "spot" because a ZQ contract settles to the
    average EFFR over its own month: most of the current month has already
    happened, so its price is dominated by the rate that actually prevailed.
    That makes it a market-derived reading of where policy *is*, with no target
    range hardcoded anywhere in this repo to go stale after the next meeting.
    """
    today = today or datetime.now(timezone.utc).date()
    path = FedPath()
    if not allow_network:
        path.notes.append("offline mode: fed funds futures not quoted")
        return path
    for y, m in months_ahead(today.replace(day=1), horizon):
        sym = zq_symbol(y, m)
        price, err = _quote_last(sym)
        point = PolicyExpectation(year=y, month=m, symbol=sym,
                                  rate=implied_rate(price), error=err)
        path.points.append(point)
    live = [p for p in path.points if p.rate is not None]
    if not live:
        path.notes.append("no fed funds futures quotes returned; policy expectations omitted")
        return path
    path.spot = path.points[0].rate
    if path.spot is None:
        # The front month can go untraded near expiry; the next one still
        # prices the same regime and is a better anchor than nothing.
        path.spot = live[0].rate
        path.notes.append(f"front ZQ contract unquoted; anchored on {live[0].label}")
    missing = [p.label for p in path.points if p.rate is None]
    if missing:
        path.notes.append(f"no quote for {len(missing)} ZQ contract(s): {', '.join(missing[:4])}")
    return path


def clean_anchor(path: FedPath, meetings: list[date], *, today: date | None = None) -> str:
    """A note when the front contract straddles a meeting, else "".

    ZQ settles to a *monthly average*, so in a month containing an FOMC
    decision the front contract is a blend of the rate before the move and the
    rate after it. Read as "where policy is now" it is biased toward wherever
    the market thinks the meeting lands, which quietly shrinks every move
    measured against it. The number is still the right anchor — it is the only
    market-derived one — but a reader comparing it to the target range should
    be told why it sits between two quarter-point marks.
    """
    today = today or datetime.now(timezone.utc).date()
    if not path.points or path.spot is None:
        return ""
    front = path.points[0]
    if any(d.year == front.year and d.month == front.month and d >= today for d in meetings):
        return (f"{front.label} ZQ settles to the month's *average* EFFR and that month "
                "holds an FOMC decision, so the spot anchor is a blend of the rate "
                "before and after it")
    return ""


_MONTHS = ("January February March April May June July August September "
           "October November December").split()


def fomc_meetings(fetcher: Fetcher, *, today: date | None = None) -> tuple[list[date], list[str]]:
    """Scheduled FOMC decision dates from the Fed's own calendar page, plus notes.

    Scraped rather than hardcoded: a table of meeting dates baked into source
    is correct for about a year and then silently points at the past, and every
    "cuts priced by the March meeting" line downstream inherits that error.
    The decision lands on the *second* day of a two-day meeting, so that is the
    date returned.
    """
    today = today or datetime.now(timezone.utc).date()
    resp = fetcher.fetch(FOMC_CALENDAR, ttl=86_400)
    if not resp or not resp.ok:
        return [], ["FOMC calendar unavailable; meeting dates omitted"]
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", resp.text))
    out: list[date] = []
    for ym in re.finditer(r"(20\d\d) FOMC Meetings", text):
        year = int(ym.group(1))
        end = text.find("FOMC Meetings", ym.end())
        segment = text[ym.end(): end if end > 0 else ym.end() + 4000]
        for m in re.finditer(r"\b(" + "|".join(_MONTHS) + r") (\d{1,2})-(\d{1,2})(?:,\s*(20\d\d))?", segment):
            month = _MONTHS.index(m.group(1)) + 1
            first, second = int(m.group(2)), int(m.group(3))
            # A meeting spanning a month boundary ("April 28-29" never does, but
            # "December 31-January 1" could) shows a second day below the first.
            day = second if second >= first else first
            # The page ends each year's table with a note naming the first
            # meeting of the year *after* next ("January 25-26, 2028"). It sits
            # inside the previous heading's segment, so without honouring the
            # explicit year that note lands as a phantom meeting twelve months
            # early — and every "cuts priced by the January meeting" line
            # downstream would then anchor on a date that does not exist.
            try:
                out.append(date(int(m.group(4)) if m.group(4) else year, month, day))
            except ValueError:
                continue
    out = sorted(set(out))
    upcoming = [d for d in out if d >= today]
    if not upcoming:
        return [], ["FOMC calendar page listed no future meetings; it may have moved"]
    return upcoming, []
