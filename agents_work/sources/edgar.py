"""SEC EDGAR: filings and XBRL fundamentals.

EDGAR is the highest-quality free financial source there is, and the only one
here that is contractually stable. It is also the one with a real rate limit
and a mandatory User-Agent, both handled in `netcache`.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from ..netcache import Fetcher

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
COMPANY_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
INDEX_JSON = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/index.json"

# Forms an equity research reader actually cares about, in priority order.
INTERESTING = ("10-K", "10-Q", "8-K", "S-1", "424B4", "DEF 14A", "20-F", "6-K")

# (concept tag, human label, unit). Companies tag revenue inconsistently, so
# revenue is a list of fallbacks tried in order — this is the single most
# common reason a naive XBRL scraper returns nothing for a real company.
# Ordered candidates, but order is only a tiebreak — `_best_series` picks the
# tag that actually yields a current TTM, because filers migrate between these
# and EDGAR keeps serving the abandoned one.
REVENUE_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "RevenuesNetOfInterestExpense",   # banks
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
)
NET_INCOME_TAGS = (
    "NetIncomeLoss",
    "ProfitLoss",
    "NetIncomeLossAvailableToCommonStockholdersBasic",
)
EPS_TAGS = ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted")

# A quarter, generously: 13 weeks is 91 days, but 4-4-5 calendars and 52/53-week
# fiscal years stretch it either way.
QUARTER_MIN_DAYS, QUARTER_MAX_DAYS = 60, 100
# Consecutive quarters should meet within a day; fiscal calendars occasionally
# leave a few. More than this is a real hole, not a calendar artefact.
MAX_PERIOD_GAP_DAYS = 5
# Past this, the newest quarter EDGAR holds is old enough to say so out loud.
STALE_TTM_DAYS = 200
# A share count older than this cannot describe today's market cap.
STALE_SHARES_DAYS = 400


def _dedup_durations(points: list[dict]) -> dict[tuple[date, date], float]:
    """(start, end) -> value, keeping the most recently *filed* restatement."""
    best: dict[tuple[date, date], tuple[float, str]] = {}
    for p in points:
        if "start" not in p or "end" not in p:
            continue
        try:
            start = datetime.strptime(p["start"], "%Y-%m-%d").date()
            end = datetime.strptime(p["end"], "%Y-%m-%d").date()
            val = float(p["val"])
        except (KeyError, TypeError, ValueError):
            continue
        filed = str(p.get("filed") or "")
        prev = best.get((start, end))
        if prev is None or filed >= prev[1]:
            best[(start, end)] = (val, filed)
    return {k: v[0] for k, v in best.items()}


def derive_quarters(points: list[dict]) -> dict[tuple[date, date], float]:
    """Every quarter obtainable from a concept's facts, explicit or implied.

    Companies with a non-calendar fiscal year almost never tag Q4 by itself:
    the 10-K carries the full year and the last 10-Q carries nine months, and
    Q4 is the difference. The same subtraction recovers any quarter reported
    only inside a year-to-date figure, so two facts sharing a start date and
    differing by about a quarter yield the stub between them.
    """
    durations = _dedup_durations(points)
    quarters = {k: v for k, v in durations.items()
                if QUARTER_MIN_DAYS <= (k[1] - k[0]).days <= QUARTER_MAX_DAYS}

    by_start: dict[date, list[tuple[date, float]]] = defaultdict(list)
    for (start, end), val in durations.items():
        by_start[start].append((end, val))
    for ends in by_start.values():
        ends.sort()
        for (short_end, short_val), (long_end, long_val) in zip(ends, ends[1:]):
            gap = (long_end - short_end).days
            if not QUARTER_MIN_DAYS <= gap <= QUARTER_MAX_DAYS:
                continue
            implied = (short_end + timedelta(days=1), long_end)
            quarters.setdefault(implied, long_val - short_val)
    return quarters


@dataclass
class Filing:
    form: str
    filed: str
    report_date: str
    accession: str
    document: str
    description: str = ""

    @property
    def url(self) -> str:
        acc = self.accession.replace("-", "")
        cik = int(self.accession.split("-")[0].lstrip("0") or 0)
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{self.document}"


@dataclass
class Fundamentals:
    revenue_ttm: float | None = None
    revenue_prior_ttm: float | None = None
    net_income_ttm: float | None = None
    equity: float | None = None
    assets: float | None = None
    eps_diluted_ttm: float | None = None
    shares: float | None = None
    periods: list[str] = field(default_factory=list)
    revenue_tag: str = ""
    ttm_end: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def revenue_growth(self) -> float | None:
        if self.revenue_ttm and self.revenue_prior_ttm:
            return self.revenue_ttm / self.revenue_prior_ttm - 1.0
        return None

    @property
    def net_margin(self) -> float | None:
        if self.revenue_ttm and self.net_income_ttm is not None and self.revenue_ttm != 0:
            return self.net_income_ttm / self.revenue_ttm
        return None


@dataclass
class CompanyProfile:
    ticker: str
    cik: int | None = None
    name: str = ""
    sic: str = ""
    exchange: str = ""
    filings: list[Filing] = field(default_factory=list)
    fundamentals: Fundamentals = field(default_factory=Fundamentals)
    notes: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.cik is not None


class Edgar:
    def __init__(self, fetcher: Fetcher) -> None:
        self.f = fetcher
        self._ticker_map: dict[str, tuple[int, str]] | None = None
        self._facts_cache: dict[int, dict] = {}

    # -- ticker -> CIK ---------------------------------------------------
    def ticker_map(self) -> dict[str, tuple[int, str]]:
        if self._ticker_map is not None:
            return self._ticker_map
        # The map changes at most daily; a day of staleness is harmless.
        resp = self.f.fetch(TICKERS_URL, ttl=86_400)
        data = resp.json({}) if resp and resp.ok else {}
        out: dict[str, tuple[int, str]] = {}
        for entry in (data or {}).values():
            try:
                out[str(entry["ticker"]).upper()] = (int(entry["cik_str"]), entry["title"])
            except (KeyError, TypeError, ValueError):
                continue
        self._ticker_map = out
        return out

    def cik_for(self, ticker: str) -> tuple[int, str] | None:
        return self.ticker_map().get(ticker.upper())

    # -- filings ---------------------------------------------------------
    def profile(self, ticker: str, *, filing_limit: int = 8,
                forms: tuple[str, ...] = INTERESTING) -> CompanyProfile:
        prof = CompanyProfile(ticker=ticker.upper())
        hit = self.cik_for(ticker)
        if not hit:
            prof.notes.append(f"{ticker.upper()} is not in the SEC ticker file (foreign or delisted?)")
            return prof
        prof.cik, prof.name = hit

        resp = self.f.fetch(SUBMISSIONS.format(cik=prof.cik), ttl=3600)
        if not resp or not resp.ok:
            prof.notes.append("EDGAR submissions feed unavailable; filings section is empty")
            return prof
        data = resp.json({}) or {}
        prof.sic = data.get("sicDescription", "")
        exchanges = data.get("exchanges") or []
        prof.exchange = exchanges[0] if exchanges else ""

        recent = (data.get("filings") or {}).get("recent") or {}
        cols = ("form", "filingDate", "reportDate", "accessionNumber",
                "primaryDocument", "primaryDocDescription")
        if not all(c in recent for c in cols):
            prof.notes.append("EDGAR returned an unexpected submissions shape; filings skipped")
            return prof
        n = len(recent["form"])
        for i in range(n):
            form = recent["form"][i]
            if form not in forms:
                continue
            prof.filings.append(Filing(
                form=form,
                filed=recent["filingDate"][i],
                report_date=recent["reportDate"][i] or "",
                accession=recent["accessionNumber"][i],
                document=recent["primaryDocument"][i],
                description=recent["primaryDocDescription"][i] or "",
            ))
            if len(prof.filings) >= filing_limit:
                break
        return prof

    # -- fundamentals ----------------------------------------------------
    def company_facts(self, cik: int) -> dict:
        """Every XBRL fact EDGAR holds for a company, in one request.

        Deliberately not the per-tag `companyconcept` endpoint: that needs one
        request per tag (eight per ticker here, against a 10 req/s limit) and
        was observed returning nothing for tags that companyfacts serves
        happily — Coca-Cola's `Revenues` among them.
        """
        if cik in self._facts_cache:
            return self._facts_cache[cik]
        resp = self.f.fetch(COMPANY_FACTS.format(cik=cik), ttl=86_400)
        facts = ((resp.json({}) or {}).get("facts") or {}) if resp and resp.ok else {}
        self._facts_cache[cik] = facts
        return facts

    @staticmethod
    def _points(facts: dict, tag: str, ns: str = "us-gaap") -> list[dict]:
        units = ((facts.get(ns) or {}).get(tag) or {}).get("units") or {}
        for key in ("USD", "USD/shares", "shares"):
            if key in units:
                return units[key]
        return next(iter(units.values()), [])

    def _best_series(self, facts: dict, tags: tuple[str, ...]) -> tuple[str, list[dict]]:
        """Of several tags meaning the same thing, the one that actually yields
        the most recent complete four-quarter TTM.

        Taking the first tag that returns *anything* is the trap. NVDA stopped
        tagging revenue as RevenueFromContractWithCustomerExcludingAssessedTax
        after FY2022, but EDGAR still serves those old facts, so first-match-wins
        reported FY2020 revenue in a 2026 brief — off by 20x and entirely
        plausible-looking. Ranking by "gives me a current TTM" is the property
        that matters, so that is what is measured.
        """
        complete: tuple[str, str, list[dict]] | None = None   # (ttm_end, tag, points)
        partial: tuple[date, str, list[dict]] | None = None   # (latest quarter, tag, points)
        for tag in tags:
            points = self._points(facts, tag)
            if not points:
                continue
            value, periods = self._ttm(points)
            if value is not None and periods:
                ttm_end = periods[0].split("..")[1]
                if complete is None or ttm_end > complete[0]:
                    complete = (ttm_end, tag, points)
                continue
            quarters = derive_quarters(points)
            if quarters:
                latest = max(end for _, end in quarters)
                if partial is None or latest > partial[0]:
                    partial = (latest, tag, points)
        if complete:
            return complete[1], complete[2]
        if partial:
            return partial[1], partial[2]
        return "", []

    @staticmethod
    def _ttm(points: list[dict], *, before: date | None = None) -> tuple[float | None, list[str]]:
        """Sum the four most recent *contiguous* quarters. None if they do not exist.

        Two failure modes this guards, both of which otherwise produce a
        confident wrong number rather than an error:

        * Overlap — a 10-K carries a full-year duration alongside quarterly
          ones, so summing blindly double-counts.
        * Holes — most fiscal-year filers never tag Q4 on its own; it exists
          only as (FY - 9M year-to-date). A greedy scan that takes the next
          non-overlapping quarter jumps the hole and returns a 15-month "TTM".
          `derive_quarters` reconstructs the stub; contiguity is then enforced,
          and an unfillable hole returns None instead of a plausible lie.
        """
        quarters = derive_quarters(points)
        if before is not None:
            quarters = {k: v for k, v in quarters.items() if k[1] <= before}
        if not quarters:
            return None, []
        ordered = sorted(quarters.items(), key=lambda kv: kv[0][1], reverse=True)

        chosen: list[float] = []
        used: list[str] = []
        cursor: date | None = None  # start of the last quarter taken
        for (start, end), val in ordered:
            if cursor is not None and not (0 <= (cursor - end).days <= MAX_PERIOD_GAP_DAYS):
                continue  # overlaps something already counted, or leaves a hole
            chosen.append(val)
            used.append(f"{start}..{end}")
            cursor = start
            if len(chosen) == 4:
                break
        if len(chosen) < 4:
            return None, used
        return sum(chosen), used

    @staticmethod
    def _latest_instant(points: list[dict]) -> float | None:
        value, _, _ = Edgar._latest_instant_values(points)
        return value

    @staticmethod
    def _latest_instant_values(points: list[dict]) -> tuple[float | None, int, str]:
        """(value at the newest instant, how many distinct values share it, that date).

        The count is what catches multi-class filers: Berkshire reports a cover
        -page share count for class A *and* class B on the same date, and
        silently taking the first gives a market cap off by three orders of
        magnitude. One number cannot describe two share classes trading at
        different prices, so the caller is told rather than guessed at.
        """
        instants = [p for p in points if "start" not in p and p.get("end")]
        if not instants:
            return None, 0, ""
        newest = max(p["end"] for p in instants)
        values = []
        for p in instants:
            if p["end"] != newest:
                continue
            try:
                values.append(float(p["val"]))
            except (TypeError, ValueError):
                continue
        if not values:
            return None, 0, ""
        return values[0], len(set(values)), newest

    def fundamentals(self, cik: int, *, today: date | None = None) -> Fundamentals:
        f = Fundamentals()
        today = today or datetime.now(timezone.utc).date()
        facts = self.company_facts(cik)
        if not facts:
            f.notes.append("EDGAR company facts unavailable; this brief has no fundamentals")
            return f

        f.revenue_tag, rev_points = self._best_series(facts, REVENUE_TAGS)
        if not rev_points:
            f.notes.append(
                "no us-gaap total-revenue tag for this filer — banks and integrated "
                "energy majors often report it only in a company-specific namespace")
        else:
            f.revenue_ttm, f.periods = self._ttm(rev_points)
            if f.periods:
                f.ttm_end = f.periods[0].split("..")[1]
            if f.revenue_ttm is None:
                anchor = f.ttm_end or "the latest filing"
                f.notes.append(
                    f"revenue TTM unavailable: the four quarters before {anchor} are not "
                    "contiguous in EDGAR and the gap could not be derived from "
                    "year-to-date facts")
            else:
                window_start = date.fromisoformat(f.periods[-1].split("..")[0])
                f.revenue_prior_ttm, _ = self._ttm(rev_points, before=window_start)

        _, ni_points = self._best_series(facts, NET_INCOME_TAGS)
        if ni_points:
            f.net_income_ttm, _ = self._ttm(ni_points)
        _, eps_points = self._best_series(facts, EPS_TAGS)
        if eps_points:
            f.eps_diluted_ttm, _ = self._ttm(eps_points)

        f.equity = self._latest_instant(self._points(facts, "StockholdersEquity"))
        f.assets = self._latest_instant(self._points(facts, "Assets"))
        # dei carries the cover-page share count, which is present for filers
        # that never tag CommonStockSharesOutstanding in us-gaap.
        stale_share_counts: list[str] = []
        for tag, ns in (("EntityCommonStockSharesOutstanding", "dei"),
                        ("CommonStockSharesOutstanding", "us-gaap"),
                        ("WeightedAverageNumberOfDilutedSharesOutstanding", "us-gaap")):
            shares, classes, as_of = self._latest_instant_values(self._points(facts, tag, ns=ns))
            if shares is None:
                continue
            if classes > 1:
                # One number cannot describe two classes trading at different
                # prices, so say so instead of picking one.
                f.notes.append(
                    f"{classes} share classes reported on {as_of} under {tag}; share "
                    "count, market cap and P/S are omitted rather than guessed")
                break
            if (today - date.fromisoformat(as_of)).days > STALE_SHARES_DAYS:
                # Berkshire's cover-page count stops in 2011 — every later one
                # is dimensioned per class and absent from companyfacts. A
                # fifteen-year-old share count silently ruins a market cap.
                stale_share_counts.append(f"{tag} last reported {as_of}")
                continue
            f.shares = shares
            break
        if f.shares is None and stale_share_counts:
            f.notes.append("no current share count in EDGAR (" +
                           "; ".join(stale_share_counts) +
                           "); market cap and P/S are omitted")

        if f.ttm_end:
            age = (today - date.fromisoformat(f.ttm_end)).days
            if age > STALE_TTM_DAYS:
                f.notes.append(
                    f"most recent quarter in EDGAR ends {f.ttm_end} ({age} days ago) — "
                    "fundamentals may pre-date a filing that has not been indexed yet")
        return f

    # -- filing documents ------------------------------------------------
    def filing_documents(self, cik: int, accession: str) -> list[dict]:
        acc = accession.replace("-", "")
        resp = self.f.fetch(INDEX_JSON.format(cik=cik, acc=acc), ttl=86_400)
        if not resp or not resp.ok:
            return []
        items = ((resp.json({}) or {}).get("directory") or {}).get("item") or []
        return [i for i in items if isinstance(i, dict)]

    def earnings_release(self, cik: int, filings: list[Filing]) -> tuple[str, str] | None:
        """(text, url) of the most recent 8-K earnings exhibit, or None.

        Earnings *transcripts* have no free licensable source. The closest
        honest substitute is the company's own earnings release, filed as an
        EX-99 exhibit to an 8-K under Item 2.02 — so that is what this returns,
        and the brief labels it as a release, not a transcript.
        """
        for filing in filings:
            if filing.form != "8-K":
                continue
            docs = self.filing_documents(cik, filing.accession)
            exhibits = [d for d in docs
                        if "ex-99" in d.get("name", "").lower()
                        or "ex99" in d.get("name", "").lower()]
            for doc in exhibits:
                acc = filing.accession.replace("-", "")
                url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc['name']}"
                resp = self.f.fetch(url, ttl=86_400)
                if resp and resp.ok and len(resp.text) > 500:
                    return html_to_text(resp.text), url
        return None


_HTML_TAG = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.S | re.I)
_WS = re.compile(r"[ \t\r\f\v]+")
_BLANK = re.compile(r"\n{3,}")


def html_to_text(html: str, *, limit: int = 20_000) -> str:
    """EDGAR exhibits are HTML. This is deliberately crude: the text goes to a
    model for summarising, not to a parser, so tag soup tolerance beats fidelity."""
    import html as _html

    text = _HTML_TAG.sub(" ", html)
    text = _html.unescape(text)
    text = _WS.sub(" ", text)
    text = _BLANK.sub("\n\n", text)
    return text.strip()[:limit]
