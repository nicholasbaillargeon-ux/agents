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

# --- the enterprise value bridge ----------------------------------------
#
# EV = market cap + total debt + preferred + minority interest - cash.
# Every term after the first is an XBRL lookup, and every one of them has a
# filer-specific tagging convention. The rule followed throughout: when two
# tags could mean the same thing, never add them — pick one and record which,
# because adding a subtotal to its own component is how a comps table reports
# a company as twice as levered as it is.

# Cash. Short-term investments are held separately: whether they net against
# EV is a judgement call, so the caller is told both numbers and the choice.
CASH_TAGS = (
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    "CashAndDueFromBanks",
)
SHORT_TERM_INVESTMENT_TAGS = (
    "ShortTermInvestments",
    "MarketableSecuritiesCurrent",
    "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
    "OtherShortTermInvestments",
)

# Debt, as three mutually exclusive strategies in priority order. They are
# alternatives, never addends: `LongTermDebt` in us-gaap is the *total* carrying
# amount including current maturities, so adding it to `LongTermDebtCurrent`
# double-counts the current portion — which for a filer with a big near-term
# maturity wall is a material overstatement of leverage, in the direction that
# makes a company look cheap on EV/EBITDA.
DEBT_STRATEGIES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("combined tag", ("DebtLongtermAndShorttermCombinedAmount",), ()),
    ("LongTermDebt (incl. current) + short-term borrowings",
     ("LongTermDebt",), ("ShortTermBorrowings", "CommercialPaper", "OtherShortTermBorrowings")),
    ("noncurrent + current + short-term borrowings",
     ("LongTermDebtNoncurrent", "LongTermDebtCurrent"),
     ("ShortTermBorrowings", "CommercialPaper", "OtherShortTermBorrowings")),
)

MINORITY_TAGS = ("MinorityInterest",)
PREFERRED_TAGS = ("PreferredStockValue", "PreferredStockLiquidationPreferenceValue")

# EBITDA is not a GAAP line and is never tagged. It is built, and the build is
# operating income plus D&A -- both of which *are* tagged, inconsistently.
OPERATING_INCOME_TAGS = ("OperatingIncomeLoss",)

# D&A, as mutually exclusive strategies for the same reason debt is. A filer
# either reports one combined line or splits depreciation from amortisation;
# Microsoft does the latter and tags no combined line at all, which is why a
# flat tag list returned nothing and silently cost MSFT its EBITDA.
DA_STRATEGIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("combined", ("DepreciationDepletionAndAmortization",)),
    ("combined incl. accretion", ("DepreciationAmortizationAndAccretionNet",)),
    ("combined", ("DepreciationAndAmortization",)),
    ("combined ex. deferred commissions",
     ("DepreciationDepletionAndAmortizationExcludingAmortizationOfDeferredSalesCommissions",)),
    ("depreciation + intangible amortisation",
     ("Depreciation", "AmortizationOfIntangibleAssets")),
)

# How stale a balance-sheet instant may be before it is refused. A filer that
# reports only annually is thirteen months behind at worst, so this sits just
# past that. Everything older is an abandoned tag, and EDGAR serves those
# forever: Microsoft's combined-debt tag stops in 2015 and JP Morgan's
# LongTermDebt in 2014, both still first in any fixed-priority list.
STALE_INSTANT_DAYS = 450

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
    shares_basis: str = ""          # which tag, or the derivation, produced it
    periods: list[str] = field(default_factory=list)
    revenue_tag: str = ""
    ttm_end: str = ""
    # --- enterprise value bridge ---
    cash: float | None = None
    short_term_investments: float | None = None
    total_debt: float | None = None
    debt_basis: str = ""            # which DEBT_STRATEGIES branch produced it
    da_basis: str = ""              # which DA_STRATEGIES branch produced it
    minority_interest: float | None = None
    preferred: float | None = None
    operating_income_ttm: float | None = None
    depreciation_amortization_ttm: float | None = None
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

    @property
    def ebitda_ttm(self) -> float | None:
        """Operating income plus D&A. None if either leg is missing.

        Not approximated from net income: adding back interest and tax from
        whatever tags happen to exist produces a number that is EBITDA-shaped
        and wrong, and a wrong EBITDA is worse than a blank cell because the
        multiple built on it looks perfectly reasonable.
        """
        if self.operating_income_ttm is None or self.depreciation_amortization_ttm is None:
            return None
        return self.operating_income_ttm + self.depreciation_amortization_ttm

    @property
    def ebitda_margin(self) -> float | None:
        e = self.ebitda_ttm
        if e is None or not self.revenue_ttm:
            return None
        return e / self.revenue_ttm

    @property
    def net_cash(self) -> float | None:
        """Cash (plus short-term investments) less total debt. Negative = net debt."""
        if self.total_debt is None and self.cash is None:
            return None
        cash = (self.cash or 0.0) + (self.short_term_investments or 0.0)
        return cash - (self.total_debt or 0.0)

    def enterprise_value(self, market_cap: float | None) -> float | None:
        """Market cap + debt + preferred + minorities - cash and equivalents.

        Returns None without a market cap or without a debt figure: an EV that
        silently treats unknown debt as zero is a market cap wearing a
        different label, and it lands in the comps table looking authoritative.
        """
        if market_cap is None or self.total_debt is None:
            return None
        cash = (self.cash or 0.0) + (self.short_term_investments or 0.0)
        return (market_cap + self.total_debt + (self.preferred or 0.0)
                + (self.minority_interest or 0.0) - cash)


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
    def _instant_class_values(points: list[dict]) -> list[float]:
        """Every distinct value reported at the newest instant.

        For a cover-page share count this is one entry per share class, which
        is what makes it possible to test whether a derived consolidated count
        is in the same units as the classes or in the units of one of them.
        """
        instants = [p for p in points if "start" not in p and p.get("end")]
        if not instants:
            return []
        newest = max(p["end"] for p in instants)
        out: list[float] = []
        for p in instants:
            if p["end"] != newest:
                continue
            try:
                v = float(p["val"])
            except (TypeError, ValueError):
                continue
            if v not in out:
                out.append(v)
        return out

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

    def _instant_from(self, facts: dict, tags: tuple[str, ...], *, ns: str = "us-gaap",
                      today: date | None = None) -> tuple[float | None, str, str]:
        """(value, tag used, as-of date) for the *most recent* usable instant.

        Ranked by date, not by list order, for exactly the reason revenue is:
        filers migrate between these tags and EDGAR keeps serving the abandoned
        one forever. Taking the first tag that returns anything gave Microsoft
        a 2015 debt balance and JP Morgan a 2018 cash balance, both of which
        look entirely plausible in a comps table and are off by a decade.

        Anything older than STALE_INSTANT_DAYS is refused outright rather than
        returned as a best effort — a balance-sheet item that old cannot
        describe the enterprise value of a company trading today.
        """
        today = today or datetime.now(timezone.utc).date()
        best: tuple[str, float, str] | None = None   # (as_of, value, tag)
        for tag in tags:
            value, classes, as_of = self._latest_instant_values(self._points(facts, tag, ns=ns))
            if value is None or not as_of:
                continue
            if classes > 1:
                # Two values at the same instant means the line is dimensioned
                # (by class, by segment). Summing them is as likely to be wrong
                # as right, so it is skipped and the next tag gets a turn.
                continue
            if (today - date.fromisoformat(as_of)).days > STALE_INSTANT_DAYS:
                continue
            if best is None or as_of > best[0]:
                best = (as_of, value, tag)
        if best is None:
            return None, "", ""
        return best[1], best[2], best[0]

    def _total_debt(self, facts: dict, *, today: date | None = None
                    ) -> tuple[float | None, str, str]:
        """(total debt, how it was built, as-of). Strategies are alternatives.

        Two rules, both learned from a wrong number:

        * The primaries of different strategies are never summed. `LongTermDebt`
          in us-gaap already includes current maturities, so adding it to
          `LongTermDebtCurrent` double-counts the maturity wall — in the
          direction that makes a levered company look cheap on EV/EBITDA.
        * The winning strategy is the one whose components are *current*, not
          the one listed first. Microsoft still serves a combined-debt tag last
          filed in 2015; taking it produced $31.8B against a true $40.3B.

        Within a strategy, a component staler than the primary is dropped
        rather than added: Microsoft's `ShortTermBorrowings` is a zero from
        2018, and adding a stale zero is silently asserting the line is empty.
        """
        today = today or datetime.now(timezone.utc).date()
        candidates: list[tuple[str, float, str, str]] = []   # (as_of, total, label, used)
        for label, primaries, extras in DEBT_STRATEGIES:
            total, used, as_of = 0.0, [], ""
            for tag in primaries:
                value, _, when = self._instant_from(facts, (tag,), today=today)
                if value is None:
                    continue
                total += value
                used.append(tag)
                as_of = max(as_of, when)
            if not used:
                continue
            for tag in extras:
                value, _, when = self._instant_from(facts, (tag,), today=today)
                # A component from an older balance sheet than the primary is
                # not part of the same balance sheet. Dropping it is the
                # conservative error; adding it asserts a number nobody filed.
                if value is None or when < as_of:
                    continue
                total += value
                used.append(tag)
                as_of = max(as_of, when)
            candidates.append((as_of, total, label, " + ".join(used)))
        if not candidates:
            return None, "", ""
        as_of, total, label, used = max(candidates, key=lambda c: c[0])
        return total, f"{label} [{used}] as of {as_of}", as_of

    def _ttm_group(self, facts: dict, strategies) -> tuple[float | None, str]:
        """Sum a strategy's tags into one TTM, preferring the freshest strategy.

        Same shape as `_best_series` and for the same reason, but over groups:
        the answer may be one tag or the sum of two, and which it is depends on
        the filer rather than on anything knowable in advance.
        """
        best: tuple[str, float, str] | None = None   # (ttm_end, value, label)
        for label, tags in strategies:
            total, end, used = 0.0, "", []
            for tag in tags:
                points = self._points(facts, tag)
                if not points:
                    continue
                value, periods = self._ttm(points)
                if value is None or not periods:
                    continue
                total += value
                used.append(tag)
                end = max(end, periods[0].split("..")[1])
            if not used:
                continue
            if len(used) < len(tags):
                # A split strategy that found only one of its legs is not the
                # strategy it is named after. Alphabet tags `Depreciation` and
                # no intangible amortisation line, so the figure is depreciation
                # alone — true, and materially different from D&A for a filer
                # that amortises acquired intangibles.
                label = f"{used[0]} only (no {', '.join(t for t in tags if t not in used)})"
            if best is None or end > best[0]:
                best = (end, total, f"{label} [{' + '.join(used)}]")
        if best is None:
            return None, ""
        return best[1], best[2]

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

        f.equity, _, _ = self._instant_from(
            facts, ("StockholdersEquity",
                    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
            today=today)
        f.assets, _, _ = self._instant_from(facts, ("Assets",), today=today)
        # dei carries the cover-page share count, which is present for filers
        # that never tag CommonStockSharesOutstanding in us-gaap.
        stale_share_counts: list[str] = []
        multi_class: list[str] = []
        for tag, ns in (("EntityCommonStockSharesOutstanding", "dei"),
                        ("CommonStockSharesOutstanding", "us-gaap"),
                        ("WeightedAverageNumberOfDilutedSharesOutstanding", "us-gaap")):
            shares, classes, as_of = self._latest_instant_values(self._points(facts, tag, ns=ns))
            if shares is None:
                continue
            if classes > 1:
                # One number cannot describe two classes, so this tag is no
                # use — but the next one may be consolidated, so keep going
                # rather than abandoning the search. Stopping here is what cost
                # Meta its market cap: its cover page is filed per class and
                # its diluted weighted-average count, one line below, is not.
                multi_class.append(f"{classes} share classes reported on {as_of} under {tag}")
                continue
            if (today - date.fromisoformat(as_of)).days > STALE_SHARES_DAYS:
                # Berkshire's cover-page count stops in 2011 — every later one
                # is dimensioned per class and absent from companyfacts. A
                # fifteen-year-old share count silently ruins a market cap.
                stale_share_counts.append(f"{tag} last reported {as_of}")
                continue
            f.shares, f.shares_basis = shares, f"{tag} ({as_of})"
            break

        if f.shares is None and f.net_income_ttm and f.eps_diluted_ttm:
            # Last resort, and pure arithmetic on two filed facts: diluted EPS
            # is net income over diluted shares, so the quotient is the share
            # count the filer itself used. It is consolidated across classes by
            # construction, which is exactly what the per-class cover page is
            # not.
            #
            # The caveat is real and is recorded: this count is in the economic
            # units of the class the EPS is reported in. For a filer whose
            # classes are economically identical (Meta, Alphabet) that is the
            # whole company. For one where they are not — Berkshire's B is
            # 1/1500 of an A — multiplying it by the *wrong* class's price is
            # off by that ratio, so the basis is printed wherever the number is.
            derived = f.net_income_ttm / f.eps_diluted_ttm
            # Is the derived count in the same units as the classes on the
            # cover page, or in the units of one of them? Test it rather than
            # assume it. Where the classes are economically equal (Meta,
            # Alphabet) the derived count lands close to their sum. Where they
            # are not — Berkshire's B is 1/1500 of an A — diluted EPS is stated
            # per A-equivalent and the derived count comes out three orders of
            # magnitude below the sum. Multiplying that by the B-class price is
            # the wrong answer by exactly the conversion ratio, and nothing
            # downstream would notice, so the ratio is what gates the fallback.
            class_counts = self._instant_class_values(
                self._points(facts, "EntityCommonStockSharesOutstanding", ns="dei")) or \
                self._instant_class_values(
                    self._points(facts, "CommonStockSharesOutstanding"))
            total_classes = sum(class_counts) if class_counts else 0.0
            ratio = derived / total_classes if total_classes else None
            if derived <= 0:
                pass
            elif ratio is not None and not (0.5 <= ratio <= 2.0):
                f.notes.append(
                    f"diluted EPS implies {derived:,.0f} shares against {total_classes:,.0f} "
                    "on the cover page across all classes — the classes are not "
                    "economically equal, so no single share count describes this "
                    "filer and market cap, EV and P/S are omitted")
            else:
                f.shares, f.shares_basis = derived, "derived: net income ÷ diluted EPS"
                f.notes.append(
                    "share count derived from net income ÷ diluted EPS because the "
                    "cover page is filed per class"
                    + (f" ({multi_class[0]})" if multi_class else "")
                    + (f"; cross-checked against the {total_classes:,.0f} shares on the "
                       f"cover page (ratio {ratio:.2f})" if ratio else ""))
        if f.shares is None and multi_class:
            f.notes.append(
                "; ".join(multi_class) + "; no consolidated count and no diluted "
                "EPS to derive one from, so share count, market cap and P/S are "
                "omitted rather than guessed")
        if f.shares is None and stale_share_counts:
            f.notes.append("no current share count in EDGAR (" +
                           "; ".join(stale_share_counts) +
                           "); market cap and P/S are omitted")

        # --- enterprise value bridge ---
        f.cash, cash_tag, cash_as_of = self._instant_from(facts, CASH_TAGS, today=today)
        f.short_term_investments, _, _ = self._instant_from(
            facts, SHORT_TERM_INVESTMENT_TAGS, today=today)
        f.total_debt, f.debt_basis, debt_as_of = self._total_debt(facts, today=today)
        f.minority_interest, _, _ = self._instant_from(facts, MINORITY_TAGS, today=today)
        f.preferred, _, _ = self._instant_from(facts, PREFERRED_TAGS, today=today)

        if f.total_debt is None:
            f.notes.append(
                "no us-gaap debt tag matched, so enterprise value is omitted — "
                "financials and REITs frequently tag borrowings only in a "
                "company-specific namespace")
        if f.cash is None:
            f.notes.append("no cash tag matched; the EV bridge nets no cash")

        # A balance sheet from a materially older filing than the income
        # statement makes the bridge mix two dates. Small gaps are normal (the
        # cover-page and the statements are filed together); a large one is not.
        for label, when in (("cash", cash_as_of), ("debt", debt_as_of)):
            if not when or not f.ttm_end:
                continue
            gap = (date.fromisoformat(f.ttm_end) - date.fromisoformat(when)).days
            if gap > 200:
                f.notes.append(
                    f"{label} is as of {when}, {gap} days before the {f.ttm_end} "
                    "income statement — the EV bridge spans two filings")

        _, oi_points = self._best_series(facts, OPERATING_INCOME_TAGS)
        if oi_points:
            f.operating_income_ttm, _ = self._ttm(oi_points)
        f.depreciation_amortization_ttm, f.da_basis = self._ttm_group(facts, DA_STRATEGIES)
        if f.operating_income_ttm is None:
            f.notes.append(
                "no OperatingIncomeLoss TTM: EBITDA and EV/EBITDA are omitted "
                "(banks and insurers do not report an operating income line)")
        elif f.depreciation_amortization_ttm is None:
            f.notes.append("no D&A TTM in EDGAR; EBITDA is omitted rather than "
                           "approximated by operating income")

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
