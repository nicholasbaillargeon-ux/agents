"""Job boards: Greenhouse, Lever, Ashby.

The board slugs below were each verified to return HTTP 200 on 2026-08-20. They
rot — firms move ATS vendors — so `Boards.fetch_all` reports which sources
answered and which came back empty, and an empty board is a logged event rather
than a silently shorter result list.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ..netcache import Fetcher

log = logging.getLogger(__name__)

GREENHOUSE = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
LEVER = "https://api.lever.co/v0/postings/{slug}?mode=json"
ASHBY = "https://api.ashbyhq.com/posting-api/job-board/{slug}"

# (vendor, slug, display name, sector)
REGISTRY: tuple[tuple[str, str, str, str], ...] = (
    ("greenhouse", "janestreet", "Jane Street", "quant"),
    ("greenhouse", "jumptrading", "Jump Trading", "quant"),
    ("greenhouse", "imc", "IMC Trading", "quant"),
    ("greenhouse", "optiver", "Optiver", "quant"),
    ("greenhouse", "akunacapital", "Akuna Capital", "quant"),
    ("greenhouse", "oldmissioncapital", "Old Mission Capital", "quant"),
    ("greenhouse", "virtu", "Virtu Financial", "quant"),
    ("greenhouse", "flowtraders", "Flow Traders", "quant"),
    ("greenhouse", "squarepointcapital", "Squarepoint Capital", "quant"),
    ("greenhouse", "stripe", "Stripe", "fintech"),
    ("greenhouse", "robinhood", "Robinhood", "fintech"),
    ("greenhouse", "coinbase", "Coinbase", "fintech"),
    ("greenhouse", "gemini", "Gemini", "fintech"),
    ("greenhouse", "mercury", "Mercury", "fintech"),
    ("greenhouse", "databricks", "Databricks", "ai"),
    ("greenhouse", "scaleai", "Scale AI", "ai"),
    ("greenhouse", "figma", "Figma", "ai"),
    ("lever", "plaid", "Plaid", "fintech"),
    ("ashby", "ramp", "Ramp", "fintech"),
    ("ashby", "openai", "OpenAI", "ai"),
)

INTERN_PATTERNS = re.compile(
    r"\b(intern|internship|co-?op|new\s?grad|campus|university|student|"
    r"summer\s?20\d\d|placement|apprentice)\b", re.I)

# Words that mean "this is the kind of work the user is aiming at".
RELEVANT = {
    "quantitative": 3, "quant": 3, "trading": 3, "trader": 3, "research": 2,
    "software": 2, "engineer": 2, "engineering": 2, "developer": 2,
    "machine learning": 3, "ml": 2, "ai": 2, "data": 1, "python": 2,
    "c++": 2, "infrastructure": 1, "platform": 1, "backend": 1, "systems": 1,
}
IRRELEVANT = {
    "sales": -4, "recruiting": -4, "recruiter": -4, "marketing": -3, "legal": -4,
    "compliance": -2, "hr ": -3, "people ": -2, "design": -2, "accounting": -3,
    "office": -2, "executive assistant": -4, "customer support": -4, "brand": -3,
}


@dataclass
class Posting:
    company: str
    title: str
    location: str
    url: str
    sector: str = ""
    posted: str = ""
    source: str = ""
    score: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Stable identity for the nightly diff. The URL carries the ATS id;
        titles and locations get edited in place and would false-positive."""
        return re.sub(r"[?#].*$", "", self.url) or f"{self.company}:{self.title}"

    @property
    def is_internship(self) -> bool:
        return bool(INTERN_PATTERNS.search(self.title))

    def as_dict(self) -> dict:
        return {"company": self.company, "title": self.title, "location": self.location,
                "url": self.url, "sector": self.sector, "posted": self.posted,
                "source": self.source, "score": self.score}


def score(posting: Posting) -> int:
    """Deterministic relevance score. The LLM re-ranks later; this decides what
    is even worth spending a token on, and works when the LLM is down."""
    text = f"{posting.title} {posting.location}".lower()
    total, reasons = 0, []
    if posting.is_internship:
        total += 6
        reasons.append("internship/new-grad title")
    for word, pts in RELEVANT.items():
        if word in text:
            total += pts
            reasons.append(f"+{pts} {word}")
    for word, pts in IRRELEVANT.items():
        if word in text:
            total += pts
            reasons.append(f"{pts} {word}")
    if re.search(r"\b(new york|nyc|manhattan|chicago|remote|boston)\b", text):
        total += 1
        reasons.append("+1 target metro")
    posting.score = total
    posting.reasons = reasons
    return total


class Boards:
    def __init__(self, fetcher: Fetcher) -> None:
        self.f = fetcher
        self.source_status: dict[str, str] = {}

    def fetch_all(self, registry=REGISTRY, *, ttl: int = 3600) -> list[Posting]:
        out: list[Posting] = []
        for vendor, slug, company, sector in registry:
            try:
                got = self._fetch_one(vendor, slug, company, sector, ttl)
            except Exception as e:  # noqa: BLE001 - one bad board never kills the sweep
                log.warning("board %s/%s blew up: %s", vendor, slug, e)
                self.source_status[company] = f"error: {type(e).__name__}"
                continue
            self.source_status[company] = f"{len(got)} postings" if got else "no postings returned"
            out.extend(got)
        return out

    def _fetch_one(self, vendor: str, slug: str, company: str, sector: str,
                   ttl: int) -> list[Posting]:
        url = {"greenhouse": GREENHOUSE, "lever": LEVER, "ashby": ASHBY}[vendor].format(slug=slug)
        resp = self.f.fetch(url, ttl=ttl)
        if not resp or not resp.ok:
            return []
        data = resp.json()
        if data is None:
            return []
        parser = {"greenhouse": _parse_greenhouse, "lever": _parse_lever,
                  "ashby": _parse_ashby}[vendor]
        postings = parser(data, company, sector)
        for p in postings:
            p.source = vendor
        return postings


def _parse_greenhouse(data, company: str, sector: str) -> list[Posting]:
    out = []
    for j in (data or {}).get("jobs", []) or []:
        loc = (j.get("location") or {}).get("name", "") or ""
        out.append(Posting(company=company, title=(j.get("title") or "").strip(),
                           location=loc, url=j.get("absolute_url", ""),
                           sector=sector, posted=(j.get("updated_at") or "")[:10]))
    return [p for p in out if p.title and p.url]


def _parse_lever(data, company: str, sector: str) -> list[Posting]:
    out = []
    for j in data or []:
        cats = j.get("categories") or {}
        out.append(Posting(company=company, title=(j.get("text") or "").strip(),
                           location=cats.get("location", "") or "",
                           url=j.get("hostedUrl", ""), sector=sector,
                           posted=_ms_date(j.get("createdAt"))))
    return [p for p in out if p.title and p.url]


def _parse_ashby(data, company: str, sector: str) -> list[Posting]:
    out = []
    for j in (data or {}).get("jobs", []) or []:
        url = j.get("jobUrl") or j.get("applyUrl") or j.get("externalLink") or ""
        out.append(Posting(company=company, title=(j.get("title") or "").strip(),
                           location=j.get("location", "") or "", url=url, sector=sector,
                           posted=(j.get("publishedAt") or "")[:10]))
    return [p for p in out if p.title and p.url]


def _ms_date(ms) -> str:
    try:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return ""
