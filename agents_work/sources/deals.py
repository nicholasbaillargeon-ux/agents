"""M&A headlines, and what can be read off them without a model.

Three keyless feeds, because no single one is enough: Google News is broad and
noisy, PR Newswire and BusinessWire carry the actual deal announcements — the
press release is where the price, the structure and the advisors are stated,
and it is the only free source that names who advised whom.

"Acquisition" is a badly overloaded word in a news feed. Defence procurement
offices, hospital systems, customer-acquisition marketing and job titles all
use it, and they outnumber real M&A in an unfiltered query. `looks_like_deal`
is the gate that keeps the deal book from filling with Space Force contracting
news; it is deliberately keyword-based and deliberately strict, because a model
verdict on every headline costs money and a false positive costs attention.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ..netcache import Fetcher
from .news import Headline, _parse

log = logging.getLogger(__name__)

# The newswires answer a browser and 403 anything that looks like a script, so
# these two feeds get a browser string. The SEC contact UA stays the default
# everywhere else: it is a courtesy that edgar.py depends on.
BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
NEEDS_BROWSER_UA = ("prnewswire.com", "businesswire.com")

FEEDS = {
    "Google News": ("https://news.google.com/rss/search?q="
                    "%22to+acquire%22+OR+%22acquisition+of%22+OR+%22merger+with%22+OR+"
                    "%22definitive+agreement%22+when:2d&hl=en-US&gl=US&ceid=US:en"),
    "PR Newswire": ("https://www.prnewswire.com/rss/financial-services-latest-news/"
                    "acquisitions-mergers-and-takeovers-list.rss"),
}
# BusinessWire's M&A feed is deliberately absent. It serves curl happily and
# 403s httpx with byte-identical headers, which means it is fingerprinting the
# TLS handshake rather than reading the User-Agent. Shelling out to curl for
# one feed would put a second HTTP path outside the cache and the rate limiter
# for no gain: PR Newswire carries the same announcements, and Google News
# indexes BusinessWire releases anyway. Recorded here so nobody re-adds it,
# watches it fail, and assumes the User-Agent needs another tweak.
DROPPED_FEEDS = {"BusinessWire": "403s httpx regardless of headers (TLS fingerprinting)"}

# A headline must contain one of these to be considered a deal at all.
DEAL_TERMS = (
    "to acquire", "acquires", "acquisition of", "to buy", "buys", "merger",
    "merges with", "to merge", "definitive agreement", "takeover", "tender offer",
    "to combine", "all-cash", "all-stock", "stake in", "majority stake",
    "go private", "take private", "bid for", "agrees to purchase", "carve-out",
)

# ...and none of these, which are the ways the word shows up in non-M&A news.
# Each entry here was pulled off a live feed, not imagined.
NOISE_TERMS = (
    "customer acquisition", "talent acquisition", "user acquisition",
    "acquisition executive", "acquisition officer", "acquisition office",
    "acquisition strategy", "acquisition cost", "land acquisition",
    "data acquisition", "acquisition corp announces", "language acquisition",
    "space force", "air force", "defense department", "pentagon",
    "procurement", "acquisition reform", "acquisition workforce",
    "merger of equals talks stall",  # rumour-stage chatter, not an announcement
    # Civic, property and nonprofit uses of the same words, all pulled off a
    # live feed: land banks, parish consolidations and council announcements
    # outnumbered real deals in the first unfiltered sweep.
    "land bank", "councilman", "councilwoman", "city council", "county board",
    "school district", "diocese", "parish", "township", "city of",
    "public library", "fire district", "housing authority", "redevelopment",
)

# Non-USD currency prefixes. A release quoting "A$2.8 billion" parses to 2.8e9
# and is then silently compared against dollar thresholds 35% too high.
CURRENCY_PREFIX = {
    "A$": "AUD", "C$": "CAD", "NZ$": "NZD", "HK$": "HKD", "S$": "SGD",
    "R$": "BRL", "\u20ac": "EUR", "\u00a3": "GBP", "\u00a5": "JPY", "\u20b9": "INR",
    # Word forms too: releases outside the US write "AUD $1 billion" as often
    # as "A$1 billion", and the prefix form alone missed every one of them.
    "AUD": "AUD", "CAD": "CAD", "EUR": "EUR", "GBP": "GBP", "JPY": "JPY",
}

# Deal value, in the shapes releases actually write it.
_VALUE = re.compile(
    r"(?:\$|US\$|USD\s*)\s*([\d,]+(?:\.\d+)?)\s*(billion|bn|b|million|mm|m|trillion|tn|t)?\b",
    re.I)
_SCALE = {"trillion": 1e12, "tn": 1e12, "t": 1e12,
          "billion": 1e9, "bn": 1e9, "b": 1e9,
          "million": 1e6, "mm": 1e6, "m": 1e6}

# "Acme to acquire Beta" / "Acme acquires Beta" / "Acme to buy Beta"
#
# The connective group is what makes this useful on a real feed: press offices
# almost never write "Acme to acquire Beta", they write "Acme Agrees to
# Acquire Beta", "Acme Announces Acquisition of Beta", "Acme Completes
# Acquisition of Beta". Without it the pattern matched about one headline in
# eight on a live sweep.
_CONNECT = (r"(?:\s+(?:agrees?|has\s+agreed|announces?|completes?|closes?|plans?|"
            r"intends?|moves?|is\s+set|sets?|enters?\s+into|signs?|to\s+enter)"
            r"(?:\s+(?:a|an|the))?(?:\s+definitive)?(?:\s+agreement)?)?")
_PAIR = re.compile(
    r"^(?P<a>[A-Z][\w&.,'\- ]{1,60}?)" + _CONNECT + r"\s+(?:to\s+)?"
    r"(?:acquire|acquires|acquisition\s+of|buy|buys|purchase|purchases|"
    r"merge\s+with|merger\s+with|combine\s+with)\s+"
    r"(?P<t>[A-Z][\w&.,'\- ]{1,60}?)"
    r"(?:\s+(?:for|in|at|to)\b|[,;:(]|$)", re.I)

# "...acted as financial advisor to..." is the phrase that names the banks.
#
# Matched as a phrase with a backward window rather than as a sentence. Bank
# names are full of full stops — "Goldman Sachs & Co. LLC", "J.P. Morgan
# Securities LLC", "Robert W. Baird & Co." — so a sentence pattern bounded by
# `[^.]` starts *after* the name it is looking for and returns "LLC".
_ADVISOR_PHRASE = re.compile(
    r"\b(?:acted|acting|is\s+acting|served|serving|is\s+serving|is|are|acts)\s+as\s+"
    r"(?:the\s+)?(?:exclusive\s+|lead\s+|sole\s+|joint\s+|financial\s+and\s+)*"
    r"financial\s+advis[oe]r", re.I)
# How far back the advisor's name may sit before the phrase. Long enough for
# "Goldman Sachs & Co. LLC and Morgan Stanley & Co. LLC", short enough not to
# reach the previous sentence's subject.
_ADVISOR_LOOKBACK = 150
_NAME_TOKEN = re.compile(r"\b([A-Z][\w&.'\-]*(?:\s+(?:&\s+)?[A-Z][\w&.'\-]*){0,4})")
# Full stops that do not end a sentence. Without this the backward window runs
# past the previous sentence and files "Beta Corp. Cantor Fitzgerald" — the
# target company glued to its counterparty's banker — as one advisor.
# Only the stops that genuinely sit *inside* a name. "Co." does — "Goldman
# Sachs & Co. LLC" — so it stays. "Inc.", "Corp." and "Ltd." are deliberately
# absent: they terminate a company name, so in "...advisor to Beta Corp. Cantor
# Fitzgerald is acting..." the stop after Corp really is the end of a sentence,
# and treating it as an abbreviation glued the target to the next bank.
_ABBREVIATIONS = {"co", "no", "jr", "sr", "st", "mr", "ms", "dr", "u.s", "n.a"}
_SENTENCE_END = re.compile(r"(?<=[a-z0-9)\]])\.\s+(?=[A-Z])")


def _trim_to_sentence(window: str) -> str:
    """The tail of `window` after the last real sentence break.

    "Co." and "J.P." are full stops inside a name, so the test is what precedes
    the stop: a known abbreviation or a single initial keeps the sentence open.
    """
    cut = 0
    for m in _SENTENCE_END.finditer(window):
        before = window[:m.start()].rsplit(" ", 1)[-1].lower().rstrip(".")
        if before in _ABBREVIATIONS or len(before) <= 1:
            continue
        cut = m.end()
    return window[cut:]
# Words that begin a clause rather than a name, and corporate suffixes that are
# never the whole name.
_NOT_A_NAME = {"the", "acted", "acting", "served", "serving", "financial", "exclusive",
               "lead", "sole", "joint", "advisor", "adviser", "legal", "counsel",
               "llc", "inc", "lp", "llp", "plc", "ltd", "corp", "company", "co",
               "securities", "and", "also", "in", "addition", "to", "a", "an"}


# Consideration language. A dollar figure in a release body only means the
# deal price when it sits next to one of these; every other figure in the text
# is revenue, market size, or the transaction volume the target processes.
_CONSIDERATION = re.compile(
    r"(?:\b(?:for|valued\s+at|value\s+of|purchase\s+price\s+of|total\s+consideration\s+of|"
    r"consideration\s+of|enterprise\s+value\s+of|equity\s+value\s+of|transaction\s+valued\s+at|"
    r"acquisition\s+price\s+of|deal\s+worth|worth\s+up\s+to)\s+"
    r"(?:approximately\s+|about\s+|up\s+to\s+|around\s+|nearly\s+)?)"
    r"((?:A|C|NZ|HK|S|R|US)?\$|USD\s*|AUD\s*|CAD\s*|EUR\s*|GBP\s*|\u20ac|\u00a3)?\s*"
    r"([\d,]+(?:\.\d+)?)\s*(billion|bn|million|mm|trillion)?", re.I)


def consideration_value(text: str) -> tuple[float | None, str]:
    """(deal price, currency) read from consideration language only.

    `parse_value` takes the largest dollar figure in its input, which is the
    right rule for a headline — the only number in a headline is the price —
    and badly wrong for a release body. Run over a full release it returned
    "AUD $1 billion in certificate transactions" as the price of Formbay and
    "$1 billion in annualized marketplace sales" as the price of four Amazon
    brands. Both landed in the book as a clean $1.00B.

    So a body figure counts only when the sentence says it is what somebody
    paid. Nothing matching means undisclosed, which is the honest answer and
    the common one.
    """
    for m in _CONSIDERATION.finditer(text):
        try:
            v = float(m.group(2).replace(",", ""))
        except (TypeError, ValueError, AttributeError):
            continue
        unit = (m.group(3) or "").lower()
        if unit:
            v *= _SCALE.get(unit, 1.0)
        elif v < 1_000_000:
            # "for 5 cents on the dollar", "for 30,000 shares" — a bare number
            # this small next to "for" is not a deal price.
            continue
        raw = (m.group(1) or "").strip().upper().rstrip("$")
        currency = {"A": "AUD", "C": "CAD", "NZ": "NZD", "HK": "HKD", "S": "SGD",
                    "R": "BRL", "AUD": "AUD", "CAD": "CAD", "EUR": "EUR",
                    "GBP": "GBP", "\u20ac": "EUR", "\u00a3": "GBP"}.get(raw, "")
        return v, currency
    return None, ""


def detect_currency(text: str) -> str:
    """The non-USD currency a figure is quoted in, or "" for dollars.

    Checked before any threshold is applied. A$2.8B is not $2.8B, and a size
    filter that treats them as equal admits deals it was configured to exclude
    while reporting a number that is simply wrong.
    """
    for prefix, code in CURRENCY_PREFIX.items():
        if prefix in text:
            return code
    return ""


def _ua_for(url: str) -> dict | None:
    return ({"User-Agent": BROWSER_UA}
            if any(h in url for h in NEEDS_BROWSER_UA) else None)


def parse_value(text: str) -> float | None:
    """The largest dollar figure in the text, in dollars. None if there is none.

    Largest, not first: a release headline reads "…for $4.1 billion, including
    the assumption of $600 million of debt", and the first match is the part,
    not the whole. Equity value is what a comps table wants and it is the
    bigger of the two often enough that "largest" is the better guess — the
    figure is carried as `value_note` alongside so a reader can check it.
    """
    best: float | None = None
    for m in _VALUE.finditer(text):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        unit = (m.group(2) or "").lower()
        if unit:
            v *= _SCALE.get(unit, 1.0)
        elif v < 1000:
            # A bare "$12" in a headline is a per-share price, not a deal value.
            continue
        if best is None or v > best:
            best = v
    return best


def looks_like_deal(title: str) -> bool:
    low = title.lower()
    if any(n in low for n in NOISE_TERMS):
        return False
    return any(t in low for t in DEAL_TERMS)


def parse_parties(title: str) -> tuple[str, str]:
    """(acquirer, target) read off the headline, or ("", "") if it is not that shape.

    Deliberately shallow. A headline is one sentence written by a press office
    and roughly a third of them follow this pattern; the rest are left to the
    model, which is given the release body and can do better. Guessing harder
    here would mean guessing wrong silently, and an acquirer and target the
    wrong way round is the single worst error this thing can make.
    """
    # Google News appends " - Outlet Name". The outlet can itself contain
    # hyphens and dots ("fin-news.com"), which the obvious `[\w .]+$` misses,
    # leaving the outlet glued to the target company's name.
    clean = re.sub(r"\s+-\s+[\w.\-' ]+$", "", title).strip()
    m = _PAIR.match(clean)
    if not m:
        return "", ""
    a, t = m.group("a").strip(" ,;:"), m.group("t").strip(" ,;:")
    if not a or not t or a.lower() == t.lower():
        return "", ""
    return a, t


def find_advisors(text: str) -> list[str]:
    """Bank names from the advisor sentences of a release. Order preserved.

    Returns the sentences' named entities rather than a curated bank list, so a
    boutique nobody hardcoded still shows up — which is the entire point when
    the name you are watching for is a mid-size advisor.
    """
    out: list[str] = []
    for phrase in _ADVISOR_PHRASE.finditer(text):
        # Backwards only. What follows "financial advisor to" is the *client*,
        # and reading forwards files the target company as its own banker.
        window = _trim_to_sentence(
            text[max(0, phrase.start() - _ADVISOR_LOOKBACK): phrase.start()])
        for name in _NAME_TOKEN.findall(window):
            name = name.strip(" ,;&")
            head = name.lower().split()[0].rstrip(".") if name else ""
            if len(name) < 3 or head in _NOT_A_NAME:
                continue
            if name not in out:
                out.append(name)
    return out


# The dateline that opens a wire release: "NEW YORK, Sept. 15, 2026
# /PRNewswire/ --". Everything above it is site chrome.
_DATELINE = re.compile(
    r"/(?:PRNewswire|PR Newswire|CNW|GLOBE NEWSWIRE|ACCESSWIRE)[\w\-/]*/\s*-{1,2}|"
    r"\(BUSINESS WIRE\)|\(GLOBE NEWSWIRE\)", re.I)
# ...and the footer the release ends with.
_FOOTER = re.compile(
    r"\n\s*(?:SOURCE\s+[A-Z]|View original content|View source version|"
    r"Related Links|Also from this source)", re.I)


def strip_boilerplate(text: str) -> str:
    """The release itself, without the newswire's site furniture.

    Not cosmetic. PR Newswire renders its entire industry taxonomy into the
    navigation of every page — "Semiconductors", "Financial Technology",
    "Data Analytics" and forty more — so a keyword filter run over the raw
    page text matches every sector on every release. That is exactly how a
    women's fashion acquisition entered the book flagged as semiconductors.
    It also puts two thousand characters of menu in front of the model.

    Anchored on the dateline because every wire release has one and nothing
    above it is ever content. If no dateline is found the text is returned
    unchanged: a release this cannot parse is still worth reading.
    """
    m = _DATELINE.search(text)
    if m:
        # Back up to the start of the dateline's own line, so the city and date
        # survive — they are the only timestamp inside the release body.
        start = text.rfind("\n", 0, m.start()) + 1
        text = text[start:]
    end = _FOOTER.search(text)
    if end:
        text = text[: end.start()]
    return text.strip()


@dataclass
class Deal:
    """One candidate deal, as far as free sources take it."""
    title: str
    url: str
    source: str
    feed: str = ""
    when: str = ""
    acquirer: str = ""
    target: str = ""
    value_usd: float | None = None
    currency: str = ""          # "" means USD; anything else means value_usd is NOT dollars
    advisors: list[str] = field(default_factory=list)
    body: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Stable identity for the seen-table: the same deal reported by four
        outlets is one deal, so the parties are the key when they parsed and the
        headline only when they did not."""
        if self.acquirer and self.target:
            return f"{self.acquirer.lower()}|{self.target.lower()}"
        return re.sub(r"\s+-\s+[\w.\-' ]+$", "", self.title).lower().strip()

    @property
    def value_label(self) -> str:
        v = self.value_usd
        if v is None:
            return "undisclosed"
        unit = self.currency or "$"
        sep = "" if unit == "$" else " "
        if v >= 1e9:
            return f"{unit}{sep}{v / 1e9:,.1f}B"
        return f"{unit}{sep}{v / 1e6:,.0f}M"

    @property
    def is_usd(self) -> bool:
        return not self.currency

    @property
    def body_fetchable(self) -> bool:
        """Whether this URL can yield a release body at all.

        Google News no longer publishes the publisher's URL in its feed: the
        link is `news.google.com/rss/articles/<opaque token>`, which resolves
        only through an internal batchexecute endpoint. So an aggregator item
        is a headline and nothing more, permanently — and spending an
        enrichment slot on one costs a round trip and returns nothing.

        This is load-bearing rather than cosmetic. Ranked purely on "has
        parties and a price", every enrichment slot in the first live sweep
        went to a Google News item and none to PR Newswire, so the run read
        twenty-two bodies and got zero.
        """
        return "news.google.com" not in self.url


class Deals:
    def __init__(self, fetcher: Fetcher) -> None:
        self.f = fetcher
        self.notes: list[str] = []

    def headlines(self, *, limit_per_feed: int = 40, max_age_hours: float = 36) -> list[Deal]:
        out: list[Deal] = []
        seen_titles: set[str] = set()
        for feed_name, url in FEEDS.items():
            resp = self.f.fetch(url, ttl=900, headers=_ua_for(url))
            if not resp or not resp.ok:
                self.notes.append(f"{feed_name} M&A feed unavailable")
                continue
            heads: list[Headline] = _parse(resp.text, limit=limit_per_feed,
                                           max_age_hours=max_age_hours)
            kept = 0
            for h in heads:
                if not looks_like_deal(h.title):
                    continue
                norm = re.sub(r"\W+", " ", h.title.lower()).strip()
                if norm in seen_titles:
                    continue
                seen_titles.add(norm)
                a, t = parse_parties(h.title)
                out.append(Deal(title=h.title, url=h.url, source=h.source, feed=feed_name,
                                when=h.when, acquirer=a, target=t,
                                value_usd=parse_value(h.title),
                                currency=detect_currency(h.title)))
                kept += 1
            log.info("%s: %d headlines, %d look like deals", feed_name, len(heads), kept)
        return out

    def enrich(self, deal: Deal, *, max_chars: int = 30_000) -> Deal:
        """Fetch the release body and read price and advisors out of it.

        Only worth doing for the handful of deals that survive the filters —
        it is one HTTP round trip each, and Google News URLs are redirects that
        frequently land on a paywall, which is why a failure here is a note on
        the deal rather than a failure of the run.
        """
        from .edgar import html_to_text  # noqa: PLC0415 - shared tag-soup stripper

        if not deal.body_fetchable:
            deal.notes.append("aggregator link: no release body is reachable, "
                              "so this one-pager is headline-only")
            return deal
        resp = self.f.fetch(deal.url, ttl=86_400, headers=_ua_for(deal.url))
        if not resp or not resp.ok:
            deal.notes.append("release body unavailable; headline-only")
            return deal
        text = strip_boilerplate(html_to_text(resp.text, limit=max_chars))
        if len(text) < 400:
            deal.notes.append("release body too short to read (paywall or redirect)")
            return deal
        deal.body = text
        deal.advisors = find_advisors(text)
        if deal.value_usd is None:
            # The headline carried no price. The body may state one — but only
            # where it says so, never just the biggest number present.
            deal.value_usd, currency = consideration_value(text)
            if deal.value_usd is not None:
                deal.currency = currency or detect_currency(text[:2000])
        if not (deal.acquirer and deal.target):
            a, t = parse_parties(text.strip().split("\n")[0][:200])
            if a and t:
                deal.acquirer, deal.target = a, t
        return deal
