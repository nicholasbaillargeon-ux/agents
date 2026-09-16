"""AI news, from the feeds that answer this host.

Nine keyless feeds in three tiers, because they are not interchangeable:

- **lab** — the organisation announcing its own work. A release post is the
  primary document; everything else in the day's coverage is a rewrite of it.
- **press** — outlets that report on the labs. They carry the things labs do
  not announce about themselves: lawsuits, layoffs, funding rounds, failures.
- **aggregator** — one Google News query. Broad, noisy, and the only way news
  from a lab with no feed of its own arrives at all: Anthropic publishes no
  RSS (see `DROPPED_FEEDS`), so its releases reach this brief only because the
  query names it.

Parsed with the stdlib, both RSS and Atom, so the suite needs no network and no
extra dependency. Feeds in the wild are malformed often enough that every field
access here is defensive.

Two things make this harder than "fetch nine feeds and print the titles":

1. **"AI" is the most overloaded token in a 2026 news feed.** A parish task
   force, a Coast Guard research hub and a stock listicle all match the word,
   and on a live sweep they outnumbered the model releases. `looks_like_ai` and
   `NOISE_TERMS` are the gate; both were fitted to what actually came back.
2. **One story arrives nine times.** Google News indexes TechCrunch, which
   rewrote the OpenAI post. A list of items is not a list of stories, and the
   difference is the whole brief: `cluster` turns the former into the latter.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

from ..netcache import Fetcher

log = logging.getLogger(__name__)

# Only the wire-style outlets need this; the SEC contact string is the default
# everywhere else because edgar.py depends on it as a courtesy.
BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


@dataclass(frozen=True)
class Feed:
    name: str
    url: str
    tier: str            # lab | press | aggregator
    browser_ua: bool = False


# The Google News query names the labs individually on purpose. A bare
# "artificial intelligence" query returns the parish task forces; naming
# OpenAI, Anthropic, DeepMind and Mistral is what makes this feed carry the
# announcements of the one major lab with no feed of its own.
GOOGLE_NEWS_QUERY = (
    '"artificial intelligence" OR "AI model" OR "language model" OR OpenAI OR '
    'Anthropic OR DeepMind OR "Mistral AI" OR "AI chips" when:2d')

FEEDS: tuple[Feed, ...] = (
    Feed("OpenAI", "https://openai.com/news/rss.xml", "lab"),
    Feed("Google DeepMind", "https://deepmind.google/blog/rss.xml", "lab"),
    Feed("Google AI", "https://blog.google/technology/ai/rss/", "lab"),
    Feed("NVIDIA", "https://blogs.nvidia.com/feed/", "lab"),
    Feed("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/",
         "press"),
    Feed("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
         "press"),
    Feed("MIT Technology Review",
         "https://www.technologyreview.com/topic/artificial-intelligence/feed", "press"),
    # 429s the default user agent and serves the same feed to a browser string.
    Feed("VentureBeat AI", "https://venturebeat.com/category/ai/feed/", "press",
         browser_ua=True),
    Feed("Google News", "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en",
         "aggregator"),
)

# Probed live on 2026-09-16 and deliberately absent. Recorded so nobody re-adds
# one, watches it fail, and assumes the user agent needs another tweak.
DROPPED_FEEDS = {
    "Anthropic": "publishes no RSS at all (/news/rss.xml is a 404) — covered via the "
                 "Google News query instead",
    "Meta AI": "ai.meta.com/blog/rss/ returns HTTP 400",
    "Microsoft AI": "blogs.microsoft.com/ai/feed/ returns HTTP 410 Gone",
    "Reuters Technology": "HTTP 401 — the feed is behind their licensing wall",
    "Hugging Face blog": "862 items, almost all community tutorials; a release "
                         "announcement is a needle in it and the noise gate cannot "
                         "tell the difference from a title",
    "AWS Machine Learning": "vendor how-to marketing rather than news; every item "
                            "is a walkthrough of a service that shipped months ago",
    "arXiv cs.AI": "634 submissions a day is not news. Ranking them needs a "
                   "different filter than this one and a model call per batch",
}

# What a tier is worth when two feeds carry the same story: the lab's own post
# is the document the rewrites are about.
TIER_WEIGHT = {"lab": 3, "press": 1, "aggregator": 0}

# Google News hands out opaque `CBMi...` redirect tokens rather than the
# publisher's URL. They open fine in a browser and cannot be resolved
# programmatically, so a direct link always wins when the cluster has one.
AGGREGATOR_HOSTS = ("news.google.com",)

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
# Google News appends " - Outlet" to every title, with a plain hyphen.
#
# Em and en dashes are deliberately not separators, and the strip runs on the
# aggregator feed only. A publisher's own headline uses a dash as punctuation --
# "The AI data center e-waste problem is huge - and getting bigger" -- and a
# greedy suffix rule filed "and getting bigger" as the outlet and truncated the
# headline at the dash.
# Greedy head, so the *last* " - " is the separator: outlet names contain
# hyphens and dots ("fin-news.com") and headlines contain neither spaced
# hyphen nor outlet.
_OUTLET_SUFFIX = re.compile(r"^(.*\S)\s+-\s+(\S.{1,39})$")


@dataclass
class Item:
    """One entry from one feed."""

    title: str
    url: str
    outlet: str = ""
    feed: str = ""
    tier: str = "press"
    published: datetime | None = None
    summary: str = ""

    @property
    def age_hours(self) -> float | None:
        if not self.published:
            return None
        return (datetime.now(timezone.utc) - self.published).total_seconds() / 3600

    @property
    def when(self) -> str:
        age = self.age_hours
        if age is None:
            return "undated"
        if age < 1:
            return f"{int(age * 60)}m ago"
        if age < 48:
            return f"{int(age)}h ago"
        return f"{int(age / 24)}d ago"

    @property
    def is_aggregator_link(self) -> bool:
        return any(h in self.url for h in AGGREGATOR_HOSTS)

    def as_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "outlet": self.outlet,
                "feed": self.feed, "tier": self.tier, "summary": self.summary,
                "published": self.published.isoformat() if self.published else ""}


# --- relevance ------------------------------------------------------------
#
# Word-boundary matching, never substring. "ai" is inside Dubai, Shanghai,
# chair, said and email; "ml" is inside HTML; "gpt" is fine either way. The
# scout learned this the expensive way — two-letter tokens carry the most
# weight and collect the most false positives — and an AI feed is where the
# two-letter token is the subject itself.
AI_TERMS = (
    "ai", "a.i.", "artificial intelligence", "machine learning", "deep learning",
    "neural network", "neural net", "llm", "large language model", "language model",
    "foundation model", "frontier model", "generative ai", "genai", "agi",
    "chatgpt", "gpt", "claude", "gemini", "llama", "mistral", "copilot",
    "openai", "anthropic", "deepmind", "hugging face", "midjourney",
    "stable diffusion", "transformer", "inference", "fine-tune", "fine-tuning",
    "training run", "ai model", "ai agent", "agentic", "chatbot", "diffusion model",
    "computer vision", "nlp", "tpu", "gpu", "ai chip", "superintelligence",
)

# Titles that match an AI term and are still not AI news. Every entry was
# pulled off a live sweep of these nine feeds, not imagined.
NOISE_TERMS = (
    # Retail-investor content farms. The single largest category by volume.
    "stocks to buy", "stock to buy", "best ai stocks", "ai stocks", "price target",
    "penny stock", "motley fool", "buy the dip", "shares to watch", "zacks",
    "should you buy", "billionaire", "dividend", "prediction: ", "wall street bets",
    # SEO listicles and evergreen how-tos.
    "best ai tools", "top 10", "top 5", "top 7", "best free", "how to use",
    "step-by-step", "beginner's guide", "ultimate guide", "tips and tricks",
    "prompts to try", "chatgpt prompts", "coupon", "deal of the day", "discount code",
    # Civic and institutional uses of the words, exactly as the deal book found
    # them: a task force is a press release, not a development.
    "task force", "advisory council", "ribbon cutting", "proclamation",
    "essay contest", "call for papers", "scholarship", "summit registration",
    "webinar", "awards finalists", "award winners", "names honorees",
    "city council", "school district", "county board", "diocese", "parish",
    "chamber of commerce", "high school", "township",
    # Horoscopes and lifestyle filler that mention an assistant in passing.
    "horoscope", "zodiac", "recipe", "workout plan", "dating app tips",
)

# Outlets whose entire output is retail-investor content. Filtering by outlet
# rather than by title is what catches the ones written to look like news:
# "Anthropic IPO Date: What Investors Need to Know" contains no noise term at
# all, and arrived on a day Anthropic had announced nothing.
NOISE_OUTLETS = {
    "the motley fool", "motley fool", "zacks", "zacks investment research",
    "benzinga", "investorplace", "insider monkey", "simply wall st",
    "24/7 wall st", "barchart", "tipranks", "stocktwits", "invezz",
}

_SIGNIFICANT = {
    # Something shipped. Both forms of every verb: a feed headline is as
    # likely to say "Introducing Gemini 3.8" as "Google introduces Gemini
    # 3.8", and the release the brief exists to lead with scored zero on the
    # first live sweep because only the third-person form was listed.
    "releases": 4, "launches": 4, "unveils": 4, "introduces": 3, "announces": 3,
    "introducing": 3, "launching": 3, "releasing": 3, "unveiling": 3,
    "announcing": 2, "rolling out": 3, "ships": 2, "shipping": 2,
    "open-sources": 4, "open source": 3, "general availability": 3, "rolls out": 3,
    "now available": 3, "preview": 2, "update": 1, "benchmark": 3, "outperforms": 2,
    "state of the art": 3, "beats": 2,
    # Money and corporate structure.
    "raises": 4, "funding round": 4, "valuation": 3, "acquires": 4, "acquisition": 3,
    "ipo": 3, "merger": 3, "invests": 3, "billion": 2, "layoffs": 3, "resigns": 3,
    "steps down": 3, "hires": 1, "partnership": 2, "deal with": 2,
    # Compute, the constraint everything else runs into.
    "chip": 3, "chips": 3, "data center": 3, "datacenter": 3, "compute": 2,
    "gpu": 2, "tpu": 2, "supply": 1, "export controls": 4, "capacity": 1,
    # Law, policy and safety — the half of AI news that labs do not announce.
    "lawsuit": 4, "sues": 4, "sued": 3, "settlement": 3, "ruling": 3, "court": 2,
    "regulation": 3, "regulator": 3, "ai act": 4, "executive order": 4, "ban": 3,
    "investigation": 3, "antitrust": 4, "copyright": 3, "safety": 2,
    "jailbreak": 3, "breach": 3, "deepfake": 2, "misinformation": 2,
    "hallucination": 2, "evaluation": 2, "red team": 3,
}

# Opinion and speculation. Not noise — a good essay is worth reading — but it
# is not the day's news and should not outrank a release under the cap.
_SOFT = {
    "opinion": -3, "op-ed": -3, "commentary": -2, "column": -2, "explainer": -1,
    "could": -1, "may soon": -2, "rumor": -2, "rumour": -2, "reportedly": -1,
    "what to expect": -2, "here's why": -2, "here is why": -2, "the case for": -2,
    # A lab's feed is also its marketing channel, and the lab bonus would
    # otherwise float a workshop announcement above a rival's model release.
    "workshop": -3, "workshops": -3, "empowering": -2, "celebrating": -2,
    "our commitment": -3, "how to": -2, "guide to": -2, "lessons from": -2,
    "meet the": -2, "behind the scenes": -2, "spotlight": -2, "q&a": -2,
    "for everyone": -2, "societal impact": -2, "everyday life": -2,
}


def _word_regex(word: str) -> re.Pattern:
    """`word` as a whole-word pattern, tolerating punctuation-tailed tokens.

    Copied in spirit from `sources.jobs`: `\\b` cannot follow "a.i.", hence the
    conditional edges rather than a blanket `\\b...\\b`.
    """
    left = r"\b" if word[:1].isalnum() else ""
    right = r"\b" if word[-1:].isalnum() else ""
    return re.compile(left + re.escape(word) + right, re.I)


_AI_RE = tuple(_word_regex(w) for w in AI_TERMS)
_NOISE_RE = tuple(_word_regex(w) for w in NOISE_TERMS)
_SIGNIFICANT_RE = {w: _word_regex(w) for w in _SIGNIFICANT}
_SOFT_RE = {w: _word_regex(w) for w in _SOFT}


def looks_like_ai(text: str) -> bool:
    """Is this about AI at all?

    Whole words only. Substring matching reads Dubai, chair, said, email and
    Shanghai as artificial intelligence, and those are ordinary words in an
    ordinary news feed — the filter would pass everything.
    """
    return any(r.search(text or "") for r in _AI_RE)


def is_noise(text: str, outlet: str = "") -> bool:
    """One of the ways an AI term shows up in something that is not AI news."""
    if (outlet or "").strip().lower() in NOISE_OUTLETS:
        return True
    return any(r.search(text or "") for r in _NOISE_RE)


# A model family next to a version number. The single most reliable signal
# that a headline is about something that shipped rather than something
# somebody thinks: "Gemini 3.8 Live" is an event, "Gemini could soon" is not.
MODEL_VERSION_POINTS = 4
_MODEL_VERSION = re.compile(
    r"\b(gpt|claude|gemini|gemma|llama|grok|mistral|qwen|deepseek|phi|sora|"
    r"dall-?e|midjourney|flux|nova|jamba|kimi|olmo|granite|titan)"
    r"[\s\-]?v?\d", re.I)


def significance(text: str) -> tuple[int, list[str]]:
    """(score, reasons) for one title. Deterministic, and works with no model.

    This decides what is worth a model call and what order the brief is in, so
    it has to be defensible on its own: the LLM only ever writes about stories
    this function already chose.
    """
    total, reasons = 0, []
    for word, pts in _SIGNIFICANT.items():
        if _SIGNIFICANT_RE[word].search(text):
            total += pts
            reasons.append(f"+{pts} {word}")
    if _MODEL_VERSION.search(text):
        total += MODEL_VERSION_POINTS
        reasons.append(f"+{MODEL_VERSION_POINTS} names a model version")
    for word, pts in _SOFT.items():
        if _SOFT_RE[word].search(text):
            total += pts
            reasons.append(f"{pts} {word}")
    return total, reasons


# --- parsing --------------------------------------------------------------

_ATOM = "{http://www.w3.org/2005/Atom}"
_CONTENT_NS = "{http://purl.org/rss/1.0/modules/content/}"


def strip_html(raw: str, *, limit: int = 600) -> str:
    """Feed summaries are HTML. The model gets text, or nothing.

    A Google News description is a block of `<a>` tags and nothing else, so an
    entry whose summary is all markup comes back empty rather than as a list of
    outlet names the model would read as content.
    """
    text = _WS.sub(" ", html.unescape(_TAG.sub(" ", raw or ""))).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rfind(" ")
    return text[: cut if cut > limit * 0.6 else limit].rstrip() + "…"


def split_outlet(title: str) -> tuple[str, str]:
    """('Headline', 'Outlet') for the ' - Outlet' suffix Google News appends."""
    m = _OUTLET_SUFFIX.match((title or "").strip())
    if not m:
        return (title or "").strip(), ""
    return m.group(1).strip(), m.group(2).strip()


def _text(node, tag: str | None = None) -> str:
    if node is None:
        return ""
    el = node if tag is None else node.find(tag)
    if el is None:
        return ""
    return html.unescape(_TAG.sub("", el.text or "")).strip()


def _date(raw: str) -> datetime | None:
    """RFC-822 (RSS) or ISO-8601 (Atom). Naive stamps are read as UTC."""
    if not raw:
        return None
    dt = None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        dt = None
    if dt is None:
        try:
            dt = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _atom_link(entry) -> str:
    """The alternate href, not the first link: Atom entries carry self, replies
    and edit links too, and the first one is regularly not the article."""
    links = entry.findall(f"{_ATOM}link")
    for rel in ("alternate", None):
        for link in links:
            if link.get("rel", "alternate") == (rel or link.get("rel", "alternate")):
                if rel is None or link.get("rel", "alternate") == "alternate":
                    href = link.get("href")
                    if href:
                        return href
    return _text(entry, f"{_ATOM}id")


def parse_feed(xml_text: str, *, feed: str = "", tier: str = "press",
               limit: int = 80, strip_outlet: bool | None = None) -> list[Item]:
    """Items out of one feed body, RSS or Atom.

    Both shapes are here because the choice is the publisher's, not ours: The
    Verge serves Atom `<entry>` elements and everything else serves RSS
    `<item>`. A parser that iterated `item` alone read The Verge as an empty
    feed — which is indistinguishable, in the coverage table, from a dead one.
    """
    # Only the aggregator glues the outlet onto the headline; doing it
    # everywhere costs a publisher's own headline its subtitle.
    if strip_outlet is None:
        strip_outlet = tier == "aggregator"
    split = split_outlet if strip_outlet else (lambda t: ((t or "").strip(), ""))
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as e:
        log.warning("malformed feed body from %s: %s", feed or "?", e)
        return []

    out: list[Item] = []
    for node in root.iter("item"):
        title, outlet = split(_text(node, "title"))
        if not title:
            continue
        out.append(Item(
            title=title,
            url=_text(node, "link"),
            outlet=outlet or _text(node.find("source")) or _domain(_text(node, "link")),
            feed=feed, tier=tier,
            published=_date(_text(node, "pubDate")),
            summary=strip_html(_raw(node, "description") or _raw(node, f"{_CONTENT_NS}encoded")),
        ))
        if len(out) >= limit:
            return out

    for node in root.iter(f"{_ATOM}entry"):
        title, outlet = split(_text(node, f"{_ATOM}title"))
        if not title:
            continue
        url = _atom_link(node)
        out.append(Item(
            title=title, url=url,
            outlet=outlet or _domain(url), feed=feed, tier=tier,
            published=_date(_text(node, f"{_ATOM}published")
                            or _text(node, f"{_ATOM}updated")),
            summary=strip_html(_raw(node, f"{_ATOM}summary")
                               or _raw(node, f"{_ATOM}content")),
        ))
        if len(out) >= limit:
            break
    return out


def _raw(node, tag: str) -> str:
    el = node.find(tag)
    return (el.text or "") if el is not None else ""


def _domain(url: str) -> str:
    m = re.search(r"https?://([^/]+)", url or "")
    return m.group(1).replace("www.", "") if m else "unknown"


# --- clustering -----------------------------------------------------------
#
# Two headlines are the same story when they share two *distinctive* tokens.
# Not Jaccard over the whole title: outlets rewrite every word they can, so
# "OpenAI releases GPT-5.5 for agentic coding" and "OpenAI's GPT-5.5 lands with
# better agents" overlap on a third of their words and are obviously one story.
# And not one shared token either — "OpenAI releases GPT-5.5" and "OpenAI sued
# over training data" share `openai` and are obviously two.

_STOP = {
    # Ordinary English.
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could", "do",
    "does", "for", "from", "has", "have", "how", "in", "into", "is", "it", "its",
    "just", "may", "more", "most", "new", "no", "not", "now", "of", "on", "one",
    "or", "out", "over", "said", "says", "should", "than", "that", "the", "their",
    "them", "there", "they", "this", "to", "up", "was", "we", "what", "when",
    "which", "who", "why", "will", "with", "you", "your", "here", "after",
    "before", "about", "why", "first", "last", "next", "two", "three",
    # Words every headline in *this* feed contains, which makes them useless as
    # evidence that two headlines are about the same thing.
    "ai", "artificial", "intelligence", "model", "models", "llm", "tech",
    "technology", "company", "startup", "report", "study", "news", "launch",
    "launches", "release", "releases", "announces", "announced", "update",
}
_TOKEN = re.compile(r"[a-z0-9][a-z0-9.\-+]*")

# A token carried by this share of the day's headlines is not evidence that two
# of them report the same event. `_STOP` holds the words that are always
# common; this handles the ones that are common *today*, which is the same
# problem and cannot be written down in advance. On an ordinary day "openai",
# "anthropic" and "claude" are each in a quarter of the titles, and two
# headlines sharing only those are two stories about one company -- which is
# how a warning about Claude's safety was filed as part of a Claude product
# launch.
COMMON_SHARE = 0.10
COMMON_FLOOR = 3
# Below this many headlines, frequency is not evidence of anything: in a batch
# of three items about one launch, that launch's own words are in every title.
# A real sweep carries a hundred-odd items, so this only ever binds on a short
# feed day -- when there is nothing to disambiguate anyway.
COMMON_MIN_BATCH = 20


def common_tokens(titles: list[str], *,
                  min_batch: int = COMMON_MIN_BATCH) -> frozenset[str]:
    """Tokens too widespread in this batch to identify a story."""
    if len(titles) < min_batch:
        return frozenset()
    counts: dict[str, int] = {}
    for title in titles:
        for tok in distinctive(title):
            counts[tok] = counts.get(tok, 0) + 1
    threshold = max(COMMON_FLOOR, int(len(titles) * COMMON_SHARE))
    return frozenset(tok for tok, n in counts.items() if n >= threshold)


def distinctive(title: str) -> set[str]:
    """The tokens worth matching on: names, products, version numbers.

    Version numbers survive intact — `gpt-5.5` is one token, not three — because
    the version *is* the story more often than the product name is.
    """
    out = set()
    for tok in _TOKEN.findall((title or "").lower()):
        tok = tok.strip(".-+")
        # A bare digit survives where a bare letter does not: "Corvid 4" and
        # "GPT 5" put the version in a one-character token, and dropping it
        # throws away the half of the name that says *which* release this is.
        if (len(tok) < 2 and not tok.isdigit()) or tok in _STOP:
            continue
        # Possessives and plurals: "OpenAI's" and "agents" must match "OpenAI"
        # and "agent" or half the rewrites cluster separately.
        tok = re.sub(r"'s$|’s$", "", tok)
        # A trailing -us, -is, -ss or -as is part of the word rather than a
        # plural: stripping it turns Corvus into "corvu" and analysis into
        # "analysi". Harmless while both sides agree, and unreadable the moment
        # the token lands in a run-log key.
        if len(tok) > 4 and tok.endswith("s") and tok[-2:] not in ("us", "is", "ss", "as"):
            tok = tok[:-1]
        out.add(tok)
    return out


MIN_SHARED = 2
MIN_CONTAINMENT = 0.4


def same_story(a: str, b: str, common: frozenset[str] = frozenset()) -> bool:
    """Do these two headlines report the same event?

    Two shared tokens, neither of them one everybody is writing about today.
    Deliberately conservative in one direction: a title with fewer than two
    distinctive tokens never merges with anything. Splitting one story into two
    rows costs the reader a duplicate; merging two stories into one row means
    the brief silently drops an event, which is the failure worth avoiding.
    """
    ta, tb = distinctive(a) - common, distinctive(b) - common
    if len(ta) < MIN_SHARED or len(tb) < MIN_SHARED:
        return False
    shared = ta & tb
    if len(shared) < MIN_SHARED:
        return False
    return len(shared) / min(len(ta), len(tb)) >= MIN_CONTAINMENT


@dataclass
class Story:
    """One event, and every item that reported it."""

    items: list[Item] = field(default_factory=list)
    score: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def origin(self) -> Item:
        """The first item to report it. Defines the story's identity.

        Stable as the cluster grows, which the *best* item is not: tomorrow's
        follow-up could outrank today's first report and silently re-key a
        story the run log has already seen.
        """
        return min(self.items, key=lambda i: (i.published or _FAR_FUTURE, i.url))

    @property
    def best(self) -> Item:
        """The item to link to: the lab's own post over a rewrite, and any
        direct link over a Google News redirect token."""
        return max(self.items, key=lambda i: (
            TIER_WEIGHT.get(i.tier, 0), 0 if i.is_aggregator_link else 1,
            len(i.summary)))

    @property
    def key(self) -> str:
        return story_key(self.origin.title)

    @property
    def title(self) -> str:
        return self.best.title

    @property
    def url(self) -> str:
        return self.best.url

    @property
    def published(self) -> datetime | None:
        dates = [i.published for i in self.items if i.published]
        return min(dates) if dates else None

    @property
    def when(self) -> str:
        return self.origin.when if self.published is None else _when(self.published)

    @property
    def outlets(self) -> list[str]:
        seen, out = set(), []
        for i in sorted(self.items, key=lambda i: -TIER_WEIGHT.get(i.tier, 0)):
            name = i.outlet or i.feed
            if name and name.lower() not in seen:
                seen.add(name.lower())
                out.append(name)
        return out

    @property
    def summary(self) -> str:
        return max((i.summary for i in self.items), key=len, default="")

    @property
    def from_lab(self) -> bool:
        return any(i.tier == "lab" for i in self.items)

    def as_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "outlets": self.outlets,
                "score": self.score, "corroboration": len(self.outlets),
                "from_lab": self.from_lab, "summary": self.summary,
                "published": self.published.isoformat() if self.published else "",
                "story_key": self.key}


_FAR_FUTURE = datetime(9999, 1, 1, tzinfo=timezone.utc)


def _when(dt: datetime) -> str:
    age = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    if age < 1:
        return f"{int(age * 60)}m ago"
    if age < 48:
        return f"{int(age)}h ago"
    return f"{int(age / 24)}d ago"


def story_key(title: str) -> str:
    """A stable id for a story, from its first report's headline.

    Deliberately *not* computed with `common_tokens`: which tokens are common
    depends on the rest of the day's feed, so a key that used them would change
    between runs and the run log would meet the same story as a new one.
    """
    toks = sorted(distinctive(title))[:8]
    return "|".join(toks) if toks else (title or "").strip().lower()[:80]


# One outlet is one vote. Reuters and Google News carrying the same wire copy
# is not corroboration, so votes are counted per outlet, not per item.
CORROBORATION_POINTS = 2
LAB_POINTS = 4
MAX_CORROBORATION = 8


def cluster(items: list[Item]) -> list[Story]:
    """Items in, stories out, scored and ordered.

    Single pass with a first-match merge. Feeds arrive in tier order, so the
    lab's own post is usually the seed of its own cluster; nothing depends on
    that, but it makes the common case cheap.
    """
    common = common_tokens([i.title for i in items])
    stories: list[Story] = []
    for item in items:
        for story in stories:
            if any(same_story(item.title, other.title, common) for other in story.items):
                story.items.append(item)
                break
        else:
            stories.append(Story(items=[item]))

    for story in stories:
        base, reasons = significance(story.title)
        extra = min((len(story.outlets) - 1) * CORROBORATION_POINTS, MAX_CORROBORATION)
        if extra > 0:
            reasons.append(f"+{extra} carried by {len(story.outlets)} outlets")
        lab = LAB_POINTS if story.from_lab else 0
        if lab:
            reasons.append(f"+{lab} announced by the lab itself")
        story.score = base + extra + lab
        story.reasons = reasons
    stories.sort(key=lambda s: (-s.score, s.published or _FAR_FUTURE))
    return stories


# --- fetching -------------------------------------------------------------

class AIFeeds:
    """Every feed, with a per-feed status line the brief can print.

    A feed that answered carries an item count. Anything else is a hole in the
    sweep, and the reason travels with the name, because they are different
    jobs: "unreachable" is a retry tomorrow, "HTTP 429" is a user-agent
    problem, and "no items returned" from a feed that served a body is a
    format change to go and look at today.

    `failures` is deliberately narrower than "did not contribute". A lab blog
    publishes a few posts a month, so "answered, nothing inside the window" is
    its ordinary state — counting that as a failure would put a banner on every
    brief, every day, and a warning that is always on is not a warning.
    """

    def __init__(self, fetcher: Fetcher) -> None:
        self.f = fetcher
        self.source_status: dict[str, str] = {}
        self.failures: dict[str, str] = {}
        self.answered: set[str] = set()

    def fetch_all(self, feeds: tuple[Feed, ...] = FEEDS, *, ttl: int = 900,
                  max_age_hours: float = 30.0, limit_per_feed: int = 80,
                  now: datetime | None = None) -> list[Item]:
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=max_age_hours)
        out: list[Item] = []
        # Lab feeds first so their post seeds its own cluster and the rewrites
        # join it, rather than the other way round.
        for feed in sorted(feeds, key=lambda f: -TIER_WEIGHT.get(f.tier, 0)):
            url = feed.url.format(q=_quote(GOOGLE_NEWS_QUERY)) if "{q}" in feed.url else feed.url
            headers = {"User-Agent": BROWSER_UA} if feed.browser_ua else None
            resp = self.f.fetch(url, headers=headers, ttl=ttl)
            if resp is None:
                # One retry before writing a feed off. Seven boards timing out
                # at once is what a busy minute looks like, not a dead feed.
                resp = self.f.fetch(url, headers=headers, ttl=0)
            if resp is None:
                self._fail(feed.name, "unreachable (network or timeout)")
                continue
            if not resp.ok:
                self._fail(feed.name, f"HTTP {resp.status}")
                continue
            items = parse_feed(resp.text, feed=feed.name, tier=feed.tier,
                               limit=limit_per_feed)
            if not items:
                self._fail(feed.name, "no items returned")
                continue
            self.answered.add(feed.name)
            # Undated items are kept. A feed that stops publishing dates should
            # show up as rows with an em dash, not as a feed that went quiet.
            fresh = [i for i in items if i.published is None or i.published >= cutoff]
            self.source_status[feed.name] = (
                f"{len(fresh)} in window ({len(items)} in feed)" if fresh
                else f"no items in the window ({len(items)} in feed, all older)")
            out.extend(fresh)
        return out

    def _fail(self, name: str, why: str) -> None:
        self.source_status[name] = why
        self.failures[name] = why


def _quote(q: str) -> str:
    from urllib.parse import quote_plus  # noqa: PLC0415 - only needed here

    return quote_plus(q)
