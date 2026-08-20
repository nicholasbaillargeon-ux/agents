"""Headlines via Google News RSS — keyless, and the only free news feed that
answered on this host (Yahoo's RSS and Stooq both refuse it).

Parsed with the stdlib so the test suite needs no network and no extra
dependency. RSS in the wild is malformed often enough that every field access
here is defensive.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from xml.etree import ElementTree

from ..netcache import Fetcher

log = logging.getLogger(__name__)

RSS = ("https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
_TAG = re.compile(r"<[^>]+>")


@dataclass
class Headline:
    title: str
    url: str
    source: str
    published: datetime | None = None

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


class News:
    def __init__(self, fetcher: Fetcher) -> None:
        self.f = fetcher

    def search(self, query: str, *, limit: int = 8, max_age_hours: float | None = None,
               ttl: int = 900) -> list[Headline]:
        resp = self.f.fetch(RSS.format(q=quote_plus(query)), ttl=ttl)
        if not resp or not resp.ok:
            log.warning("news feed unavailable for %r", query)
            return []
        return _parse(resp.text, limit=limit, max_age_hours=max_age_hours)

    def for_ticker(self, ticker: str, company: str = "", **kw) -> list[Headline]:
        q = f'"{company}" OR {ticker} stock' if company else f"{ticker} stock"
        return self.search(q, **kw)


def _parse(xml_text: str, *, limit: int = 8, max_age_hours: float | None = None) -> list[Headline]:
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as e:
        log.warning("malformed RSS: %s", e)
        return []
    out: list[Headline] = []
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
              if max_age_hours else None)
    for item in root.iter("item"):
        title = _text(item, "title")
        if not title:
            continue
        published = _date(_text(item, "pubDate"))
        if cutoff and published and published < cutoff:
            continue
        out.append(Headline(
            title=title,
            url=_text(item, "link"),
            source=_text(item.find("source"), None) or _domain(_text(item, "link")),
            published=published,
        ))
        if len(out) >= limit:
            break
    return out


def _text(node, tag: str | None) -> str:
    if node is None:
        return ""
    el = node if tag is None else node.find(tag)
    if el is None or el.text is None:
        return ""
    return html.unescape(_TAG.sub("", el.text)).strip()


def _date(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _domain(url: str) -> str:
    m = re.search(r"https?://([^/]+)", url or "")
    return m.group(1).replace("www.", "") if m else "unknown"
