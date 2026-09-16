"""The Morning Tape: the sections of the briefing that are about the market
rather than about your watchlist.

Not a separate agent. The briefing is already a pre-bell product with a timer,
a dashboard row and a notes commit behind it; a second agent covering the same
minute of the day would duplicate all three and give you two documents to read
instead of one. So this is a set of section builders the briefing calls, and
everything it adds lands in the same brief.

What it adds, in the order it is read: the Treasury curve and the day's move in
basis points, the curve spreads a desk quotes, what fed funds futures price for
the next three meetings, overnight M&A, and one talking point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from ..brief import table
from ..grounding import ungrounded
from ..llm import LLMUnavailable
from ..sources.deals import Deals
from ..sources.rates import (BP, SPREADS, Treasury, clean_anchor, fed_path,
                             fomc_meetings)
from .base import Context

log = logging.getLogger(__name__)

# How many upcoming FOMC decisions to price. Three covers roughly six months,
# which is as far out as the strip is liquid enough to be worth quoting.
MEETINGS_SHOWN = 3
# Deals shown in the brief. The deal *book* keeps everything; this is the
# skim-on-a-phone list.
DEALS_SHOWN = 6

TAPE_SYSTEM = (
    "You brief one experienced reader on the day's macro tape before the open. "
    "You are given levels and moves that are already correct — your job is to "
    "say what they add up to, not to restate them. Use only the figures given. "
    "Never predict, never advise, and if the tape is quiet say so.\n\n"
    "Return JSON with exactly two keys:\n"
    '  "tape": 2-3 sentences on what moved and what it implies about growth, '
    "inflation or policy expectations. Name the specific levels that matter.\n"
    '  "talking_point": one thing this reader could say in an interview today '
    "that shows they follow markets. State the observation, then why it matters "
    "in one clause. 2-3 sentences, no preamble, no 'you could say that'. It must "
    "follow from the data given, not from general knowledge."
)


def _bp_change(now: float | None, prev: float | None) -> float | None:
    if now is None or prev is None:
        return None
    return (now - prev) * BP


def _fmt_bp(v: float | None) -> str:
    if v is None:
        return "n/a"
    arrow = "▲" if v > 0.5 else ("▼" if v < -0.5 else "—")
    return f"{arrow} {v:+.0f}bp"


def _fmt_pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}%"


@dataclass
class TapeData:
    """Everything the tape sections fetched, for the lede and for the push."""
    curve: object = None
    prior: object = None
    spreads: list[tuple[str, float | None, float | None]] = field(default_factory=list)
    path: object = None
    meetings: list[date] = field(default_factory=list)
    deals: list = field(default_factory=list)
    anchor_note: str = ""
    talking_point: str = ""
    tape_lede: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def ten_year(self) -> float | None:
        return self.curve.get("10Y") if self.curve else None

    @property
    def two_ten(self) -> float | None:
        return self.curve.spread_bp("10Y", "2Y") if self.curve else None


def collect(ctx: Context, *, today: date | None = None) -> TapeData:
    """Fetch the tape. Every source here degrades to a note rather than raising."""
    today = today or datetime.now(timezone.utc).date()
    data = TapeData()

    tsy = Treasury(ctx.fetcher)
    data.curve, data.prior = tsy.latest(today=today)
    data.notes.extend(tsy.notes)
    if data.curve:
        for label, long_leg, short_leg in SPREADS:
            now = data.curve.spread_bp(long_leg, short_leg)
            was = data.prior.spread_bp(long_leg, short_leg) if data.prior else None
            data.spreads.append((label, now, None if was is None else now - was))

    data.meetings, meeting_notes = fomc_meetings(ctx.fetcher, today=today)
    data.notes.extend(meeting_notes)
    data.path = fed_path(today=today, allow_network=not ctx.offline)
    data.notes.extend(data.path.notes)
    if data.path and data.meetings:
        data.anchor_note = clean_anchor(data.path, data.meetings, today=today)

    src = Deals(ctx.fetcher)
    try:
        data.deals = src.headlines()
    except Exception as e:  # noqa: BLE001 - a bad feed must not cost the brief
        log.warning("deal feed failed: %s", e)
        data.notes.append(f"M&A feeds failed ({type(e).__name__}); deal section empty")
    data.notes.extend(src.notes)
    return data


def rate_rows(data: TapeData) -> list[list]:
    if not data.curve:
        return []
    rows = []
    for label in ("3M", "2Y", "5Y", "10Y", "30Y"):
        now = data.curve.get(label)
        if now is None:
            continue
        rows.append([label, _fmt_pct(now),
                     _fmt_bp(_bp_change(now, data.prior.get(label) if data.prior else None))])
    return rows


def spread_rows(data: TapeData) -> list[list]:
    return [[label, "n/a" if now is None else f"{now:+.0f}bp",
             _fmt_bp(chg),
             "inverted" if now is not None and now < 0 else ""]
            for label, now, chg in data.spreads]


def fed_rows(data: TapeData) -> list[list]:
    """One row per upcoming FOMC decision: what the strip prices for that month."""
    if not data.path or data.path.spot is None or not data.meetings:
        return []
    rows = []
    for meeting in data.meetings[:MEETINGS_SHOWN]:
        rate = data.path.at(meeting.year, meeting.month)
        if rate is None:
            continue
        bp = data.path.move_bp(meeting.year, meeting.month)
        steps = data.path.steps(meeting.year, meeting.month)
        direction = "cuts" if steps is not None and steps < 0 else "hikes"
        rows.append([
            f"{meeting:%d %b %Y}",
            f"{rate:.2f}%",
            "n/a" if bp is None else f"{bp:+.0f}bp",
            "n/a" if steps is None else f"{abs(steps):.1f} × 25bp {direction}",
        ])
    return rows


def deal_rows(data: TapeData, *, limit: int = DEALS_SHOWN) -> list[list]:
    """Named, priced deals first — those are the ones worth a sentence."""
    ranked = sorted(
        data.deals,
        key=lambda d: (bool(d.acquirer and d.target), d.value_usd or 0.0),
        reverse=True)
    seen, rows = set(), []
    for d in ranked:
        if d.key in seen:
            continue
        seen.add(d.key)
        parties = (f"{d.acquirer} → {d.target}" if d.acquirer and d.target
                   else "_(parties not parsed)_")
        rows.append([parties, d.value_label, f"[{d.title[:80]}]({d.url})", d.when])
        if len(rows) >= limit:
            break
    return rows


def add_sections(ctx: Context, brief, data: TapeData) -> None:
    """Append the tape to a brief that already has futures and movers on it."""
    rows = rate_rows(data)
    if rows:
        stamp = f" — Treasury par curve, {data.curve.as_of:%d %b}" if data.curve else ""
        brief.add("Rates" + stamp, table(["Tenor", "Level", "Change"], rows))
        brief.source("US Treasury daily par yield curve",
                     "https://home.treasury.gov/resource-center/data-chart-center/interest-rates",
                     note="keyless XML feed; levels in percent, changes in basis points")
    if data.spreads:
        brief.add("Curve", table(["Spread", "Level", "Change", "Note"], spread_rows(data)))

    frows = fed_rows(data)
    if frows:
        body = table(["FOMC decision", "Priced rate", "vs spot", "Implied"], frows)
        spot = data.path.spot
        body += (f"\n\nSpot anchor: front fed funds future implies **{spot:.2f}%**. "
                 "Each row is what the ZQ contract for that meeting's month prices.")
        if data.anchor_note:
            body += f"\n\n_Caveat: {data.anchor_note}._"
        brief.add("What the strip prices", body)
        brief.source("CME 30-day fed funds futures (ZQ) via Yahoo Finance",
                     note="implied rate = 100 − price; settles to the month's average EFFR")
        brief.source("FOMC meeting calendar",
                     "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
                     note="scraped, not hardcoded — dates move")

    drows = deal_rows(data)
    if drows:
        brief.add("M&A overnight", table(["Parties", "Value", "Headline", "Age"], drows))
        brief.source("Google News RSS + PR Newswire M&A feed",
                     note="keyword-gated to deal announcements; see the deal book for the full set")
    elif not data.notes:
        brief.add("M&A overnight", "_No new deal announcements matched in the last 36 hours._")


def narrate(ctx: Context, data: TapeData, tape_context: str = "") -> tuple[str, str, list[str]]:
    """(tape lede, talking point, figures neither is supported by).

    One model call for both, because they read the same tape and two calls
    would let them contradict each other on the same morning.
    """
    if not ctx.llm.available:
        return "", "", []
    lines: list[str] = []
    if data.curve:
        lines.append(f"Treasury curve as of {data.curve.as_of}:")
        for label in ("3M", "2Y", "5Y", "10Y", "30Y"):
            now = data.curve.get(label)
            if now is None:
                continue
            chg = _bp_change(now, data.prior.get(label) if data.prior else None)
            lines.append(f"  {label}: {now:.2f}%" +
                         (f", {chg:+.0f}bp on the day" if chg is not None else ""))
    for label, now, chg in data.spreads:
        if now is None:
            continue
        lines.append(f"{label} spread: {now:+.0f}bp" +
                     (f", {chg:+.0f}bp on the day" if chg is not None else ""))
    if data.path and data.path.spot is not None:
        lines.append(f"Fed funds futures imply {data.path.spot:.2f}% spot.")
        for meeting in data.meetings[:MEETINGS_SHOWN]:
            bp = data.path.move_bp(meeting.year, meeting.month)
            rate = data.path.at(meeting.year, meeting.month)
            if bp is None or rate is None:
                continue
            lines.append(f"  by the {meeting:%d %b %Y} meeting: {rate:.2f}% "
                         f"({bp:+.0f}bp vs spot)")
    deals = deal_rows(data, limit=5)
    if deals:
        lines.append("Overnight M&A:")
        lines.extend(f"  {r[0]} — {r[1]} — {r[2].split('](')[0].lstrip('[')}" for r in deals)
    prompt = (tape_context + "\n" if tape_context else "") + "\n".join(lines)
    try:
        parsed = ctx.llm.json(prompt, system=TAPE_SYSTEM, max_tokens=700, default=None)
    except LLMUnavailable as e:
        log.warning("tape narration skipped: %s", e)
        return "", "", []
    if not isinstance(parsed, dict):
        return "", "", []
    tape = str(parsed.get("tape") or "").strip()
    point = str(parsed.get("talking_point") or "").strip()
    return tape, point, ungrounded(tape + "\n" + point, prompt)


def push_body(data: TapeData, equity_line: str = "") -> str:
    """The ninety-second read, for a phone lock screen.

    Deliberately not the brief. The brief is five tables and is read at a desk;
    this is the part that has to survive being read while walking, so it is one
    line of levels, the model's two sentences, the talking point, and the
    biggest deal — in that order, because that is the order they stop being
    worth reading.
    """
    head: list[str] = []
    if equity_line:
        head.append(equity_line)
    if data.ten_year is not None:
        chg = _bp_change(data.ten_year, data.prior.get("10Y") if data.prior else None)
        head.append(f"**10y** {data.ten_year:.2f}%" +
                    (f" {_fmt_bp(chg)}" if chg is not None else ""))
    if data.two_ten is not None:
        head.append(f"**2s10s** {data.two_ten:+.0f}bp")

    parts: list[str] = []
    if head:
        parts.append(" · ".join(head))
    if data.path and data.path.spot is not None and data.meetings:
        for meeting in data.meetings[:1]:
            rate = data.path.at(meeting.year, meeting.month)
            bp = data.path.move_bp(meeting.year, meeting.month)
            if rate is None or bp is None:
                continue
            steps = data.path.steps(meeting.year, meeting.month) or 0.0
            word = "cuts" if steps < 0 else "hikes"
            parts.append(f"**Fed** {meeting:%d %b}: {rate:.2f}% priced, {bp:+.0f}bp vs "
                         f"{data.path.spot:.2f}% spot ({abs(steps):.1f}×25bp {word})")
    if data.tape_lede:
        parts.append(data.tape_lede)
    if data.talking_point:
        parts.append(f"**Talking point.** {data.talking_point}")
    rows = deal_rows(data, limit=1)
    if rows:
        headline = rows[0][2].split("](")[0].lstrip("[")
        parts.append(f"**Top deal:** {rows[0][0]} — {rows[0][1]} · {headline}")
    return "\n\n".join(parts) if parts else "No tape available this morning."
