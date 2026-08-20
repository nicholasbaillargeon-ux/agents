"""Agent 3 — market open briefing.

Runs on a timer before the bell: futures, macro, overnight headlines, watchlist
movers, and which of your names report today. Written to be readable in thirty
seconds, in the same shape as the Morning Scorecard.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from ..brief import Brief, table
from ..llm import LLMUnavailable
from ..sources.news import News
from ..sources.prices import FUTURES, MACRO, YIELDS, PriceSource
from ..store import Run, record
from .base import AgentResult, Context

log = logging.getLogger(__name__)

NAME = "briefing"

SYSTEM = (
    "You write the two-sentence lede of a pre-market note for one experienced "
    "reader. You are given the overnight tape and headlines. State what moved "
    "and the most plausible reason visible in the data you were given. If the "
    "tape is quiet, say it is quiet — do not manufacture a narrative. Never "
    "predict. Two sentences, no heading, no preamble."
)

MARKET_HOLIDAY_HINT = (
    "US markets are closed today (weekend) — this brief covers the last session.")


def _arrow(pct: float | None) -> str:
    if pct is None:
        return "·"
    return "▲" if pct > 0.05 else ("▼" if pct < -0.05 else "—")


def change_label(q) -> str:
    """The move, in the instrument's own units.

    Yields move in basis points. Reporting a 4.66 -> 4.70 move as "+0.86%" is
    arithmetically true and journalistically wrong: the first model given that
    line opened its brief with "yields spiked 92 basis points overnight".
    """
    if q.symbol in YIELDS:
        if q.last is None or q.prev_close is None:
            return "n/a"
        bp = (q.last - q.prev_close) * 100.0
        return f"{_arrow(bp)} {bp:+.0f}bp"
    if q.change_pct is None:
        return "n/a"
    return f"{_arrow(q.change_pct)} {q.change_pct:+.2f}%"


def _quote_rows(quotes) -> list[list]:
    rows = []
    for q in quotes:
        rows.append([
            q.label or q.symbol,
            f"{q.last:,.2f}" if q.last is not None else "n/a",
            change_label(q),
            q.error or "",
        ])
    return rows


def earnings_today(symbols: list[str], *, today: date | None = None) -> tuple[list[list], str | None]:
    """Which watchlist names report today. Returns (rows, degradation)."""
    today = today or datetime.now(timezone.utc).date()
    try:
        import yfinance as yf
    except ImportError:
        return [], "yfinance unavailable: earnings calendar skipped"
    # ETFs have no earnings calendar and yfinance logs a 404 at ERROR level for
    # each one. That is an expected answer here, not an incident, and it should
    # not fill the journal every weekday morning.
    yf_log = logging.getLogger("yfinance")
    previous_level = yf_log.level
    yf_log.setLevel(logging.CRITICAL)
    rows, failures = [], 0
    for sym in symbols:
        try:
            cal = yf.Ticker(sym).calendar or {}
            dates = cal.get("Earnings Date") or []
            if not isinstance(dates, (list, tuple)):
                dates = [dates]
            for d in dates:
                d = d.date() if hasattr(d, "date") else d
                if d == today:
                    rows.append([sym, str(d), "confirmed" if len(dates) == 1 else "estimated"])
        except Exception as e:  # noqa: BLE001 - one bad symbol must not kill the brief
            failures += 1
            log.debug("earnings lookup failed for %s: %s", sym, e)
    yf_log.setLevel(previous_level)
    note = (f"earnings calendar unavailable for {failures}/{len(symbols)} symbols"
            if failures else None)
    return rows, note


def build_brief(ctx: Context, *, watchlist: list[str] | None = None,
                today: date | None = None) -> tuple[Brief, dict]:
    watchlist = [s.upper() for s in (watchlist or ctx.cfg.watchlist)]
    today = today or datetime.now(timezone.utc).date()
    prices = PriceSource(ctx.cfg.lake_dir if ctx.cfg.has_lake else None,
                         allow_network=not ctx.offline)
    news = News(ctx.fetcher)

    brief = Brief(title=f"Market open briefing — {today:%A %d %B %Y}", agent=NAME,
                  target=f"open-{today.isoformat()}", tags=["markets", "morning"])
    for d in ctx.base_degradations():
        brief.degrade(d)
    if today.weekday() >= 5:
        brief.degrade(MARKET_HOLIDAY_HINT)

    futures = prices.quotes(FUTURES)
    macro = prices.quotes(MACRO)
    movers = prices.movers(watchlist, top=len(watchlist))
    headlines = news.search("stock market futures premarket", limit=8, max_age_hours=18, ttl=600)
    if ctx.offline:
        # The calendar is a live lookup with no cache worth serving stale.
        er_rows, er_note = [], "offline mode: earnings calendar not checked"
    else:
        er_rows, er_note = earnings_today(watchlist, today=today)
    if er_note:
        brief.degrade(er_note)

    data = {"futures": futures, "macro": macro, "movers": movers,
            "headlines": headlines, "earnings": er_rows, "watchlist": watchlist}

    # Lede first — it is the part read on a phone.
    lede = _lede(ctx, data)
    if lede:
        brief.add("Overnight", lede)

    brief.add("Futures", table(["Contract", "Last", "Change", "Note"], _quote_rows(futures)))
    brief.add("Macro", table(["Instrument", "Last", "Change", "Note"], _quote_rows(macro)))

    if movers:
        rows = [[q.symbol, f"{q.last:,.2f}" if q.last is not None else "n/a",
                 f"{_arrow(q.change_pct)} {q.change_pct:+.2f}%"
                 if q.change_pct is not None else "n/a", q.error or ""]
                for q in movers]
        brief.add("Watchlist", table(["Symbol", "Last", "Change", "Note"], rows))
    else:
        brief.degrade("no watchlist quotes returned")

    brief.add("Reporting today",
              table(["Symbol", "Date", "Confidence"], er_rows) if er_rows
              else "_Nothing on the watchlist reports today._")

    if headlines:
        brief.add("Overnight headlines", "\n".join(
            f"- **{h.when}** [{h.title}]({h.url}) — _{h.source}_" for h in headlines))
        brief.source("Google News RSS", note="market headlines, last 18 hours")
    else:
        brief.degrade("no overnight headlines retrieved")

    for note in prices.notes:
        brief.degrade(note)
    brief.source("Yahoo Finance via yfinance", note="futures, macro and watchlist quotes")
    brief.extra_meta["watchlist"] = ",".join(watchlist)
    return brief, data


def _lede(ctx: Context, data: dict) -> str:
    if not ctx.llm.available:
        return ""
    tape = []
    for q in data["futures"] + data["macro"]:
        if q.last is not None and q.prev_close is not None:
            # The level goes in too: "+4bp" means nothing without "4.70%".
            tape.append(f"{q.label}: now {q.last:,.2f}, "
                        f"{change_label(q).split(' ', 1)[-1]} on the session")
    for q in data["movers"][:6]:
        if q.change_pct is not None:
            tape.append(f"{q.symbol}: {q.change_pct:+.2f}%")
    heads = "\n".join(f"- {h.title}" for h in data["headlines"][:8])
    prompt = (f"TAPE:\n" + "\n".join(tape) + f"\n\nHEADLINES:\n{heads or '(none retrieved)'}")
    try:
        return ctx.llm.complete(prompt, system=SYSTEM, max_tokens=300, fast=True)
    except LLMUnavailable as e:
        log.warning("lede skipped: %s", e)
        return ""


def run(ctx: Context, *, watchlist: list[str] | None = None, commit: bool = True,
        today: date | None = None) -> AgentResult:
    started = datetime.now(timezone.utc)
    res = AgentResult(agent=NAME, target="open")
    try:
        brief, data = build_brief(ctx, watchlist=watchlist, today=today)
    except Exception as e:  # noqa: BLE001
        log.exception("briefing failed")
        res.ok, res.error = False, f"{type(e).__name__}: {e}"
        record(ctx.db, Run(agent=NAME, ok=False, error=res.error,
                           started_at=started.timestamp()))
        return res

    res.brief = brief
    res.degradations = list(brief.degradations)
    res.artifact = brief.write(ctx.cfg.out_dir / NAME)
    if commit:
        try:
            cr = ctx.notes.commit_file(f"briefings/{brief.filename}", brief.render(),
                                       f"market open briefing {brief.date}")
            res.data["commit"] = {"sha": cr.sha, "committed": cr.committed}
        except Exception as e:  # noqa: BLE001
            res.degrade(f"could not commit briefing: {e}")

    live = [q for q in data["futures"] if q.ok]
    res.summary = (f"{len(live)}/{len(data['futures'])} futures priced, "
                   f"{len(data['headlines'])} headlines, "
                   f"{len(data['earnings'])} reporting today")
    record(ctx.db, Run(agent=NAME, target=res.target, ok=True, artifact=str(res.artifact),
                       summary=res.summary, degradations=res.degradations,
                       started_at=started.timestamp(),
                       duration_s=(datetime.now(timezone.utc) - started).total_seconds()))
    return res
