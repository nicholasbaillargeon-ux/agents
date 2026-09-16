"""Agent 7 — the deal book.

The agent does the clipping; you do the thinking. It watches M&A feeds, and
for anything matching your filters — size, sector, or an advisor you are
watching — it drafts a structured one-pager into Postgres: parties, price,
implied multiple where disclosed, rationale, financing, open questions. Then it
stops. The `my_view` column is yours and no sweep ever writes to it, because a
library of fifteen deals you have actually formed a view on is the product, and
a library of fifteen deals a model summarised is not.

Cost discipline mirrors the scout: cheap gates first, and a deal already in the
book is never re-read by a model. The order is keyword gate, dedupe, ask
Postgres what it already has, fetch the release body only for what is left,
filter on size and advisors, and only then spend a model call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..brief import Brief, table
from ..dealstore import (DealBookUnavailable, DealRecord, connect, ensure_schema,
                         list_deals, upsert)
from ..sources.deals import Deal, Deals
from ..store import Run, record
from .base import AgentResult, Context, finalize

log = logging.getLogger(__name__)

NAME = "dealbook"

# Ceilings, not targets. Each release body is one HTTP round trip and each
# one-pager is one model call, so a noisy news day cannot turn into an
# unbounded bill.
MAX_ENRICH = 30
MAX_DRAFTS = 12

STRUCTURE_SYSTEM = (
    "You read one M&A press release and fill in a deal one-pager for an analyst "
    "who will form their own view. Return JSON only, with these keys:\n"
    '  "is_ma": true only if this announces a merger, acquisition, minority '
    "stake, take-private or divestiture of a business. Hiring, contract awards, "
    "product launches and fund closings are false.\n"
    '  "acquirer", "target": legal names as written in the release.\n'
    '  "sector": two or three words for the target\'s industry.\n'
    '  "value_usd": total announced consideration in US dollars as a number, or '
    "null if undisclosed or quoted in another currency.\n"
    '  "consideration": one of "cash", "stock", "mixed", "undisclosed".\n'
    '  "implied_multiple": the multiple the release discloses (e.g. "14.2x '
    'EV/EBITDA 2026E"), or "" if none is stated. Never compute one yourself.\n'
    '  "rationale": the stated strategic logic, one or two sentences, in your '
    "own words but claiming nothing the release does not.\n"
    '  "financing": how it is being paid for, or "" if not stated.\n'
    '  "open_questions": 2-4 questions a buy-side analyst would still have. '
    "These must be answerable in principle and specific to this deal — no "
    '"will it create value?".\n'
    '  "advisors": [{"name": "...", "side": "acquirer"|"target"|"unclear"}] for '
    "every financial advisor named.\n"
    "Use only the release. Where it is silent, say so with null or an empty "
    "string. Do not estimate, do not infer a price, do not add market context."
)


@dataclass
class DealbookResult:
    scanned: int = 0
    candidates: int = 0
    already_known: int = 0
    enriched: int = 0
    drafted: int = 0
    stored: int = 0
    new: list[DealRecord] = field(default_factory=list)
    flagged: list[DealRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def matches_advisor(deal: Deal, watched: list[str]) -> str:
    """The watched advisor named in this release, or "".

    Matched on the release body as well as the parsed advisor list, because the
    sentence naming a boutique is sometimes phrased in a way `find_advisors`
    does not catch, and a false negative here is the one failure that defeats
    the point of watching a name at all.
    """
    haystacks = [a.lower() for a in deal.advisors] + [deal.body.lower()]
    for name in watched:
        needle = name.strip().lower()
        if not needle:
            continue
        if any(needle in h for h in haystacks):
            return name.strip()
    return ""


HEADLINE_ONLY_SYSTEM = (
    "You are given only a news headline about an M&A deal — no press release. "
    "Return JSON with the same keys as a full one-pager, but fill ONLY what the "
    'headline itself states: "is_ma", "acquirer", "target", "sector", '
    '"value_usd", and "open_questions". Set "consideration", '
    '"implied_multiple", "rationale", "financing" to empty strings and '
    '"advisors" to []. Inventing a rationale from the company names is the '
    "specific failure to avoid: a one-pager that reads as though someone read "
    "the release, when nobody did, is worse than a stub. The open questions "
    "should be exactly what you would need the release to answer."
)


def matches_sector(deal: Deal, sectors: list[str]) -> str:
    """The watched sector keyword this deal hits, or "".

    Keyword-matched on the headline and the release body, deliberately before
    any model call, for the same reason the scout gates before its verdicts: a
    deal rejected here costs nothing, and a model asked to classify forty
    headlines a day costs money every day.

    The headline is weighted by being checked first only for reporting; both
    are searched, because a release's own body names its industry far more
    reliably than a press-office headline does.
    """
    haystack = (deal.title + " " + deal.body[:6000]).lower()
    for term in sectors:
        if term and term in haystack:
            return term
    return ""


def passes_filters(deal: Deal, *, min_usd: float, advisor: str,
                   sector: str = "") -> tuple[bool, str]:
    """(keep, why). An advisor match overrides the size floor.

    Size is only applied to dollar figures. A deal announced in Australian
    dollars has a `value_usd` that is not dollars, so comparing it to a dollar
    threshold is meaningless — it is kept and labelled rather than silently
    admitted or silently dropped.
    """
    if advisor:
        return True, f"{advisor} is advising"
    if sector:
        return True, f"watched sector: {sector}"
    if not deal.is_usd:
        return True, f"value quoted in {deal.currency}; size filter not applied"
    if deal.value_usd is None:
        # A headline-only item with no price is a rumour round-up or a tuck-in
        # too small to name a number. Neither is worth a page in the book.
        return False, "no disclosed value and no watched advisor"
    if deal.value_usd < min_usd:
        return False, f"{deal.value_label} below the {min_usd / 1e6:,.0f}M floor"
    return True, f"{deal.value_label} clears the size floor"


def draft(ctx: Context, deal: Deal, *, advisor: str, why: str) -> DealRecord | None:
    """One structured one-pager, or None if the model says this is not a deal."""
    # A thin, honestly-labelled stub beats a full-looking page written from a
    # headline: the second is indistinguishable from a real read at a glance.
    if deal.body:
        prompt = f"HEADLINE: {deal.title}\n\nRELEASE:\n{deal.body[:18000]}"
        system = STRUCTURE_SYSTEM
    else:
        prompt = f"HEADLINE: {deal.title}\nSOURCE: {deal.source}"
        system = HEADLINE_ONLY_SYSTEM
    # 1200 truncated a long advisor list mid-object on the first live run, and
    # a cut-off object is unparseable JSON — the whole one-pager, lost.
    parsed = ctx.llm.json(prompt, system=system, max_tokens=2000, default=None)
    if not isinstance(parsed, dict):
        return None
    if parsed.get("is_ma") is False:
        return None

    def s(key: str) -> str:
        return str(parsed.get(key) or "").strip()

    value = parsed.get("value_usd")
    try:
        value = float(value) if value is not None else None
    except (TypeError, ValueError):
        value = None
    # The model read the release; the regex read the headline. Where a release
    # was actually available, the model's silence is evidence — it means the
    # release states no price, and overriding that with a regex hit from
    # somewhere else in the text is how "$1 billion in marketplace sales"
    # became a deal price. So the fallback applies only to headline-only items,
    # where there was no release for the model to be silent about.
    if value is None and deal.is_usd and not deal.body:
        value = deal.value_usd

    advisors = []
    for a in (parsed.get("advisors") or []):
        if isinstance(a, dict) and str(a.get("name") or "").strip():
            advisors.append({"name": str(a["name"]).strip(),
                             "side": str(a.get("side") or "unclear").strip()})
    for name in deal.advisors:
        if not any(name.lower() == a["name"].lower() for a in advisors):
            advisors.append({"name": name, "side": "unclear"})

    questions = [str(q).strip() for q in (parsed.get("open_questions") or [])
                 if str(q).strip()]
    return DealRecord(
        deal_key=deal.key,
        headline=deal.title,
        url=deal.url,
        source=deal.source or deal.feed,
        acquirer=s("acquirer") or deal.acquirer,
        target=s("target") or deal.target,
        sector=s("sector"),
        value_usd=value,
        currency=deal.currency,
        consideration=s("consideration"),
        implied_multiple=s("implied_multiple"),
        rationale=s("rationale"),
        financing=s("financing"),
        open_questions=questions
        + ([] if deal.body else ["_Headline-only: no release body was reachable, "
                                 "so structure, financing and advisors are unfilled._"])
        + ([f"_Matched because: {why}_"] if why else []),
        advisors=advisors,
        flagged_advisor=advisor,
    )


def sweep(ctx: Context, *, max_enrich: int = MAX_ENRICH,
          max_drafts: int = MAX_DRAFTS) -> DealbookResult:
    out = DealbookResult()
    src = Deals(ctx.fetcher)
    raw = src.headlines()
    out.notes.extend(src.notes)
    out.scanned = len(raw)

    # Dedupe: one deal reported by four outlets is one deal.
    unique: dict[str, Deal] = {}
    for d in raw:
        if d.key not in unique:
            unique[d.key] = d
    out.candidates = len(unique)

    known: set[str] = set()
    conn = None
    try:
        cm = connect(ctx.cfg.dealbook_dsn)
        conn = cm.__enter__()
        ensure_schema(conn)
        known = {r["deal_key"] for r in list_deals(conn, limit=5000)}
    except DealBookUnavailable as e:
        out.notes.append(f"deal book not persisted: {e}")
        cm = None

    try:
        fresh = [d for d in unique.values() if d.key not in known]
        out.already_known = len(unique) - len(fresh)

        # Fetchable first, substantive second. The ordering matters more than
        # it looks: ranked on "named both parties and a price" alone, the whole
        # enrichment budget went to Google News items whose bodies cannot be
        # fetched at all, and the PR Newswire releases — the only ones carrying
        # financing, multiples and advisor names — were never reached.
        fresh.sort(key=lambda d: (d.body_fetchable,
                                  bool(d.acquirer and d.target),
                                  d.value_usd or 0.0), reverse=True)

        drafts: list[tuple[Deal, str, str]] = []
        for deal in fresh[:max_enrich]:
            src.enrich(deal)
            out.enriched += 1
            advisor = matches_advisor(deal, ctx.cfg.deal_advisors)
            sector = matches_sector(deal, ctx.cfg.deal_sectors)
            keep, why = passes_filters(deal, min_usd=ctx.cfg.deal_min_usd,
                                       advisor=advisor, sector=sector)
            if keep:
                drafts.append((deal, advisor, why))

        if not ctx.llm.available:
            out.notes.append("no LLM: one-pagers not drafted, so nothing was stored")
            return out

        for deal, advisor, why in drafts[:max_drafts]:
            rec = draft(ctx, deal, advisor=advisor, why=why)
            out.drafted += 1
            if rec is None:
                continue
            out.new.append(rec)
            if rec.flagged_advisor:
                out.flagged.append(rec)
            if conn is not None:
                try:
                    upsert(conn, rec)
                    out.stored += 1
                except Exception as e:  # noqa: BLE001 - one bad row must not kill the sweep
                    log.warning("upsert failed for %s: %s", rec.deal_key, e)
                    out.notes.append(f"could not store {rec.deal_key}: {type(e).__name__}")
        if len(drafts) > max_drafts:
            out.notes.append(
                f"{len(drafts) - max_drafts} more deals passed the filters than the "
                f"{max_drafts}-draft budget allows; they are not marked seen and "
                "will be picked up on the next run")
        return out
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)


def _one_pager(rec: DealRecord) -> str:
    bits = [f"**{rec.acquirer or '?'} → {rec.target or '?'}**"]
    meta = []
    if rec.value_usd is not None:
        unit = rec.currency or "$"
        sep = "" if unit == "$" else " "
        meta.append(f"{unit}{sep}{rec.value_usd / 1e9:,.2f}B" if rec.value_usd >= 1e9
                    else f"{unit}{sep}{rec.value_usd / 1e6:,.0f}M")
    else:
        meta.append("undisclosed")
    if rec.consideration:
        meta.append(rec.consideration)
    if rec.sector:
        meta.append(rec.sector)
    bits.append(" · ".join(meta))
    if rec.flagged_advisor:
        bits.append(f"🚩 **{rec.flagged_advisor} is advising.**")
    if rec.implied_multiple:
        bits.append(f"- **Multiple disclosed:** {rec.implied_multiple}")
    if rec.rationale:
        bits.append(f"- **Rationale:** {rec.rationale}")
    if rec.financing:
        bits.append(f"- **Financing:** {rec.financing}")
    if rec.advisors:
        bits.append("- **Advisors:** " + ", ".join(
            f"{a['name']} ({a['side']})" for a in rec.advisors))
    if rec.open_questions:
        bits.append("- **Open questions:**\n" + "\n".join(
            f"  - {q}" for q in rec.open_questions))
    bits.append(f"- [{rec.headline[:110]}]({rec.url}) — _{rec.source}_")
    return "\n".join(bits)


def build_brief(ctx: Context, result: DealbookResult) -> Brief:
    today = datetime.now(timezone.utc).date()
    brief = Brief(title=f"Deal book — {today:%A %d %B %Y}", agent=NAME,
                  target=f"deals-{today.isoformat()}", tags=["m&a", "dealbook"])
    for d in ctx.base_degradations():
        brief.degrade(d)
    for n in result.notes:
        brief.degrade(n)

    if result.flagged:
        brief.add("Flagged advisors",
                  "\n\n---\n\n".join(_one_pager(r) for r in result.flagged))
    others = [r for r in result.new if not r.flagged_advisor]
    if others:
        brief.add("New one-pagers", "\n\n---\n\n".join(_one_pager(r) for r in others))
    if not result.new:
        brief.add("New one-pagers",
                  "_Nothing new cleared the filters this run._")

    brief.add("Funnel", table(
        ["Stage", "Count"],
        [["Headlines scanned", result.scanned],
         ["Distinct deals", result.candidates],
         ["Already in the book", result.already_known],
         ["Release bodies read", result.enriched],
         ["One-pagers drafted", result.drafted],
         ["Stored in Postgres", result.stored]]))
    brief.add("Your turn",
              "The agent stops here. Add your own view with "
              "`agents deals --note <id> \"...\"` — nothing in a later sweep "
              "will overwrite it. `agents deals --list` shows what is waiting.")
    brief.source("Google News RSS + PR Newswire M&A feed", note="deal announcements")
    brief.extra_meta["stored"] = result.stored
    brief.extra_meta["flagged"] = len(result.flagged)
    return brief


def run(ctx: Context, *, commit: bool = True, max_enrich: int = MAX_ENRICH,
        max_drafts: int = MAX_DRAFTS) -> AgentResult:
    started = datetime.now(timezone.utc)
    res = AgentResult(agent=NAME, target="deals")
    try:
        result = sweep(ctx, max_enrich=max_enrich, max_drafts=max_drafts)
        brief = build_brief(ctx, result)
    except Exception as e:  # noqa: BLE001
        log.exception("dealbook failed")
        res.ok, res.error = False, f"{type(e).__name__}: {e}"
        record(ctx.db, Run(agent=NAME, ok=False, error=res.error,
                           started_at=started.timestamp()))
        return res

    res.brief = brief
    finalize(ctx, brief, res)
    res.artifact = brief.write(ctx.cfg.out_dir / NAME)
    if commit:
        try:
            cr = ctx.notes.commit_file(f"dealbook/{brief.filename}", brief.render(),
                                       f"deal book {brief.date}")
            res.data["commit"] = {"sha": cr.sha, "committed": cr.committed}
        except Exception as e:  # noqa: BLE001
            res.degrade(f"could not commit deal book: {e}")

    res.summary = (f"{result.scanned} headlines, {result.candidates} distinct, "
                   f"{result.enriched} read, {result.stored} stored, "
                   f"{len(result.flagged)} advisor-flagged")
    res.data.update({"stored": result.stored, "new": len(result.new),
                     "flagged": len(result.flagged), "scanned": result.scanned})
    record(ctx.db, Run(agent=NAME, target=res.target, ok=True, artifact=str(res.artifact),
                       summary=res.summary, degradations=res.degradations,
                       started_at=started.timestamp(),
                       duration_s=(datetime.now(timezone.utc) - started).total_seconds()))
    return res
