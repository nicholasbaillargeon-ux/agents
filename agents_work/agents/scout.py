"""Agent 4 — internship scout.

Sweeps quant / fintech / AI job boards nightly and surfaces only what is new
since the last run.

The diff is the product. A scout that re-lists yesterday's ninety postings is
worse than no scout, because you stop reading it. `store.mark_new` does the
diff in one transaction, so a crash mid-run cannot half-remember a batch and
silently drop those postings out of tomorrow's list too.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from ..brief import Brief, table
from ..llm import LLMUnavailable
from ..sources.jobs import REGISTRY, Boards, Posting, posting_key, score
from ..store import Run, mark_new, record, seen_count, seen_since, unseen_keys
from .base import AgentResult, Context

log = logging.getLogger(__name__)

NAME = "scout"

# How many new postings one brief will show. Steady-state nightly volume
# across these boards is well under this, so the cap only binds after a
# registry change adds a board -- and when it does, the brief says so and the
# remainder queue rather than vanishing.
DISPLAY_LIMIT = 100

# How many of the day's "apply" verdicts to lift out as a shortlist.
TOP_MATCHES = 10

SYSTEM = (
    "You triage job postings for one candidate: an undergraduate targeting "
    "quant / fintech / AI internships, strongest in Python, data pipelines, and "
    "full-slice deployment. For each posting you are given, return a JSON array "
    "of objects with keys `url`, `verdict` (one of: apply, maybe, skip), and "
    "`why` (at most 15 words, concrete). Judge fit, not prestige. A posting "
    "that is senior, non-technical, or in an unrelated function is `skip` even "
    "at a famous firm. Return ONLY the JSON array."
)


# One verdict costs the model roughly 40 tokens of JSON. Asking for a hundred in
# a single completion overruns any sane max_tokens and the reply is truncated
# mid-array — which parses as nothing, so a larger cap would silently cost every
# verdict rather than a few. Batching keeps each reply comfortably inside its
# budget and makes one bad batch cost only that batch.
RANK_BATCH = 25
TOKENS_PER_VERDICT = 80


def rank(ctx: Context, postings: list[Posting], *, limit: int | None = None,
         batch_size: int = RANK_BATCH) -> dict[str, dict]:
    """LLM verdicts keyed by posting key. Empty dict when the model is
    unavailable — the deterministic score already ordered the list."""
    if not postings or not ctx.llm.available:
        return {}
    wanted = postings if limit is None else postings[:limit]
    known = {p.key for p in wanted}
    out: dict[str, dict] = {}

    for start in range(0, len(wanted), batch_size):
        batch = wanted[start:start + batch_size]
        # Send the canonical key, not the raw URL: verdicts come back keyed by
        # whatever was sent, and the caller looks them up by key.
        lines = [f"- url: {p.key}\n  company: {p.company}\n  title: {p.title}\n"
                 f"  location: {p.location}" for p in batch]
        verdicts = ctx.llm.json(
            "Postings:\n" + "\n".join(lines), system=SYSTEM, fast=True,
            max_tokens=max(1000, len(batch) * TOKENS_PER_VERDICT), default=[])
        if not isinstance(verdicts, list):
            log.warning("batch %d returned %s, not a list", start // batch_size,
                        type(verdicts).__name__)
            continue
        for v in verdicts:
            if not isinstance(v, dict) or not v.get("url"):
                continue
            # Canonicalise whatever the model echoed back — it re-adds tracking
            # parameters and trailing slashes. A url that still does not resolve
            # to a posting is dropped rather than guessed at: on the boards that
            # put identity in the query string, a near-match would attach one
            # role's verdict to another's row.
            key = posting_key(str(v["url"]))
            if key not in known:
                log.debug("verdict for an unrecognised url dropped: %s", v["url"])
                continue
            out[key] = {"verdict": str(v.get("verdict", "")).lower(),
                        "why": str(v.get("why", ""))[:120]}
    return out


VERDICT_ORDER = {"apply": 0, "maybe": 1, "": 2, "skip": 3}


def build_brief(ctx: Context, *, registry=REGISTRY, min_score: int = 4,
                internships_only: bool = True, limit: int = DISPLAY_LIMIT,
                use_llm: bool = True) -> tuple[Brief, dict]:
    boards = Boards(ctx.fetcher)
    all_postings = boards.fetch_all(registry, ttl=1800)

    for p in all_postings:
        score(p)
    candidates = [p for p in all_postings
                  if p.score >= min_score and (p.is_internship or not internships_only)]
    candidates.sort(key=lambda p: -p.score)

    # Claim only what this brief will actually show. Marking every candidate
    # seen and then displaying the first `limit` of them means the remainder are
    # remembered as shown without ever having been, and they never surface again.
    by_key = {}
    for p in candidates:
        by_key.setdefault(p.key, p)
    fresh = unseen_keys(ctx.db, NAME, list(by_key))
    new_postings = [by_key[k] for k in fresh[:limit]]
    backlog = len(fresh) - len(new_postings)

    # Rank before claiming, so the verdict is stored alongside the posting and a
    # later run of the same day can render it without asking the model again.
    verdicts = rank(ctx, new_postings, limit=len(new_postings)) if use_llm else {}
    if new_postings:
        new_postings.sort(key=lambda p: (
            VERDICT_ORDER.get(verdicts.get(p.key, {}).get("verdict", ""), 2), -p.score))
    mark_new(ctx.db, NAME, {p.key: {**p.as_dict(), **verdicts.get(p.key, {})}
                            for p in new_postings})

    today = datetime.now(timezone.utc).date()
    surfaced = seen_since(ctx.db, NAME, _start_of_day(today))
    brief = Brief(title=f"Internship scout — {len(surfaced)} "
                        f"{'match' if len(surfaced) == 1 else 'matches'} today",
                  agent=NAME, target=f"scout-{today.isoformat()}",
                  tags=["internships", "search"])
    for d in ctx.base_degradations():
        brief.degrade(d)

    dead = [c for c, status in boards.source_status.items() if not status.endswith("postings")]
    if dead:
        brief.degrade(f"{len(dead)} board(s) returned nothing: {', '.join(sorted(dead))}")

    # The digest covers the whole day, not this run. A second run of the same
    # day used to overwrite the morning's findings with its own smaller list --
    # first when it found nothing, then, once that was guarded, when it found
    # the 29 postings the display cap had queued and replaced 100 rows with 29.
    # Rendering the union makes the file monotonic within a day and removes the
    # special case rather than guarding it.
    # A 129-row table is fourteen chunks to a retriever and a wall to a reader,
    # and neither of them starts at the top. The shortlist is the answer to
    # "which should I apply to" in one place small enough to be read or
    # retrieved whole.
    shortlist = sorted((p for p in surfaced if p.get("verdict") == "apply"),
                       key=lambda p: -p.get("score", 0))[:TOP_MATCHES]
    if shortlist:
        brief.add("Worth applying to", "\n".join(
            f"{i}. **{p.get('company', '')}** — [{p.get('title', '')}]({p.get('url', '')})"
            f" · {p.get('location') or 'location not stated'}"
            f" · score {p.get('score', 0)}"
            + (f" · _{p['why']}_" if p.get("why") else "")
            for i, p in enumerate(shortlist, 1)))

    if surfaced:
        rows = [[p.get("company", ""),
                 f"[{p.get('title', '')}]({p.get('url', '')})",
                 p.get("location") or "—", p.get("score", 0),
                 p.get("verdict") or "—", p.get("why", "")]
                for p in surfaced]
        brief.add(f"Surfaced today ({len(rows)})",
                  table(["Company", "Role", "Location", "Score", "Verdict", "Why"], rows))
    else:
        brief.add("Surfaced today",
                  "_Nothing new. Every qualifying posting on these boards was already "
                  "surfaced before today._")

    brief.add("Coverage", table(
        ["Company", "Sector", "Result"],
        [[company, next((s for v, sl, c, s in registry if c == company), ""), status]
         for company, status in sorted(boards.source_status.items())]))

    # Why there are no verdicts matters. "Unavailable" when the real reason is
    # "nothing new to rank" is the brief crying wolf about its own model, and a
    # warning that fires on a healthy run stops being read.
    if not new_postings:
        verdict_note = "no new postings to rank this run"
    elif verdicts:
        unranked = [p for p in new_postings if p.key not in verdicts]
        verdict_note = "verdicts are model-assigned"
        if unranked:
            verdict_note += (f" for {len(new_postings) - len(unranked)} of "
                             f"{len(new_postings)} rows; the model returned nothing "
                             f"usable for the other {len(unranked)}")
    elif not use_llm:
        verdict_note = "model ranking was switched off for this run"
    else:
        verdict_note = "model ranking was unavailable this run"

    brief.add("How this was filtered", (
        f"- {len(all_postings)} postings pulled from {len(registry)} boards\n"
        f"- {len(candidates)} passed the keyword filter "
        f"(score ≥ {min_score}{', internship titles only' if internships_only else ''})\n"
        f"- {len(new_postings)} shown as new this run "
        f"({len(surfaced)} surfaced today in total)"
        + (f", {backlog} more queued for the next run (capped at {limit} a night)"
           if backlog else "")
        + f"; {seen_count(ctx.db, NAME)} postings tracked in total\n"
        f"- {verdict_note.capitalize()}; scores are deterministic keyword weights."))
    brief.source("Greenhouse / Lever / Ashby public job board APIs",
                 note="each board's own feed, no scraping")
    brief.extra_meta.update({"new": len(new_postings), "today": len(surfaced),
                             "scanned": len(all_postings),
                             "candidates": len(candidates), "backlog": backlog})
    data = {"all": all_postings, "candidates": candidates, "new": new_postings,
            "verdicts": verdicts, "status": boards.source_status, "backlog": backlog,
            "surfaced_today": surfaced}
    return brief, data


def _start_of_day(day) -> float:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()


def run(ctx: Context, *, commit: bool = True, **kw) -> AgentResult:
    started = datetime.now(timezone.utc)
    res = AgentResult(agent=NAME, target="boards")
    try:
        brief, data = build_brief(ctx, **kw)
    except Exception as e:  # noqa: BLE001
        log.exception("scout failed")
        res.ok, res.error = False, f"{type(e).__name__}: {e}"
        record(ctx.db, Run(agent=NAME, ok=False, error=res.error,
                           started_at=started.timestamp()))
        return res

    res.brief = brief
    res.degradations = list(brief.degradations)
    res.data = {"new": [p.as_dict() for p in data["new"]],
                "scanned": len(data["all"]), "status": data["status"]}
    # No guard against overwriting: the digest is the day's union, so a run that
    # adds nothing renders the same rows and a run that adds something renders
    # more. It can only grow within a day.
    res.artifact = brief.write(ctx.cfg.out_dir / NAME)
    if commit:
        try:
            cr = ctx.notes.commit_file(
                f"scout/{brief.filename}", brief.render(),
                f"scout: {len(data['new'])} new, {len(data['surfaced_today'])} today "
                f"{brief.date}")
            res.data["commit"] = {"sha": cr.sha, "committed": cr.committed}
        except Exception as e:  # noqa: BLE001
            res.degrade(f"could not commit scout report: {e}")
    res.summary = (f"{len(data['new'])} new of {len(data['candidates'])} qualifying, "
                   f"{len(data['surfaced_today'])} today, {len(data['all'])} scanned")
    record(ctx.db, Run(agent=NAME, target=res.target, ok=True, artifact=str(res.artifact),
                       summary=res.summary, degradations=res.degradations,
                       started_at=started.timestamp(),
                       duration_s=(datetime.now(timezone.utc) - started).total_seconds()))
    return res
