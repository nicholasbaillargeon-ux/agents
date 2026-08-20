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
from datetime import datetime, timezone

from ..brief import Brief, table
from ..llm import LLMUnavailable
from ..sources.jobs import REGISTRY, Boards, Posting, score
from ..store import Run, mark_new, record, seen_count
from .base import AgentResult, Context

log = logging.getLogger(__name__)

NAME = "scout"

SYSTEM = (
    "You triage job postings for one candidate: an undergraduate targeting "
    "quant / fintech / AI internships, strongest in Python, data pipelines, and "
    "full-slice deployment. For each posting you are given, return a JSON array "
    "of objects with keys `url`, `verdict` (one of: apply, maybe, skip), and "
    "`why` (at most 15 words, concrete). Judge fit, not prestige. A posting "
    "that is senior, non-technical, or in an unrelated function is `skip` even "
    "at a famous firm. Return ONLY the JSON array."
)


def rank(ctx: Context, postings: list[Posting], *, limit: int = 20) -> dict[str, dict]:
    """LLM verdicts keyed by URL. Empty dict when the model is unavailable —
    the deterministic score already ordered the list."""
    if not postings or not ctx.llm.available:
        return {}
    lines = [f"- url: {p.url}\n  company: {p.company}\n  title: {p.title}\n"
             f"  location: {p.location}" for p in postings[:limit]]
    verdicts = ctx.llm.json(
        "Postings:\n" + "\n".join(lines), system=SYSTEM, fast=True,
        max_tokens=2000, default=[])
    if not isinstance(verdicts, list):
        return {}
    out = {}
    for v in verdicts:
        if isinstance(v, dict) and v.get("url"):
            out[v["url"]] = {"verdict": str(v.get("verdict", "")).lower(),
                             "why": str(v.get("why", ""))[:120]}
    return out


VERDICT_ORDER = {"apply": 0, "maybe": 1, "": 2, "skip": 3}


def build_brief(ctx: Context, *, registry=REGISTRY, min_score: int = 4,
                internships_only: bool = True, limit: int = 25,
                use_llm: bool = True) -> tuple[Brief, dict]:
    boards = Boards(ctx.fetcher)
    all_postings = boards.fetch_all(registry, ttl=1800)

    for p in all_postings:
        score(p)
    candidates = [p for p in all_postings
                  if p.score >= min_score and (p.is_internship or not internships_only)]
    candidates.sort(key=lambda p: -p.score)

    fresh_keys = set(mark_new(ctx.db, NAME, {p.key: p.as_dict() for p in candidates}))
    new_postings = [p for p in candidates if p.key in fresh_keys][:limit]

    verdicts = rank(ctx, new_postings) if use_llm else {}
    if new_postings:
        new_postings.sort(key=lambda p: (
            VERDICT_ORDER.get(verdicts.get(p.key, {}).get("verdict", ""), 2), -p.score))

    today = datetime.now(timezone.utc).date()
    brief = Brief(title=f"Internship scout — {len(new_postings)} new "
                        f"{'match' if len(new_postings) == 1 else 'matches'}",
                  agent=NAME, target=f"scout-{today.isoformat()}",
                  tags=["internships", "search"])
    for d in ctx.base_degradations():
        brief.degrade(d)

    dead = [c for c, status in boards.source_status.items() if not status.endswith("postings")]
    if dead:
        brief.degrade(f"{len(dead)} board(s) returned nothing: {', '.join(sorted(dead))}")

    if new_postings:
        rows = []
        for p in new_postings:
            v = verdicts.get(p.key, {})
            rows.append([
                p.company, f"[{p.title}]({p.url})", p.location or "—",
                p.score, (v.get("verdict") or "—"), v.get("why", ""),
            ])
        brief.add("New since last run",
                  table(["Company", "Role", "Location", "Score", "Verdict", "Why"], rows))
    else:
        brief.add("New since last run",
                  "_Nothing new. Every qualifying posting on these boards was already "
                  "surfaced in an earlier run._")

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
        verdict_note = "verdicts are model-assigned"
    elif not use_llm:
        verdict_note = "model ranking was switched off for this run"
    else:
        verdict_note = "model ranking was unavailable this run"

    brief.add("How this was filtered", (
        f"- {len(all_postings)} postings pulled from {len(registry)} boards\n"
        f"- {len(candidates)} passed the keyword filter "
        f"(score ≥ {min_score}{', internship titles only' if internships_only else ''})\n"
        f"- {len(new_postings)} were new; "
        f"{seen_count(ctx.db, NAME)} postings tracked in total\n"
        f"- {verdict_note.capitalize()}; scores are deterministic keyword weights."))
    brief.source("Greenhouse / Lever / Ashby public job board APIs",
                 note="each board's own feed, no scraping")
    brief.extra_meta.update({"new": len(new_postings), "scanned": len(all_postings),
                             "candidates": len(candidates)})
    data = {"all": all_postings, "candidates": candidates, "new": new_postings,
            "verdicts": verdicts, "status": boards.source_status}
    return brief, data


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
    res.artifact = brief.write(ctx.cfg.out_dir / NAME)
    if commit:
        try:
            cr = ctx.notes.commit_file(f"scout/{brief.filename}", brief.render(),
                                       f"scout: {len(data['new'])} new {brief.date}")
            res.data["commit"] = {"sha": cr.sha, "committed": cr.committed}
        except Exception as e:  # noqa: BLE001
            res.degrade(f"could not commit scout report: {e}")

    res.summary = (f"{len(data['new'])} new of {len(data['candidates'])} qualifying, "
                   f"{len(data['all'])} scanned")
    record(ctx.db, Run(agent=NAME, target=res.target, ok=True, artifact=str(res.artifact),
                       summary=res.summary, degradations=res.degradations,
                       started_at=started.timestamp(),
                       duration_s=(datetime.now(timezone.utc) - started).total_seconds()))
    return res
