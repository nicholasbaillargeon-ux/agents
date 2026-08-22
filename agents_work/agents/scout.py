"""Agent 4 — internship scout.

Sweeps the quant, bank, broker, exchange, fintech, AI and enterprise-IT job
boards in `sources.jobs.REGISTRY` nightly and surfaces only what is new since
the last run.

The diff is the product. A scout that re-lists yesterday's ninety postings is
worse than no scout, because you stop reading it. `store.mark_new` does the
diff in one transaction, so a crash mid-run cannot half-remember a batch and
silently drop those postings out of tomorrow's list too.

Each row also carries how long the posting has been open, taken from the
board's own publication date. It is the one field that separates "this opened
last night" from "this has been sitting there since March and is probably
filled", and it is not derivable from the diff: the day the scout first *saw* a
posting is the day the board was added, not the day the job opened.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone

from ..brief import Brief, table
from ..llm import LLMUnavailable
from ..sources.jobs import (REGISTRY, VENDOR_LABELS, Boards, Posting, days_open,
                            format_days_open, location_in_range, posting_key, score)
from ..store import Run, mark_new, record, seen_count, seen_since, unseen_keys
from .base import AgentResult, Context

log = logging.getLogger(__name__)

NAME = "scout"

# How many new postings one brief will show. Steady-state nightly volume
# across these boards is well under this, so the cap only binds after a
# registry change adds a board -- and when it does, the brief says so and the
# remainder queue rather than vanishing. It was raised from 100 when the
# registry went from twenty boards to eighty; the first few nights after that
# change are a backlog drain, not a steady state.
DISPLAY_LIMIT = 150

# How many of the day's "apply" verdicts to lift out as a shortlist.
TOP_MATCHES = 10

# Postings open longer than this are not shown. Three weeks is roughly the point
# where an internship req has already collected the applications it is going to
# read, and the scout that surfaces it is spending the reader's attention on a
# race that is over. Applied after the range gate and before the model, so a
# stale posting costs neither a row nor a verdict.
MAX_DAYS_OPEN = 21

SYSTEM = (
    "You triage job postings for one candidate: a junior-year computer "
    "engineering undergraduate aiming at AI and finance, strongest in Python, "
    "data pipelines, and full-slice deployment, and comfortable across the "
    "hardware/software line. For each posting you are given, return a JSON "
    "array of objects with keys `url`, `verdict` (one of: apply, maybe, skip), "
    "and `why` (at most 15 words, concrete). Judge fit, not prestige. A posting "
    "that is senior, non-technical, or in an unrelated function is `skip` even "
    "at a famous firm. The candidate will only work in the United States, "
    "Canada, or London: `skip` anything sited elsewhere, and where a posting "
    "names several offices, judge it on the ones in that range. Some boards "
    "publish no usable location at all -- treat those on their merits rather "
    "than assuming the worst. `days_open` is how long the posting has been "
    "live; every posting you are shown opened within the last three weeks, so "
    "use it only to break ties between otherwise equal roles. Return ONLY the "
    "JSON array."
)


# Asking for a hundred verdicts in a single completion overruns any sane
# max_tokens and the reply is truncated mid-array — which parses as nothing, so
# a larger cap would silently cost every verdict rather than a few. Batching
# keeps each reply inside its budget and makes one bad batch cost only that
# batch.
RANK_BATCH = 25

# A flat 80 tokens a verdict was fitted to Greenhouse urls and broke the night
# the Workday and Oracle boards landed: their urls run past a hundred
# characters, the model echoes each one back, and batches blew the 2000-token
# cap, truncated mid-array and parsed as nothing. The budget is now measured off
# the batch it is for.
#
# Two characters to a token, not the usual four. That is not a typo: this output
# is a JSON array of long urls, and urls tokenise badly — every slash, hyphen
# and id fragment is its own token. A four-chars-a-token estimate was measured
# against a real reply that spent 2294 tokens on 6003 characters, and hit the
# cap anyway. The ceiling costs nothing when it is not reached, and costs the
# whole batch when it is too low.
CHARS_PER_TOKEN = 2
CHARS_PER_VERDICT = 120   # the JSON scaffolding plus a 15-word `why`
BATCH_TOKEN_FLOOR = 1000


def rank_budget(batch: list[Posting]) -> int:
    """max_tokens for one ranking batch, sized to the urls it must echo."""
    chars = sum(len(p.key) for p in batch) + len(batch) * CHARS_PER_VERDICT
    return max(BATCH_TOKEN_FLOOR, chars // CHARS_PER_TOKEN)


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
                 f"  location: {p.location}\n"
                 f"  days_open: {format_days_open(p.posted, p.posted_is_floor)}"
                 for p in batch]
        verdicts = ctx.llm.json(
            "Postings:\n" + "\n".join(lines), system=SYSTEM, fast=True,
            max_tokens=rank_budget(batch), default=[])
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
                max_days_open: int = MAX_DAYS_OPEN,
                use_llm: bool = True) -> tuple[Brief, dict]:
    boards = Boards(ctx.fetcher)
    all_postings = boards.fetch_all(registry, ttl=1800)

    for p in all_postings:
        score(p)
    qualifying = [p for p in all_postings
                  if p.score >= min_score and (p.is_internship or not internships_only)]
    # Range first, then age. Both are hard gates rather than score adjustments,
    # and both run before `rank` so an out-of-range or stale posting costs no
    # verdict: the last sweep carried 19 Singapore rows and 99 postings older
    # than three weeks that the model was paying to read and reject.
    in_range = [p for p in qualifying if location_in_range(p.title, p.location)]
    # Counted, not hidden: these are the rows the gate let through without ever
    # confirming a place, and the reader deserves to know how much of the range
    # line rests on them.
    unlocated = sum(1 for p in in_range if _names_no_place(p.location))
    candidates = [p for p in in_range if _fresh_enough(p, max_days_open)]
    candidates.sort(key=lambda p: -p.score)

    # Claim only what this brief will actually show. Marking every candidate
    # seen and then displaying the first `limit` of them means the remainder are
    # remembered as shown without ever having been, and they never surface again.
    by_key = {}
    for p in candidates:
        by_key.setdefault(p.key, p)
    fresh = unseen_keys(ctx.db, NAME, list(by_key))
    # Score first, then freshness. Score is the fit signal and stays primary,
    # but when the cap binds -- and after a registry change it binds for several
    # nights -- an equally good posting that opened yesterday is the one worth
    # reading tonight; the older one keeps its place in the queue.
    fresh.sort(key=lambda k: (-by_key[k].score, _age(by_key[k])))
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

    # A board that answered carries a count; anything else is a hole in the
    # sweep. The reason travels with the name, because "unreachable" is a retry
    # tomorrow and "no postings returned" is a registry edit today.
    dead = {c: status for c, status in boards.source_status.items()
            if not re.match(r"\d+ postings", status)}
    if dead:
        brief.degrade(f"{len(dead)} board(s) contributed nothing: "
                      + ", ".join(f"{c} ({s})" for c, s in sorted(dead.items())))

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
            f" · {_age_phrase(p, today)}"
            + (f" · _{p['why']}_" if p.get("why") else "")
            for i, p in enumerate(shortlist, 1)))

    if surfaced:
        rows = [[p.get("company", ""),
                 f"[{p.get('title', '')}]({p.get('url', '')})",
                 p.get("location") or "—",
                 format_days_open(p.get("posted", ""), p.get("posted_is_floor", False), today),
                 p.get("score", 0),
                 p.get("verdict") or "—", p.get("why", "")]
                for p in surfaced]
        brief.add(f"Surfaced today ({len(rows)})",
                  table(["Company", "Role", "Location", "Days open", "Score",
                         "Verdict", "Why"], rows))
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

    vendors = sorted({v for v, *_ in registry})
    vendor_names = [VENDOR_LABELS.get(v, v.title()) for v in vendors]
    undated = [p for p in surfaced if not p.get("posted")]
    brief.add("How this was filtered", (
        f"- {len(all_postings)} postings pulled from {len(registry)} boards "
        f"across {len(vendors)} ATS vendors ({', '.join(vendor_names)})\n"
        f"- {len(qualifying)} passed the keyword filter "
        f"(score ≥ {min_score}{', internship titles only' if internships_only else ''})\n"
        f"- {len(in_range)} of those are sited in the United States, Canada or "
        f"London"
        + (f", counting {unlocated} whose board publishes no usable location"
           if unlocated else "")
        + f"\n- {len(candidates)} of those opened within the last "
        f"{max_days_open} days"
        + (f"; {len(in_range) - len(candidates)} were older and are not shown"
           if len(candidates) < len(in_range) else "")
        + "\n"
        f"- {len(new_postings)} shown as new this run "
        f"({len(surfaced)} surfaced today in total)"
        + (f", {backlog} more queued for the next run (capped at {limit} a night)"
           if backlog else "")
        + f"; {seen_count(ctx.db, NAME)} postings tracked in total\n"
        f"- {verdict_note.capitalize()}; scores are deterministic keyword weights\n"
        f"- Days open counts from the board's own publication date"
        + (f"; {len(undated)} of {len(surfaced)} rows show — because their board "
           f"publishes none" if undated else "")
        + (". Workday reports an age bucket rather than a date, so its rows can "
           "read \"30+\"" if any(p.get("posted_is_floor") for p in surfaced) else ".")))
    # Named off the registry rather than hard-coded: the sweep already counts
    # the vendors two lines above, and a Sources line that listed one the
    # registry no longer uses would be the brief citing a feed it never read.
    brief.source(" / ".join(vendor_names) + " public job board APIs",
                 note="each board's own feed, no scraping")
    fresh_today = sum(1 for p in surfaced if days_open(p.get("posted", ""), today) == 0)
    brief.extra_meta.update({"new": len(new_postings), "today": len(surfaced),
                             "scanned": len(all_postings), "boards": len(registry),
                             "candidates": len(candidates), "backlog": backlog,
                             "posted_today": fresh_today})
    data = {"all": all_postings, "candidates": candidates, "new": new_postings,
            "verdicts": verdicts, "status": boards.source_status, "backlog": backlog,
            "surfaced_today": surfaced}
    return brief, data


# An undated posting sorts as if it were a month old rather than as if it were
# brand new: boards that publish no date are mostly ones that never did, and
# letting them jump the queue would push genuinely fresh rows off the page.
UNDATED_AGE = 30


def _age_phrase(payload: dict, today: date | None = None) -> str:
    """How long the posting has been open, in words, for the shortlist.

    The table gets a number in its own column; a numbered list does not, so
    "open 3 days" beats a bare "3" wedged between a score and a verdict."""
    n = days_open(payload.get("posted", ""), today)
    if n is None:
        return "posting date not published"
    plus = "+" if payload.get("posted_is_floor") else ""
    if n == 0 and not plus:
        return "posted today"
    return f"open {n}{plus} day" + ("" if n == 1 and not plus else "s")


# Location text that is a filing convention rather than a place: Cloudflare's
# "In-Office", Capital One's "8 Locations", a bare "Remote" with no country.
_NO_PLACE_RE = re.compile(
    r"\s*(in-?office|on-?site|remote|virtual|hybrid|flexible|multiple locations|"
    r"\d+ locations)\s*$", re.I)


def _names_no_place(location: str) -> bool:
    """Did the board give a location that names nowhere?"""
    return not location.strip() or bool(_NO_PLACE_RE.fullmatch(location))


def _fresh_enough(p: Posting, max_days: int) -> bool:
    """Inside the freshness window?

    A Workday floor works without a special case: its "30+" bucket parses to 30,
    which is already over the limit, while a "7+" bucket could still be inside it
    and is kept. An undated posting is kept too -- there were none in the last
    366 tracked, but a board that stops publishing dates should show up as rows
    with an em dash in the Days open column, not as a board that went quiet.
    """
    n = p.days_open
    return n is None or n <= max_days


def _age(p: Posting) -> int:
    n = p.days_open
    return UNDATED_AGE if n is None else n


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
