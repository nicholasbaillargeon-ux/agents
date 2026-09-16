"""Agent 8 — the daily AI brief.

Nine AI feeds in, one page out: what happened in AI since yesterday, ordered by
what it changes rather than by what was loudest, with every story's coverage
named and the model's commentary kept to the parts a model can be wrong about
harmlessly.

Three things this agent is built around, in the order they bite:

**A list of items is not a list of stories.** Nine feeds carrying one
announcement is one event. The clustering lives in `sources.aifeeds`; what
matters here is that the brief is a list of *stories*, each showing every
headline that fed it — so a bad merge is visible on the page rather than
silently swallowing an event.

**The diff is the product, and the day's digest only grows.** A brief that
re-lists yesterday's news stops being read, so a story is shown once, on the
day it first appears. Identity is per *item*, not per story: a story is new
only when none of its headlines have been seen before, which means tomorrow's
follow-up joins today's story rather than resurrecting it. The day's file is
rendered from everything surfaced today rather than from one run's delta, so a
second run can only add to it — the scout learned that one the hard way.

**The facts are assembled before the model is asked anything.** Every story,
its sources and its ordering come from the feeds and a deterministic score. The
model writes a lede, one line per story about why it matters, and a short watch
list — and every figure it uses is checked back against the story list it was
given. With no model the brief is the same page minus those three parts.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from ..brief import Brief, table
from ..grounding import ungrounded
from ..notify import lan_host
from ..sources.aifeeds import (DROPPED_FEEDS, FEEDS, AIFeeds, Story, cluster,
                               is_noise, looks_like_ai, story_key)
from ..store import Run, mark_new, record, seen_count, seen_since, unseen_keys
from .base import AgentResult, Context, finalize

log = logging.getLogger(__name__)

NAME = "ainews"

# The window is wider than the cadence on purpose. The timer runs once a day;
# if a run slips or a machine sleeps, a 24-hour window would leave a hole no
# later run ever fills. The overlap costs nothing because `seen` — not the
# window — is what stops a story being reported twice.
WINDOW_HOURS = 30.0

# How many stories one brief will show, and how many get a model sentence.
# The cap on the second is the cost control: the feeds carry a hundred-odd
# items a day and about eight of them are worth a paragraph.
DISPLAY_LIMIT = 40
TOP_STORIES = 8

# Below this a story is listed but not written about. Set at the point where a
# single-outlet item with no release, money, legal or compute signal in its
# headline falls out — those are the ones the model has nothing to say about
# except to restate the title.
ANALYSIS_FLOOR = 3

SYSTEM = (
    "You write the daily AI brief for one reader who already follows the field "
    "closely: they know what a frontier model is, who the labs are, and what "
    "shipped last month. You are given today's stories, already selected and "
    "ordered, with the outlets that carried each one. Rules, in priority "
    "order:\n"
    "1. Use ONLY what is in the story list. Never introduce a fact, a number "
    "or a product that is not there, and never estimate one.\n"
    "2. Say what changed, not what was announced. 'Anthropic folded Cowork "
    "into Claude, retiring a product six months old' beats 'Anthropic "
    "announced changes to its product line'.\n"
    "3. Where the day is thin, say it is thin. Do not manufacture a theme "
    "across unrelated stories, and do not predict.\n"
    "4. No hedging boilerplate, no 'as an AI', no sign-off."
)

# The visible answer is well under a thousand tokens. The budget is several
# times that because the endpoint returns a thinking block against the same
# cap — the research agent measured 2,527 output tokens to show 1,224 of prose,
# and sizing this to the prose alone truncates the last field of the JSON,
# which then parses as nothing at all.
ANALYSIS_TOKENS = 4000


def _dossier(rows: list[dict]) -> str:
    """The exact factual surface handed to the model.

    Plain text on purpose: it is the thing to read when the brief says
    something odd.
    """
    lines = []
    for n, r in enumerate(rows, 1):
        lines.append(f"{n}. [{r['when']} · {', '.join(r['outlets'][:4])}] {r['title']}")
        if r.get("summary"):
            lines.append(f"   {r['summary']}")
        others = [t for t in r.get("headlines", []) if t != r["title"]][:4]
        if others:
            lines.append("   also reported as: " + " | ".join(others))
    return "\n".join(lines)


def analyse(ctx: Context, rows: list[dict]) -> tuple[dict, list[str], str]:
    """(analysis, ungrounded figures, degradation).

    One call, JSON, because the per-story line has to come back *attached* to a
    story: prose with the stories restated in it would have to be matched back
    by title, and a model that paraphrases a headline breaks that match
    silently.
    """
    if not rows or not ctx.llm.available:
        # An absent model is already reported by `base_degradations`, in one
        # line, on every brief. Saying it again here is the banner crying wolf
        # about something the reader has already been told.
        return {}, [], ""

    dossier = _dossier(rows)
    prompt = (
        f"TODAY'S STORIES:\n{dossier}\n\n---\n\n"
        "Return ONLY a JSON object with these keys:\n"
        '  "lede": two or three sentences on what today actually means. Lead '
        "with the most consequential story, name it, and say what it changes. "
        "If several stories share a thread, say what the thread is.\n"
        '  "stories": an array of {"n": <the number above>, "why": "<at most '
        '25 words on why this one matters>"} — one entry for each numbered '
        "story, in the same order.\n"
        '  "watch": two or three short strings, each naming something '
        "specific to watch for next, grounded in a story above.\n"
    )
    analysis = ctx.llm.json(prompt, system=SYSTEM, default=None,
                            max_tokens=ANALYSIS_TOKENS)
    if analysis is None and getattr(ctx.llm, "last_stop_reason", None) == "max_tokens":
        # The reply was cut off mid-JSON, which parses as nothing at all.
        # Thinking length varies run to run, so the same eight stories can fit
        # one morning and not the next.
        log.info("analysis truncated at the token cap; retrying with more room")
        analysis = ctx.llm.json(prompt, system=SYSTEM, default=None,
                                max_tokens=ANALYSIS_TOKENS * 2)
    if not isinstance(analysis, dict):
        return {}, [], "model commentary omitted: nothing usable came back"

    lede = str(analysis.get("lede") or "").strip()
    watch = [str(w).strip() for w in (analysis.get("watch") or []) if str(w).strip()]
    whys: dict[int, str] = {}
    for entry in analysis.get("stories") or []:
        if not isinstance(entry, dict):
            continue
        try:
            n = int(entry.get("n"))
        except (TypeError, ValueError):
            continue
        why = str(entry.get("why") or "").strip()
        if why and 1 <= n <= len(rows):
            whys[n] = why

    out = {"lede": lede, "watch": watch, "whys": whys}
    # Everything the model wrote, checked against everything it was shown.
    flagged = ungrounded("\n".join([lede, *watch, *whys.values()]), dossier)
    missing = [n for n in range(1, len(rows) + 1) if n not in whys]
    note = ""
    if not lede:
        note = "the model returned no lede"
    elif missing:
        note = (f"{len(missing)} of {len(rows)} top stories came back without a line "
                "on why they matter")
    return out, flagged, note


def _funnel(counts: dict, feeds, hours: float, dead: dict) -> str:
    return (
        f"- {counts['items']} items pulled from {len(feeds)} feeds in the last "
        f"{hours:.0f} hours"
        + (f"; {len(dead)} feed(s) contributed nothing" if dead else "")
        + f"\n- {counts['ai']} of those are about AI at all "
        f"({counts['items'] - counts['ai']} matched no AI term as a whole word)\n"
        f"- {counts['clean']} survived the noise filter "
        f"({counts['ai'] - counts['clean']} were stock listicles, task-force press "
        "releases or lifestyle filler)\n"
        f"- {counts['stories']} distinct stories, once every rewrite of the same "
        "event is folded together\n"
        f"- {counts['new']} of those had not been reported here before "
        f"({counts['today']} surfaced today in total)"
        + (f", {counts['backlog']} more queued for the next run "
           f"(capped at {counts['limit']} a brief)" if counts["backlog"] else "")
        + f"; {counts['tracked']} headlines tracked in total\n"
        "- Ordering is a deterministic score: release and legal language in the "
        "headline, a model version, how many independent outlets carried it, and "
        "whether the lab announced it itself. The model never chooses the order")


def build_brief(ctx: Context, *, feeds=FEEDS, hours: float = WINDOW_HOURS,
                limit: int = DISPLAY_LIMIT, top: int = TOP_STORIES,
                use_llm: bool = True, today: date | None = None) -> tuple[Brief, dict]:
    today = today or datetime.now(timezone.utc).date()
    source = AIFeeds(ctx.fetcher)
    items = source.fetch_all(feeds, ttl=900, max_age_hours=hours)

    ai_items = [i for i in items if looks_like_ai(f"{i.title} {i.summary}")]
    clean = [i for i in ai_items if not is_noise(i.title, i.outlet)]
    stories = cluster(clean)

    # One pass over the run log for every headline in the sweep, not one query
    # per story: `unseen_keys` reads the whole table each call.
    unseen = set(unseen_keys(ctx.db, NAME, [story_key(i.title)
                                            for s in stories for i in s.items]))
    # Claim only what this brief will show. Marking every story seen and then
    # displaying the first `limit` of them means the remainder are remembered
    # as reported without ever having been, and they never surface again.
    by_key: dict[str, Story] = {}
    for s in stories:
        by_key.setdefault(s.key, s)
    fresh = [k for k, s in by_key.items()
             if all(story_key(i.title) in unseen for i in s.items)]
    fresh.sort(key=lambda k: -by_key[k].score)
    new_stories = [by_key[k] for k in fresh[:limit]]
    backlog = len(fresh) - len(new_stories)

    mark_new(ctx.db, NAME, {story_key(i.title): {**s.as_dict(), "item_title": i.title}
                            for s in new_stories for i in s.items})

    # Everything surfaced today, from every run of the day — not this run's
    # delta. The model then writes over the day's top stories rather than over
    # the ones this particular run happened to add, so a second run of the day
    # rewrites the same page instead of a thinner one.
    surfaced = _stories_today(ctx, today)
    headline_rows = [r for r in surfaced if r["score"] >= ANALYSIS_FLOOR][:top]
    if use_llm:
        analysis, flagged, analysis_note = analyse(ctx, headline_rows)
    else:
        analysis, flagged, analysis_note = {}, [], (
            "model commentary was switched off for this run")
    for n, row in enumerate(headline_rows, 1):
        row["why"] = analysis.get("whys", {}).get(n, "")

    brief = Brief(
        title=f"AI brief — {today:%A %d %B %Y}", agent=NAME,
        target=f"ai-{today.isoformat()}", tags=["ai", "news", "daily"])
    for d in ctx.base_degradations():
        brief.degrade(d)

    # A feed that answered and had nothing inside the window is a quiet feed,
    # not a broken one: a lab blog publishes a handful of posts a month. Only
    # the feeds that failed to answer are a hole in the sweep, and a banner
    # that fires every day for a healthy lab feed is a banner nobody reads.
    dead = dict(source.failures)
    if dead:
        brief.degrade(f"{len(dead)} feed(s) could not be read: "
                      + ", ".join(f"{n} ({s})" for n, s in sorted(dead.items())))
    if analysis_note:
        brief.degrade(analysis_note)
    if not ctx.push.available and not ctx.offline:
        # This brief goes to a phone as well as to the notes repo, so one that
        # was written and never sent is a degraded brief — the same line the
        # morning tape takes, and for the same reason.
        brief.degrade("no phone push configured: the brief was written but not sent")
    if not items and not surfaced:
        brief.degrade("no feed returned anything: this brief is empty because the "
                      "sweep failed, not because the day was quiet")

    lede = analysis.get("lede", "")
    if lede:
        if flagged:
            lede += ("\n\n_Not found in today's story list: "
                     + ", ".join(f"`{x}`" for x in flagged)
                     + ". The commentary is model-written; the stories below are not._")
            brief.extra_meta["ungrounded_figures"] = len(flagged)
        brief.add("The day", lede)

    if headline_rows:
        brief.add(f"Top stories ({len(headline_rows)})", "\n\n".join(
            _headline_block(n, r) for n, r in enumerate(headline_rows, 1)))
    elif surfaced:
        brief.add("Top stories", (
            "_Nothing today cleared the bar for a write-up: every story is a single "
            "outlet with no release, funding, legal or compute news in its headline. "
            "They are all listed below._"))

    shown = {r["story_key"] for r in headline_rows}
    rest = [r for r in surfaced if r["story_key"] not in shown]
    if rest:
        brief.add(f"Also today ({len(rest)})", table(
            ["Story", "Carried by", "When", "Score"],
            [[f"[{r['title']}]({r['url']})" if r["url"] else r["title"],
              ", ".join(r["outlets"][:3]) or "—", r["when"], r["score"]]
             for r in rest]))
    elif not surfaced:
        brief.add("Today", (
            "_Nothing new. Every story these feeds carried today had already been "
            "reported here._" if items else
            "_No feed answered this run, so this brief knows nothing about today._"))

    watch = analysis.get("watch", [])
    if watch:
        brief.add("What to watch", "\n".join(f"- {w}" for w in watch))

    brief.add("Coverage", table(
        ["Feed", "Tier", "Result"],
        [[f.name, f.tier, source.source_status.get(f.name, "not fetched")]
         for f in feeds]))

    counts = {"items": len(items), "ai": len(ai_items), "clean": len(clean),
              "stories": len(stories), "new": len(new_stories),
              "today": len(surfaced), "backlog": backlog, "limit": limit,
              "tracked": seen_count(ctx.db, NAME)}
    brief.add("How this was filtered", _funnel(counts, feeds, hours, dead))
    brief.add("Feeds not used", "\n".join(
        f"- **{name}** — {why}" for name, why in sorted(DROPPED_FEEDS.items())), level=3)

    for f in feeds:
        if f.name in source.answered:
            brief.source(f.name, note=f"{f.tier} feed")
    brief.extra_meta.update({"stories": len(stories), "new": len(new_stories),
                             "today": len(surfaced), "items": len(items),
                             "feeds_live": len(source.answered)})
    data = {"items": items, "stories": stories, "new": new_stories,
            "headline": headline_rows, "surfaced_today": surfaced,
            "status": source.source_status, "counts": counts,
            "analysis": analysis, "backlog": backlog}
    return brief, data


# How many stories reach the phone. Three fit on a lock screen; the fourth is
# already below the fold in the notification shade.
PUSH_STORIES = 3


def push_body(lede: str, rows: list[dict], counts: dict) -> str:
    """The thirty-second version, for a phone.

    Deliberately not the brief. The brief is eighty rows and is read at a desk;
    this is what survives being read on a lock screen, so it is the lede, three
    headlines and the line each one earned.

    **No links.** A Google News redirect token runs to five hundred characters,
    so three of them would eat a third of ntfy's body limit and push the
    stories themselves past the truncation — the reader would get a wall of
    `CBMi...` and no news. The notification's tap target is the dashboard,
    where every link already is.
    """
    parts: list[str] = []
    if lede:
        parts.append(lede.split("\n\n")[0])
    for n, r in enumerate(rows[:PUSH_STORIES], 1):
        line = f"**{n}. {r['title']}**"
        why = (r.get("why") or "").strip()
        if why:
            line += f"\n{why}"
        parts.append(line)
    tail = (f"_{counts['today']} stories today from {counts['items']} items._"
            if counts.get("today") else
            "_Nothing new in the feeds since the last brief._")
    parts.append(tail)
    return "\n\n".join(parts)


def _headline_block(n: int, r: dict) -> str:
    """One story, with every headline that reported it.

    The member headlines are on the page on purpose. Clustering is a heuristic;
    printing what it merged is what makes a wrong merge a visible duplicate
    line rather than an event that quietly disappeared.
    """
    head = (f"**{n}. [{r['title']}]({r['url']})**  \n"
            f"_{r['when']} · {', '.join(r['outlets'][:5])}"
            + (" · announced by the lab" if r.get("from_lab") else "") + "_")
    why = r.get("why") or ""
    body = f"\n\n{why}" if why else ""
    summary = f"\n\n{r['summary']}" if r.get("summary") and not why else ""
    others = [t for t in r.get("headlines", []) if t != r["title"]][:3]
    also = ("\n\n" + "\n".join(f"- _also:_ {t}" for t in others)) if others else ""
    return head + body + summary + also


def _stories_today(ctx: Context, today: date) -> list[dict]:
    """The day's stories, from every run of the day, best first.

    Rendered from the run log rather than from this run's list so the file can
    only grow within a day: a second run that finds nothing renders the same
    page, and one that finds something renders more. The scout replaced a
    hundred-row digest with a twenty-nine-row one exactly once before this
    became the rule.

    One row per *headline* goes into the log, so they are folded back into one
    row per story here — and the headlines travel with it, which is what the
    brief prints under each story.
    """
    out: dict[str, dict] = {}
    for row in seen_since(ctx.db, NAME, _start_of_day(today)):
        key = row.get("story_key") or row.get("key", "")
        kept = out.setdefault(key, {**row, "story_key": key, "headlines": [],
                                    "when": _when_label(row.get("published", ""))})
        title = row.get("item_title")
        if title and title not in kept["headlines"]:
            kept["headlines"].append(title)
    return sorted(out.values(), key=lambda r: (-r.get("score", 0), r.get("title", "")))


def _when_label(published: str) -> str:
    if not published:
        return "—"
    try:
        dt = datetime.fromisoformat(published)
    except ValueError:
        return "—"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    if age < 1:
        return f"{int(age * 60)}m ago"
    if age < 48:
        return f"{int(age)}h ago"
    return f"{int(age / 24)}d ago"


def _start_of_day(day: date) -> float:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()


def run(ctx: Context, *, commit: bool = True, **kw) -> AgentResult:
    started = datetime.now(timezone.utc)
    res = AgentResult(agent=NAME, target="ai")
    try:
        brief, data = build_brief(ctx, **kw)
    except Exception as e:  # noqa: BLE001
        log.exception("ainews failed")
        res.ok, res.error = False, f"{type(e).__name__}: {e}"
        record(ctx.db, Run(agent=NAME, target=res.target, ok=False, error=res.error,
                           started_at=started.timestamp()))
        return res

    res.brief = brief
    finalize(ctx, brief, res)
    # No guard against overwriting: the file is the day's union, so a second
    # run of the same day can only add rows to it.
    res.artifact = brief.write(ctx.cfg.out_dir / NAME)
    res.data.update({"new": data["counts"]["new"], "scanned": data["counts"]["items"]})
    if commit:
        try:
            cr = ctx.notes.commit_file(
                f"ai-news/{brief.filename}", brief.render(),
                f"AI brief: {len(data['new'])} new, {len(data['surfaced_today'])} today "
                f"{brief.date}")
            res.data["commit"] = {"sha": cr.sha, "committed": cr.committed,
                                  "pushed": cr.pushed}
            if cr.push_error:
                res.degrade(f"commit stayed local: {cr.push_error}")
        except Exception as e:  # noqa: BLE001 — a git failure must not lose the brief
            log.warning("commit failed: %s", e)
            res.degrade(f"could not commit AI brief: {e}")

    # The push goes out after the brief is written and committed, never before:
    # a phone buzzing about a brief that then failed to persist is worse than a
    # late buzz, because the notification is the only copy the reader saw.
    counts = data["counts"]
    pushed = ctx.push.send(
        push_body(data["analysis"].get("lede", ""), data["headline"], counts),
        title=f"AI brief - {brief.date}",
        click=f"http://{lan_host()}:{ctx.cfg.port}/view/{NAME}/{res.artifact.name}",
        # Priority 3, where the morning tape is 4. The tape is time-critical —
        # it is worth reading before the bell or not at all — and this is a
        # read-when-you-can. A daily brief that buzzes like a market alert
        # teaches the reader to silence the topic both agents share.
        tags="robot", priority=3)
    if not pushed.sent and ctx.push.available:
        # Not configured is already on the brief; configured and failing is a
        # new fact and belongs on this run.
        res.degrade(pushed.degradation)
        brief.degrade(pushed.degradation)
    res.data["pushed"] = pushed.sent

    res.summary = (f"{counts['new']} new of {counts['stories']} stories, "
                   f"{counts['today']} today, {counts['items']} items from "
                   f"{len(data['status'])} feeds" + (", pushed" if pushed.sent else ""))
    record(ctx.db, Run(agent=NAME, target=res.target, ok=True, artifact=str(res.artifact),
                       summary=res.summary, degradations=res.degradations,
                       started_at=started.timestamp(),
                       duration_s=(datetime.now(timezone.utc) - started).total_seconds()))
    return res
