"""Internship scout: the nightly diff is the product."""

from __future__ import annotations

import json
import re


import pytest

from agents_work.agents import scout
from agents_work.sources.jobs import Boards, Posting, score
from agents_work.store import mark_new, seen_count

REGISTRY = (("greenhouse", "quantco", "QuantCo", "quant"),
            ("lever", "fintechco", "FintechCo", "fintech"),
            ("ashby", "aico", "AICo", "ai"))


def greenhouse(*titles):
    return {"jobs": [{"title": t, "location": {"name": "New York, NY"},
                      "absolute_url": f"https://boards.greenhouse.io/quantco/jobs/{i}",
                      "updated_at": "2026-08-19T10:00:00Z"}
                     for i, t in enumerate(titles, 1)]}


def lever(*titles):
    return [{"text": t, "categories": {"location": "Remote"},
             "hostedUrl": f"https://jobs.lever.co/fintechco/{i}",
             "createdAt": 1_755_000_000_000} for i, t in enumerate(titles, 1)]


def ashby(*titles):
    return {"jobs": [{"title": t, "location": "Chicago",
                      "jobUrl": f"https://jobs.ashbyhq.com/aico/{i}",
                      "publishedAt": "2026-08-18T00:00:00Z"}
                     for i, t in enumerate(titles, 1)]}


def wire(fetcher, gh=("Quantitative Trading Intern - Summer 2027",),
         lv=("Software Engineer Intern",), ash=("Machine Learning Intern",)):
    fetcher.route("boards/quantco/jobs", greenhouse(*gh))
    fetcher.route("postings/fintechco", lever(*lv))
    fetcher.route("job-board/aico", ashby(*ash))
    return fetcher


# -- scoring -----------------------------------------------------------------

@pytest.mark.benchmark
@pytest.mark.parametrize("title,wanted", [
    ("Quantitative Trading Intern - Summer 2027", True),
    ("Software Engineering Intern", True),
    ("Machine Learning Intern", True),
    ("Campus Recruiting Intern", False),
    ("Sales Development Intern", False),
    ("Legal Intern", False),
    ("Brand Marketing Intern", False),
])
def test_relevance_filter(title, wanted):
    """S4: prestige is not fit."""
    p = Posting(company="X", title=title, location="New York, NY", url="u")
    score(p)
    assert (p.score >= 4) is wanted, f"{title} scored {p.score} ({p.reasons})"


@pytest.mark.parametrize("title", [
    "Summer 2027 Intern", "Quant Co-op", "New Grad Software Engineer",
    "University Trading Programme", "Campus Analyst", "Student Developer",
])
def test_internship_titles_are_recognised_in_their_many_forms(title):
    assert Posting(company="X", title=title, location="", url="u").is_internship


def test_senior_roles_are_not_internships():
    assert not Posting(company="X", title="Senior Quantitative Researcher",
                       location="", url="u").is_internship


# -- the diff ----------------------------------------------------------------

@pytest.mark.benchmark
def test_first_run_surfaces_everything_second_run_surfaces_nothing(ctx, fetcher):
    """S1: a scout that re-lists yesterday's postings stops being read."""
    wire(fetcher)
    brief1, data1 = scout.build_brief(ctx, registry=REGISTRY, use_llm=False)
    assert len(data1["new"]) == 3

    brief2, data2 = scout.build_brief(ctx, registry=REGISTRY, use_llm=False)
    assert data2["new"] == [], "run 2 must find nothing new"
    # The digest is the day's union, so it still lists what run 1 surfaced. That
    # is the point: re-running must not empty the morning's brief.
    assert len(data2["surfaced_today"]) == 3
    assert "QuantCo" in brief2.render()


@pytest.mark.benchmark
def test_one_added_posting_surfaces_exactly_one(ctx, fetcher):
    """S1."""
    wire(fetcher)
    scout.build_brief(ctx, registry=REGISTRY, use_llm=False)
    wire(fetcher, gh=("Quantitative Trading Intern - Summer 2027",
                      "Quantitative Research Intern - Summer 2027"))
    _, data = scout.build_brief(ctx, registry=REGISTRY, use_llm=False)
    assert len(data["new"]) == 1
    assert data["new"][0].title.startswith("Quantitative Research")


@pytest.mark.benchmark
def test_identity_survives_a_retitled_posting(ctx, fetcher):
    """S3: ATS teams edit titles and locations in place all the time."""
    wire(fetcher)
    scout.build_brief(ctx, registry=REGISTRY, use_llm=False)
    edited = greenhouse("Quant Trading Internship (Summer 2027) - NYC")
    edited["jobs"][0]["location"]["name"] = "Chicago, IL"
    fetcher.route("boards/quantco/jobs", edited)
    _, data = scout.build_brief(ctx, registry=REGISTRY, use_llm=False)
    assert data["new"] == []


def test_query_string_does_not_change_identity():
    a = Posting("X", "T", "L", "https://x/jobs/1?gh_src=abc")
    b = Posting("X", "T", "L", "https://x/jobs/1")
    assert a.key == b.key


@pytest.mark.benchmark
def test_the_diff_is_atomic(ctx):
    """S2: a crash mid-batch must not half-remember, or those postings vanish
    from every future diff without ever having been shown.

    The failure is real rather than mocked: an unserialisable payload raises
    inside the loop, after the first key has already been written.
    """
    items = {"key0": {"fine": 1},
             "key1": {"unserialisable": {1, 2}},
             "key2": {"fine": 2}}
    with pytest.raises(TypeError):
        mark_new(ctx.db, "scout", items)
    assert seen_count(ctx.db, "scout") == 0

    # and the whole batch is still available to a later, healthy run
    items["key1"] = {"fine": 3}
    assert len(mark_new(ctx.db, "scout", items)) == 3


def test_mark_new_returns_only_the_new_keys(ctx):
    assert set(mark_new(ctx.db, "a", {"x": {}, "y": {}})) == {"x", "y"}
    assert mark_new(ctx.db, "a", {"x": {}}) == []
    assert mark_new(ctx.db, "b", {"x": {}}) == ["x"]   # per-agent namespacing


# -- board health ------------------------------------------------------------

@pytest.mark.benchmark
def test_a_dead_board_is_reported_not_silently_dropped(ctx, fetcher):
    """S5: board slugs rot when firms change ATS vendor. A board that answers
    with an empty list is the shape that migration takes -- the old slug keeps
    resolving for a while and simply has nothing on it."""
    wire(fetcher)
    fetcher.route("postings/fintechco", [])
    brief, data = scout.build_brief(ctx, registry=REGISTRY, use_llm=False)
    assert data["status"]["FintechCo"] == "no postings returned"
    assert any("contributed nothing" in d for d in brief.degradations)
    assert "FintechCo" in brief.render()


def test_a_board_returning_junk_does_not_kill_the_sweep(ctx, fetcher):
    wire(fetcher)
    fetcher.route("boards/quantco/jobs", "<html>not json</html>")
    _, data = scout.build_brief(ctx, registry=REGISTRY, use_llm=False)
    assert len(data["all"]) == 2          # the other two boards still answered


def test_all_three_vendor_shapes_parse(ctx, fetcher):
    wire(fetcher)
    postings = Boards(fetcher).fetch_all(REGISTRY)
    assert {p.source for p in postings} == {"greenhouse", "lever", "ashby"}
    assert all(p.title and p.url for p in postings)
    assert any(p.location == "Remote" for p in postings)


def test_postings_without_a_url_are_dropped(ctx, fetcher):
    fetcher.route("boards/quantco/jobs", {"jobs": [{"title": "Intern", "absolute_url": ""}]})
    assert Boards(fetcher).fetch_all(REGISTRY[:1]) == []


# -- verdicts and output -----------------------------------------------------

def test_llm_verdicts_reorder_but_never_gate(ctx, fetcher):
    wire(fetcher)
    ctx.llm.default_response = (
        '[{"url": "https://jobs.ashbyhq.com/aico/1", "verdict": "apply", "why": "direct fit"}]')
    brief, data = scout.build_brief(ctx, registry=REGISTRY, use_llm=True)
    assert len(data["new"]) == 3                      # nothing was filtered out
    assert data["new"][0].company == "AICo"           # but the "apply" floated up
    assert "direct fit" in brief.render()


def test_unparseable_verdicts_degrade_to_score_order(ctx, fetcher):
    wire(fetcher)
    ctx.llm.default_response = "I could not do that."
    _, data = scout.build_brief(ctx, registry=REGISTRY, use_llm=True)
    assert data["verdicts"] == {}
    assert [p.score for p in data["new"]] == sorted(
        (p.score for p in data["new"]), reverse=True)


def test_run_writes_an_artifact_and_records_the_run(ctx, fetcher):
    wire(fetcher)
    res = scout.run(ctx, registry=REGISTRY, use_llm=False, commit=False)
    assert res.ok
    assert res.artifact.is_file()
    assert res.data["scanned"] == 3
    assert "3 new" in res.summary


@pytest.mark.benchmark
def test_an_empty_diff_does_not_blame_the_model(ctx, fetcher):
    """S6 (regression). With nothing new the brief said model verdicts were
    "unavailable this run". The model was fine; there was nothing to rank, and a
    warning that fires on a healthy run stops being read."""
    wire(fetcher)
    scout.build_brief(ctx, registry=REGISTRY, use_llm=True)      # seeds the diff
    brief, data = scout.build_brief(ctx, registry=REGISTRY, use_llm=True)
    text = brief.render()
    assert data["new"] == []
    assert "unavailable" not in text
    assert "No new postings to rank" in text


def test_a_genuinely_absent_model_is_still_reported(cfg, ctx, fetcher):
    """S6: the real failure must survive the fix for the false one."""
    from agents_work.llm import FakeLLM
    ctx.llm = FakeLLM(cfg, available=False)
    wire(fetcher)
    brief, data = scout.build_brief(ctx, registry=REGISTRY, use_llm=True)
    assert data["new"]
    assert "Model ranking was unavailable this run" in brief.render()


def test_switching_the_model_off_says_so(ctx, fetcher):
    """S6."""
    wire(fetcher)
    brief, _ = scout.build_brief(ctx, registry=REGISTRY, use_llm=False)
    assert "switched off" in brief.render()


# -- registry hygiene (no network) -------------------------------------------

def test_registry_is_well_formed():
    from agents_work.sources.jobs import REGISTRY as LIVE, SECTORS, VENDORS
    assert len(LIVE) >= 15
    assert all(vendor in VENDORS for vendor, *_ in LIVE)
    assert all(sector in SECTORS for *_, sector in LIVE)
    assert all(company and sector for *_, company, sector in LIVE)
    keys = [(vendor, slug) for vendor, slug, *_ in LIVE]
    assert len(keys) == len(set(keys)), "duplicate board in the registry"
    names = [company for *_, company, _ in LIVE]
    assert len(names) == len(set(names)), "duplicate company in the registry"


@pytest.mark.parametrize("vendor,parts,shape", [
    ("workday", 3, "tenant/wdN/Site"),
    ("oracle", 2, "host/siteNumber"),
    ("eightfold", 2, "sub/domain"),
])
def test_compound_slugs_have_the_shape_their_adapter_expects(vendor, parts, shape):
    """A registry typo in one of these reads as a dead board on the night it
    lands, not at import; the adapters raise on the wrong shape and this is the
    check that runs without the network."""
    from agents_work.sources.jobs import REGISTRY as LIVE
    rows = [(slug, company) for v, slug, company, _ in LIVE if v == vendor]
    assert rows, f"no {vendor} boards left in the registry"
    for slug, company in rows:
        assert len(slug.split("/")) == parts, f"{company}: {slug!r} is not {shape}"


@pytest.mark.benchmark
def test_registry_covers_every_sector_it_claims_to():
    """S13: the point of the expansion is that banks, brokers, exchanges and
    enterprise IT are all reachable, not just the quant/fintech/AI startups the
    original three vendors happened to cover."""
    from agents_work.sources.jobs import REGISTRY as LIVE
    sectors = {sector for *_, sector in LIVE}
    assert {"quant", "fintech", "ai", "bank", "broker", "exchange",
            "enterprise"} <= sectors
    # every sector has enough boards to survive one of them going dark
    for sector in sectors:
        assert sum(1 for *_, s in LIVE if s == sector) >= 2, f"{sector} has one board"


@pytest.mark.benchmark
def test_cantor_fitzgerald_and_its_adjacent_desks_are_covered():
    """S13: the user asked for Cantor Fitzgerald and firms like it by name, and
    none of them is on Greenhouse/Lever/Ashby -- they are the reason the Oracle
    and Workday adapters exist."""
    from agents_work.sources.jobs import REGISTRY as LIVE
    names = {company for *_, company, _ in LIVE}
    assert any("Cantor" in n for n in names)
    brokers = {c for _, _, c, s in LIVE if s in ("broker", "bank", "exchange")}
    assert len(brokers) >= 6, brokers


@pytest.mark.benchmark
def test_verdicts_attach_when_identity_lives_in_the_query_string(ctx, fetcher):
    """S7 (regression). Six firms point every posting at one generic careers page
    and distinguish them only by `gh_jid`. Verdicts keyed on a stripped url
    missed every one, and the brief still claimed they were model-assigned."""
    fetcher.route("boards/quantco/jobs", {"jobs": [
        {"title": "Quantitative Trading Internship (Summer 2027)",
         "location": {"name": "Chicago, IL"},
         "absolute_url": "https://www.jumptrading.com/hr/job?gh_jid=7982619",
         "updated_at": "2026-08-19T10:00:00Z"},
        {"title": "Campus Quantitative Research Intern",
         "location": {"name": "New York, NY"},
         "absolute_url": "https://www.jumptrading.com/hr/job?gh_jid=7848371",
         "updated_at": "2026-08-19T10:00:00Z"}]})
    ctx.llm.default_response = (
        '[{"url": "https://www.jumptrading.com/hr/job?gh_jid=7982619",'
        ' "verdict": "apply", "why": "exactly the target role"},'
        ' {"url": "https://www.jumptrading.com/hr/job?gh_jid=7848371",'
        ' "verdict": "maybe", "why": "research rather than trading"}]')
    brief, data = scout.build_brief(ctx, registry=REGISTRY[:1], use_llm=True)

    assert len(data["new"]) == 2, "the two roles collapsed into one key"
    verdicts = {p.key: data["verdicts"].get(p.key, {}).get("verdict") for p in data["new"]}
    assert set(verdicts.values()) == {"apply", "maybe"}
    assert "exactly the target role" in brief.render()


@pytest.mark.benchmark
def test_verdicts_survive_the_model_re_adding_a_tracking_parameter(ctx, fetcher):
    """S7."""
    fetcher.route("boards/quantco/jobs", {"jobs": [{
        "title": "Quant Research Intern", "location": {"name": "NYC"},
        "absolute_url": "https://x.com/jobs/1", "updated_at": "2026-08-19T10:00:00Z"}]})
    ctx.llm.default_response = (
        '[{"url": "https://x.com/jobs/1/?utm_source=chat", "verdict": "maybe",'
        ' "why": "adjacent"}]')
    _, data = scout.build_brief(ctx, registry=REGISTRY[:1], use_llm=True)
    assert data["verdicts"]["https://x.com/jobs/1"]["verdict"] == "maybe"


@pytest.mark.benchmark
def test_an_ambiguous_verdict_url_is_dropped_not_guessed(ctx, fetcher):
    """S7: attaching one role's verdict to another's row is worse than no verdict."""
    fetcher.route("boards/quantco/jobs", {"jobs": [
        {"title": "Quant Trading Intern", "location": {"name": "NYC"},
         "absolute_url": "https://www.jumptrading.com/hr/job?gh_jid=1",
         "updated_at": "2026-08-19T10:00:00Z"},
        {"title": "Quant Research Intern", "location": {"name": "NYC"},
         "absolute_url": "https://www.jumptrading.com/hr/job?gh_jid=2",
         "updated_at": "2026-08-19T10:00:00Z"}]})
    ctx.llm.default_response = (
        '[{"url": "https://www.jumptrading.com/hr/job", "verdict": "apply", "why": "x"}]')
    _, data = scout.build_brief(ctx, registry=REGISTRY[:1], use_llm=True)
    assert data["verdicts"] == {}


def test_posting_key_keeps_identity_and_drops_tracking():
    from agents_work.sources.jobs import posting_key
    # identity survives
    assert posting_key("https://j.com/hr/job?gh_jid=1") != posting_key("https://j.com/hr/job?gh_jid=2")
    assert posting_key("https://g.com/embed?for=gemini&token=8065112") == \
        "https://g.com/embed?for=gemini&token=8065112"
    # tracking does not
    assert posting_key("https://x/jobs/1?utm_source=a&gh_src=b") == "https://x/jobs/1"
    assert posting_key("https://x/jobs/1?t=gh_src%3D&gh_jid=9") == "https://x/jobs/1?gh_jid=9"
    # cosmetic differences do not
    assert posting_key("https://x/jobs/1/#apply") == posting_key("https://x/jobs/1")
    assert posting_key("https://x/j?b=2&a=1") == posting_key("https://x/j?a=1&b=2")
    assert posting_key("") == ""


@pytest.mark.benchmark
def test_only_displayed_postings_are_marked_seen(ctx, fetcher):
    """S8 (regression). `mark_new` claimed every candidate while the brief showed
    the first `limit`. The remainder were remembered as shown without ever having
    been, so they could never appear in any future diff — a silent cap that read
    as "nothing new"."""
    fetcher.route("boards/quantco/jobs", {"jobs": [
        {"title": f"Quantitative Trading Intern {i}", "location": {"name": "New York, NY"},
         "absolute_url": f"https://x.com/jobs/{i}", "updated_at": "2026-08-19T10:00:00Z"}
        for i in range(10)]})

    brief, data = scout.build_brief(ctx, registry=REGISTRY[:1], limit=4, use_llm=False)
    assert len(data["new"]) == 4
    assert data["backlog"] == 6
    assert "6 more queued for the next run" in brief.render()
    assert seen_count(ctx.db, "scout") == 4

    # the queued six are still new tomorrow, not swallowed
    _, second = scout.build_brief(ctx, registry=REGISTRY[:1], limit=4, use_llm=False)
    assert len(second["new"]) == 4
    assert second["backlog"] == 2
    third = scout.build_brief(ctx, registry=REGISTRY[:1], limit=4, use_llm=False)[1]
    assert len(third["new"]) == 2 and third["backlog"] == 0
    assert scout.build_brief(ctx, registry=REGISTRY[:1], limit=4, use_llm=False)[1]["new"] == []


def test_backlog_is_silent_when_there_is_none(ctx, fetcher):
    wire(fetcher)
    brief, data = scout.build_brief(ctx, registry=REGISTRY, use_llm=False)
    assert data["backlog"] == 0
    assert "queued for the next run" not in brief.render()


@pytest.mark.benchmark
def test_every_displayed_row_is_ranked(ctx, fetcher):
    """S9 (regression). The brief displayed 25 rows and asked the model about 20,
    so five carried a dash while the summary claimed verdicts were assigned."""
    fetcher.route("boards/quantco/jobs", {"jobs": [
        {"title": f"Quantitative Trading Intern {i}", "location": {"name": "New York, NY"},
         "absolute_url": f"https://x.com/jobs/{i}", "updated_at": "2026-08-19T10:00:00Z"}
        for i in range(25)]})
    ctx.llm.default_response = json.dumps([
        {"url": f"https://x.com/jobs/{i}", "verdict": "apply", "why": "fit"}
        for i in range(25)])
    brief, data = scout.build_brief(ctx, registry=REGISTRY[:1], limit=25, use_llm=True)
    assert len(data["new"]) == 25
    unranked = [p for p in data["new"] if p.key not in data["verdicts"]]
    assert unranked == [], f"{len(unranked)} displayed rows never reached the model"
    assert "—" not in brief.render().split("## Coverage")[0].split("| Why |")[-1]


@pytest.mark.benchmark
def test_partial_ranking_is_declared(ctx, fetcher):
    """S9: when the model genuinely answers for only some rows, say how many."""
    fetcher.route("boards/quantco/jobs", {"jobs": [
        {"title": f"Quantitative Trading Intern {i}", "location": {"name": "New York, NY"},
         "absolute_url": f"https://x.com/jobs/{i}", "updated_at": "2026-08-19T10:00:00Z"}
        for i in range(4)]})
    ctx.llm.default_response = json.dumps(
        [{"url": "https://x.com/jobs/0", "verdict": "apply", "why": "fit"}])
    brief, data = scout.build_brief(ctx, registry=REGISTRY[:1], limit=4, use_llm=True)
    assert len(data["verdicts"]) == 1
    assert "for 1 of 4 rows" in brief.render()


# -- batched ranking ---------------------------------------------------------

@pytest.mark.benchmark
def test_ranking_batches_so_a_large_run_still_gets_verdicts(ctx, fetcher, cfg):
    """S10 (regression risk). One verdict is ~40 tokens of JSON; asking for a
    hundred in one completion truncates the array mid-element, which parses as
    nothing. A raised cap would then cost every verdict, not a few."""
    from agents_work.llm import FakeLLM
    postings = [Posting("QuantCo", f"Quant Intern {i}", "NYC", f"https://x.com/jobs/{i}")
                for i in range(60)]
    replies = []
    for start in range(0, 60, 25):
        chunk = postings[start:start + 25]
        replies.append(json.dumps([{"url": p.url, "verdict": "apply", "why": "fit"}
                                   for p in chunk]))
    ctx.llm = FakeLLM(cfg, replies)

    verdicts = scout.rank(ctx, postings)
    assert len(ctx.llm.prompts) == 3, "did not batch"
    assert len(verdicts) == 60, "verdicts lost between batches"
    assert all(len(p) < 40_000 for p in ctx.llm.prompts)


def test_a_failed_batch_costs_only_that_batch(ctx, fetcher, cfg):
    from agents_work.llm import FakeLLM
    postings = [Posting("QuantCo", f"Quant Intern {i}", "NYC", f"https://x.com/jobs/{i}")
                for i in range(50)]
    good = json.dumps([{"url": p.url, "verdict": "apply", "why": "fit"}
                       for p in postings[:25]])
    ctx.llm = FakeLLM(cfg, [good, "I cannot comply with that."])
    verdicts = scout.rank(ctx, postings)
    assert len(verdicts) == 25
    assert all(p.key in verdicts for p in postings[:25])


def test_max_tokens_scales_with_the_batch(ctx, cfg, monkeypatch):
    seen = {}
    from agents_work.llm import FakeLLM
    ctx.llm = FakeLLM(cfg, ["[]"])
    real = ctx.llm.json
    monkeypatch.setattr(ctx.llm, "json",
                        lambda prompt, **kw: seen.update(kw) or real(prompt, **kw))
    scout.rank(ctx, [Posting("C", f"T{i}", "L", f"https://x/{i}") for i in range(25)])
    assert seen["max_tokens"] >= 25 * 40


def test_the_display_cap_is_reachable_from_the_cli(ctx, fetcher, monkeypatch, cfg):
    from agents_work import cli
    monkeypatch.setattr(cli, "load_config", lambda **kw: cfg)
    captured = {}
    monkeypatch.setattr(cli.scout, "run", lambda ctx, **kw: captured.update(kw) or
                        type("R", (), {"agent": "scout", "target": "", "ok": True,
                                       "summary": "", "artifact": None, "degradations": [],
                                       "error": None, "data": {}})())
    cli.main(["scout", "--limit", "250"])
    assert captured["limit"] == 250
    captured.clear()
    cli.main(["scout"])
    assert "limit" not in captured, "the CLI should not override the module default"


def test_the_default_display_cap_is_generous():
    assert scout.DISPLAY_LIMIT >= 100


# -- S11: the day's digest only grows -----------------------------------------

@pytest.mark.benchmark
def test_a_second_run_does_not_shrink_the_days_digest(ctx, fetcher):
    """S11 (regression). A day's brief is named for the day, so a second run
    overwrites the first. The run that surfaced 104 roles was replaced first by
    one that surfaced none, then -- once that was guarded -- by one that
    surfaced the 29 the display cap had queued, turning 100 rows into 29. The
    digest is the day's union now, so it cannot shrink."""
    fetcher.route("boards/quantco/jobs", {"jobs": [
        {"title": f"Quantitative Trading Intern {i}", "location": {"name": "New York, NY"},
         "absolute_url": f"https://x.com/jobs/{i}", "updated_at": "2026-08-19T10:00:00Z"}
        for i in range(10)]})

    first = scout.run(ctx, registry=REGISTRY[:1], limit=6, use_llm=False, commit=False)
    assert "6 new" in first.summary
    assert first.artifact.read_text().count("Quantitative Trading Intern") == 6

    # the queued four arrive next run and are added, not substituted
    second = scout.run(ctx, registry=REGISTRY[:1], limit=6, use_llm=False, commit=False)
    text = second.artifact.read_text()
    assert "4 new" in second.summary and "10 today" in second.summary
    assert text.count("Quantitative Trading Intern") == 10
    assert "Surfaced today (10)" in text

    # and a run with nothing new leaves all ten in place
    third = scout.run(ctx, registry=REGISTRY[:1], limit=6, use_llm=False, commit=False)
    assert "0 new" in third.summary
    assert third.artifact.read_text().count("Quantitative Trading Intern") == 10


@pytest.mark.benchmark
def test_a_verdict_survives_into_a_later_run(ctx, fetcher):
    """S11: verdicts are stored with the posting, so the day's digest keeps them
    without paying the model again for rows it already judged."""
    wire(fetcher)
    ctx.llm.default_response = (
        '[{"url": "https://jobs.ashbyhq.com/aico/1", "verdict": "apply",'
        ' "why": "direct fit"}]')
    scout.run(ctx, registry=REGISTRY, use_llm=True, commit=False)

    ctx.llm.default_response = "[]"
    second = scout.run(ctx, registry=REGISTRY, use_llm=True, commit=False)
    assert second.data["new"] == []
    assert "direct fit" in second.artifact.read_text()


def test_an_empty_first_run_of_the_day_still_writes(ctx, fetcher):
    res = scout.run(ctx, registry=REGISTRY, use_llm=False, commit=False)
    assert res.artifact.is_file()
    assert "Nothing new" in res.artifact.read_text()


@pytest.mark.benchmark
def test_the_digest_leads_with_a_shortlist(ctx, fetcher):
    """S12. A 129-row table is fourteen chunks to a retriever and a wall to a
    reader, and neither starts at the top: asked which roles to prioritise, the
    analyst kept quoting an arbitrary mid-table slice. The shortlist puts the
    answer in one place small enough to read or retrieve whole."""
    fetcher.route("boards/quantco/jobs", {"jobs": [
        {"title": f"Quantitative Trading Intern {i}", "location": {"name": "New York, NY"},
         "absolute_url": f"https://x.com/jobs/{i}", "updated_at": "2026-08-19T10:00:00Z"}
        for i in range(14)]})
    ctx.llm.default_response = json.dumps(
        [{"url": f"https://x.com/jobs/{i}", "verdict": "apply" if i < 12 else "skip",
          "why": f"reason {i}"} for i in range(14)])

    brief, data = scout.build_brief(ctx, registry=REGISTRY[:1], use_llm=True)
    text = brief.render()
    shortlist = text.split("## Worth applying to")[1].split("##")[0]

    assert shortlist.count("https://x.com/jobs/") == scout.TOP_MATCHES
    assert "reason 0" in shortlist
    assert "jobs/13" not in shortlist, "a skip verdict reached the shortlist"
    assert len(shortlist) < 2000, "the shortlist must fit one retrieval chunk"
    assert text.index("Worth applying to") < text.index("Surfaced today")


def test_the_shortlist_is_absent_when_nothing_is_worth_applying_to(ctx, fetcher):
    wire(fetcher)
    ctx.llm.default_response = json.dumps(
        [{"url": "https://jobs.ashbyhq.com/aico/1", "verdict": "skip", "why": "no"}])
    brief, _ = scout.build_brief(ctx, registry=REGISTRY, use_llm=True)
    assert "Worth applying to" not in brief.render()


# -- days open ---------------------------------------------------------------

def _gh_dated(first_published, updated="2026-08-20T10:00:00Z", n=1):
    return {"jobs": [{"title": f"Quantitative Trading Intern {i}",
                      "location": {"name": "New York, NY"},
                      "absolute_url": f"https://x.com/jobs/{i}",
                      "first_published": first_published, "updated_at": updated}
                     for i in range(n)]}


@pytest.mark.benchmark
def test_days_open_counts_from_first_publication_not_last_edit(ctx, fetcher):
    """S14: `updated_at` moves every time a recruiter touches the requisition.
    Reading the age off it would reset the clock on a role that has been open
    since March and report it as posted today -- the exact opposite of what the
    column is for."""
    fetcher.route("boards/quantco/jobs",
                  _gh_dated("2026-03-01T09:00:00Z", updated="2026-08-20T10:00:00Z"))
    _, data = scout.build_brief(ctx, registry=REGISTRY[:1], use_llm=False)
    assert data["new"][0].posted == "2026-03-01"
    assert data["new"][0].days_open > 150


@pytest.mark.benchmark
def test_the_brief_shows_how_long_each_posting_has_been_open(ctx, fetcher):
    """S15: the column the shortlist and the table both carry."""
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date()
    fetcher.route("boards/quantco/jobs", _gh_dated(today.isoformat()))
    ctx.llm.default_response = json.dumps(
        [{"url": "https://x.com/jobs/0", "verdict": "apply", "why": "fit"}])
    brief, _ = scout.build_brief(ctx, registry=REGISTRY[:1], use_llm=True)
    text = brief.render()
    assert "Days open" in text
    assert "posted today" in text        # shortlist phrasing
    assert "| 0 |" in text               # table cell


def test_an_undated_posting_says_so_rather_than_claiming_zero(ctx, fetcher):
    """A blank cell in a numeric column reads as zero, and zero is a claim."""
    fetcher.route("boards/quantco/jobs", {"jobs": [
        {"title": "Quantitative Trading Intern", "location": {"name": "NYC"},
         "absolute_url": "https://x.com/jobs/1"}]})
    brief, data = scout.build_brief(ctx, registry=REGISTRY[:1], use_llm=False)
    assert data["new"][0].posted == ""
    assert data["new"][0].days_open is None
    assert "—" in brief.render()
    assert "publishes none" in brief.render()


@pytest.mark.parametrize("phrase,expected_days,floor", [
    ("Posted Today", 0, False),
    ("Posted Yesterday", 1, False),
    ("Posted 7 Days Ago", 7, False),
    ("Posted 30+ Days Ago", 30, True),
    ("", None, False),
])
def test_workdays_relative_age_becomes_a_real_date(phrase, expected_days, floor):
    """Workday publishes a bucket, not a date. Storing the phrase would leave a
    posting reading '30+ days' forever; storing the date it implies keeps
    ageing."""
    from datetime import date
    from agents_work.sources.jobs import days_open, workday_posted
    today = date(2026, 8, 20)
    posted, is_floor = workday_posted(phrase, today)
    assert is_floor is floor
    assert days_open(posted, today) == expected_days


def test_a_floored_age_is_rendered_as_a_floor():
    from datetime import date
    from agents_work.sources.jobs import format_days_open
    today = date(2026, 8, 20)
    assert format_days_open("2026-07-21", True, today) == "30+"
    assert format_days_open("2026-07-21", False, today) == "30"
    assert format_days_open("", False, today) == "—"


@pytest.mark.benchmark
def test_the_freshest_posting_wins_the_last_slot_when_the_cap_binds(ctx, fetcher):
    """S16: after a registry change the display cap binds for several nights.
    Among equally-scoring postings the one that opened yesterday is the one
    worth reading tonight; the older one keeps its place in the queue."""
    fetcher.route("boards/quantco/jobs", {"jobs": [
        {"title": "Quantitative Trading Intern old", "location": {"name": "NYC"},
         "absolute_url": "https://x.com/jobs/old",
         "first_published": "2025-01-01T00:00:00Z"},
        {"title": "Quantitative Trading Intern new", "location": {"name": "NYC"},
         "absolute_url": "https://x.com/jobs/new",
         "first_published": "2026-08-19T00:00:00Z"}]})
    _, data = scout.build_brief(ctx, registry=REGISTRY[:1], limit=1, use_llm=False)
    assert [p.title for p in data["new"]] == ["Quantitative Trading Intern new"]
    assert data["backlog"] == 1
    # and the older one is still there tomorrow rather than swallowed
    _, second = scout.build_brief(ctx, registry=REGISTRY[:1], limit=1, use_llm=False)
    assert [p.title for p in second["new"]] == ["Quantitative Trading Intern old"]


# -- the vendors added for banks, brokers and enterprise IT -------------------

WORKDAY_URL = "wday/cxs/acme/Careers/jobs"
ORACLE_URL = "recruitingCEJobRequisitions"
EIGHTFOLD_URL = "eightfold.ai/api/apply"


def test_workday_boards_parse_and_paginate(ctx, fetcher):
    """Workday hands out twenty postings a request, so a board of thirty is two
    requests and a parser that stopped at the first would lose a third of it."""
    from agents_work.sources.jobs import Boards
    page1 = {"total": 30, "jobPostings": [
        {"title": f"Software Engineer Intern {i}", "locationsText": "New York",
         "externalPath": f"/job/NY/SWE-Intern-{i}_R{i}", "postedOn": "Posted 3 Days Ago"}
        for i in range(20)]}
    page2 = {"total": 30, "jobPostings": [
        {"title": f"Software Engineer Intern {i}", "locationsText": "New York",
         "externalPath": f"/job/NY/SWE-Intern-{i}_R{i}", "postedOn": "Posted Today"}
        for i in range(20, 30)]}
    fetcher.route(WORKDAY_URL + '|{"appliedFacets": {}, "limit": 20, "offset": 0', page1)
    fetcher.route(WORKDAY_URL + '|{"appliedFacets": {}, "limit": 20, "offset": 20', page2)
    got = Boards(fetcher).fetch_all((("workday", "acme/wd1/Careers", "Acme", "bank"),))
    assert len(got) == 30
    assert got[0].url.startswith("https://acme.wd1.myworkdayjobs.com/en-US/Careers/job/")
    assert got[0].days_open == 3


def test_a_workday_board_too_big_to_sweep_says_so(ctx, fetcher):
    """A truncated board that reports itself as complete is the silent cap this
    codebase keeps refusing to ship."""
    from agents_work.sources.jobs import Boards
    big = {"total": 2000, "jobPostings": [
        {"title": "Quant Intern", "locationsText": "NY",
         "externalPath": "/job/NY/Quant-Intern_R1", "postedOn": "Posted Today"}]}
    boards = Boards(fetcher)
    fetcher.route(WORKDAY_URL, big)
    boards.fetch_all((("workday", "acme/wd1/Careers", "Acme", "bank"),))
    assert "early-career search of 2000" in boards.source_status["Acme"]


def test_oracle_boards_parse(ctx, fetcher):
    """Cantor Fitzgerald, BGC and Lazard all live here."""
    from agents_work.sources.jobs import Boards
    fetcher.route(ORACLE_URL, {"items": [{"TotalJobsCount": 2, "requisitionList": [
        {"Id": "249697", "Title": "2027 Summer Analyst - Technology",
         "PrimaryLocation": "New York, NY, United States", "PostedDate": "2026-08-18"},
        {"Id": "249698", "Title": "Senior Broker", "PrimaryLocation": "London",
         "PostedDate": "2026-08-19"}]}]})
    got = Boards(fetcher).fetch_all(
        (("oracle", "h.oraclecloud.com/CX_1003", "Cantor Fitzgerald / BGC", "broker"),))
    assert [p.title for p in got] == ["2027 Summer Analyst - Technology", "Senior Broker"]
    assert got[0].url.endswith("/sites/CX_1003/job/249697")
    assert got[0].posted == "2026-08-18"
    assert got[0].is_internship, "'2027 Summer Analyst' is how a bank says intern"


def test_eightfold_boards_parse(ctx, fetcher):
    from agents_work.sources.jobs import Boards
    fetcher.route(EIGHTFOLD_URL, {"count": 1, "positions": [
        {"name": "Quantitative Developer Intern", "location": "New York",
         "canonicalPositionUrl": "https://mlp.eightfold.ai/careers/job/7559",
         "t_create": 1_755_000_000}]})
    got = Boards(fetcher).fetch_all((("eightfold", "mlp/mlp.com", "Millennium", "quant"),))
    assert len(got) == 1 and got[0].posted == "2025-08-12"


def test_smartrecruiters_boards_parse(ctx, fetcher):
    from agents_work.sources.jobs import Boards
    fetcher.route("smartrecruiters.com/v1/companies", {"totalFound": 1, "content": [
        {"id": "7439", "name": "Technology Analyst Programme",
         "releasedDate": "2026-08-01T09:00:00.000Z",
         "location": {"city": "London", "region": "England", "country": "gb"}}]})
    got = Boards(fetcher).fetch_all((("smartrecruiters", "acme", "Acme", "bank"),))
    assert got[0].url == "https://jobs.smartrecruiters.com/acme/7439"
    assert got[0].location == "London, England, gb"
    assert got[0].posted == "2026-08-01"


@pytest.mark.parametrize("vendor,slug", [
    ("workday", "acme"), ("oracle", "host"), ("eightfold", "sub")])
def test_a_malformed_compound_slug_fails_that_board_only(ctx, fetcher, vendor, slug):
    from agents_work.sources.jobs import Boards
    boards = Boards(fetcher)
    wire(fetcher)
    got = boards.fetch_all(((vendor, slug, "Acme", "bank"),) + REGISTRY)
    assert boards.source_status["Acme"].startswith("error: ValueError")
    assert len(got) == 3, "the other boards still answered"


# -- scoring -----------------------------------------------------------------

@pytest.mark.benchmark
@pytest.mark.parametrize("title", [
    "Retail Sales Intern",           # "ai" inside "Retail"
    "HTML Email Marketing Intern",   # "ml" inside "HTML"
    "Maintenance Technician Intern",  # "ai" inside "Maintenance"
    "Chair of the Audit Committee",
])
def test_short_keywords_do_not_match_inside_other_words(title):
    """S17 (regression). Scoring was a substring test, and the two-letter
    entries -- "ai" and "ml", worth two points each -- matched inside "Retail",
    "HTML", "Maintenance" and "Chair". The false positives landed on exactly the
    weights that decide whether a posting is worth spending a model call on."""
    p = Posting(company="X", title=title, location="", url="u")
    score(p)
    assert not any(r.endswith((" ai", " ml")) for r in p.reasons), p.reasons


@pytest.mark.benchmark
@pytest.mark.parametrize("title", ["Retail Sales Intern", "HTML Email Marketing Intern"])
def test_a_phantom_keyword_no_longer_lifts_a_posting_over_the_bar(title):
    """S17: the substring match was worth enough on its own to carry an
    irrelevant posting past the score >= 4 gate and into the model's batch."""
    p = Posting(company="X", title=title, location="", url="u")
    score(p)
    assert p.score < 4, f"{title} scored {p.score} ({p.reasons})"


def test_c_plus_plus_still_matches_despite_the_word_boundaries():
    p = Posting(company="X", title="C++ Developer Intern", location="", url="u")
    score(p)
    assert any("c++" in r for r in p.reasons), p.reasons


@pytest.mark.benchmark
@pytest.mark.parametrize("title,wanted", [
    ("2027 Engineering Summer Analyst", True),
    ("Technology Analyst Program - 2027", True),
    ("Global Markets Graduate Programme", True),
    ("Off-Cycle Analyst - Quantitative Research", True),
    ("Early Career Software Engineer", True),
    ("Senior Equity Research Analyst", False),
    ("Analyst, Investment Banking", False),
    ("Vice President, Technology", False),
])
def test_bank_shaped_internship_titles_are_recognised(title, wanted):
    """S18: banks and brokers do not say "intern". Getting this wrong in either
    direction is costly -- miss it and Cantor Fitzgerald contributes nothing,
    over-match it and every senior analyst in the firm lands in the brief."""
    assert Posting(company="X", title=title, location="", url="u").is_internship is wanted


# -- board health, continued -------------------------------------------------

def test_a_failed_request_is_not_reported_as_a_dead_board(ctx, fetcher):
    """S19: an empty board and a failed request produced the same status, and
    that status was "no postings returned" -- the one that means "this firm has
    moved ATS vendor, go probe candidates". Naming the failure is the difference
    between a retry tomorrow and an afternoon spent probing firms that never
    went anywhere."""
    from agents_work.sources.jobs import Boards
    wire(fetcher)
    fetcher.route("postings/fintechco", None)                     # request fails
    fetcher.route("job-board/aico", {"jobs": []})                 # board is empty
    boards = Boards(fetcher)
    boards.fetch_all(REGISTRY)
    assert boards.source_status["FintechCo"] == "unreachable (network or timeout)"
    assert boards.source_status["AICo"] == "no postings returned"

    brief, _ = scout.build_brief(ctx, registry=REGISTRY, use_llm=False)
    degradation = next(d for d in brief.degradations if "contributed nothing" in d)
    assert "FintechCo (unreachable" in degradation
    assert "AICo (no postings returned)" in degradation


def test_an_http_error_names_its_status(ctx, fetcher):
    """S19: a 429 is a rate limit and a 404 is a moved slug. Same empty result,
    different fix."""
    from agents_work.sources.jobs import Boards
    wire(fetcher)
    fetcher.route("boards/quantco/jobs", "rate limited", 429)
    boards = Boards(fetcher)
    boards.fetch_all(REGISTRY)
    assert boards.source_status["QuantCo"] == "HTTP 429"


@pytest.mark.benchmark
def test_a_transient_failure_is_retried_before_the_board_is_written_off(ctx, fetcher):
    """S19: one retry rescues the timeout without hiding the migration -- a
    board that is genuinely gone fails both attempts and still reports."""
    from agents_work.sources.jobs import Boards
    calls = {"n": 0}
    real = fetcher.fetch

    def flaky(url, **kw):
        if "boards/quantco/jobs" in url:
            calls["n"] += 1
            if calls["n"] == 1:
                return None
        return real(url, **kw)

    wire(fetcher)
    fetcher.fetch = flaky
    boards = Boards(fetcher)
    got = boards.fetch_all(REGISTRY[:1])
    assert calls["n"] == 2, "the first failure was never retried"
    assert boards.source_status["QuantCo"] == "1 postings"
    assert len(got) == 1


# -- ranking budget ----------------------------------------------------------

@pytest.mark.benchmark
def test_the_token_budget_covers_the_urls_the_model_must_echo(ctx, cfg, monkeypatch):
    """S10 (regression). The budget was a flat 80 tokens a verdict, fitted to
    Greenhouse urls. Workday and Oracle urls run past a hundred characters and
    the model echoes each one back, so on the first live sweep three batches
    blew the 2000-token cap, truncated mid-array and parsed as nothing -- 75
    postings displayed with no verdict at all."""
    long_url = ("https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite"
                "/job/US-CA-Santa-Clara/Senior-Deep-Learning-Software-Engineer-Intern"
                "-Summer-2027_JR1998877")
    short = [Posting("C", f"T{i}", "L", f"https://x/{i}") for i in range(25)]
    long = [Posting("C", f"T{i}", "L", f"{long_url}{i}") for i in range(25)]
    assert scout.rank_budget(long) > scout.rank_budget(short)
    # the ceiling must clear the urls themselves with the verdicts still to
    # write, and urls tokenise at roughly two characters a token, not four
    assert scout.rank_budget(long) > sum(len(p.key) for p in long) // 2

    seen = {}
    from agents_work.llm import FakeLLM
    ctx.llm = FakeLLM(cfg, ["[]"])
    real = ctx.llm.json
    monkeypatch.setattr(ctx.llm, "json",
                        lambda prompt, **kw: seen.update(kw) or real(prompt, **kw))
    scout.rank(ctx, long)
    assert seen["max_tokens"] == scout.rank_budget(long)


@pytest.mark.benchmark
def test_a_board_that_reports_a_caveat_is_not_counted_as_dead(ctx, fetcher):
    """S19 (regression). The health check was `status.endswith("postings")`, and
    the moment a status grew a caveat -- "271 postings (early-career search of
    1859)" -- seven perfectly healthy boards were named in the degradation line
    as having returned nothing, in a brief that listed their postings two
    sections above."""
    from agents_work.sources.jobs import Boards
    wire(fetcher)
    boards = Boards(fetcher)
    boards.source_status = {"BigCo": "271 postings (early-career search of 1859)",
                            "OldCo": "no postings returned"}
    dead = {c: s for c, s in boards.source_status.items()
            if not re.match(r"\d+ postings", s)}
    assert dead == {"OldCo": "no postings returned"}

    fetcher.route("boards/quantco/jobs", {"jobs": [
        {"title": f"Quantitative Trading Intern {i}", "location": {"name": "NYC"},
         "absolute_url": f"https://x.com/jobs/{i}",
         "first_published": "2026-08-19T00:00:00Z"} for i in range(3)]})
    brief, _ = scout.build_brief(ctx, registry=REGISTRY[:1], use_llm=False)
    assert not any("contributed nothing" in d for d in brief.degradations), \
        brief.degradations


def test_the_sources_line_names_the_vendors_actually_swept(ctx, fetcher):
    """A hard-coded vendor list drifts the moment the registry does, and a brief
    that cites a feed it never read is worse than one that cites fewer."""
    wire(fetcher)
    brief, _ = scout.build_brief(ctx, registry=REGISTRY, use_llm=False)
    cited = next(s.label for s in brief.sources if "job board" in s.label)
    assert "Greenhouse" in cited and "Lever" in cited and "Ashby" in cited
    assert "Workday" not in cited, "no Workday board is in this registry"


def test_every_vendor_the_registry_uses_has_an_adapter_and_a_label():
    from agents_work.sources.jobs import REGISTRY as LIVE, VENDOR_LABELS, VENDORS
    used = {v for v, *_ in LIVE}
    assert used <= set(VENDORS)
    assert used <= set(VENDOR_LABELS)
    # and no adapter is carried for a vendor nothing uses
    assert used == set(VENDORS), f"unused adapters: {set(VENDORS) - used}"
