"""Internship scout: the nightly diff is the product."""

from __future__ import annotations

import json


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
    assert data2["new"] == []
    assert "Nothing new" in brief2.render()


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
    """S5: board slugs rot when firms change ATS vendor."""
    wire(fetcher)
    fetcher.route("postings/fintechco", None)
    brief, data = scout.build_brief(ctx, registry=REGISTRY, use_llm=False)
    assert data["status"]["FintechCo"] == "no postings returned"
    assert any("returned nothing" in d for d in brief.degradations)
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
    from agents_work.sources.jobs import REGISTRY as LIVE
    assert len(LIVE) >= 15
    assert all(vendor in ("greenhouse", "lever", "ashby") for vendor, *_ in LIVE)
    assert all(company and sector for *_, company, sector in LIVE)
    keys = [(vendor, slug) for vendor, slug, *_ in LIVE]
    assert len(keys) == len(set(keys)), "duplicate board in the registry"
    names = [company for *_, company, _ in LIVE]
    assert len(names) == len(set(names)), "duplicate company in the registry"


def test_registry_covers_every_supported_sector():
    from agents_work.sources.jobs import REGISTRY as LIVE
    assert {sector for *_, sector in LIVE} == {"quant", "fintech", "ai"}


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
