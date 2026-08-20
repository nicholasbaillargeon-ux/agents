"""Internship scout: the nightly diff is the product."""

from __future__ import annotations


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
def test_verdicts_survive_a_query_string_in_the_url(ctx, fetcher):
    """S7 (regression). Greenhouse hands back `...?gh_jid=123`. The diff keys on
    the URL with the query stripped, so verdicts keyed on the raw URL missed
    every row — and the brief still claimed "verdicts are model-assigned" while
    printing a dash in each one."""
    fetcher.route("boards/quantco/jobs", {"jobs": [{
        "title": "Quantitative Trading Internship (Summer 2027)",
        "location": {"name": "Chicago, IL"},
        "absolute_url": "https://www.optiver.com/join-us/jobs/8623923002/?gh_jid=8623923002",
        "updated_at": "2026-08-19T10:00:00Z"}]})
    ctx.llm.default_response = (
        '[{"url": "https://www.optiver.com/join-us/jobs/8623923002/",'
        ' "verdict": "apply", "why": "exactly the target role"}]')
    brief, data = scout.build_brief(ctx, registry=REGISTRY[:1], use_llm=True)
    assert data["verdicts"], "no verdicts parsed at all"
    assert data["verdicts"][data["new"][0].key]["verdict"] == "apply"
    assert "exactly the target role" in brief.render()


@pytest.mark.benchmark
def test_verdicts_survive_the_model_re_adding_a_query_string(ctx, fetcher):
    """S7."""
    fetcher.route("boards/quantco/jobs", {"jobs": [{
        "title": "Quant Research Intern", "location": {"name": "NYC"},
        "absolute_url": "https://x.com/jobs/1", "updated_at": "2026-08-19T10:00:00Z"}]})
    ctx.llm.default_response = (
        '[{"url": "https://x.com/jobs/1?utm=abc", "verdict": "maybe", "why": "adjacent"}]')
    _, data = scout.build_brief(ctx, registry=REGISTRY[:1], use_llm=True)
    assert data["verdicts"]["https://x.com/jobs/1"]["verdict"] == "maybe"


def test_posting_key_is_the_one_canonicaliser():
    from agents_work.sources.jobs import posting_key
    assert posting_key("https://x/jobs/1?a=b#frag") == "https://x/jobs/1"
    assert posting_key("https://x/jobs/1") == "https://x/jobs/1"
    assert posting_key("") == ""
    assert Posting("C", "T", "L", "https://x/jobs/1?a=b").key == posting_key("https://x/jobs/1")
