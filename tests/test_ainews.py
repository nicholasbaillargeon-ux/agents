"""The daily AI brief: the feed parsing, the story clustering, and the page.

The gates here are the failures a headline-driven brief actually has. Two of
them produce a *plausible* brief rather than an error, which is why they are
pinned: a story that quietly disappeared into another story's cluster, and a
day's page that got shorter when it was re-run.

Every fixture date is relative to now. Absolute literals fall out of the
freshness window as real time passes, which is how the scout's fixtures
silently stopped exercising anything.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

from agents_work.agents import ainews
from agents_work.agents.ainews import push_body
from agents_work.notify import MAX_BODY, FakePush, Push, PushResult
from agents_work.sources.aifeeds import (Feed, Item, cluster, common_tokens,
                                         distinctive, is_noise, looks_like_ai,
                                         parse_feed, same_story, significance,
                                         split_outlet, story_key)


def _ago(hours: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)


def rss(*entries: tuple) -> str:
    """An RSS body. entries are (title, link, hours_ago[, description])."""
    items = []
    for entry in entries:
        title, link, hours = entry[0], entry[1], entry[2]
        desc = entry[3] if len(entry) > 3 else ""
        items.append(
            f"<item><title>{title}</title><link>{link}</link>"
            f"<pubDate>{format_datetime(_ago(hours))}</pubDate>"
            f"<description>{desc}</description></item>")
    return f"<?xml version='1.0'?><rss version='2.0'><channel>{''.join(items)}</channel></rss>"


def atom(*entries: tuple) -> str:
    """An Atom body — The Verge serves one, and nothing else here does."""
    items = []
    for title, link, hours in entries:
        items.append(
            f"<entry><title>{title}</title>"
            f"<link rel='alternate' href='{link}'/>"
            f"<published>{_ago(hours).isoformat()}</published>"
            f"<summary>a summary of {title}</summary></entry>")
    return ("<?xml version='1.0'?><feed xmlns='http://www.w3.org/2005/Atom'>"
            f"{''.join(items)}</feed>")


LAB = Feed("TestLab", "https://lab.test/rss", "lab")
PRESS = Feed("TestPress", "https://press.test/rss", "press")
PRESS2 = Feed("TestPress2", "https://press2.test/atom", "press")
WIRE = Feed("TestWire", "https://wire.test/rss?q={q}", "aggregator")
TEST_FEEDS = (LAB, PRESS, PRESS2, WIRE)

ANALYSIS = json.dumps({
    "lede": "Corvus shipped Corvid 4 and the rest of the day was lawyers.",
    "stories": [{"n": 1, "why": "A frontier release with a version number."},
                {"n": 2, "why": "The suit now names 550 publications."}],
    "watch": ["Whether Corvid 4 ships with a public eval card."],
})


def wire_up(fetcher, *, lab=None, press=None, press2=None, wire=None):
    fetcher.route("lab.test", lab if lab is not None else rss())
    fetcher.route("press.test", press if press is not None else rss())
    fetcher.route("press2.test", press2 if press2 is not None else atom())
    fetcher.route("wire.test", wire if wire is not None else rss())
    return fetcher


# -- parsing -----------------------------------------------------------------

@pytest.mark.benchmark
def test_an_atom_feed_is_read_like_an_rss_one():
    """N10. (regression) The Verge serves Atom entries and every other feed
    serves RSS items. A parser that iterated `item` alone read it as empty,
    which is indistinguishable in the coverage table from a dead feed."""
    items = parse_feed(atom(("Corvus ships Corvid 4", "https://verge.test/a", 3)),
                       feed="TestPress2", tier="press")
    assert [i.title for i in items] == ["Corvus ships Corvid 4"]
    assert items[0].url == "https://verge.test/a"
    assert items[0].published is not None
    assert items[0].summary


@pytest.mark.benchmark
def test_a_publishers_own_headline_keeps_its_subtitle():
    """N9. (regression) Google News appends ' - Outlet' with a plain hyphen.
    Stripping any dash on every feed filed 'and getting bigger' as the outlet
    of a Verge headline and truncated the headline at the dash."""
    headline = "The AI data center e-waste problem is huge — and getting bigger"
    kept = parse_feed(rss((headline, "https://verge.test/x", 2)),
                      feed="TestPress2", tier="press")
    assert kept[0].title == headline
    assert kept[0].outlet != "and getting bigger"

    wire = parse_feed(rss(("Corvus ships Corvid 4 - Reuters", "https://n.test/1", 2)),
                      feed="TestWire", tier="aggregator")
    assert wire[0].title == "Corvus ships Corvid 4"
    assert wire[0].outlet == "Reuters"


def test_the_outlet_split_survives_hyphenated_outlets():
    """Outlet names contain hyphens and dots. The separator is the spaced
    hyphen, taken from the right — the deal book learned the same thing about
    the same feed."""
    assert split_outlet("Corvus ships Corvid 4 - fin-news.com") == (
        "Corvus ships Corvid 4", "fin-news.com")


def test_a_markup_only_summary_comes_back_empty():
    """Google News descriptions are a block of anchor tags and nothing else. A
    list of outlet names is not a summary, and the model would read it as one."""
    body = rss(("Corvus ships Corvid 4", "https://n.test/1", 1,
                "&lt;a href=&quot;https://x.test&quot;&gt;Reuters&lt;/a&gt;"))
    assert parse_feed(body, feed="TestWire", tier="aggregator")[0].summary == "Reuters"


# -- N2: "AI" is a word ------------------------------------------------------

@pytest.mark.parametrize("title", [
    "Dubai opens a new terminal", "The chair of the committee said no",
    "HTML and CSS for beginners", "Email marketing in Shanghai",
    "How to maintain a retail portfolio", "Thailand details a trade deal",
])
@pytest.mark.benchmark
def test_ai_is_matched_as_a_word_not_a_substring(title):
    """N2. (regression risk, inherited) 'ai' is inside Dubai, chair, said,
    email and Shanghai, and 'ml' is inside HTML. The scout scored a retail
    posting as an AI role this way; in a feed whose whole subject is the
    two-letter token, substring matching passes everything."""
    assert looks_like_ai(title) is False


@pytest.mark.parametrize("title", [
    "An AI-powered assistant ships", "A.I. researchers publish",
    "Corvus releases a new AI model", "The LLM is open-weight now",
    "Machine learning at the edge", "GPT-6 lands",
])
def test_real_ai_language_is_kept(title):
    assert looks_like_ai(title) is True


# -- N3: the other meanings --------------------------------------------------

@pytest.mark.parametrize("title,outlet", [
    ("3 AI Stocks to Buy Before They Soar", ""),
    ("Best AI Tools for students in 2026", ""),
    ("U.S. Bishops Establish a Task Force on Artificial Intelligence", ""),
    ("Your AI horoscope for the week", ""),
    ("City council hears AI presentation", ""),
    ("Corvus IPO Date: What Investors Need to Know", "The Motley Fool"),
])
@pytest.mark.benchmark
def test_the_other_meanings_of_ai_are_refused(title, outlet):
    """N3. A parish task force, a stock listicle and a horoscope all match an
    AI term. On the first live sweep they outnumbered the model releases, and
    the retail-investor ones carry no noise word at all — only an outlet."""
    assert is_noise(title, outlet) is True


@pytest.mark.parametrize("title", [
    "Corvus releases Corvid 4", "Regulators open an antitrust investigation into Corvus",
    "Corvus raises at a $40 billion valuation",
])
def test_real_ai_news_is_not_noise(title):
    assert is_noise(title) is False


def test_a_model_version_outranks_an_opinion_piece():
    """The ordering has to work with no model at all, because the model never
    chooses it — it only writes about what the score already picked."""
    release, _ = significance("Introducing Corvid 4 Live")
    opinion, _ = significance("Opinion: why Corvid could soon matter")
    assert release > opinion


# -- N1 / N4: one story, and only one --------------------------------------

@pytest.mark.benchmark
def test_one_event_reported_nine_times_is_one_story():
    """N1. Nine feeds carrying one announcement is one event. The lab's own
    post supplies the link even when an aggregator reported it first, because
    a Google News redirect token cannot be resolved to the publisher."""
    items = [
        Item("Corvus Weighing Corvid 4 Launch This Week",
             "https://news.google.com/rss/articles/CBMi1", "Reuters", "TestWire",
             "aggregator", _ago(9)),
        Item("Introducing Corvid 4", "https://lab.test/corvid-4", "lab.test",
             "TestLab", "lab", _ago(6), summary="Corvid 4 is our new model."),
        Item("Corvus launches Corvid 4 with longer context",
             "https://press.test/corvid", "TestPress", "TestPress", "press", _ago(5)),
    ]
    stories = cluster(items)
    assert len(stories) == 1
    story = stories[0]
    assert len(story.items) == 3
    assert story.url == "https://lab.test/corvid-4", "the lab's own post is the link"
    assert story.from_lab is True
    # Identity comes from the first report, which does not move when tomorrow's
    # coverage arrives.
    assert story.key == story_key("Corvus Weighing Corvid 4 Launch This Week")


@pytest.mark.benchmark
def test_two_stories_about_one_lab_do_not_merge():
    """N4. (regression) On any given day the lab's name is in a quarter of the
    headlines, so two titles sharing only 'Corvus' and 'Corvid' are two stories
    about one company. Merging them filed a warning about a product as part of
    that product's launch, and the warning vanished from the page."""
    titles = [
        "Corvus merges Corvid chat and Cowork in one interface",
        "Corvus is folding Cowork into Corvid chat",
        "Rival chief warns Corvus Corvid is risky",
        "Corvus sued over Corvid training data",
        "Corvus raises at a record valuation",
        "Corvid tops the Corvus leaderboard again",
    ]
    # min_batch is lowered here on purpose: the frequency rule is switched off
    # for a handful of headlines, because in a batch of three about one launch
    # that launch's own words are in every title. A real sweep carries a
    # hundred-odd items and never reaches for this.
    common = common_tokens(titles, min_batch=1)
    assert "corvus" in common and "corvid" in common
    assert same_story(titles[0], titles[1], common) is True
    assert same_story(titles[0], titles[2], common) is False
    assert same_story(titles[0], titles[3], common) is False
    assert same_story(titles[1], titles[5], common) is False
    # Without the day's common tokens, the lab's name alone fuses two unrelated
    # stories: this is the merge the frequency rule exists to refuse.
    assert same_story(titles[1], titles[5]) is True


def test_a_headline_with_nothing_distinctive_never_merges():
    assert same_story("AI model news", "AI model report") is False


def test_a_story_key_does_not_depend_on_the_rest_of_the_day():
    """The key is written to the run log. If it moved with the day's token
    frequencies, tomorrow's run would meet the same story as a new one."""
    title = "Corvus launches Corvid 4 with longer context"
    assert story_key(title) == story_key(title)
    assert "corvid" in distinctive(title)


# -- the page ----------------------------------------------------------------

def _feeds_with_a_release(hours: float = 4.0):
    return {
        "lab": rss(("Introducing Corvid 4", "https://lab.test/corvid-4", hours,
                    "Corvid 4 is our new frontier model.")),
        "press": rss(("Corvus launches Corvid 4, an AI model with longer context",
                      "https://press.test/corvid", hours - 1),
                     ("Copyright suit against Corvus AI grows to 550 publications",
                      "https://press.test/suit", hours)),
        "press2": atom(("Corvus launches Corvid 4, its new AI model",
                        "https://press2.test/corvid", hours - 2)),
        "wire": rss(("Corvid 4 AI model is here - Reuters",
                     "https://news.google.com/rss/articles/CBMi7", hours + 1)),
    }


def test_the_brief_is_a_list_of_stories_not_of_items(ctx, fetcher):
    ctx.llm.default_response = ANALYSIS
    wire_up(fetcher, **_feeds_with_a_release())
    brief, data = ainews.build_brief(ctx, feeds=TEST_FEEDS)
    assert data["counts"]["items"] == 5
    assert data["counts"]["stories"] == 2
    text = brief.render()
    assert "Corvid 4" in text and "550 publications" in text
    assert "## Coverage" in text and "## Sources" in text


@pytest.mark.benchmark
def test_the_model_writes_last_and_is_checked(ctx, fetcher):
    """N8. The stories, their order and the coverage table are assembled before
    the model is asked anything, and every figure it writes is checked back
    against the story list it was given."""
    ctx.llm.default_response = json.dumps({
        "lede": "Corvus shipped Corvid 4 and raised at a $90 billion valuation.",
        "stories": [{"n": 1, "why": "A frontier release."}],
        "watch": ["An eval card."],
    })
    wire_up(fetcher, **_feeds_with_a_release())
    brief, _ = ainews.build_brief(ctx, feeds=TEST_FEEDS)
    text = brief.render()
    assert "Corvus shipped Corvid 4" in text
    # $90 billion is in the prose and in no story. It is named, not removed:
    # the brief cannot tell a derived figure from an invented one.
    assert "Not found in today's story list" in text
    assert "$90 billion" in text
    assert brief.extra_meta["ungrounded_figures"] == 1


@pytest.mark.benchmark
def test_with_no_model_the_stories_still_render(ctx, fetcher):
    """N8. An LLM outage costs the commentary, not the brief."""
    wire_up(fetcher, **_feeds_with_a_release())
    brief, data = ainews.build_brief(ctx, feeds=TEST_FEEDS, use_llm=False)
    text = brief.render()
    assert "Corvid 4" in text
    assert "## Coverage" in text
    assert "The day" not in text
    assert data["counts"]["stories"] == 2
    assert any("switched off" in d for d in brief.degradations)


# -- N5 / N6: the diff, and the day ----------------------------------------

@pytest.mark.benchmark
def test_a_story_is_reported_once(ctx, fetcher):
    """N5. The diff is the product: a brief that re-lists yesterday's news
    stops being read. Identity is per headline, so a follow-up article joins a
    story already reported rather than resurrecting it."""
    feeds = _feeds_with_a_release()
    wire_up(fetcher, **feeds)
    _, first = ainews.build_brief(ctx, feeds=TEST_FEEDS, use_llm=False)
    assert first["counts"]["new"] == 2

    _, second = ainews.build_brief(ctx, feeds=TEST_FEEDS, use_llm=False)
    assert second["counts"]["new"] == 0, "the same sweep surfaced the same stories twice"

    # A new headline about a story already reported is not a new story.
    feeds["press"] = rss(
        ("Corvus launches Corvid 4, an AI model with longer context",
         "https://press.test/corvid", 3),
        ("Copyright suit against Corvus AI grows to 550 publications",
         "https://press.test/suit", 4),
        ("Corvid 4 AI model draws a longer context comparison",
         "https://press.test/corvid-followup", 1))
    wire_up(fetcher, **feeds)
    _, third = ainews.build_brief(ctx, feeds=TEST_FEEDS, use_llm=False)
    assert third["counts"]["new"] == 0, "a follow-up resurrected a story already told"


@pytest.mark.benchmark
def test_the_days_page_only_grows(ctx, fetcher):
    """N6. (regression, inherited) The scout replaced a hundred-row digest with
    a twenty-nine-row one because it rendered one run's delta. The page is the
    day's union: a second run can add stories and can never remove one."""
    feeds = _feeds_with_a_release()
    wire_up(fetcher, **feeds)
    first, _ = ainews.build_brief(ctx, feeds=TEST_FEEDS, use_llm=False)
    assert "550 publications" in first.render()

    # A later run of the same day, with a different story in the feeds.
    feeds["press"] = rss(("Regulators open an antitrust case into Corvus AI",
                          "https://press.test/anti", 1))
    wire_up(fetcher, **feeds)
    second, data = ainews.build_brief(ctx, feeds=TEST_FEEDS, use_llm=False)
    text = second.render()
    assert "antitrust" in text, "the new story is missing"
    assert "550 publications" in text, "the morning's story fell off the page"
    assert "Corvid 4" in text
    assert data["counts"]["today"] >= 3


# -- N7: coverage ------------------------------------------------------------

@pytest.mark.benchmark
def test_a_feed_that_could_not_be_read_is_named(ctx, fetcher):
    """N7. A feed that failed is a hole in the sweep, and the reason travels
    with the name: unreachable is a retry tomorrow, HTTP 429 is a user-agent
    problem, and an empty body from a feed that answered is a format change."""
    feeds = _feeds_with_a_release()
    fetcher.route("lab.test", feeds["lab"])
    fetcher.route("press.test", None)               # unreachable, twice over
    fetcher.route("press2.test", "not xml at all")  # answered with a non-feed
    fetcher.route("wire.test", "", 429)
    brief, _ = ainews.build_brief(ctx, feeds=TEST_FEEDS, use_llm=False)
    text = brief.render()
    assert "degraded: true" in text
    assert "unreachable" in text and "HTTP 429" in text and "no items returned" in text
    assert "TestPress" in text


@pytest.mark.benchmark
def test_a_quiet_feed_is_not_a_broken_one(ctx, fetcher):
    """N7, the other half. A lab publishes a few posts a month, so 'answered,
    nothing inside the window' is its ordinary state. Reporting that as a
    failure puts a banner on every brief, and a warning that is always on is
    not a warning."""
    feeds = _feeds_with_a_release()
    feeds["lab"] = rss(("An older lab post about an AI model",
                        "https://lab.test/old", 400))
    wire_up(fetcher, **feeds)
    brief, _ = ainews.build_brief(ctx, feeds=TEST_FEEDS, use_llm=False)
    assert not any("could not be read" in d for d in brief.degradations)
    assert "no items in the window" in brief.render()


def test_an_empty_sweep_says_the_sweep_failed(ctx, fetcher):
    """Nothing from any feed is a failure, not a quiet day, and the brief has
    to say which — the two look identical on the page otherwise."""
    brief, _ = ainews.build_brief(ctx, feeds=TEST_FEEDS, use_llm=False)
    assert any("sweep failed" in d for d in brief.degradations)


# -- the run -----------------------------------------------------------------

def test_a_run_writes_an_artifact_and_a_row(ctx, fetcher):
    ctx.llm.default_response = ANALYSIS
    wire_up(fetcher, **_feeds_with_a_release())
    res = ainews.run(ctx, feeds=TEST_FEEDS, commit=False)
    assert res.ok and res.artifact.is_file()
    assert "Corvid 4" in res.artifact.read_text()
    assert res.data["new"] == 2


def test_the_brief_is_committed_to_the_notes_repo(ctx, fetcher):
    ctx.llm.default_response = ANALYSIS
    wire_up(fetcher, **_feeds_with_a_release())
    res = ainews.run(ctx, feeds=TEST_FEEDS, commit=True)
    assert res.ok
    committed = list((ctx.cfg.notes_repo / "ai-news").glob("*.md"))
    assert len(committed) == 1
    assert "AI brief" in committed[0].read_text()
    assert ctx.notes.log(5)[0]["subject"].startswith("AI brief:")


# -- the phone ---------------------------------------------------------------

class FailingPush(FakePush):
    """Configured, reachable topic, and the server said no."""

    def send(self, body, **kw):
        self.sent.append({"body": body, **kw})
        return PushResult(False, "HTTP 503 from https://ntfy.sh")


@pytest.mark.benchmark
def test_the_push_carries_the_headlines_and_no_links(ctx, fetcher):
    """N11. A Google News redirect token runs to five hundred characters, so
    three links would eat a third of ntfy's body limit and push the stories
    themselves past the truncation: a wall of `CBMi...` and no news. The tap
    target is the dashboard, where every link already is."""
    ctx.llm.default_response = ANALYSIS
    ctx.push = FakePush()
    wire_up(fetcher, **_feeds_with_a_release())
    res = ainews.run(ctx, feeds=TEST_FEEDS, commit=False)

    assert res.data["pushed"] is True
    sent = ctx.push.sent[0]
    body = sent["body"]
    assert "Corvus shipped Corvid 4" in body, "the lede is the first thing read"
    assert "Corvid 4" in body and "A frontier release" in body
    assert "http" not in body and "CBMi" not in body
    assert len(body) <= MAX_BODY
    # The notification points at this brief, not at a page the phone cannot route to.
    assert sent["click"].endswith(f"/view/ainews/{res.artifact.name}")
    assert "localhost" not in sent["click"] or sent["click"].startswith("http://localhost")
    assert res.summary.endswith("pushed")


def test_the_push_body_survives_having_nothing_to_say():
    body = push_body("", [], {"today": 0, "items": 0})
    assert "Nothing new" in body and body.strip()


@pytest.mark.benchmark
def test_a_phone_that_did_not_buzz_is_reported(ctx, fetcher):
    """N12. (M12, for this agent) A push that was configured and failed is a
    new fact about the run; one that was never configured is a fact about the
    brief. Both are said out loud — a notification nobody received is the one
    failure the reader cannot notice by themselves."""
    ctx.llm.default_response = ANALYSIS
    wire_up(fetcher, **_feeds_with_a_release())

    ctx.push = FailingPush()
    res = ainews.run(ctx, feeds=TEST_FEEDS, commit=False)
    assert res.data["pushed"] is False
    assert any("HTTP 503" in d for d in res.degradations)

    ctx.push = Push(None)
    brief, _ = ainews.build_brief(ctx, feeds=TEST_FEEDS)
    assert any("not sent" in d for d in brief.degradations)
    assert "degraded: true" in brief.render()
