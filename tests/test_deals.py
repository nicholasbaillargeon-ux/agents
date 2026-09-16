"""M&A feed parsing and the deal book.

The gates here are the four wrong records that reached Postgres on the first
live sweeps: a fashion brand filed as semiconductors, an Australian-dollar deal
counted as dollars, "$1 billion in marketplace sales" recorded as a purchase
price, and a whole enrichment budget spent on links whose bodies cannot be
fetched.
"""

from __future__ import annotations

import pytest

from agents_work.agents import dealbook
from agents_work.sources.deals import (Deal, consideration_value, detect_currency,
                                       find_advisors, looks_like_deal, parse_parties,
                                       parse_value, strip_boilerplate)


@pytest.mark.parametrize("title", [
    "Acme to acquire Beta Corp for $4.1 billion",
    "Acme Agrees to Acquire Beta Corp",
    "Acme Announces Acquisition of Beta Corp",
    "Acme Completes Acquisition of Beta Corp",
    "Acme enters into a definitive agreement to acquire Beta Corp",
])
def test_press_office_phrasings_all_parse(title):
    """(regression) Requiring the verb next to the acquirer matched about one
    headline in eight on a live feed: releases say 'Agrees to', 'Announces',
    'Completes' before it."""
    assert parse_parties(title) == ("Acme", "Beta Corp")


def test_the_outlet_suffix_is_not_part_of_the_target():
    """Google News appends ' - Outlet'. Outlets contain hyphens and dots, which
    the obvious pattern misses, gluing the outlet onto the company name."""
    assert parse_parties("Creative Planning To Acquire RVK - fin-news.com") == (
        "Creative Planning", "RVK")


@pytest.mark.parametrize("title", [
    "Space Force creates office to accelerate innovative tech acquisition",
    "AFNWC welcomes new portfolio acquisition executive",
    "Land bank nearing acquisition of Lowry Middle School",
    "Councilman Green Celebrates Acquisition of Former Church Property",
    "Startup raises $40M to improve customer acquisition",
])
@pytest.mark.benchmark
def test_the_other_meanings_of_acquisition_are_refused(title):
    """D1."""
    assert looks_like_deal(title) is False


@pytest.mark.parametrize("title", [
    "Acme to acquire Beta Corp", "Acme and Beta announce merger",
    "Acme agrees definitive agreement with Beta", "Acme takes majority stake in Beta",
])
def test_real_deal_language_is_kept(title):
    assert looks_like_deal(title) is True


@pytest.mark.benchmark
def test_a_foreign_currency_is_labelled_not_treated_as_dollars():
    """D4. A$2.8B is not $2.8B. A size filter that treats them as equal admits
    deals it was configured to exclude and reports a number that is wrong."""
    title = "Brookfield Agrees to Acquire Reliance Worldwide in A$2.8 Billion Deal"
    assert detect_currency(title) == "AUD"
    deal = Deal(title=title, url="", source="", value_usd=parse_value(title),
                currency=detect_currency(title))
    assert deal.is_usd is False
    assert "AUD" in deal.value_label


def test_word_form_currencies_are_caught_too():
    assert detect_currency("facilitating more than AUD $1 billion in transactions") == "AUD"


@pytest.mark.parametrize("text,expected", [
    ("to acquire Beta for approximately $4.1 billion in cash", 4.1e9),
    ("transaction valued at $275 million", 275e6),
    ("Grab To Acquire 60% Of Atome Financial For $1.49 Billion", 1.49e9),
    ("total consideration of $900 million", 900e6),
])
def test_consideration_language_yields_the_price(text, expected):
    assert consideration_value(text)[0] == pytest.approx(expected)


@pytest.mark.parametrize("text", [
    "facilitating more than AUD $1 billion in certificate transactions in 2025.",
    "a reported near $1 billion in annualized marketplace sales.",
    "serving customers across a $40 billion addressable market",
])
@pytest.mark.benchmark
def test_business_scale_figures_are_not_a_purchase_price(text):
    """D3. (regression) `parse_value` takes the largest figure in its input, which
    is right for a headline and wrong for a release body. Run over a full
    release it filed 'AUD $1 billion in certificate transactions' as the price
    of Formbay and '$1 billion in annualized marketplace sales' as the price of
    four Amazon brands. Both landed in the book as a clean $1.00B."""
    assert consideration_value(text)[0] is None
    assert parse_value(text) is not None      # the looser rule still matches


PRN_PAGE = """WSG Brands Acquires Nasty Gal

Accessibility Statement  Skip Navigation
Tech Computer & Electronics Data Analytics Financial Technology Semiconductors
Entertainment & Media All Entertainment

NEW YORK , Sept. 15, 2026 /PRNewswire/ -- WSG Brands today announced its
acquisition of Nasty Gal, the iconic fashion brand.

SOURCE WSG Brands
Related Links  View original content
"""


@pytest.mark.benchmark
def test_site_navigation_is_stripped_from_a_release_body():
    """D2. (regression) PR Newswire renders its whole industry taxonomy into every
    page's navigation, so a keyword filter over the raw text matched every
    sector on every release — which is how a women's fashion acquisition
    entered the book flagged as semiconductors."""
    body = strip_boilerplate(PRN_PAGE)
    assert body.startswith("NEW YORK")
    assert "semiconductors" not in body.lower()
    assert "Nasty Gal" in body
    assert "SOURCE WSG Brands" not in body


def test_a_page_with_no_dateline_is_returned_intact():
    assert strip_boilerplate("A short note with no wire dateline.").startswith("A short")


def test_the_sector_gate_reads_the_stripped_body():
    clean = Deal(title="WSG Brands Acquires Nasty Gal", url="", source="",
                 body=strip_boilerplate(PRN_PAGE))
    assert dealbook.matches_sector(clean, ["semiconductor", "fintech"]) == ""
    raw = Deal(title="WSG Brands Acquires Nasty Gal", url="", source="", body=PRN_PAGE)
    assert dealbook.matches_sector(raw, ["semiconductor"]) == "semiconductor"


@pytest.mark.benchmark
def test_aggregator_links_are_known_to_be_unfetchable():
    """D5. (regression) Ranked on 'named both parties and a price' alone, every
    enrichment slot in the first live sweep went to Google News items whose
    bodies cannot be fetched at all — 22 read, 0 bodies — and the PR Newswire
    releases carrying financing, multiples and advisor names were never
    reached."""
    google = Deal(title="x", url="https://news.google.com/rss/articles/CBMiabc", source="")
    wire = Deal(title="x", url="https://www.prnewswire.com/news-releases/x.html", source="")
    assert google.body_fetchable is False
    assert wire.body_fetchable is True
    assert sorted([google, wire], key=lambda d: d.body_fetchable, reverse=True)[0] is wire


def test_advisors_are_read_out_of_the_advisor_sentence():
    text = ("Goldman Sachs & Co. LLC acted as exclusive financial advisor to Beta Corp. "
            "Cantor Fitzgerald is acting as financial advisor to Acme.")
    names = find_advisors(text)
    assert any("Goldman" in n for n in names)
    assert any("Cantor" in n for n in names)


@pytest.mark.parametrize("text,expected", [
    ("Goldman Sachs & Co. LLC acted as exclusive financial advisor to Beta Corp. "
     "Cantor Fitzgerald is acting as financial advisor to Acme.",
     ["Goldman Sachs & Co. LLC", "Cantor Fitzgerald"]),
    ("J.P. Morgan Securities LLC served as financial advisor to the Company.",
     ["J.P. Morgan Securities LLC"]),
    ("Evercore and Centerview Partners are acting as financial advisors to Target Inc.",
     ["Evercore", "Centerview Partners"]),
    ("Robert W. Baird & Co. acted as financial advisor to Seller.",
     ["Robert W. Baird & Co."]),
])
@pytest.mark.benchmark
def test_bank_names_survive_the_full_stops_inside_them(text, expected):
    """D6. (regression) Bank names are full of full stops — 'Goldman Sachs & Co.
    LLC', 'J.P. Morgan Securities LLC', 'Robert W. Baird & Co.' — so a sentence
    pattern bounded by [^.] began *after* the name and returned 'LLC'. And a
    backward window that ignores sentence ends glues the previous clause's
    client to the next bank: 'Beta Corp. Cantor Fitzgerald'."""
    assert find_advisors(text) == expected


def test_the_client_after_advisor_to_is_never_read_as_the_advisor():
    """What follows 'financial advisor to' is who was advised."""
    names = find_advisors("Evercore acted as financial advisor to Nasty Gal Holdings.")
    assert names == ["Evercore"]


def test_a_watched_advisor_is_found_in_the_body_even_if_unparsed():
    deal = Deal(title="x", url="", source="",
                body="Cantor Fitzgerald served as financial advisor to the Company.")
    assert dealbook.matches_advisor(deal, ["Cantor Fitzgerald", "BGC"]) == "Cantor Fitzgerald"
    assert dealbook.matches_advisor(deal, ["Evercore"]) == ""


@pytest.mark.benchmark
def test_a_watched_advisor_overrides_the_size_floor():
    """D8. The size floor exists to keep tuck-ins out. A deal the firm you are
    interviewing with is advising is worth a page at any size."""
    small = Deal(title="x", url="", source="", value_usd=5e6)
    keep, why = dealbook.passes_filters(small, min_usd=250e6, advisor="Cantor Fitzgerald")
    assert keep and "Cantor" in why


def test_a_watched_sector_also_overrides_the_size_floor():
    small = Deal(title="x", url="", source="", value_usd=5e6)
    keep, why = dealbook.passes_filters(small, min_usd=250e6, advisor="", sector="fintech")
    assert keep and "fintech" in why


def test_an_undisclosed_tuck_in_with_no_hook_is_dropped():
    deal = Deal(title="x", url="", source="")
    keep, why = dealbook.passes_filters(deal, min_usd=250e6, advisor="")
    assert keep is False and "no disclosed value" in why


def test_a_foreign_currency_deal_is_kept_and_the_size_filter_is_not_applied():
    deal = Deal(title="x", url="", source="", value_usd=2.8e9, currency="AUD")
    keep, why = dealbook.passes_filters(deal, min_usd=250e6, advisor="")
    assert keep and "AUD" in why


def test_the_same_deal_from_four_outlets_is_one_key():
    a = Deal(title="Acme to acquire Beta - Reuters", url="", source="",
             acquirer="Acme", target="Beta")
    b = Deal(title="ACME TO ACQUIRE BETA - Bloomberg", url="", source="",
             acquirer="ACME", target="BETA")
    assert a.key == b.key
