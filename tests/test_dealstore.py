"""The deal book's Postgres store.

The property worth a test is the one the whole design exists for: a sweep may
refresh what the agent knows about a deal, and must never touch what you think
about it. Everything else here is plumbing.

Runs against a scratch schema in the configured database and drops it
afterwards, so it can never see or damage the real book. Skipped when Postgres
is not reachable, like every other optional capability in this repo.
"""

from __future__ import annotations

import os

import pytest

from agents_work.dealstore import (DealBookUnavailable, DealRecord, annotate, connect,
                                   counts, ensure_schema, get_deal, health, list_deals,
                                   set_status, upsert)

DSN = os.getenv("AGENTS_DEALBOOK_DSN")
SCHEMA = "dealbook_test"


@pytest.fixture
def conn():
    if not DSN:
        pytest.skip("AGENTS_DEALBOOK_DSN not configured")
    try:
        cm = connect(DSN)
        c = cm.__enter__()
    except DealBookUnavailable as e:
        pytest.skip(f"Postgres unreachable: {e}")
    try:
        with c.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
            cur.execute(f"CREATE SCHEMA {SCHEMA}")
            cur.execute(f"SET search_path TO {SCHEMA}")
        c.commit()
        ensure_schema(c)
        yield c
    finally:
        with c.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        c.commit()
        cm.__exit__(None, None, None)


def _rec(**kw) -> DealRecord:
    base = dict(deal_key="acme|beta", headline="Acme to acquire Beta",
                acquirer="Acme", target="Beta", sector="fintech", value_usd=4.1e9,
                open_questions=["How is it financed?"], advisors=[{"name": "Cantor",
                                                                   "side": "target"}],
                flagged_advisor="Cantor")
    base.update(kw)
    return DealRecord(**base)


def test_a_deal_is_inserted_once_and_updated_thereafter(conn):
    first_id, was_new = upsert(conn, _rec())
    assert was_new is True
    second_id, was_new = upsert(conn, _rec(headline="Acme to acquire Beta (revised)"))
    assert second_id == first_id and was_new is False
    assert get_deal(conn, first_id)["headline"].endswith("(revised)")


@pytest.mark.benchmark
def test_your_view_survives_every_later_sweep(conn):
    """D7. The whole point of the book. A second sweep re-reads the same press
    release; if it overwrote your annotation, the library would reset itself
    every morning and there would be nothing to revise."""
    deal_id, _ = upsert(conn, _rec())
    assert annotate(conn, deal_id, "Multiple looks full versus the 2024 comp.")
    upsert(conn, _rec(headline="Acme to acquire Beta — updated", sector="payments"))
    row = get_deal(conn, deal_id)
    assert row["my_view"] == "Multiple looks full versus the 2024 comp."
    assert row["status"] == "reviewed"
    assert row["reviewed_at"] is not None
    assert row["sector"] == "payments"          # the agent's half did refresh


def test_jsonb_columns_round_trip(conn):
    deal_id, _ = upsert(conn, _rec())
    row = get_deal(conn, deal_id)
    assert row["open_questions"] == ["How is it financed?"]
    assert row["advisors"] == [{"name": "Cantor", "side": "target"}]


def test_listing_filters_by_status_and_by_flag(conn):
    a, _ = upsert(conn, _rec(deal_key="a|b", flagged_advisor="Cantor"))
    b, _ = upsert(conn, _rec(deal_key="c|d", flagged_advisor="", headline="Plain deal"))
    assert {r["id"] for r in list_deals(conn, flagged_only=True)} == {a}
    assert {r["id"] for r in list_deals(conn, status="new")} == {a, b}
    set_status(conn, b, "archived")
    assert {r["id"] for r in list_deals(conn, status="new")} == {a}


def test_counts_summarise_the_book(conn):
    upsert(conn, _rec(deal_key="a|b"))
    deal_id, _ = upsert(conn, _rec(deal_key="c|d", flagged_advisor=""))
    annotate(conn, deal_id, "worth watching")
    c = counts(conn)
    assert c["total"] == 2 and c["flagged"] == 1 and c["annotated"] == 1


def test_annotating_a_deal_that_does_not_exist_reports_rather_than_raises(conn):
    assert annotate(conn, 999_999, "nobody") is False


def test_no_dsn_is_a_normal_state_not_an_exception():
    with pytest.raises(DealBookUnavailable):
        with connect(None):
            pass
    assert "no AGENTS_DEALBOOK_DSN" in health(None)
