"""Phone delivery. Absence is a normal state; a dead phone is not a dead brief."""

from __future__ import annotations

import pytest

from agents_work.notify import MAX_BODY, FakePush, Push, clip


@pytest.mark.benchmark
def test_no_topic_means_not_sent_and_says_why():
    """M12."""
    result = Push(None).send("body")
    assert result.sent is False
    assert "no AGENTS_NTFY_TOPIC" in result.degradation


def test_an_offline_run_sends_nothing():
    assert Push("topic", enabled=False).send("body").sent is False


def test_a_short_body_is_untouched():
    assert clip("a short brief") == "a short brief"


def test_a_long_body_is_cut_at_a_paragraph_and_says_it_was_cut():
    """A notification that ends mid-number is worse than a short one: the
    reader cannot tell whether 'the 10-year is at 4.' is a cut or a typo."""
    body = "\n\n".join(["paragraph " + "x" * 200 for _ in range(40)])
    out = clip(body)
    assert len(out) <= MAX_BODY
    assert out.endswith("full brief on the dashboard.")
    # The kept text ends where a paragraph did, not mid-token.
    kept = out.split("\n\n…truncated")[0]
    assert kept.endswith("x" * 200)


def test_the_cut_prefers_a_paragraph_boundary():
    body = "first\n\n" + "y" * (MAX_BODY - 100) + "\n\n" + "tail" * 100
    assert "…truncated" in clip(body)


def test_a_fake_push_records_what_would_reach_the_phone():
    push = FakePush()
    assert push.send("body", title="Morning tape", tags="chart").sent is True
    assert push.sent[0]["title"] == "Morning tape"
    assert push.sent[0]["body"] == "body"


def test_an_unavailable_fake_push_reports_rather_than_records():
    push = FakePush(available=False)
    assert push.send("body").sent is False
    assert push.sent == []


def test_a_non_ascii_title_does_not_break_the_request():
    """ntfy headers are latin-1 on the wire; an em dash in a title is enough to
    make the client raise before the request is ever sent."""
    push = Push("topic", server="http://127.0.0.1:1")   # nothing listening
    result = push.send("body", title="Morning tape — 2026-09-16")
    assert result.sent is False
    assert "ConnectError" in result.reason or "Error" in result.reason
