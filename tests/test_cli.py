"""The `agents` entry point."""

from __future__ import annotations

import json

import pytest

from agents_work import cli


@pytest.fixture(autouse=True)
def _use_test_config(monkeypatch, cfg):
    monkeypatch.setattr(cli, "load_config", lambda **kw: cfg)


def test_status_on_an_empty_log_prints_nothing_and_succeeds(capsys):
    assert cli.main(["status"]) == 0
    assert capsys.readouterr().out == ""


def test_status_lists_recorded_runs(capsys, ctx):
    from agents_work.store import Run, record
    record(ctx.db, Run(agent="research", target="NVDA", ok=True, summary="P/E 33.2"))
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "research" in out and "NVDA" in out and "P/E 33.2" in out


def test_index_rebuilds_and_reports(capsys, vault):
    assert cli.main(["index"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["chunks"] > 0 and stats["files"] == 10


def test_ask_prints_the_answer(capsys):
    assert cli.main(["ask", "what did I conclude about NVDA"]) == 0
    out = capsys.readouterr().out
    assert "backlog" in out or "FAKE" in out


def test_json_output_is_machine_readable(capsys):
    cli.main(["--json", "ask", "what did I conclude about NVDA"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["agent"] == "analyst"
    assert "citations" in payload["data"]


def test_missing_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code == 2


def test_unknown_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit):
        cli.main(["teleport"])


def test_research_accepts_several_tickers(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(cli.research, "run", lambda ctx, t, **kw: seen.append(t) or
                        type("R", (), {"agent": "research", "target": t, "ok": True,
                                       "summary": "", "artifact": None,
                                       "degradations": [], "error": None, "data": {}})())
    assert cli.main(["research", "NVDA", "AAPL"]) == 0
    assert seen == ["NVDA", "AAPL"]
