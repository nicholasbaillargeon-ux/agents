"""The dashboard. Read-only, and it must stay read-only."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agents_work.agents import briefing
from agents_work.web.app import create_app


@pytest.fixture
def client(cfg, offline_ctx):
    briefing.run(offline_ctx, commit=False)      # give the dashboard something to show
    return TestClient(create_app(cfg))


def test_dashboard_lists_every_agent(client):
    body = client.get("/").text
    assert client.get("/").status_code == 200
    for title in ("Portfolio research", "Backtest runner", "Market open briefing",
                  "Internship scout", "Personal RAG analyst"):
        assert title in body


def test_dashboard_shows_the_last_run_and_its_degradations(client):
    body = client.get("/").text
    assert "futures priced" in body
    assert "offline mode" in body


def test_agent_index_lists_artifacts(client):
    r = client.get("/agents/briefing")
    assert r.status_code == 200
    assert ".md" in r.text


def test_unknown_agent_is_404(client):
    assert client.get("/agents/nope").status_code == 404


def test_view_renders_markdown_tables(client):
    name = client.get("/api/health").json()
    listing = client.get("/agents/briefing").text
    filename = listing.split('href="/view/briefing/')[1].split('"')[0]
    body = client.get(f"/view/briefing/{filename}").text
    assert "<table>" in body
    assert "<h2>" in body


def test_raw_returns_the_markdown(client):
    listing = client.get("/agents/briefing").text
    filename = listing.split('href="/view/briefing/')[1].split('"')[0]
    r = client.get(f"/raw/briefing/{filename}")
    assert r.status_code == 200
    assert r.text.startswith("---")


@pytest.mark.benchmark
@pytest.mark.parametrize("attack", [
    "../../../../etc/passwd", "..%2f..%2f.env", "....//....//.env",
    "/etc/passwd", "%2e%2e%2f.env",
])
def test_path_traversal_is_refused(client, attack):
    """X6. The filename comes from the URL; a bare Path() here serves the .env."""
    r = client.get(f"/raw/briefing/{attack}")
    assert r.status_code in (307, 404), r.text[:200]
    assert "AGENTS_LLM_API_KEY" not in r.text
    assert "root:" not in r.text


def test_health_reports_capabilities(client):
    payload = client.get("/api/health").json()
    assert payload["ok"] is True
    assert set(payload["capabilities"]) >= {"llm", "price_lake", "sandbox_image"}
    assert set(payload["agents"]) == {"research", "backtest", "briefing", "scout", "analyst"}
    assert payload["agents"]["briefing"]["artifacts"] >= 1


def test_runs_api_filters_by_agent(client):
    assert all(r["agent"] == "briefing"
               for r in client.get("/api/runs?agent=briefing").json())
    assert client.get("/api/runs?limit=1").json().__len__() <= 1


def test_dashboard_survives_an_empty_database(cfg):
    body = TestClient(create_app(cfg)).get("/").text
    assert "no runs recorded yet" in body
    assert "never run" in body


def test_static_stylesheet_is_served(client):
    r = client.get("/static/style.css")
    assert r.status_code == 200
    assert "prefers-color-scheme" in r.text


def test_there_are_no_write_routes(cfg):
    app = create_app(cfg)
    methods = {m for route in app.routes for m in getattr(route, "methods", set())}
    assert methods <= {"GET", "HEAD"}
