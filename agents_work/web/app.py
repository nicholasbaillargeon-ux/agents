"""A read-only dashboard over the agents' output.

Read-only on purpose. These agents commit to a git repo, write to a run log and
spend money at an LLM endpoint; putting a "run now" button on an unauthenticated
LAN page would make all three trivially reachable. The timers own execution, the
CLI owns ad-hoc runs, and this page owns *looking at what happened* — which is
the part you actually want at 8am on a phone.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from markdown_it import MarkdownIt

from ..agents.backtest import docker_available
from ..config import load_config
from ..gitsink import NotesRepo
from ..store import connect, recent

AGENTS = {
    "research": ("Portfolio research", "Ticker in, one-page brief out — filings, "
                 "fundamentals and headlines, committed to the notes repo."),
    "backtest": ("Backtest runner", "A strategy in plain English, generated, run in a "
                 "sandbox, scored on Sharpe and drawdown."),
    "briefing": ("Market open briefing", "Futures, macro, watchlist movers and today's "
                 "earnings, before the bell."),
    "scout": ("Internship scout", "Quant / fintech / AI boards swept nightly; only what "
              "is new since last run."),
    "analyst": ("Personal RAG analyst", "Questions answered from your own notes and "
                "briefs, with citations."),
}

_md = MarkdownIt("commonmark", {"html": True, "linkify": False}).enable("table")


def create_app(cfg=None) -> FastAPI:
    cfg = cfg or load_config()
    cfg.ensure_dirs()
    app = FastAPI(title="agents_work", docs_url=None, redoc_url=None)

    static = Path(__file__).parent / "static"
    static.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static), name="static")

    def db():
        return connect(cfg.db_path)

    def artifacts(agent: str) -> list[Path]:
        d = cfg.out_dir / agent
        if not d.is_dir():
            return []
        return sorted(d.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)

    def safe_artifact(agent: str, name: str) -> Path:
        """Resolve inside the agent's output directory or 404.

        `name` arrives from the URL; a bare `Path(name)` here would serve
        ../../../.env to anyone who asked for it.
        """
        if agent not in AGENTS:
            raise HTTPException(404, "no such agent")
        base = (cfg.out_dir / agent).resolve()
        target = (base / name).resolve()
        if not str(target).startswith(str(base) + "/") or not target.is_file():
            raise HTTPException(404, "no such brief")
        return target

    # -- pages -----------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        conn = db()
        try:
            runs = recent(conn, limit=25)
            last_by_agent = {}
            for row in recent(conn, limit=500):
                last_by_agent.setdefault(row["agent"], row)
        finally:
            conn.close()

        cards = []
        for agent, (title, blurb) in AGENTS.items():
            row = last_by_agent.get(agent)
            files = artifacts(agent)
            latest = files[0] if files else None
            if row is None:
                state, when, detail = "idle", "never run", ""
            else:
                state = "ok" if row["ok"] else "fail"
                when = _ago(row["started_at"])
                detail = row["summary"] or row["error"] or ""
            link = (f'<a class="cta" href="/view/{agent}/{latest.name}">read latest</a>'
                    if latest else '<span class="cta muted">no output yet</span>')
            degr = "".join(f'<li>{html.escape(d)}</li>'
                           for d in (row["degradations"] if row else []))
            cards.append(f"""
            <article class="card">
              <header>
                <span class="pill {state}">{state}</span>
                <h2>{html.escape(title)}</h2>
              </header>
              <p class="blurb">{html.escape(blurb)}</p>
              <p class="detail">{html.escape(detail)}</p>
              {f'<ul class="degr">{degr}</ul>' if degr else ''}
              <footer><span class="when">{html.escape(when)}</span>
                <a href="/agents/{agent}">{len(files)} brief{"" if len(files) == 1 else "s"}</a>
                {link}</footer>
            </article>""")

        run_rows = "".join(
            f"<tr><td>{_ago(r['started_at'])}</td><td>{html.escape(r['agent'])}</td>"
            f"<td>{html.escape((r['target'] or '')[:40])}</td>"
            f"<td class='{'ok' if r['ok'] else 'fail'}'>{'ok' if r['ok'] else 'failed'}</td>"
            f"<td>{r['duration_s']:.1f}s</td>"
            f"<td>{html.escape((r['summary'] or r['error'] or '')[:90])}</td></tr>"
            for r in runs) or "<tr><td colspan='6' class='muted'>no runs recorded yet</td></tr>"

        commits = NotesRepo(cfg.notes_repo, remote=cfg.git_remote).log(limit=8)
        commit_rows = "".join(
            f"<tr><td><code>{html.escape(c['sha'])}</code></td>"
            f"<td>{html.escape(c['date'][:16])}</td>"
            f"<td>{html.escape(c['subject'])}</td></tr>" for c in commits
        ) or "<tr><td colspan='3' class='muted'>no commits yet</td></tr>"

        return _page("agents_work", f"""
        <section class="health">{_health_pills(cfg)}</section>
        <section class="grid">{"".join(cards)}</section>
        <section>
          <h2>Recent runs</h2>
          <div class="scroll"><table>
            <thead><tr><th>When</th><th>Agent</th><th>Target</th><th>Result</th>
            <th>Took</th><th>Summary</th></tr></thead>
            <tbody>{run_rows}</tbody></table></div>
        </section>
        <section>
          <h2>Notes repo</h2>
          <p class="blurb">Every brief is committed to
             <code>{html.escape(str(cfg.notes_repo))}</code>
             {'and pushed to <code>' + html.escape(cfg.git_remote) + '</code>' if cfg.git_remote else '(local only)'}.</p>
          <div class="scroll"><table>
            <thead><tr><th>Commit</th><th>Date</th><th>Subject</th></tr></thead>
            <tbody>{commit_rows}</tbody></table></div>
        </section>""")

    @app.get("/agents/{agent}", response_class=HTMLResponse)
    def agent_index(agent: str) -> str:
        if agent not in AGENTS:
            raise HTTPException(404, "no such agent")
        title, blurb = AGENTS[agent]
        items = "".join(
            f'<li><a href="/view/{agent}/{p.name}">{html.escape(p.name)}</a>'
            f'<span class="when">{_ago(p.stat().st_mtime)}</span></li>'
            for p in artifacts(agent)) or '<li class="muted">nothing written yet</li>'
        return _page(title, f"""
        <p class="crumb"><a href="/">&larr; all agents</a></p>
        <h1>{html.escape(title)}</h1>
        <p class="blurb">{html.escape(blurb)}</p>
        <ul class="filelist">{items}</ul>""")

    @app.get("/view/{agent}/{name}", response_class=HTMLResponse)
    def view(agent: str, name: str) -> str:
        path = safe_artifact(agent, name)
        text = path.read_text(errors="replace")
        body = text.split("---", 2)[2] if text.startswith("---") else text
        return _page(name, f"""
        <p class="crumb"><a href="/">&larr; all agents</a> ·
           <a href="/agents/{agent}">{html.escape(AGENTS[agent][0])}</a> ·
           <a href="/raw/{agent}/{name}">raw</a></p>
        <article class="brief">{_md.render(body)}</article>""")

    @app.get("/raw/{agent}/{name}", response_class=PlainTextResponse)
    def raw(agent: str, name: str) -> str:
        return safe_artifact(agent, name).read_text(errors="replace")

    # -- api -------------------------------------------------------------
    @app.get("/api/health")
    def health() -> JSONResponse:
        conn = db()
        try:
            runs = recent(conn, limit=200)
        finally:
            conn.close()
        payload = {
            "ok": True,
            "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "capabilities": {
                "llm": cfg.has_llm, "price_lake": cfg.has_lake,
                "sandbox_image": docker_available(),
                "notes_remote": bool(cfg.git_remote),
                "vault_roots": [str(p) for p in cfg.vault_roots if p.is_dir()],
            },
            "degradations": cfg.degradations(),
            "agents": {
                a: {"artifacts": len(artifacts(a)),
                    "last_run": next((r["started_at"] for r in runs if r["agent"] == a), None),
                    "last_ok": next((r["ok"] for r in runs if r["agent"] == a), None)}
                for a in AGENTS},
            "runs_recorded": len(runs),
        }
        return JSONResponse(payload)

    @app.get("/api/runs")
    def api_runs(agent: str | None = None, limit: int = 50) -> JSONResponse:
        conn = db()
        try:
            return JSONResponse(recent(conn, agent=agent, limit=min(limit, 200)))
        finally:
            conn.close()

    return app


# -- rendering helpers -------------------------------------------------------

def _ago(ts: float | None) -> str:
    if not ts:
        return "never"
    delta = datetime.now(timezone.utc).timestamp() - float(ts)
    if delta < 90:
        return "just now"
    for unit, size in (("m", 60), ("h", 3600), ("d", 86400)):
        if delta < size * 60 or unit == "d":
            return f"{int(delta / size)}{unit} ago"
    return "a while ago"


def _health_pills(cfg) -> str:
    checks = [
        ("LLM", cfg.has_llm, cfg.write_model),
        ("price lake", cfg.has_lake, str(cfg.lake_dir)),
        ("sandbox", docker_available(), "container isolation"),
        ("notes repo", cfg.notes_repo.is_dir(), cfg.git_remote or "local only"),
        ("vault", any(p.is_dir() for p in cfg.vault_roots),
         f"{len(cfg.vault_roots)} root(s)"),
    ]
    return "".join(
        f'<span class="pill {"ok" if good else "fail"}" title="{html.escape(str(note))}">'
        f'{html.escape(label)}</span>' for label, good, note in checks)


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · agents_work</title>
<link rel="stylesheet" href="/static/style.css">
</head><body>
<nav><a class="brand" href="/">agents_work</a>
<span class="sub">five agents, one run log</span></nav>
<main>{body}</main>
<footer class="page">Read-only view. Runs are owned by systemd timers and the
<code>agents</code> CLI.</footer>
</body></html>"""


app = create_app()
