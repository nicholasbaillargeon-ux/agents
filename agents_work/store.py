"""Run log. Every agent invocation lands here so the dashboard and the timers
have one place to answer 'did it run, did it work, what did it produce'."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY,
    agent        TEXT    NOT NULL,
    target       TEXT    NOT NULL DEFAULT '',
    started_at   REAL    NOT NULL,
    duration_s   REAL    NOT NULL DEFAULT 0,
    ok           INTEGER NOT NULL DEFAULT 0,
    artifact     TEXT,
    summary      TEXT,
    degradations TEXT    NOT NULL DEFAULT '[]',
    error        TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_agent ON runs(agent, started_at DESC);

CREATE TABLE IF NOT EXISTS seen (
    agent      TEXT NOT NULL,
    key        TEXT NOT NULL,
    first_seen REAL NOT NULL,
    payload    TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (agent, key)
);
"""


@dataclass
class Run:
    agent: str
    target: str = ""
    started_at: float = field(default_factory=time.time)
    duration_s: float = 0.0
    ok: bool = False
    artifact: str | None = None
    summary: str | None = None
    degradations: list[str] = field(default_factory=list)
    error: str | None = None
    id: int | None = None


def connect(db_path: Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def record(conn: sqlite3.Connection, run: Run) -> int:
    cur = conn.execute(
        "INSERT INTO runs(agent,target,started_at,duration_s,ok,artifact,summary,degradations,error)"
        " VALUES(?,?,?,?,?,?,?,?,?)",
        (run.agent, run.target, run.started_at, run.duration_s, int(run.ok),
         run.artifact, run.summary, json.dumps(run.degradations), run.error),
    )
    conn.commit()
    run.id = cur.lastrowid
    return cur.lastrowid


def recent(conn: sqlite3.Connection, agent: str | None = None, limit: int = 25) -> list[dict]:
    sql = "SELECT * FROM runs"
    args: tuple = ()
    if agent:
        sql += " WHERE agent = ?"
        args = (agent,)
    sql += " ORDER BY started_at DESC LIMIT ?"
    rows = conn.execute(sql, (*args, limit)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["degradations"] = json.loads(d["degradations"] or "[]")
        d["ok"] = bool(d["ok"])
        out.append(d)
    return out


def mark_new(conn: sqlite3.Connection, agent: str, items: dict[str, dict]) -> list[str]:
    """Insert keys not seen before; return only the newly-inserted keys.

    This is the whole idea behind 'nightly diff, only new matches surfaced' —
    it lives in one transaction so a crash mid-run can't half-remember a batch
    and silently drop those matches from tomorrow's diff.
    """
    if not items:
        return []
    now = time.time()
    fresh: list[str] = []
    with conn:  # BEGIN/COMMIT — all or nothing
        for key, payload in items.items():
            cur = conn.execute(
                "INSERT OR IGNORE INTO seen(agent,key,first_seen,payload) VALUES(?,?,?,?)",
                (agent, key, now, json.dumps(payload, sort_keys=True)),
            )
            if cur.rowcount:
                fresh.append(key)
    return fresh


def seen_count(conn: sqlite3.Connection, agent: str) -> int:
    return conn.execute("SELECT COUNT(*) FROM seen WHERE agent = ?", (agent,)).fetchone()[0]


@contextmanager
def track(conn: sqlite3.Connection, agent: str, target: str = ""):
    """Context manager that logs the run whatever happens to it."""
    run = Run(agent=agent, target=target)
    t0 = time.perf_counter()
    try:
        yield run
        run.ok = run.error is None
    except Exception as e:  # noqa: BLE001 - the log is the point
        run.ok = False
        run.error = f"{type(e).__name__}: {e}"
        raise
    finally:
        run.duration_s = time.perf_counter() - t0
        record(conn, run)
