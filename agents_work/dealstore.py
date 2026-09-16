"""Postgres for the deal book.

The one store in this repo that is not SQLite, and deliberately so. A deal
one-pager is a document you come back to and edit by hand over months — JSONB
for the pieces whose shape varies by deal, an array for the open questions, a
real timestamp on your own annotation — and it is the one table another tool
might reasonably want to read while an agent is writing to it.

Absence is a normal state, as everywhere else here: with no DSN configured the
agent still fetches, still filters, still writes the brief, and reports that
nothing was persisted.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS deals (
    id               SERIAL PRIMARY KEY,
    deal_key         TEXT        NOT NULL UNIQUE,
    first_seen       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    headline         TEXT        NOT NULL,
    url              TEXT        NOT NULL DEFAULT '',
    source           TEXT        NOT NULL DEFAULT '',
    acquirer         TEXT        NOT NULL DEFAULT '',
    target           TEXT        NOT NULL DEFAULT '',
    sector           TEXT        NOT NULL DEFAULT '',
    value_usd        NUMERIC,
    currency         TEXT        NOT NULL DEFAULT '',
    consideration    TEXT        NOT NULL DEFAULT '',
    implied_multiple TEXT        NOT NULL DEFAULT '',
    rationale        TEXT        NOT NULL DEFAULT '',
    financing        TEXT        NOT NULL DEFAULT '',
    open_questions   JSONB       NOT NULL DEFAULT '[]'::jsonb,
    advisors         JSONB       NOT NULL DEFAULT '[]'::jsonb,
    flagged_advisor  TEXT        NOT NULL DEFAULT '',
    -- 'new' until you have read it, then whatever you set. The point of the
    -- book is the annotation, so status and my_view are the only columns an
    -- agent never overwrites once a human has touched them.
    status           TEXT        NOT NULL DEFAULT 'new',
    my_view          TEXT        NOT NULL DEFAULT '',
    reviewed_at      TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_deals_status ON deals(status, first_seen DESC);
CREATE INDEX IF NOT EXISTS idx_deals_flagged ON deals(flagged_advisor)
    WHERE flagged_advisor <> '';
"""

# Columns an agent re-run may refresh. `status`, `my_view` and `reviewed_at`
# are absent on purpose: a second sweep must never overwrite your own view of a
# deal with a fresh reading of the same press release.
REFRESHABLE = ("headline", "url", "source", "acquirer", "target", "sector",
               "value_usd", "currency", "consideration", "implied_multiple",
               "rationale", "financing", "open_questions", "advisors",
               "flagged_advisor")


@dataclass
class DealRecord:
    deal_key: str
    headline: str
    url: str = ""
    source: str = ""
    acquirer: str = ""
    target: str = ""
    sector: str = ""
    value_usd: float | None = None
    currency: str = ""
    consideration: str = ""
    implied_multiple: str = ""
    rationale: str = ""
    financing: str = ""
    open_questions: list[str] = field(default_factory=list)
    advisors: list[dict] = field(default_factory=list)
    flagged_advisor: str = ""

    def params(self) -> dict:
        d = {k: getattr(self, k) for k in ("deal_key", *REFRESHABLE)}
        d["open_questions"] = json.dumps(self.open_questions)
        d["advisors"] = json.dumps(self.advisors)
        return d


class DealBookUnavailable(Exception):
    """No DSN, or Postgres is not answering. Callers degrade; they do not crash."""


@contextmanager
def connect(dsn: str | None):
    """A connection, or DealBookUnavailable. Always closed."""
    if not dsn:
        raise DealBookUnavailable("no AGENTS_DEALBOOK_DSN configured")
    try:
        import psycopg  # noqa: PLC0415 - optional at import time by design
    except ImportError as e:  # pragma: no cover
        raise DealBookUnavailable("psycopg not installed") from e
    try:
        conn = psycopg.connect(dsn, connect_timeout=10)
    except Exception as e:  # noqa: BLE001 - psycopg raises a wide variety
        raise DealBookUnavailable(f"cannot reach Postgres: {type(e).__name__}") from e
    try:
        yield conn
    finally:
        conn.close()


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()


def upsert(conn, record: DealRecord) -> tuple[int, bool]:
    """(id, was_new). Refreshes the agent-owned columns, never your annotation."""
    assignments = ", ".join(f"{c} = EXCLUDED.{c}" for c in REFRESHABLE)
    columns = ", ".join(("deal_key", *REFRESHABLE))
    placeholders = ", ".join(f"%({c})s" for c in ("deal_key", *REFRESHABLE))
    sql = (f"INSERT INTO deals ({columns}) VALUES ({placeholders}) "
           f"ON CONFLICT (deal_key) DO UPDATE SET {assignments}, updated_at = now() "
           f"RETURNING id, (xmax = 0) AS inserted")
    with conn.cursor() as cur:
        cur.execute(sql, record.params())
        row = cur.fetchone()
    conn.commit()
    return int(row[0]), bool(row[1])


def _rows(cur) -> list[dict]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def list_deals(conn, *, status: str | None = None, flagged_only: bool = False,
               limit: int = 50) -> list[dict]:
    sql = "SELECT * FROM deals"
    clauses, args = [], []
    if status:
        clauses.append("status = %s")
        args.append(status)
    if flagged_only:
        clauses.append("flagged_advisor <> ''")
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY first_seen DESC LIMIT %s"
    args.append(limit)
    with conn.cursor() as cur:
        cur.execute(sql, args)
        return _rows(cur)


def get_deal(conn, deal_id: int) -> dict | None:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM deals WHERE id = %s", (deal_id,))
        rows = _rows(cur)
    return rows[0] if rows else None


def annotate(conn, deal_id: int, view: str, *, status: str = "reviewed") -> bool:
    """Record your own read of a deal. This is the half the agent does not do."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE deals SET my_view = %s, status = %s, reviewed_at = now(), "
            "updated_at = now() WHERE id = %s",
            (view, status, deal_id))
        changed = cur.rowcount
    conn.commit()
    return bool(changed)


def set_status(conn, deal_id: int, status: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("UPDATE deals SET status = %s, updated_at = now() WHERE id = %s",
                    (status, deal_id))
        changed = cur.rowcount
    conn.commit()
    return bool(changed)


def counts(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT status, COUNT(*) FROM deals GROUP BY status")
        by_status = {r[0]: int(r[1]) for r in cur.fetchall()}
        cur.execute("SELECT COUNT(*) FROM deals WHERE flagged_advisor <> ''")
        flagged = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM deals WHERE my_view <> ''")
        annotated = int(cur.fetchone()[0])
    return {"by_status": by_status, "flagged": flagged, "annotated": annotated,
            "total": sum(by_status.values())}


def health(dsn: str | None) -> str:
    """One line for `agents doctor`."""
    try:
        with connect(dsn) as conn:
            ensure_schema(conn)
            c = counts(conn)
        return (f"reachable — {c['total']} deals, {c['flagged']} advisor-flagged, "
                f"{c['annotated']} annotated")
    except DealBookUnavailable as e:
        return str(e)
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
