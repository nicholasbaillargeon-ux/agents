"""Live check that every board slug in the registry still resolves.

Skipped by default — it hits twenty third-party APIs and would make the suite
depend on their uptime. Run it deliberately, on a schedule or after a scout
brief starts reporting dead boards:

    AGENTS_LIVE=1 .venv/bin/pytest -m live -v

Slugs rot. Optiver moved from `optiver` to `optiverus` and Plaid left Lever for
Ashby within weeks of this registry being written, and both were only caught
because `Boards.fetch_all` reports empty boards instead of silently returning a
shorter list. This turns that report into a failing test.
"""

from __future__ import annotations

import os

import pytest

from agents_work.config import load_config
from agents_work.netcache import Fetcher
from agents_work.sources.jobs import REGISTRY, Boards

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("AGENTS_LIVE") != "1",
                       reason="live board check; set AGENTS_LIVE=1 to run"),
]


@pytest.fixture(scope="module")
def live_boards():
    cfg = load_config()
    cfg.ensure_dirs()
    # A short TTL, not zero: re-running the file should not re-hammer the APIs.
    fetcher = Fetcher(cfg.cache_dir, ttl=900, user_agent=cfg.sec_user_agent)
    boards = Boards(fetcher)
    postings = boards.fetch_all(REGISTRY, ttl=900)
    return boards, postings


@pytest.mark.parametrize("vendor,slug,company,sector", REGISTRY,
                         ids=[f"{c}({v}/{s})" for v, s, c, _ in REGISTRY])
def test_board_slug_still_resolves(live_boards, vendor, slug, company, sector):
    boards, _ = live_boards
    status = boards.source_status.get(company, "never attempted")
    assert status.endswith("postings"), (
        f"{company} ({vendor}/{slug}) returned {status!r} — the firm has probably "
        f"moved ATS vendor. Probe candidates and update REGISTRY in sources/jobs.py.")


def test_the_sweep_returns_a_plausible_volume(live_boards):
    _, postings = live_boards
    assert len(postings) > 500, f"only {len(postings)} postings across {len(REGISTRY)} boards"
    assert all(p.title and p.url.startswith("http") for p in postings)


def test_some_internships_are_visible(live_boards):
    """Zero internships across twenty boards means the title filter has drifted,
    not that nobody is hiring."""
    _, postings = live_boards
    interns = [p for p in postings if p.is_internship]
    assert len(interns) > 10, f"only {len(interns)} internship titles found"
