"""A polite, cached HTTP client shared by every source adapter.

Three things every one of these agents needs and would otherwise re-invent
badly: an on-disk response cache (so a re-run during development doesn't hammer
someone's API), per-host rate limiting (SEC asks for <=10 req/s and means it),
and a fetch that returns None instead of raising when a source is down.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

# Minimum seconds between requests, per host. SEC publishes its limit; the
# others are courtesy values chosen to stay well under anything unpublished.
_RATE_LIMITS = {
    "www.sec.gov": 0.11,
    "data.sec.gov": 0.11,
    "query1.finance.yahoo.com": 0.30,
    "query2.finance.yahoo.com": 0.30,
    "boards-api.greenhouse.io": 0.20,
    "api.lever.co": 0.20,
    "api.ashbyhq.com": 0.20,
    "api.smartrecruiters.com": 0.20,
}
# Workday and Oracle each serve one tenant per host, so a per-host limit is
# already a per-employer limit; these are courtesy values for hosts that
# publish none.
_HOST_SUFFIX_LIMITS = {
    ".myworkdayjobs.com": 0.30,
    ".oraclecloud.com": 0.30,
    ".eightfold.ai": 0.30,
}
_DEFAULT_RATE = 0.10

_lock = threading.Lock()
_last_call: dict[str, float] = {}


def _rate_for(host: str) -> float:
    if host in _RATE_LIMITS:
        return _RATE_LIMITS[host]
    for suffix, delay in _HOST_SUFFIX_LIMITS.items():
        if host.endswith(suffix):
            return delay
    return _DEFAULT_RATE


def _throttle(host: str) -> None:
    delay = _rate_for(host)
    with _lock:
        prev = _last_call.get(host, 0.0)
        wait = delay - (time.monotonic() - prev)
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.monotonic()


@dataclass
class Response:
    url: str
    status: int
    text: str
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self, default=None):
        try:
            return json.loads(self.text)
        except json.JSONDecodeError:
            log.warning("non-JSON body from %s (%d bytes)", self.url, len(self.text))
            return default


class Fetcher:
    """Cache-first GET. `fetch` never raises on a network or HTTP problem."""

    def __init__(self, cache_dir: Path, *, ttl: int = 900, user_agent: str = "agents_work/0.1",
                 timeout: float = 20.0, offline: bool = False) -> None:
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.user_agent = user_agent
        self.timeout = timeout
        self.offline = offline
        self.stats = {"hit": 0, "miss": 0, "error": 0}

    def _key(self, url: str, headers: dict | None, body: dict | None = None) -> Path:
        # The body is part of the identity: two Workday pages differ only by the
        # `offset` in their POST payload, and a key that ignored it would serve
        # page 1 for every page of a paginated board.
        material = url + json.dumps(headers or {}, sort_keys=True)
        if body is not None:
            material += json.dumps(body, sort_keys=True)
        h = hashlib.sha256(material.encode()).hexdigest()[:24]
        return self.cache_dir / f"{h}.json"

    def _read_cache(self, path: Path, ttl: int) -> Response | None:
        if not path.is_file():
            return None
        try:
            blob = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if ttl >= 0 and time.time() - blob.get("at", 0) > ttl:
            return None
        return Response(blob["url"], blob["status"], blob["text"], from_cache=True)

    def _write_cache(self, path: Path, resp: Response) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"url": resp.url, "status": resp.status,
                                   "text": resp.text, "at": time.time()}))
        tmp.replace(path)

    def fetch(self, url: str, *, headers: dict | None = None, ttl: int | None = None,
              params: dict | None = None, body: dict | None = None) -> Response | None:
        """GET `url`, or POST `body` to it when `body` is given.

        Workday and Oracle publish their job boards behind POST-only JSON
        endpoints, so "cache-first GET" had to grow a body rather than every
        caller growing its own client — the throttle, the on-disk cache and the
        stale-on-failure fallback are the parts that matter, and they are the
        same either way.
        """
        if params:
            url = str(httpx.URL(url).copy_merge_params(params))
        ttl = self.ttl if ttl is None else ttl
        path = self._key(url, headers, body)

        cached = self._read_cache(path, ttl)
        if cached is not None:
            self.stats["hit"] += 1
            return cached

        if self.offline:
            # Offline mode still serves stale cache: a stale brief beats none.
            stale = self._read_cache(path, -1)
            if stale is not None:
                self.stats["hit"] += 1
                return stale
            self.stats["error"] += 1
            return None

        hdrs = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        hdrs.update(headers or {})
        _throttle(httpx.URL(url).host)
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                r = (client.post(url, headers=hdrs, json=body) if body is not None
                     else client.get(url, headers=hdrs))
            resp = Response(url, r.status_code, r.text)
        except httpx.HTTPError as e:
            log.warning("fetch failed %s: %s", url, e)
            self.stats["error"] += 1
            stale = self._read_cache(path, -1)
            if stale is not None:
                log.info("serving stale cache for %s", url)
                self.stats["hit"] += 1
                return stale
            return None

        self.stats["miss"] += 1
        if resp.ok:
            self._write_cache(path, resp)
        else:
            log.warning("HTTP %d from %s", resp.status, url)
        return resp


# The cache is the only piece of state here with no ceiling. Keys are a hash of
# the URL, so a nightly re-run of the same board overwrites its own entry rather
# than adding one -- what actually accumulates is *variety*: every ticker ever
# researched leaves a permanent `companyfacts` blob behind, and those run to 12MB
# apiece (59 of them were 45MB of a 85MB cache the first day). Nothing ever read
# them again. So the cap is on total bytes, evicting least-recently-written first.
#
# mtime is the right recency signal even though a cache *hit* does not touch it:
# an entry still in use gets rewritten every time its TTL lapses, so its mtime
# stays fresh, while one that is never requested again ages out. That is exactly
# the ticker-researched-once case this exists to collect.
CACHE_TMP_MAX_AGE = 3600


def prune_cache(cache_dir: Path, max_bytes: int, *, dry_run: bool = False) -> dict:
    """Evict least-recently-written entries until the cache fits `max_bytes`.

    Returns a summary rather than logging one, so the caller decides whether a
    routine eviction is worth a line of output. Never raises: a cache that
    cannot be pruned is a disk-space problem, not a reason to fail a run that
    has already produced its brief.
    """
    stats = {"kept": 0, "kept_bytes": 0, "removed": 0, "freed_bytes": 0, "tmp_removed": 0}
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return stats

    entries: list[tuple[float, str, Path, int]] = []
    now = time.time()
    for entry in cache_dir.iterdir():
        try:
            if not entry.is_file():
                continue
            st = entry.stat()
        except OSError:
            continue  # vanished under us; nothing to account for
        if entry.suffix == ".tmp":
            # A stray .tmp is a write that died between write_text and replace.
            # Age-gate it so a prune racing a live fetch cannot eat its scratch file.
            if now - st.st_mtime > CACHE_TMP_MAX_AGE and not dry_run:
                try:
                    entry.unlink()
                    stats["tmp_removed"] += 1
                except OSError:
                    pass
            continue
        # Sort newest first, breaking mtime ties by name so the eviction set is
        # deterministic -- a test that writes its fixtures in one second would
        # otherwise pick a different victim each run.
        entries.append((st.st_mtime, entry.name, entry, st.st_size))

    entries.sort(key=lambda e: (-e[0], e[1]))

    total = 0
    for mtime, _name, path, size in entries:
        if total + size <= max_bytes:
            total += size
            stats["kept"] += 1
            stats["kept_bytes"] = total
            continue
        if dry_run:
            stats["removed"] += 1
            stats["freed_bytes"] += size
            continue
        try:
            path.unlink()
        except OSError:
            # Could not remove it, so it is still occupying the budget: count it
            # as kept or the next run will evict a healthy entry to make room.
            total += size
            stats["kept"] += 1
            stats["kept_bytes"] = total
            continue
        stats["removed"] += 1
        stats["freed_bytes"] += size
    return stats
