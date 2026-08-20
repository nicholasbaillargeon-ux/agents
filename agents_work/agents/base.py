"""Shared agent plumbing: one Context object carries every capability an agent
is allowed to use, so an agent can be constructed in a test with fakes in place
of the network, the LLM, and the git repo."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..brief import Brief
from ..config import Config, load_config
from ..gitsink import NotesRepo
from ..llm import LLM
from ..netcache import Fetcher
from ..store import connect

log = logging.getLogger(__name__)


@dataclass
class AgentResult:
    agent: str
    target: str = ""
    ok: bool = True
    summary: str = ""
    artifact: Path | None = None
    brief: Brief | None = None
    data: dict = field(default_factory=dict)
    degradations: list[str] = field(default_factory=list)
    error: str | None = None

    def degrade(self, msg: str) -> "AgentResult":
        if msg and msg not in self.degradations:
            self.degradations.append(msg)
        return self


class Context:
    """Everything an agent may touch. Nothing reaches outside this object."""

    def __init__(self, cfg: Config | None = None, *, llm: LLM | None = None,
                 fetcher: Fetcher | None = None, notes: NotesRepo | None = None,
                 offline: bool = False) -> None:
        self.cfg = cfg or load_config()
        self.cfg.ensure_dirs()
        self.offline = offline
        self.llm = llm if llm is not None else LLM(self.cfg)
        self.fetcher = fetcher or Fetcher(
            self.cfg.cache_dir, ttl=self.cfg.http_cache_ttl,
            user_agent=self.cfg.sec_user_agent, offline=offline,
        )
        self.notes = notes if notes is not None else NotesRepo(
            self.cfg.notes_repo, remote=self.cfg.git_remote)
        self._conn = None

    @property
    def db(self):
        if self._conn is None:
            self._conn = connect(self.cfg.db_path)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def base_degradations(self) -> list[str]:
        out = list(self.cfg.degradations())
        if self.offline:
            out.append("offline mode: serving cached responses only")
        if not self.llm.available:
            msg = "no LLM: analysis sections omitted"
            if msg not in out and not any("no LLM" in o for o in out):
                out.append(msg)
        return out
