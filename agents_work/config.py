"""One config surface. Every agent and the web app read this and nothing else.

Everything has a default that works offline, so `import agents_work` never
depends on a secret being present. Absent capability is reported, not raised —
see `Config.degradations()`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(_ROOT / ".env")


def _paths(raw: str) -> list[Path]:
    return [Path(p).expanduser() for p in raw.split(":") if p.strip()]


def _csv(raw: str) -> list[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


@dataclass(frozen=True)
class Config:
    root: Path
    data_dir: Path
    llm_base_url: str
    llm_api_key: str | None
    write_model: str
    fast_model: str
    git_remote: str | None
    sec_user_agent: str
    lake_dir: Path
    watchlist: list[str]
    vault_roots: list[Path]
    port: int
    http_cache_ttl: int = 900
    _extra: dict = field(default_factory=dict, repr=False)

    # --- derived paths -------------------------------------------------
    @property
    def notes_repo(self) -> Path:
        return self.data_dir / "research-notes"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def out_dir(self) -> Path:
        return self.data_dir / "out"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "runs.db"

    # --- capability flags ----------------------------------------------
    @property
    def has_llm(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def has_lake(self) -> bool:
        return self.lake_dir.is_dir()

    def degradations(self) -> list[str]:
        """Human-readable list of what this process cannot do right now.

        Agents copy these into their output so a thin brief is never mistaken
        for a confident one.
        """
        out: list[str] = []
        if not self.has_llm:
            out.append("no LLM key: prose sections are omitted, data sections still render")
        if not self.has_lake:
            out.append(f"no parquet lake at {self.lake_dir}: backtests fall back to live download")
        if not any(p.is_dir() for p in self.vault_roots):
            out.append("no readable vault roots: the RAG analyst has nothing to index")
        return out

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.cache_dir, self.out_dir):
            p.mkdir(parents=True, exist_ok=True)


def load_config(**overrides) -> Config:
    data_dir = Path(os.getenv("AGENTS_DATA_DIR", _ROOT / "data")).expanduser()
    cfg = Config(
        root=_ROOT,
        data_dir=data_dir,
        llm_base_url=os.getenv("AGENTS_LLM_BASE_URL", "https://api.anthropic.com"),
        llm_api_key=os.getenv("AGENTS_LLM_API_KEY") or None,
        write_model=os.getenv("AGENTS_WRITE_MODEL", "claude-sonnet-5"),
        fast_model=os.getenv("AGENTS_FAST_MODEL", "claude-haiku-4-5"),
        git_remote=os.getenv("AGENTS_GIT_REMOTE") or None,
        sec_user_agent=os.getenv("SEC_USER_AGENT", "agents_work/0.1 (contact unset)"),
        lake_dir=Path(os.getenv("AGENTS_LAKE_DIR", "/home/nbaillar/market-lab/data/lake")).expanduser(),
        watchlist=_csv(os.getenv("AGENTS_WATCHLIST", "SPY,QQQ,AAPL,MSFT,NVDA")),
        vault_roots=_paths(os.getenv("AGENTS_VAULT_ROOTS", str(data_dir / "research-notes"))),
        port=int(os.getenv("AGENTS_PORT", "8110")),
        http_cache_ttl=int(os.getenv("AGENTS_HTTP_CACHE_TTL", "900")),
    )
    if overrides:
        from dataclasses import replace

        cfg = replace(cfg, **overrides)
    return cfg
