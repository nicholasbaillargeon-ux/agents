"""Commit agent output to a real git repository.

Local by default. Point AGENTS_GIT_REMOTE at a Gitea (or GitHub) remote and the
same commits get pushed — the agent code does not change, which is the reason
the sink is a seam and not an inline `subprocess.run(['git', ...])` in the
research agent.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

AUTHOR_NAME = "agents_work"
AUTHOR_EMAIL = "agents@localhost"


class GitError(RuntimeError):
    pass


@dataclass
class CommitResult:
    committed: bool
    sha: str | None
    message: str
    pushed: bool = False
    push_error: str | None = None


def _git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, timeout=120,
        env={"GIT_TERMINAL_PROMPT": "0", "HOME": str(Path.home()), "PATH": "/usr/bin:/bin"},
    )
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout.strip()


class NotesRepo:
    """A git repo of research notes. Idempotent to create, safe to re-init."""

    def __init__(self, path: Path, remote: str | None = None) -> None:
        self.path = Path(path)
        self.remote = remote

    def ensure(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        if not (self.path / ".git").is_dir():
            _git(self.path, "init", "-q", "-b", "main")
            (self.path / "README.md").write_text(
                "# Research notes\n\n"
                "Machine-written briefs, one file per run, committed by `agents_work`.\n"
                "Nothing here is human-reviewed unless a commit says so.\n"
            )
            _git(self.path, "add", "README.md")
            self._commit("Initialise research notes repo")
        _git(self.path, "config", "user.name", AUTHOR_NAME)
        _git(self.path, "config", "user.email", AUTHOR_EMAIL)
        if self.remote:
            existing = _git(self.path, "remote", check=False)
            if "origin" in existing.split():
                _git(self.path, "remote", "set-url", "origin", self.remote)
            else:
                _git(self.path, "remote", "add", "origin", self.remote)

    def _commit(self, message: str) -> str:
        _git(self.path, "-c", f"user.name={AUTHOR_NAME}", "-c", f"user.email={AUTHOR_EMAIL}",
             "commit", "-q", "-m", message)
        return _git(self.path, "rev-parse", "HEAD")

    def commit_file(self, relpath: str | Path, content: str, message: str,
                    *, push: bool | None = None) -> CommitResult:
        """Write, stage, commit. Identical content is a no-op, not an empty commit."""
        self.ensure()
        target = self.path / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_text() == content:
            _git(self.path, "add", "--", str(target.relative_to(self.path)))
            if not _git(self.path, "status", "--porcelain"):
                return CommitResult(False, _git(self.path, "rev-parse", "HEAD"),
                                    "unchanged — nothing committed")
        target.write_text(content)
        _git(self.path, "add", "--", str(target.relative_to(self.path)))
        if not _git(self.path, "status", "--porcelain"):
            return CommitResult(False, _git(self.path, "rev-parse", "HEAD"),
                                "unchanged — nothing committed")
        sha = self._commit(message)
        result = CommitResult(True, sha, message)

        should_push = self.remote is not None if push is None else push
        if should_push and self.remote:
            try:
                _git(self.path, "push", "-q", "-u", "origin", "main")
                result.pushed = True
            except GitError as e:
                # A dead remote must not lose the commit — it is already local.
                result.push_error = str(e)
                log.warning("commit %s stayed local: %s", sha[:8], e)
        return result

    def log(self, limit: int = 20) -> list[dict]:
        if not (self.path / ".git").is_dir():
            return []
        raw = _git(self.path, "log", f"-{limit}", "--pretty=format:%H%x1f%ad%x1f%s", "--date=iso",
                   check=False)
        out = []
        for line in raw.splitlines():
            parts = line.split("\x1f")
            if len(parts) == 3:
                out.append({"sha": parts[0][:10], "date": parts[1], "subject": parts[2]})
        return out
