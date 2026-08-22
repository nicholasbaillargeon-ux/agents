"""Agent 2 — backtest runner.

Plain-English strategy idea in; generated code, a sandboxed run, and Sharpe /
max drawdown / equity curve out.

Two decisions carry this agent:

* The model writes a **signal function only**. Execution semantics — lag,
  next-open fill, costs — live in the sandbox harness, so the classic
  LLM-backtest failure (trading on the same bar it decided on) is structurally
  impossible rather than something to review for.
* The generated code runs in a container with no network, a read-only root, a
  memory cap and a wall-clock timeout. If Docker is missing, it still runs, in
  a locked-down subprocess, and the result is labelled as such.
"""

from __future__ import annotations

import json
import logging
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..brief import Brief, table
from ..llm import LLMUnavailable
from ..sources.prices import LAKE_STALE_DAYS, coverage_gap_days
from ..store import Run, record
from .base import AgentResult, Context

log = logging.getLogger(__name__)

NAME = "backtest"
IMAGE = "agents-backtest-sandbox:latest"
SANDBOX_DIR = Path(__file__).resolve().parent.parent.parent / "sandbox"

SYSTEM = textwrap.dedent("""
    You write one Python function and nothing else.

        def signal(df: pd.DataFrame) -> pd.Series

    `df` is a daily OHLCV frame indexed by date, columns: open, high, low,
    close, adj_close, volume. Return a Series aligned to df.index whose value
    for each bar is the TARGET PORTFOLIO WEIGHT decided at that bar's close,
    in [-1, 1]. 1.0 is fully long, 0.0 flat, -1.0 fully short.

    Hard rules:
    - `pd` and `np` are already available. Import nothing. No file or network
      access. No classes. No `print`.
    - Compute only from data at or before each bar. The harness lags your
      signal by one bar and fills at the next open, so do NOT shift the result
      yourself — that would double-lag it.
    - Use `.rolling(...)`, `.ewm(...)`, `.shift(...)` and friends. Never index
      by integer position into the future.
    - Return a float Series, not booleans. NaNs are treated as flat.
    - Output ONLY the function inside one ```python fenced block. No prose.
""").strip()


@dataclass
class BacktestJob:
    idea: str
    symbols: list[str]
    start: str | None = None
    end: str | None = None
    cost_bps: float = 5.0
    slippage_bps: float = 5.0
    code: str = ""
    attempts: list[str] = field(default_factory=list)

    def payload(self) -> dict:
        return {"symbols": self.symbols, "start": self.start, "end": self.end,
                "cost_bps": self.cost_bps, "slippage_bps": self.slippage_bps,
                "code": self.code}


# --- code generation --------------------------------------------------------

def generate_code(ctx: Context, idea: str, *, error: str | None = None,
                  previous: str | None = None) -> str:
    if error and previous:
        prompt = (f"This strategy code failed.\n\n```python\n{previous}\n```\n\n"
                  f"Error:\n```\n{error[:2000]}\n```\n\n"
                  f"The original idea was: {idea}\n\n"
                  "Return the corrected function, same rules, fenced block only.")
    else:
        prompt = (f"Strategy idea, in the user's words:\n\n{idea}\n\n"
                  "Write the `signal` function that implements it.")
    text = ctx.llm.complete(prompt, system=SYSTEM, max_tokens=2000)
    return extract_code(text)


def extract_code(text: str) -> str:
    """Pull the function out of a fenced block, tolerating the model's prose."""
    import re

    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
    code = blocks[0] if blocks else text
    code = code.strip()
    if "def signal" not in code:
        raise ValueError("generated text contains no `def signal`")
    return code


BANNED = ("import ", "__import__", "open(", "exec(", "eval(", "compile(",
          "subprocess", "socket", "os.", "sys.", "globals(", "getattr(")


def static_check(code: str) -> list[str]:
    """Cheap belt to the sandbox's braces. Not a security boundary — the
    container is. This catches the honest mistakes before paying for a run."""
    problems = []
    for token in BANNED:
        if token in code:
            problems.append(f"code contains banned construct {token!r}")
    if "def signal" not in code:
        problems.append("no `signal` function defined")
    try:
        compile(code, "<strategy>", "exec")
    except SyntaxError as e:
        problems.append(f"syntax error line {e.lineno}: {e.msg}")
    return problems


# --- execution --------------------------------------------------------------

def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        proc = subprocess.run(["docker", "image", "inspect", IMAGE],
                              capture_output=True, timeout=20)
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def build_image(*, quiet: bool = True) -> tuple[bool, str]:
    if not shutil.which("docker"):
        return False, "docker is not installed"
    cmd = ["docker", "build", "-t", IMAGE, str(SANDBOX_DIR)]
    if quiet:
        cmd.insert(2, "-q")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()[-2000:]


def run_in_docker(job: BacktestJob, lake: Path, *, timeout: int = 120) -> dict:
    with tempfile.TemporaryDirectory(prefix="bt-") as tmp:
        tmpd = Path(tmp)
        (tmpd / "job").mkdir()
        (tmpd / "out").mkdir()
        (tmpd / "job" / "job.json").write_text(json.dumps(job.payload()))
        os.chmod(tmpd / "out", 0o777)
        cmd = [
            "docker", "run", "--rm",
            "--network", "none",            # generated code gets no egress
            "--memory", "1g", "--memory-swap", "1g",
            "--cpus", "1", "--pids-limit", "128",
            "--read-only", "--tmpfs", "/tmp:size=64m",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "-v", f"{tmpd / 'job'}:/job:ro",
            "-v", f"{tmpd / 'out'}:/out",
            "-v", f"{lake}:/lake:ro",
            IMAGE,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"sandbox exceeded {timeout}s wall clock",
                    "stage": "timeout", "isolation": "docker"}
        out = tmpd / "out" / "result.json"
        if not out.is_file():
            return {"ok": False, "stage": "sandbox",
                    "error": (proc.stderr or proc.stdout or "no output").strip()[-2000:],
                    "isolation": "docker"}
        result = json.loads(out.read_text())
        result["isolation"] = "docker"
        return result


def run_in_subprocess(job: BacktestJob, lake: Path, *, timeout: int = 120) -> dict:
    """Fallback when Docker is unavailable. Weaker: same kernel, same user,
    filesystem visible. rlimits cap CPU, address space and subprocesses, the
    environment is stripped, and the result is labelled `isolation: subprocess`
    so nobody mistakes it for the real thing.
    """
    with tempfile.TemporaryDirectory(prefix="bt-") as tmp:
        tmpd = Path(tmp)
        (tmpd / "job").mkdir()
        (tmpd / "out").mkdir()
        job_file = tmpd / "job" / "job.json"
        out_file = tmpd / "out" / "result.json"
        job_file.write_text(json.dumps(job.payload()))

        # The runner takes its paths from the environment. An earlier version
        # rewrote its source with str.replace and produced a file that would not
        # parse, so this path silently never ran while Docker was present.
        env = {
            "BT_JOB": str(job_file), "BT_OUT": str(out_file), "BT_LAKE": str(lake),
            "PATH": "/usr/bin:/bin", "HOME": str(tmpd),
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        def limits():  # pragma: no cover - runs in the child
            resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
            resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
            resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))

        try:
            proc = subprocess.run([sys.executable, str(SANDBOX_DIR / "runner.py")],
                                  capture_output=True, text=True, timeout=timeout,
                                  env=env, cwd=tmpd, preexec_fn=limits)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"exceeded {timeout}s wall clock",
                    "stage": "timeout", "isolation": "subprocess"}
        if not out_file.is_file():
            return {"ok": False, "stage": "sandbox", "isolation": "subprocess",
                    "error": (proc.stderr or proc.stdout or "no output").strip()[-2000:]}
        result = json.loads(out_file.read_text())
        result["isolation"] = "subprocess"
        return result


def execute(job: BacktestJob, lake: Path, *, prefer_docker: bool = True,
            timeout: int = 120) -> dict:
    if prefer_docker and docker_available():
        return run_in_docker(job, lake, timeout=timeout)
    return run_in_subprocess(job, lake, timeout=timeout)


# --- rendering --------------------------------------------------------------

def sparkline(curve: list[dict], *, width: int = 640, height: int = 120) -> str:
    """Inline SVG equity curve. No chart library in the brief's dependency
    surface, and it renders in any markdown viewer that allows raw HTML."""
    if len(curve) < 2:
        return ""
    vals = [p["equity"] for p in curve]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    step = width / (len(vals) - 1)
    pts = " ".join(f"{i * step:.1f},{height - (v - lo) / span * (height - 8) - 4:.1f}"
                   for i, v in enumerate(vals))
    base = height - (1.0 - lo) / span * (height - 8) - 4 if lo <= 1.0 <= hi else None
    baseline = (f'<line x1="0" y1="{base:.1f}" x2="{width}" y2="{base:.1f}" '
                f'stroke="#6f6f6a" stroke-dasharray="4 4" stroke-width="1"/>'
                if base is not None else "")
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
            f'role="img" aria-label="equity curve, {curve[0]["date"]} to {curve[-1]["date"]}, '
            f'ending at {vals[-1]:.2f}x">'
            f'{baseline}<polyline fill="none" stroke="#2a78d6" stroke-width="2" '
            f'points="{pts}"/></svg>')


def _pct(v, digits=1):
    return "n/a" if v is None else f"{v * 100:+.{digits}f}%"


def _num(v, digits=2):
    return "n/a" if v is None else f"{v:.{digits}f}"


def build_brief(job: BacktestJob, result: dict, *, degradations: list[str]) -> Brief:
    brief = Brief(title=f"Backtest — {job.idea[:70]}", agent=NAME,
                  target=", ".join(job.symbols), tags=["backtest", "quant"])
    for d in degradations:
        brief.degrade(d)
    if result.get("isolation") == "subprocess":
        brief.degrade("ran in a subprocess sandbox, not a container — Docker image unavailable")

    brief.add("Idea", f"> {job.idea}")

    rows = []
    for symbol, res in result.get("symbols", {}).items():
        m = res.get("metrics")
        if not m:
            rows.append([symbol, "—", "—", "—", "—", "failed"])
            continue
        rows.append([symbol, _num(m["sharpe"]), _pct(m["cagr"]), _pct(m["max_drawdown"]),
                     _pct(m["hit_rate"], 0) if m.get("hit_rate") is not None else "n/a",
                     f"{m['trades']}"])
    brief.add("Results", table(
        ["Symbol", "Sharpe", "CAGR", "Max drawdown", "Hit rate", "Trades"], rows))

    # Cost drag is the honest-vs-naive delta market-lab exists to demonstrate.
    drag_rows = []
    for symbol, res in result.get("symbols", {}).items():
        m, g = res.get("metrics"), res.get("gross_metrics")
        if m and g:
            drag_rows.append([symbol, _num(g["sharpe"]), _num(m["sharpe"]),
                              _num((g["sharpe"] or 0) - (m["sharpe"] or 0)),
                              _num(m.get("turnover_annualised"), 1)])
    if drag_rows:
        brief.add("Cost drag", table(
            ["Symbol", "Sharpe gross", "Sharpe net", "Cost of trading", "Turnover/yr"],
            drag_rows) + f"\n\nCosts charged: {job.cost_bps:.0f}bp commission + "
            f"{job.slippage_bps:.0f}bp slippage per unit of turnover.")

    for symbol, res in result.get("symbols", {}).items():
        curve = res.get("equity_curve") or []
        if len(curve) > 2:
            brief.add(f"Equity curve — {symbol}",
                      sparkline(curve) +
                      f"\n\n_{curve[0]['date']} to {curve[-1]['date']}, "
                      f"{res['bars']} bars, ending at {curve[-1]['equity']:.2f}x._")
            break  # one curve is illustration; the table is the evidence

    brief.add("Generated strategy", f"```python\n{job.code}\n```")
    if len(job.attempts) > 1:
        brief.add("Repair history",
                  "\n".join(f"{i}. {a}" for i, a in enumerate(job.attempts, 1)))
    brief.add("Method",
              "Signal decided at the close, position taken at the **next open**, held to "
              "that day's close. Turnover is charged commission plus slippage. The model "
              "wrote only the signal function; lag, fills and costs are harness-owned, so "
              "look-ahead cannot be introduced by the generated code.")
    brief.source("Price lake", note="market-lab Parquet lake, daily OHLCV")
    brief.extra_meta["isolation"] = result.get("isolation", "unknown")
    return brief


def stale_window_notes(result: dict, end: str | None) -> list[str]:
    """Degradations for symbols whose last bar falls short of the window asked for.

    The sandbox mounts the lake and reads Parquet directly, so it never passes
    through PriceSource and never sees a staleness note. A lake that stopped
    updating therefore produces a completely healthy-looking run: every symbol
    returns its full history, `ok` is true, and the only trace is a curve caption
    ending weeks early. Judge the result against the calendar instead.
    """
    by_gap: dict[str, list[str]] = {}
    for symbol, r in sorted((result.get("symbols") or {}).items()):
        if not isinstance(r, dict) or not r.get("end"):
            continue
        gap = coverage_gap_days(r["end"], end)
        if gap > LAKE_STALE_DAYS:
            by_gap.setdefault(f"{r['end']}|{gap}", []).append(symbol)
    notes = []
    for key, symbols in sorted(by_gap.items()):
        last, gap = key.split("|")
        notes.append(
            f"price data for {', '.join(symbols)} ends {last}, {gap} days before "
            f"{end or 'today'} — the backtest window is short by that much; "
            f"refresh the market-lab lake")
    return notes


# --- entry point ------------------------------------------------------------

def run(ctx: Context, idea: str, *, symbols: list[str] | None = None,
        start: str | None = "2015-01-01", end: str | None = None,
        cost_bps: float = 5.0, slippage_bps: float = 5.0,
        max_repairs: int = 2, timeout: int = 120, code: str | None = None,
        commit: bool = True) -> AgentResult:
    started = datetime.now(timezone.utc)
    symbols = [s.upper() for s in (symbols or ["SPY"])]
    res = AgentResult(agent=NAME, target=", ".join(symbols))
    job = BacktestJob(idea=idea, symbols=symbols, start=start, end=end,
                      cost_bps=cost_bps, slippage_bps=slippage_bps)
    for d in ctx.base_degradations():
        res.degrade(d)

    if not ctx.cfg.has_lake:
        res.ok = False
        res.error = f"no price lake at {ctx.cfg.lake_dir}; the sandbox has no data to read"
        record(ctx.db, Run(agent=NAME, target=res.target, ok=False, error=res.error,
                           started_at=started.timestamp()))
        return res

    # 1. code
    if code:
        job.code = code
        job.attempts.append("supplied by caller")
    else:
        try:
            job.code = generate_code(ctx, idea)
            job.attempts.append("generated from the idea")
        except (LLMUnavailable, ValueError) as e:
            res.ok = False
            res.error = f"could not generate strategy code: {e}"
            record(ctx.db, Run(agent=NAME, target=res.target, ok=False, error=res.error,
                               started_at=started.timestamp()))
            return res

    problems = static_check(job.code)
    if problems:
        res.degrade("static check flagged: " + "; ".join(problems))

    # 2. run, repairing on failure
    result = execute(job, ctx.cfg.lake_dir, timeout=timeout)
    repairs = 0
    while not result.get("ok") and repairs < max_repairs and ctx.llm.available and not code:
        err = result.get("error") or "unknown failure"
        log.info("repair %d/%d: %s", repairs + 1, max_repairs, err.splitlines()[-1:])
        try:
            job.code = generate_code(ctx, idea, error=err, previous=job.code)
        except (LLMUnavailable, ValueError) as e:
            res.degrade(f"repair attempt failed: {e}")
            break
        job.attempts.append(f"repaired after: {err.strip().splitlines()[-1][:160]}")
        repairs += 1
        result = execute(job, ctx.cfg.lake_dir, timeout=timeout)

    res.data = result
    if not result.get("ok"):
        res.ok = False
        res.error = f"[{result.get('stage', 'run')}] {result.get('error', 'unknown')}"
        res.summary = f"failed after {repairs} repair attempt(s)"
        record(ctx.db, Run(agent=NAME, target=res.target, ok=False, error=res.error,
                           summary=res.summary, started_at=started.timestamp(),
                           degradations=res.degradations,
                           duration_s=(datetime.now(timezone.utc) - started).total_seconds()))
        return res

    for note in stale_window_notes(result, job.end):
        res.degrade(note)

    brief = build_brief(job, result, degradations=res.degradations)
    res.brief = brief
    res.degradations = list(brief.degradations)
    res.artifact = brief.write(ctx.cfg.out_dir / NAME)

    if commit:
        try:
            cr = ctx.notes.commit_file(f"backtests/{brief.filename}", brief.render(),
                                       f"backtest: {idea[:60]}")
            res.data["commit"] = {"sha": cr.sha, "committed": cr.committed}
        except Exception as e:  # noqa: BLE001
            res.degrade(f"could not commit backtest: {e}")

    first = next((v for v in result["symbols"].values() if "metrics" in v), {})
    m = first.get("metrics", {})
    res.summary = (f"Sharpe {_num(m.get('sharpe'))}, "
                   f"max DD {_pct(m.get('max_drawdown'))}, "
                   f"{repairs} repair(s), {result.get('isolation')}")
    record(ctx.db, Run(agent=NAME, target=res.target, ok=True, artifact=str(res.artifact),
                       summary=res.summary, degradations=res.degradations,
                       started_at=started.timestamp(),
                       duration_s=(datetime.now(timezone.utc) - started).total_seconds()))
    return res
