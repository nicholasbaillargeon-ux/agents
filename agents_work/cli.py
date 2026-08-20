"""`agents` — one entry point for all five agents and for the timers."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime

from .agents import analyst, backtest, briefing, research, scout
from .agents.base import Context
from .config import load_config
from .store import recent


def _print_result(res, *, as_json: bool) -> int:
    if as_json:
        print(json.dumps({
            "agent": res.agent, "target": res.target, "ok": res.ok,
            "summary": res.summary, "artifact": str(res.artifact) if res.artifact else None,
            "degradations": res.degradations, "error": res.error,
            "data": {k: v for k, v in res.data.items() if k in ("commit", "new", "scanned",
                                                                "citations", "search_only")},
        }, indent=2, default=str))
        return 0 if res.ok else 1
    mark = "ok" if res.ok else "FAILED"
    print(f"[{mark}] {res.agent} {res.target}".rstrip())
    if res.summary:
        print(f"  {res.summary}")
    if res.artifact:
        print(f"  -> {res.artifact}")
    for d in res.degradations:
        print(f"  ! {d}")
    if res.error:
        print(f"  error: {res.error}", file=sys.stderr)
    return 0 if res.ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agents", description=__doc__)
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--offline", action="store_true", help="serve cached responses only")
    p.add_argument("--no-commit", action="store_true", help="skip the git commit")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("research", help="one-page research brief per ticker")
    r.add_argument("tickers", nargs="+")

    b = sub.add_parser("backtest", help="plain-English strategy -> sandboxed backtest")
    b.add_argument("idea")
    b.add_argument("--symbols", default="SPY", help="comma-separated")
    b.add_argument("--start", default="2015-01-01")
    b.add_argument("--end", default=None)
    b.add_argument("--cost-bps", type=float, default=5.0)
    b.add_argument("--slippage-bps", type=float, default=5.0)
    b.add_argument("--timeout", type=int, default=120)
    b.add_argument("--build-image", action="store_true", help="build the sandbox image first")

    m = sub.add_parser("briefing", help="market open briefing")
    m.add_argument("--watchlist", default=None, help="comma-separated override")

    s = sub.add_parser("scout", help="nightly internship diff")
    s.add_argument("--min-score", type=int, default=4)
    s.add_argument("--all-roles", action="store_true", help="not just internship titles")
    s.add_argument("--no-llm", action="store_true", help="skip model verdicts")

    a = sub.add_parser("ask", help="ask the RAG analyst")
    a.add_argument("question")
    a.add_argument("--k", type=int, default=6)
    a.add_argument("--since", default=None)
    a.add_argument("--reindex", action="store_true")

    sub.add_parser("index", help="rebuild the analyst index")
    sub.add_parser("status", help="recent runs")
    sub.add_parser("doctor", help="what works right now")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config()
    if args.cmd == "doctor":
        return _doctor(cfg)
    if args.cmd == "status":
        ctx = Context(cfg)
        for row in recent(ctx.db, limit=20):
            when = datetime.fromtimestamp(row["started_at"]).strftime("%Y-%m-%d %H:%M")
            flag = "ok " if row["ok"] else "ERR"
            print(f"{when}  {flag}  {row['agent']:<9} {row['target'][:28]:<28} "
                  f"{row['duration_s']:5.1f}s  {(row['summary'] or row['error'] or '')[:60]}")
        ctx.close()
        return 0

    ctx = Context(cfg, offline=args.offline)
    commit = not args.no_commit
    try:
        if args.cmd == "research":
            rc = 0
            for t in args.tickers:
                rc |= _print_result(research.run(ctx, t, commit=commit), as_json=args.json)
            return rc
        if args.cmd == "backtest":
            if args.build_image:
                okd, out = backtest.build_image()
                print(f"image build: {'ok' if okd else 'FAILED'} {out.splitlines()[-1] if out else ''}")
            return _print_result(backtest.run(
                ctx, args.idea, symbols=args.symbols.split(","), start=args.start,
                end=args.end, cost_bps=args.cost_bps, slippage_bps=args.slippage_bps,
                timeout=args.timeout, commit=commit), as_json=args.json)
        if args.cmd == "briefing":
            wl = args.watchlist.split(",") if args.watchlist else None
            return _print_result(briefing.run(ctx, watchlist=wl, commit=commit), as_json=args.json)
        if args.cmd == "scout":
            return _print_result(scout.run(
                ctx, min_score=args.min_score, internships_only=not args.all_roles,
                use_llm=not args.no_llm, commit=commit), as_json=args.json)
        if args.cmd == "ask":
            res = analyst.run(ctx, args.question, k=args.k, since=args.since,
                              reindex=args.reindex)
            if not args.json and res.brief:
                print(res.brief.sections[0].body if res.brief.sections else "")
                print()
            return _print_result(res, as_json=args.json)
        if args.cmd == "index":
            idx = analyst.Index(cfg.data_dir / "analyst.db")
            stats = idx.build(cfg.vault_roots)
            idx.close()
            print(json.dumps(stats, indent=2))
            return 0
    finally:
        ctx.close()
    return 2


def _doctor(cfg) -> int:
    from .agents.backtest import docker_available
    from .netcache import Fetcher

    print(f"data dir       {cfg.data_dir}")
    print(f"LLM            {'yes' if cfg.has_llm else 'NO KEY'} "
          f"({cfg.llm_base_url}, write={cfg.write_model}, fast={cfg.fast_model})")
    print(f"price lake     {'yes' if cfg.has_lake else 'MISSING'} ({cfg.lake_dir})")
    print(f"sandbox image  {'yes' if docker_available() else 'NOT BUILT — subprocess fallback'}")
    print(f"notes repo     {cfg.notes_repo} "
          f"(remote: {cfg.git_remote or 'local only'})")
    for root in cfg.vault_roots:
        n = len(list(root.rglob('*.md'))) if root.is_dir() else 0
        print(f"vault root     {'yes' if root.is_dir() else 'MISSING'}  {root} ({n} md files)")
    f = Fetcher(cfg.cache_dir, user_agent=cfg.sec_user_agent)
    probe = f.fetch("https://www.sec.gov/files/company_tickers.json", ttl=86_400)
    print(f"SEC EDGAR      {'reachable' if probe and probe.ok else 'UNREACHABLE'}")
    for d in cfg.degradations():
        print(f"  ! {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
