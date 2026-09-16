"""`agents` — one entry point for all five agents and for the timers."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime

from .agents import analyst, backtest, briefing, comps, dealbook, research, scout
from .agents.base import Context
from .config import load_config
from .netcache import prune_cache
from .store import recent

log = logging.getLogger(__name__)


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
    s.add_argument("--limit", type=int, default=None,
                   help="how many new postings to show (default 100); raise it to "
                        "drain a backlog in one run")

    c = sub.add_parser("comps", help="comparable-companies table from SEC filings")
    c.add_argument("tickers", nargs="*", help="explicit peer tickers")
    c.add_argument("--set", dest="peer_set", default=None,
                   help=f"a curated set: {', '.join(comps.PEER_SETS)}")
    c.add_argument("--peers", default=None,
                   help='describe the peer group in English, e.g. "mid-cap asset managers"')
    c.add_argument("--list-sets", action="store_true", help="print the curated sets and exit")

    d = sub.add_parser("deals", help="M&A deal book: sweep, review, annotate")
    d.add_argument("--list", dest="list_deals", action="store_true",
                   help="show the book instead of sweeping")
    d.add_argument("--show", type=int, default=None, metavar="ID",
                   help="print one deal's full one-pager")
    d.add_argument("--note", nargs=2, default=None, metavar=("ID", "TEXT"),
                   help="record your own view and mark the deal reviewed")
    d.add_argument("--status", nargs=2, default=None, metavar=("ID", "STATUS"),
                   help="set a deal's status (new/reviewed/tracking/archived)")
    d.add_argument("--flagged", action="store_true", help="with --list, only advisor-flagged")
    d.add_argument("--new-only", action="store_true", help="with --list, only unreviewed")
    d.add_argument("--max-enrich", type=int, default=dealbook.MAX_ENRICH)
    d.add_argument("--max-drafts", type=int, default=dealbook.MAX_DRAFTS)

    a = sub.add_parser("ask", help="ask the RAG analyst")
    a.add_argument("question")
    a.add_argument("--k", type=int, default=analyst.DEFAULT_K,
                   help=f"how many passages to retrieve (default {analyst.DEFAULT_K})")
    a.add_argument("--since", default=None)
    a.add_argument("--reindex", action="store_true")

    sub.add_parser("index", help="rebuild the analyst index")

    pr = sub.add_parser("prune", help="trim the HTTP cache to its size cap")
    pr.add_argument("--max-mb", type=int, default=None,
                    help="override the configured cap for this run")
    pr.add_argument("--dry-run", action="store_true",
                    help="report what would be evicted, delete nothing")
    sub.add_parser("status", help="recent runs")
    sub.add_parser("doctor", help="what works right now")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config()
    if args.cmd == "doctor":
        return _doctor(cfg)
    if args.cmd == "prune":
        cap = cfg.cache_max_bytes if args.max_mb is None else args.max_mb * 1024 * 1024
        st = prune_cache(cfg.cache_dir, cap, dry_run=args.dry_run)
        verb = "would evict" if args.dry_run else "evicted"
        print(f"cache {st['kept_bytes'] / 1048576:.1f}MB in {st['kept']} entries "
              f"(cap {cap / 1048576:.0f}MB); {verb} {st['removed']} "
              f"freeing {st['freed_bytes'] / 1048576:.1f}MB")
        return 0
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
            kw = {} if args.limit is None else {"limit": args.limit}
            return _print_result(scout.run(
                ctx, min_score=args.min_score, internships_only=not args.all_roles,
                use_llm=not args.no_llm, commit=commit, **kw), as_json=args.json)
        if args.cmd == "comps":
            if args.list_sets:
                for name, members in comps.PEER_SETS.items():
                    print(f"{name:<16} {', '.join(members)}")
                print("\nSectors this table cannot price, and why:")
                for sector, why in comps.KNOWN_HARD.items():
                    print(f"  {sector:<20} {why}")
                return 0
            label, note = "", ""
            if args.peer_set:
                if args.peer_set not in comps.PEER_SETS:
                    print(f"unknown set {args.peer_set!r}; try --list-sets", file=sys.stderr)
                    return 2
                tickers, label, note = (list(comps.PEER_SETS[args.peer_set]),
                                        args.peer_set, f"curated set `{args.peer_set}`")
            elif args.peers:
                tickers, note = comps.resolve_peers(ctx, args.peers)
                label = args.peers
                if not tickers:
                    print(f"could not resolve a peer set: {note}", file=sys.stderr)
                    return 1
            else:
                tickers = [t.upper() for t in args.tickers]
                if not tickers:
                    print("give tickers, --set, or --peers", file=sys.stderr)
                    return 2
            return _print_result(comps.run(ctx, tickers, label=label, peer_note=note,
                                           commit=commit), as_json=args.json)
        if args.cmd == "deals":
            return _deals(ctx, args, commit=commit)
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
        # After the brief is written, never before: an eviction must not be
        # able to cost a run the cached response it was about to read.
        st = prune_cache(cfg.cache_dir, cfg.cache_max_bytes)
        if st["removed"] or st["tmp_removed"]:
            log.info("cache pruned: %d entries freeing %.1fMB, %d stray tmp",
                     st["removed"], st["freed_bytes"] / 1048576, st["tmp_removed"])
    return 2


def _deals(ctx, args, *, commit: bool) -> int:
    """The review half of the deal book. The sweep is the agent; this is you."""
    from .dealstore import (DealBookUnavailable, annotate, connect, ensure_schema,
                            get_deal, list_deals, set_status)

    interactive = (args.list_deals or args.show is not None
                   or args.note is not None or args.status is not None)
    if not interactive:
        return _print_result(
            dealbook.run(ctx, commit=commit, max_enrich=args.max_enrich,
                         max_drafts=args.max_drafts), as_json=args.json)
    try:
        with connect(ctx.cfg.dealbook_dsn) as conn:
            ensure_schema(conn)
            if args.note:
                deal_id, text = int(args.note[0]), args.note[1]
                ok = annotate(conn, deal_id, text)
                print(f"deal {deal_id}: {'annotated and marked reviewed' if ok else 'not found'}")
                return 0 if ok else 1
            if args.status:
                deal_id, status = int(args.status[0]), args.status[1]
                ok = set_status(conn, deal_id, status)
                print(f"deal {deal_id}: {'status ' + status if ok else 'not found'}")
                return 0 if ok else 1
            if args.show is not None:
                row = get_deal(conn, args.show)
                if not row:
                    print(f"no deal {args.show}", file=sys.stderr)
                    return 1
                print(json.dumps(row, indent=2, default=str) if args.json
                      else _render_deal(row))
                return 0
            rows = list_deals(conn, status="new" if args.new_only else None,
                              flagged_only=args.flagged)
            if args.json:
                print(json.dumps(rows, indent=2, default=str))
                return 0
            if not rows:
                print("the book is empty — run `agents deals` to sweep")
                return 0
            for r in rows:
                flag = "!" if r["flagged_advisor"] else " "
                view = "*" if r["my_view"] else " "
                value = ("—" if r["value_usd"] is None
                         else f"{float(r['value_usd']) / 1e9:,.2f}B")
                print(f"{r['id']:>4}{flag}{view} {r['status']:<9} {value:>9} "
                      f"{(r['acquirer'] or '?')[:24]:<24} -> {(r['target'] or '?')[:24]:<24} "
                      f"{r['sector'][:18]}")
            print(f"\n{len(rows)} deals   ! = watched advisor   * = you have a view on it")
            return 0
    except DealBookUnavailable as e:
        print(f"deal book unavailable: {e}", file=sys.stderr)
        return 1


def _render_deal(r: dict) -> str:
    out = [f"#{r['id']}  {r['acquirer'] or '?'} -> {r['target'] or '?'}",
           f"  status     {r['status']}" + ("  (you have a view on this)" if r["my_view"] else ""),
           f"  sector     {r['sector'] or '—'}",
           f"  value      {'—' if r['value_usd'] is None else format(float(r['value_usd']), ',.0f')}"
           f" {r['currency'] or 'USD'}",
           f"  structure  {r['consideration'] or '—'}",
           f"  multiple   {r['implied_multiple'] or 'none disclosed'}"]
    if r["flagged_advisor"]:
        out.append(f"  FLAGGED    {r['flagged_advisor']} is advising")
    for label, key in (("rationale", "rationale"), ("financing", "financing")):
        if r[key]:
            out.append(f"  {label:<10} {r[key]}")
    if r["advisors"]:
        out.append("  advisors   " + ", ".join(
            f"{a['name']} ({a.get('side', '?')})" for a in r["advisors"]))
    if r["open_questions"]:
        out.append("  open questions:")
        out += [f"    - {q}" for q in r["open_questions"]]
    out.append(f"  source     {r['url']}")
    out.append("")
    out.append("  YOUR VIEW: " + (r["my_view"] or
                                  "(none yet — agents deals --note %d \"...\")" % r["id"]))
    return "\n".join(out)


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
    from .dealstore import health  # noqa: PLC0415
    print(f"deal book      {health(cfg.dealbook_dsn)}")
    print(f"phone push     {'ntfy topic set' if cfg.has_push else 'NOT CONFIGURED'} "
          f"({cfg.ntfy_server})")
    for d in cfg.degradations():
        print(f"  ! {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
