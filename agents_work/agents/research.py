"""Agent 1 — portfolio research.

Ticker or watchlist in; a one-page brief (thesis, risks, valuation context) out,
committed to a git repo of research notes.

The ordering matters: facts are gathered and rendered first, the model writes
*into* the document last. So a brief is never fabricated — every number in it
came from EDGAR or a price series, and if the model is unavailable you still
get the numbers with the prose sections marked absent.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..brief import Brief, table
from ..grounding import ungrounded
from ..llm import LLMUnavailable
from ..sources.edgar import Edgar
from ..sources.news import News
from ..sources.prices import PriceSource
from ..store import Run, record
from .base import AgentResult, Context

log = logging.getLogger(__name__)

NAME = "research"

SYSTEM = (
    "You are an equity research associate writing for one reader who already "
    "knows markets. You are given a factual dossier assembled from SEC filings, "
    "price history, and headlines. Rules, in priority order:\n"
    "1. Use ONLY facts present in the dossier. Never introduce a number that is "
    "not there, and never estimate one.\n"
    "2. Where the dossier is missing something material, say so explicitly in "
    "one clause — that absence is itself information.\n"
    "3. No hedging boilerplate, no 'as an AI', no investment-advice disclaimer.\n"
    "4. Be specific and falsifiable. 'Margins compressed 240bp year over year' "
    "beats 'margins came under pressure'."
)


def _fmt_money(v: float | None) -> str:
    if v is None:
        return "n/a"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= div:
            return f"${v / div:,.2f}{unit}"
    return f"${v:,.0f}"


def _fmt_pct(v: float | None, *, already_pct: bool = False) -> str:
    if v is None:
        return "n/a"
    return f"{v if already_pct else v * 100:+.1f}%"


def gather(ctx: Context, ticker: str) -> dict:
    """Everything the brief is allowed to talk about, in one dict."""
    edgar = Edgar(ctx.fetcher)
    news = News(ctx.fetcher)
    prices = PriceSource(ctx.cfg.lake_dir if ctx.cfg.has_lake else None,
                         allow_network=not ctx.offline)

    profile = edgar.profile(ticker)
    fundamentals = edgar.fundamentals(profile.cik) if profile.cik else None
    hist = prices.history(ticker, min_rows=2)
    quote = prices.quotes([ticker])[0]
    headlines = news.for_ticker(ticker, profile.name, limit=8, max_age_hours=24 * 14)
    release = edgar.earnings_release(profile.cik, profile.filings) if profile.cik else None

    last = quote.last if quote.ok else (float(hist["close"].iloc[-1]) if len(hist) else None)
    market_cap = (last * fundamentals.shares) if (last and fundamentals and fundamentals.shares) else None
    pe = (last / fundamentals.eps_diluted_ttm) if (
        last and fundamentals and fundamentals.eps_diluted_ttm and fundamentals.eps_diluted_ttm > 0) else None
    ps = (market_cap / fundamentals.revenue_ttm) if (
        market_cap and fundamentals and fundamentals.revenue_ttm) else None

    perf = {}
    if len(hist) > 1:
        closes = hist["adj_close"] if "adj_close" in hist else hist["close"]
        for label, days in (("1m", 21), ("6m", 126), ("1y", 252)):
            if len(closes) > days:
                perf[label] = (closes.iloc[-1] / closes.iloc[-1 - days] - 1) * 100
        window = closes.tail(252)
        if len(window) > 20:
            perf["52w_high_gap"] = (closes.iloc[-1] / window.max() - 1) * 100
            ret = closes.pct_change().dropna()
            if len(ret) > 20:
                perf["vol_annualised"] = float(ret.std() * (252 ** 0.5) * 100)

    return {
        "ticker": ticker.upper(), "profile": profile, "fundamentals": fundamentals,
        "history": hist, "quote": quote, "headlines": headlines, "release": release,
        "last": last, "market_cap": market_cap, "pe": pe, "ps": ps, "perf": perf,
        "price_notes": prices.notes,
    }


def _dossier(d: dict) -> str:
    """The exact factual surface handed to the model. Kept as plain text on
    purpose: it is the thing to read when a brief says something odd."""
    p, f = d["profile"], d["fundamentals"]
    lines = [f"COMPANY: {p.name or d['ticker']} ({d['ticker']}) — {p.sic or 'sector unknown'}"]
    lines.append(f"PRICE: last {d['last'] if d['last'] else 'n/a'}, "
                 f"market cap {_fmt_money(d['market_cap'])}")
    if d["perf"]:
        lines.append("PERFORMANCE: " + ", ".join(
            f"{k} {v:+.1f}%" for k, v in d["perf"].items()))
    if f:
        lines += [
            f"REVENUE TTM: {_fmt_money(f.revenue_ttm)} (prior-year TTM {_fmt_money(f.revenue_prior_ttm)}, "
            f"growth {_fmt_pct(f.revenue_growth)})",
            f"NET INCOME TTM: {_fmt_money(f.net_income_ttm)}, net margin {_fmt_pct(f.net_margin)}",
            f"DILUTED EPS TTM: {f.eps_diluted_ttm if f.eps_diluted_ttm is not None else 'n/a'}",
            f"EQUITY: {_fmt_money(f.equity)}, ASSETS: {_fmt_money(f.assets)}",
            f"MULTIPLES: P/E {d['pe']:.1f}" if d["pe"] else "MULTIPLES: P/E n/a",
        ]
        if d["ps"]:
            lines[-1] += f", P/S {d['ps']:.1f}"
        if f.periods:
            lines.append(f"TTM built from quarters ({f.revenue_tag or 'no revenue tag'}): "
                         + ", ".join(f.periods))
        for note in f.notes:
            lines.append(f"DATA CAVEAT: {note}")
    else:
        lines.append("FUNDAMENTALS: unavailable (no XBRL facts retrieved)")
    if p.filings:
        lines.append("RECENT FILINGS: " + "; ".join(
            f"{x.form} filed {x.filed}" for x in p.filings[:6]))
    if d["headlines"]:
        lines.append("HEADLINES (most recent first):")
        lines += [f"  - [{h.when}] {h.title} ({h.source})" for h in d["headlines"]]
    else:
        lines.append("HEADLINES: none retrieved")
    if d["release"]:
        lines.append("LATEST EARNINGS RELEASE (8-K EX-99 excerpt):")
        lines.append(d["release"][0][:6000])
    else:
        lines.append("EARNINGS RELEASE: not retrieved")
    return "\n".join(lines)


# The three the prompt asks for. A brief missing one is not a shorter brief, it
# is a brief whose reader does not know a section was requested and lost.
EXPECTED_SECTIONS = ("Thesis", "Risks", "Valuation context")


def _write_sections(ctx: Context, d: dict, brief: Brief) -> None:
    dossier = _dossier(d)
    prompt = (
        f"{dossier}\n\n---\n\n"
        "Write three sections in GitHub-flavoured markdown. Use these exact "
        "headings and nothing above or below them:\n\n"
        "## Thesis\nTwo short paragraphs. First the bull case as the market is "
        "pricing it, then the specific thing that would have to be true for it "
        "to work.\n\n"
        "## Risks\nThree to five bullets. Each names a concrete mechanism and, "
        "where the dossier supports it, the number that would signal it.\n\n"
        "## Valuation context\nOne paragraph placing the multiples against the "
        "company's own growth and margins. If a multiple is missing, say which "
        "and why it matters. Do not compare to peers — you have no peer data."
    )
    try:
        text = ctx.llm.complete(prompt, system=SYSTEM, max_tokens=4000)
    except LLMUnavailable as e:
        brief.degrade(f"analysis sections omitted ({e})")
        brief.add("Thesis", "_Not written — the language model was unavailable for this run. "
                            "The data sections above are complete and unaffected._")
        return
    flagged: list[str] = []
    written = set()
    for heading, body in _split_sections(text):
        brief.add(heading, body)
        written.add(heading.strip().lower())
        flagged += [f for f in ungrounded(body, dossier) if f not in flagged]

    missing = [h for h in EXPECTED_SECTIONS if h.lower() not in written]
    if missing and "analysis" not in written:
        # Either the reply was cut off at the token cap or the model ignored the
        # headings. Both leave a brief that looks finished and is not.
        why = ("the reply was cut off at the token limit"
               if getattr(ctx.llm, "last_stop_reason", None) == "max_tokens"
               else "the model did not return them")
        brief.degrade(f"analysis incomplete — {', '.join(missing)} missing ({why})")
    if flagged:
        # Not an error: a derived figure ("margins compressed 240bp") is legitimate
        # and still unverifiable from the dossier. Naming them is the point.
        brief.add("Unverified figures", (
            "These figures appear in the analysis above but not in the source dossier, "
            "so nothing here checked them: "
            + ", ".join(f"`{x}`" for x in flagged)
            + ". They are either derived from two dossier facts or invented; the brief "
            "cannot tell which."), level=3)
        brief.extra_meta["ungrounded_figures"] = len(flagged)


def _split_sections(text: str) -> list[tuple[str, str]]:
    out, heading, buf = [], None, []
    for line in text.splitlines():
        if line.startswith("## "):
            if heading:
                out.append((heading, "\n".join(buf).strip()))
            heading, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if heading:
        out.append((heading, "\n".join(buf).strip()))
    if not out:  # model ignored the headings; keep the prose rather than drop it
        out = [("Analysis", text.strip())]
    return out


def build_brief(ctx: Context, ticker: str) -> tuple[Brief, dict]:
    d = gather(ctx, ticker)
    p, f, q = d["profile"], d["fundamentals"], d["quote"]
    title = f"{p.name or d['ticker']} ({d['ticker']}) — research brief"
    brief = Brief(title=title, agent=NAME, target=d["ticker"], tags=["research", "equity"])
    for note in ctx.base_degradations():
        brief.degrade(note)
    for note in p.notes + d["price_notes"] + (f.notes if f else []):
        brief.degrade(note)
    if q.error:
        brief.degrade(q.error)

    snapshot = [
        ["Last price", f"{d['last']:,.2f}" if d["last"] else "n/a"],
        ["Change", _fmt_pct(q.change_pct, already_pct=True) if q.change_pct is not None else "n/a"],
        ["Market cap", _fmt_money(d["market_cap"])],
        ["P/E (TTM, diluted)", f"{d['pe']:.1f}" if d["pe"] else "n/a"],
        ["P/S (TTM)", f"{d['ps']:.1f}" if d["ps"] else "n/a"],
        ["Revenue TTM", _fmt_money(f.revenue_ttm) if f else "n/a"],
        ["Revenue growth YoY", _fmt_pct(f.revenue_growth) if f else "n/a"],
        ["Net margin TTM", _fmt_pct(f.net_margin) if f else "n/a"],
    ]
    for label, days in (("1m", "1m"), ("6m", "6m"), ("1y", "1y")):
        if days in d["perf"]:
            snapshot.append([f"Return {label}", f"{d['perf'][days]:+.1f}%"])
    # Everything the model is shown, the reader can see. This one was in the
    # dossier but not the table, so a brief could say "4.7% below its 52-week
    # high" with nothing on the page to check it against.
    if "52w_high_gap" in d["perf"]:
        snapshot.append(["From 52-week high", f"{d['perf']['52w_high_gap']:+.1f}%"])
    if "vol_annualised" in d["perf"]:
        snapshot.append(["Annualised vol", f"{d['perf']['vol_annualised']:.1f}%"])
    brief.add("Snapshot", table(["Metric", "Value"], snapshot))
    if f and f.periods:
        brief.add("How the TTM was assembled", (
            f"Revenue is tagged `{f.revenue_tag}` by this filer. The trailing twelve "
            "months is the sum of four contiguous quarters — "
            + ", ".join(f"`{x}`" for x in f.periods) + " — reconstructing any quarter "
            "the filer reports only inside a year-to-date figure. A gap that cannot be "
            "reconstructed yields no TTM at all rather than a shorter window "
            "presented as a year."), level=3)

    if p.filings:
        brief.add("Recent filings", table(
            ["Form", "Filed", "Period", "Link"],
            [[x.form, x.filed, x.report_date or "—", f"[{x.accession}]({x.url})"]
             for x in p.filings[:6]]))
        brief.source("SEC EDGAR filings",
                     f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={p.cik:010d}",
                     "primary source for every fundamental above")
    else:
        brief.degrade("no filings retrieved from EDGAR")

    if d["headlines"]:
        brief.add("Headlines", "\n".join(
            f"- **{h.when}** [{h.title}]({h.url}) — _{h.source}_" for h in d["headlines"]))
        brief.source("Google News RSS", note="headline discovery, last 14 days")
    else:
        brief.degrade("no headlines retrieved")

    if d["release"]:
        text, url = d["release"]
        brief.source("Latest 8-K earnings release", url,
                     "used as the earnings-materials input (no transcript source is free)")
    else:
        brief.degrade("no 8-K earnings release found; earnings commentary is from filings only")

    _write_sections(ctx, d, brief)

    brief.extra_meta.update({
        "cik": p.cik or "", "last_price": round(d["last"], 4) if d["last"] else "",
        "pe": round(d["pe"], 2) if d["pe"] else "",
        "sources": len(brief.sources),
    })
    return brief, d


def run(ctx: Context, ticker: str, *, commit: bool = True) -> AgentResult:
    """One ticker -> one committed brief."""
    started = datetime.now(timezone.utc)
    res = AgentResult(agent=NAME, target=ticker.upper())
    try:
        brief, data = build_brief(ctx, ticker)
    except Exception as e:  # noqa: BLE001
        log.exception("research failed for %s", ticker)
        res.ok, res.error = False, f"{type(e).__name__}: {e}"
        record(ctx.db, Run(agent=NAME, target=ticker.upper(), ok=False, error=res.error,
                           started_at=started.timestamp()))
        return res

    res.brief = brief
    res.degradations = list(brief.degradations)
    path = brief.write(ctx.cfg.out_dir / NAME)
    res.artifact = path

    if commit:
        try:
            commit_res = ctx.notes.commit_file(
                f"briefs/{brief.filename}", brief.render(),
                f"{brief.target}: research brief {brief.date}")
            res.data["commit"] = {"sha": commit_res.sha, "committed": commit_res.committed,
                                  "pushed": commit_res.pushed}
            if commit_res.push_error:
                res.degrade(f"commit stayed local: {commit_res.push_error}")
        except Exception as e:  # noqa: BLE001 - a git failure must not lose the brief
            log.warning("commit failed: %s", e)
            res.degrade(f"could not commit to notes repo: {e}")

    res.summary = (f"{brief.target}: "
                   f"{'P/E ' + format(data['pe'], '.1f') if data['pe'] else 'no P/E'}, "
                   f"{len(data['headlines'])} headlines, "
                   f"{len(data['profile'].filings)} filings")
    record(ctx.db, Run(agent=NAME, target=res.target, ok=True, artifact=str(path),
                       summary=res.summary, degradations=res.degradations,
                       started_at=started.timestamp(),
                       duration_s=(datetime.now(timezone.utc) - started).total_seconds()))
    return res


def run_watchlist(ctx: Context, tickers: list[str], **kw) -> list[AgentResult]:
    return [run(ctx, t, **kw) for t in tickers]
