"""Agent 6 — a comparable-companies table built from SEC filings.

The table a banking analyst builds by hand, built from EDGAR's XBRL API
instead: market cap from the share count on the cover page and a live price,
an enterprise value bridged through debt, cash, preferred and minorities, and
the multiples that follow. Every cell traces to a filed fact or is blank.

The hard part is not the arithmetic, it is that XBRL is only nominally
standard. Filers migrate between tags and EDGAR serves the abandoned ones
forever; banks report no operating income; oil majors tag revenue in a
company-specific namespace; a split-share-class filer reports two cover-page
counts. `sources/edgar.py` carries that knowledge. What this module adds is the
rule that a peer whose facts do not support a multiple gets an empty cell and a
footnote — never a plugged number — because one invented cell in a comps table
is invisible and moves the median everyone reads off it.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..brief import Brief, table
from ..grounding import ungrounded
from ..llm import LLMUnavailable
from ..sources.edgar import Edgar
from ..sources.prices import PriceSource
from ..store import Run, record
from .base import AgentResult, Context, finalize

log = logging.getLogger(__name__)

NAME = "comps"

# Curated sets, chosen for clean XBRL rather than for sector coverage. Every
# one of these was run before it was added; the ones that do not resolve (oil
# majors, most banks) are deliberately absent and listed in KNOWN_HARD.
PEER_SETS: dict[str, list[str]] = {
    "megacap-tech": ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA"],
    "semis": ["NVDA", "AMD", "AVGO", "QCOM", "TXN", "ADI", "MU"],
    "software": ["MSFT", "CRM", "ADBE", "ORCL", "NOW", "INTU", "WDAY"],
    "payments": ["V", "MA", "PYPL", "FI", "GPN", "AXP"],
    "exchanges": ["CME", "ICE", "NDAQ", "CBOE"],
    "asset-managers": ["BLK", "TROW", "BEN", "IVZ", "AMG", "APAM", "VRTS"],
    "advisory": ["LAZ", "EVR", "PJT", "HLI", "MC", "GS", "MS"],
    "med-device": ["MDT", "SYK", "BSX", "ZBH", "EW", "ABT"],
}

# Filers whose us-gaap facts do not support this table, with the reason. Named
# so a user who asks for them is told why rather than handed blank rows.
KNOWN_HARD = {
    "banks and insurers": "no OperatingIncomeLoss line, so no EBITDA; debt is funding, not leverage",
    "oil majors": "revenue tagged in a company-specific namespace (XOM, CVX)",
    "REITs": "borrowings tagged per-property; FFO, not EBITDA, is the sector multiple",
}

RESOLVE_SYSTEM = (
    "You turn a description of a peer group into US-listed tickers. Return JSON: "
    '{"tickers": ["..."], "note": "one clause on what you selected"}. '
    "Rules: 4-8 tickers, US primary listings only, ordinary common shares only. "
    "Match the size band asked for — 'mid-cap' means roughly $2-20bn, not the "
    "sector's largest names. If the description names a sector whose filers do "
    "not report an operating income line (banks, insurers, REITs), still return "
    "the tickers and say so in the note. Return tickers only, no company names."
)

COMMENT_SYSTEM = (
    "You are handed a completed comps table and the median of each column. "
    "Write three to five sentences for an analyst who already knows the sector. "
    "Say which names trade above and below the peer median and on which metric, "
    "and name the one dispersion worth explaining. Use only the numbers given — "
    "blank cells mean the filer does not report that line, which is itself worth "
    "a clause if it affects the median. No recommendations, no price targets, no "
    "preamble."
)


def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or not b:
        return None
    return a / b


@dataclass
class CompRow:
    ticker: str
    name: str = ""
    price: float | None = None
    shares: float | None = None
    market_cap: float | None = None
    enterprise_value: float | None = None
    revenue_ttm: float | None = None
    ebitda_ttm: float | None = None
    net_income_ttm: float | None = None
    eps_ttm: float | None = None
    revenue_growth: float | None = None
    ebitda_margin: float | None = None
    net_margin: float | None = None
    ttm_end: str = ""
    debt_basis: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def ev_sales(self) -> float | None:
        return _safe_div(self.enterprise_value, self.revenue_ttm)

    @property
    def ev_ebitda(self) -> float | None:
        if self.ebitda_ttm is not None and self.ebitda_ttm <= 0:
            return None      # a negative multiple is not a cheap company
        return _safe_div(self.enterprise_value, self.ebitda_ttm)

    @property
    def pe(self) -> float | None:
        if self.eps_ttm is not None and self.eps_ttm <= 0:
            return None
        return _safe_div(self.price, self.eps_ttm)

    @property
    def ps(self) -> float | None:
        return _safe_div(self.market_cap, self.revenue_ttm)

    @property
    def ok(self) -> bool:
        return self.revenue_ttm is not None or self.market_cap is not None


METRICS = (
    ("EV/Sales", "ev_sales", "x", 1),
    ("EV/EBITDA", "ev_ebitda", "x", 1),
    ("P/E", "pe", "x", 1),
    ("P/S", "ps", "x", 1),
    ("Rev growth", "revenue_growth", "%", 1),
    ("EBITDA margin", "ebitda_margin", "%", 1),
    ("Net margin", "net_margin", "%", 1),
)


def _fmt(value: float | None, unit: str, places: int) -> str:
    if value is None:
        return "—"
    if unit == "%":
        return f"{value * 100:.{places}f}%"
    return f"{value:.{places}f}x"


def _money(v: float | None) -> str:
    if v is None:
        return "—"
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(v) >= scale:
            return f"${v / scale:,.1f}{suffix}"
    return f"${v:,.0f}"


def median_of(rows: list[CompRow], attr: str) -> float | None:
    """Median across the peers that actually have the metric.

    Blank cells are dropped, not zeroed. A zero in a multiple column is not a
    company trading at zero times earnings, it is a company that did not report
    the line — and averaging it in drags the peer median toward a number no
    peer trades at.
    """
    values = [getattr(r, attr) for r in rows]
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def build_rows(ctx: Context, tickers: list[str]) -> tuple[list[CompRow], list[str]]:
    edgar = Edgar(ctx.fetcher)
    prices = PriceSource(ctx.cfg.lake_dir if ctx.cfg.has_lake else None,
                         allow_network=not ctx.offline)
    quotes = {q.symbol: q for q in prices.quotes(tickers)}
    rows, degradations = [], list(prices.notes)

    for ticker in tickers:
        row = CompRow(ticker=ticker)
        hit = edgar.cik_for(ticker)
        if not hit:
            row.notes.append("not in the SEC ticker file")
            rows.append(row)
            continue
        cik, row.name = hit
        f = edgar.fundamentals(cik)
        quote = quotes.get(ticker)
        row.price = quote.last if quote else None
        row.shares = f.shares
        row.market_cap = (row.price * row.shares
                          if row.price is not None and row.shares else None)
        row.enterprise_value = f.enterprise_value(row.market_cap)
        row.revenue_ttm = f.revenue_ttm
        row.ebitda_ttm = f.ebitda_ttm
        row.net_income_ttm = f.net_income_ttm
        row.eps_ttm = f.eps_diluted_ttm
        row.revenue_growth = f.revenue_growth
        row.ebitda_margin = f.ebitda_margin
        row.net_margin = f.net_margin
        row.ttm_end = f.ttm_end
        row.debt_basis = f.debt_basis
        row.notes.extend(f.notes)
        if quote and quote.error:
            row.notes.append(f"price: {quote.error}")
        if row.market_cap is None and row.shares is None:
            row.notes.append("no usable share count, so no market cap and no EV")
        rows.append(row)
    return rows, degradations


def comps_table(rows: list[CompRow]) -> str:
    headers = ["Ticker", "Price", "Mkt cap", "EV"] + [m[0] for m in METRICS]
    body = []
    for r in rows:
        body.append([r.ticker, "—" if r.price is None else f"{r.price:,.2f}",
                     _money(r.market_cap), _money(r.enterprise_value)]
                    + [_fmt(getattr(r, attr), unit, places)
                       for _, attr, unit, places in METRICS])
    live = [r for r in rows if r.ok]
    if live:
        body.append(["**Median**", "", "", ""]
                    + [f"**{_fmt(median_of(live, attr), unit, places)}**"
                       for _, attr, unit, places in METRICS])
    return table(headers, body, align=["---"] + ["---:"] * (len(headers) - 1))


def bridge_table(rows: list[CompRow]) -> str:
    """The EV bridge, shown rather than asserted.

    A comps table that prints an enterprise value and nothing else asks the
    reader to trust the hardest number on the page. This is the arithmetic.
    """
    body = []
    for r in rows:
        if r.market_cap is None and r.enterprise_value is None:
            continue
        bridge = (None if r.enterprise_value is None or r.market_cap is None
                  else r.enterprise_value - r.market_cap)
        body.append([r.ticker, _money(r.market_cap),
                     "—" if bridge is None else _money(bridge),
                     _money(r.enterprise_value), r.ttm_end or "—",
                     (r.debt_basis or "—")[:64]])
    return table(["Ticker", "Market cap", "+ net debt etc.", "= EV",
                  "TTM ends", "Debt basis"], body)


def resolve_peers(ctx: Context, description: str) -> tuple[list[str], str]:
    """A peer description -> tickers. Curated sets win; the model is the fallback.

    Checked against the SEC ticker file before being returned, because a model
    asked for mid-cap asset managers will happily produce a plausible ticker
    that belongs to something else entirely, and a wrong constituent is far
    harder to spot in a finished table than a missing one.
    """
    key = description.strip().lower().replace(" ", "-")
    if key in PEER_SETS:
        return list(PEER_SETS[key]), f"curated set `{key}`"
    if not ctx.llm.available:
        return [], "no LLM available to resolve a peer description"
    parsed = ctx.llm.json(f"Peer group: {description}", system=RESOLVE_SYSTEM,
                          max_tokens=400, fast=True, default=None)
    if not isinstance(parsed, dict):
        return [], "could not resolve that peer description"
    raw = [str(t).upper().strip() for t in (parsed.get("tickers") or []) if str(t).strip()]
    edgar = Edgar(ctx.fetcher)
    tickers = [t for t in raw if edgar.cik_for(t)]
    dropped = [t for t in raw if t not in tickers]
    note = str(parsed.get("note") or "").strip()
    if dropped:
        note += (f" (dropped {', '.join(dropped)}: not in the SEC ticker file)")
    return tickers, note or "model-resolved peer set"


def commentary(ctx: Context, rows: list[CompRow]) -> tuple[str, list[str]]:
    if not ctx.llm.available:
        return "", []
    live = [r for r in rows if r.ok]
    if len(live) < 2:
        return "", []
    lines = []
    for r in live:
        bits = [f"{r.ticker} ({r.name})", f"mkt cap {_money(r.market_cap)}",
                f"EV {_money(r.enterprise_value)}"]
        bits += [f"{label} {_fmt(getattr(r, attr), unit, places)}"
                 for label, attr, unit, places in METRICS]
        lines.append("; ".join(bits))
        for n in r.notes:
            lines.append(f"    note on {r.ticker}: {n}")
    lines.append("Peer medians: " + "; ".join(
        f"{label} {_fmt(median_of(live, attr), unit, places)}"
        for label, attr, unit, places in METRICS))
    prompt = "\n".join(lines)
    try:
        text = ctx.llm.complete(prompt, system=COMMENT_SYSTEM, max_tokens=600)
    except LLMUnavailable as e:
        log.warning("comps commentary skipped: %s", e)
        return "", []
    return text, ungrounded(text, prompt)


def build_brief(ctx: Context, tickers: list[str], *, label: str = "",
                peer_note: str = "") -> tuple[Brief, dict]:
    today = datetime.now(timezone.utc).date()
    title = f"Comps — {label or ', '.join(tickers)}"
    brief = Brief(title=title, agent=NAME, target=label or "-".join(t.lower() for t in tickers),
                  tags=["comps", "valuation"])
    for d in ctx.base_degradations():
        brief.degrade(d)
    if peer_note:
        brief.add("Peer set", f"{', '.join(tickers)}\n\n_{peer_note}_")

    rows, degradations = build_rows(ctx, tickers)
    for d in degradations:
        brief.degrade(d)

    text, unverified = commentary(ctx, rows)
    if text:
        if unverified:
            text += ("\n\n_Figures not found in the table above: "
                     + ", ".join(f"`{x}`" for x in unverified)
                     + ". The table is assembled from filings; this paragraph is model-written._")
            brief.extra_meta["ungrounded_figures"] = len(unverified)
        brief.add("Read", text)
    elif ctx.llm.available:
        brief.degrade("comps commentary omitted")

    brief.add("Comparable companies", comps_table(rows))
    brief.add("Enterprise value bridge", bridge_table(rows))

    footnotes = [f"- **{r.ticker}** — {n}" for r in rows for n in r.notes]
    if footnotes:
        brief.add("Why cells are blank", "\n".join(footnotes))

    priced = [r for r in rows if r.ev_ebitda is not None]
    brief.extra_meta["peers"] = ",".join(tickers)
    brief.extra_meta["ev_ebitda_coverage"] = f"{len(priced)}/{len(rows)}"
    brief.add("Method",
              "Market cap is the cover-page share count times the last trade. "
              "EV adds total debt, preferred and minority interest and subtracts "
              "cash and short-term investments. EBITDA is operating income plus "
              "D&A — never approximated from net income. Every multiple whose "
              "inputs are missing is left blank and footnoted; medians are taken "
              "over the peers that have the metric, so a blank never drags one.")
    brief.source("SEC EDGAR XBRL companyfacts", "https://data.sec.gov",
                 note=f"TTM fundamentals, as of {today}")
    brief.source("Yahoo Finance via yfinance", note="last trade for market cap")
    return brief, {"rows": rows, "tickers": tickers}


def run(ctx: Context, tickers: list[str], *, label: str = "", peer_note: str = "",
        commit: bool = True) -> AgentResult:
    started = datetime.now(timezone.utc)
    res = AgentResult(agent=NAME, target=label or ",".join(tickers))
    try:
        brief, data = build_brief(ctx, tickers, label=label, peer_note=peer_note)
    except Exception as e:  # noqa: BLE001
        log.exception("comps failed")
        res.ok, res.error = False, f"{type(e).__name__}: {e}"
        record(ctx.db, Run(agent=NAME, ok=False, error=res.error,
                           started_at=started.timestamp()))
        return res

    res.brief = brief
    finalize(ctx, brief, res)
    res.artifact = brief.write(ctx.cfg.out_dir / NAME)
    if commit:
        try:
            cr = ctx.notes.commit_file(f"comps/{brief.filename}", brief.render(),
                                       f"comps {res.target} {brief.date}")
            res.data["commit"] = {"sha": cr.sha, "committed": cr.committed}
        except Exception as e:  # noqa: BLE001
            res.degrade(f"could not commit comps: {e}")

    rows = data["rows"]
    with_ev = [r for r in rows if r.enterprise_value is not None]
    with_mult = [r for r in rows if r.ev_ebitda is not None]
    res.summary = (f"{len(rows)} peers, {len(with_ev)} with an EV bridge, "
                   f"{len(with_mult)} with EV/EBITDA")
    res.data["median_ev_ebitda"] = median_of(rows, "ev_ebitda")
    record(ctx.db, Run(agent=NAME, target=res.target, ok=True, artifact=str(res.artifact),
                       summary=res.summary, degradations=res.degradations,
                       started_at=started.timestamp(),
                       duration_s=(datetime.now(timezone.utc) - started).total_seconds()))
    return res
