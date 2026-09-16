# Benchmarks

Every gate below is enforced by a test marked `@pytest.mark.benchmark`, so
"passes the benchmarks" means `pytest -m benchmark` is green, not that someone
eyeballed the output. Run them with:

```
./tests/run_all.sh          # everything
.venv/bin/pytest -m benchmark   # just the gates
```

Each gate names the failure it exists to prevent. Several of them were written
*after* the bug they describe was found in this codebase; those are marked
**(regression)**.

## Cross-cutting

| # | Gate | Threshold |
|---|------|-----------|
| X1 | Every agent completes with no network and no LLM | 5/5 produce a non-empty artifact, none raise. The one exception is stated: a backtest with no model has no strategy code, and fails by name rather than inventing one |
| X2 | A degraded run says so | `degraded: true` in frontmatter **and** a `> Ran degraded` banner **and** the reasons in the run log |
| X3 | One run, one row | each agent invocation writes exactly one `runs` row, success or failure |
| X4 | No credential leaks | no rendered brief, run log row or API response contains the LLM key |
| X5 | Sources are attributed | every brief with data has a Sources section or an explicit degradation saying why not |
| X6 | The dashboard serves only its own output | five path-traversal shapes against `/raw/` return 404, and no response body contains the `.env` or `/etc/passwd` |

## 1 · Research agent

| # | Gate | Threshold |
|---|------|-----------|
| R1 | TTM is four **contiguous** quarters | exact match against a hand-computed fixture |
| R2 | A missing fiscal Q4 is reconstructed from (FY − 9M YTD) **(regression)** | NVDA-shaped facts yield 253.49B, not the 229.43B a gap-jumping scan returns |
| R3 | A stale tag never wins **(regression)** | given a 2020-vintage tag and a current one, the current one is chosen |
| R4 | An unfillable gap yields no number | `revenue_ttm is None` plus a note — never a 15-month "TTM" |
| R5 | Multi-class and stale share counts are refused **(regression)** | Berkshire-shaped facts omit market cap rather than using a 2011 count |
| R6 | Model figures are checked against the dossier | a fabricated figure is listed under "Unverified figures"; a grounded one is not |
| R7 | Brief structure | Snapshot, filings table, sources, parseable frontmatter |
| R9 | A brief that lost a section says so **(regression)** | a real MSFT brief shipped with Thesis only and `degraded: false`; missing sections are now named, and a reply cut off at the token cap is distinguished from a model that ignored the headings |
| R8 | The reader sees what the model saw | every performance figure in the dossier is also in the Snapshot table — a cited number the reader cannot check on the page is worse than one never offered |

## 2 · Backtest agent

| # | Gate | Threshold |
|---|------|-----------|
| B1 | The overnight gap belongs to the book, not the entry **(regression)** | on a synthetic series where the gap is predictable from the prior close, a gap-chasing strategy earns **0.0** — the old attribution earned all of it |
| B2 | Signals cannot act on their own bar | `sign(today's return)` on a random walk yields \|Sharpe\| < 1.0 |
| B3 | Costs are charged, exactly | net = gross − turnover × (commission + slippage)/1e4, to 1e-12 |
| B4 | Metrics are correct | Sharpe, CAGR, max drawdown match closed-form values on a constructed series |
| B5 | Generated code is contained | no network, no filesystem, non-root, memory and wall-clock capped; a network attempt fails inside the sandbox |
| B6 | Isolation is never overstated | a subprocess fallback run is labelled `isolation: subprocess` in the brief and the metadata |
| B7 | Broken code is repaired, then given up on | a failing strategy triggers ≤ `max_repairs` regenerations and then reports failure honestly |

## 3 · Market open briefing

| # | Gate | Threshold |
|---|------|-----------|
| M1 | Renders with every source dead | futures/macro/watchlist tables present, each cell `n/a`, degradations non-empty |
| M2 | Non-trading days are labelled | a Saturday brief carries the closed-market note |
| M3 | Only today's earnings | a symbol reporting tomorrow does not appear |
| M4 | Movers are ranked by absolute move | −4% outranks +1% |
| M5 | Instruments are quoted in their own units **(regression)** | a 4.66→4.70 move on the 10y renders `+4bp`, never `+0.86%` — a model read the percent form as 92 basis points |
| M6 | The lede is checked against its own inputs | figures the lede uses that appear in neither the tape nor the headline titles are named under it; ones drawn from a headline are not, or the warning becomes noise |
| M7 | Curve spreads are basis points | a 4.67→5.00 gap renders `+33bp`, never the 7% a percentage change would give |
| M8 | The feed's display duplicate is not a second tenor | `BC_30YEARDISPLAY` repeats `BC_30YEAR`; the curve holds one 30Y |
| M9 | FOMC dates are scraped, and a trailing note keeps its own year **(regression)** | the page closes each year's table with a meeting in the year *after* next; read under the enclosing heading it became a phantom meeting twelve months early |
| M10 | A meeting inside the front contract's month is declared a blend | ZQ settles to a monthly average, so the spot anchor straddles the move and understates everything measured against it |
| M11 | The push leads with levels and carries the talking point | levels, then the model's read, then the point — the order they stop being worth reading on a lock screen |
| M12 | An unconfigured or failed push is reported, never silent | a phone that did not buzz is a degradation on the run, not an absence of one |

## 3b · Enterprise value bridge

| # | Gate | Threshold |
|---|------|-----------|
| V1 | The most recent instant wins, not the first tag listed **(regression)** | MSFT's combined-debt tag stops in 2015 and JPM's cash tag in 2018; first-match-wins gave MSFT $31.8B against a true $40.3B |
| V2 | A subtotal is never added to its own component | `LongTermDebt` already includes current maturities; adding `LongTermDebtCurrent` double-counts the maturity wall |
| V3 | A stale component is dropped, not added as a zero **(regression)** | Apple's `ShortTermBorrowings` is a zero last filed in 2018; adding it asserts a line nobody filed |
| V4 | A split-D&A filer still gets EBITDA **(regression)** | Microsoft tags `Depreciation` and `AmortizationOfIntangibleAssets` separately and no combined line, which silently cost it its EBITDA |
| V5 | Share classes that are not economically equal refuse one count | Berkshire's B is 1/1500 of an A; the derived count times the wrong class's price is off by that ratio |

## 4 · Internship scout

| # | Gate | Threshold |
|---|------|-----------|
| S1 | The diff is the product | run 1 surfaces N, an identical run 2 surfaces 0, one added posting surfaces exactly 1 |
| S2 | The diff is atomic | a batch that fails mid-write leaves no partially-remembered keys |
| S3 | Identity survives edits | a posting whose title and location change is still not "new" |
| S4 | Relevance filter | recruiting/sales/legal internships score below threshold; quant/SWE internships above |
| S5 | A dead board is reported | a board returning nothing appears in Coverage and in the degradations |
| S6 | An empty diff does not blame the model **(regression)** | with nothing new the brief says "no new postings to rank", never "unavailable" — and a genuinely absent model still is reported |
| S7 | Identity in the query string is preserved **(regression)** | six firms point every posting at one careers page and differ only by `gh_jid`/`id`/`token`; stripping the query collapsed 107 Jump Trading roles to one key. Tracking params are dropped, identifying ones kept, and an unresolvable verdict url is dropped rather than attached to the wrong row |
| S8 | Nothing is claimed that is not shown **(regression)** | with more qualifying postings than the display cap, only the shown ones are marked seen and the brief states how many are queued |
| S9 | Every displayed row is ranked **(regression)** | the ranking cap matches the display cap, and a genuinely partial answer says "for N of M rows" rather than leaving silent dashes |
| S12 | The digest leads with a shortlist | the day's `apply` verdicts, best first, in one section small enough to read or retrieve whole; a 129-row table is fourteen chunks and neither a reader nor a retriever starts at the top |
| S11 | The day's digest only grows **(regression)** | the brief renders the day's union rather than one run's delta, so a second run cannot shrink it. The 104-match digest was replaced first by a run finding none, then by one finding the 29 the display cap had queued — 100 rows became 29. Verdicts are stored with the posting so later runs keep them |
| S10 | Ranking batches | 60 postings produce 3 model calls and 60 verdicts; one unusable batch costs only its own 25. A single completion for a hundred verdicts truncates mid-array and parses as nothing, so a raised cap would cost *every* verdict. `max_tokens` is measured off the urls in the batch rather than assumed: a flat 80-a-verdict budget fitted to Greenhouse urls blew the cap on the first Workday/Oracle sweep and cost 75 rows their verdict |
| S13 | The registry reaches past Greenhouse | Cantor Fitzgerald and its adjacent desks are covered, and every sector has at least two boards. Banks, brokers, exchanges and enterprise IT run on Workday / Oracle Recruiting / Eightfold, so three-vendor coverage could not see them at all |
| S14 | Days open counts from first publication **(regression risk)** | the age comes from Greenhouse's `first_published`, not `updated_at` — the latter moves whenever a recruiter touches the requisition and would report a role open since March as posted today |
| S15 | The age is shown, never implied | shortlist and table both carry it; a board that publishes no date renders "—" and the brief says why, because a blank cell in a numeric column reads as zero |
| S16 | Freshness breaks score ties under the cap | with the display cap binding, an equally-scoring posting opened yesterday displaces one opened last year, and the older one stays queued rather than being dropped |
| S17 | Keyword scoring matches words, not substrings **(regression)** | "ai" inside "Retail" and "ml" inside "HTML" were worth two points each — enough to carry a retail-sales posting past the score gate and into a model call |
| S18 | Bank-shaped internship titles are recognised | "2027 Summer Analyst", "Technology Analyst Program", "Off-Cycle Analyst" count; "Senior Equity Research Analyst" and "Analyst, Investment Banking" do not. Missing this makes the broker boards contribute nothing; over-matching puts every senior analyst in the brief |
| S19 | A failed request is not reported as a dead board **(regression)** | the coverage row names the failure — "unreachable (network or timeout)", "HTTP 429", "no postings returned" — and one retry precedes writing a board off. Seven Workday/Oracle boards timed out at once on the first live sweep and all read as "no postings returned", which is the status that sends you probing ATS vendors for firms that had not moved |
| S20 | The range is the United States | an out-of-range posting never reaches the model: Singapore, Amsterdam, Bengaluru — and, since 2026-09-15, Toronto and London — are gone before a verdict is spent on them, while a posting listing several offices survives on the US one. The title is read as well as the location field, because Cloudflare files every internship under "In-Office" and puts the city in the title; a board that names no place at all is kept rather than guessed at, and counted in the brief. Only unambiguously foreign names are denied: Vancouver, Waterloo, Manchester and Birmingham all exist in the US too and are left to fall through |
| S22 | Only postings an undergraduate can take | a PhD, master's or MBA title is filtered before `rank`, on the same terms as an out-of-range one, and costs no verdict. `graduate` alone never gates: "Graduate Analyst Programme" and "new graduate" are what banks call the roles a final-year undergraduate applies to, and denying the bare word would delete the exact population the scout exists to find |
| S21 | Nothing open longer than three weeks is shown | past three weeks a requisition has collected the applications it will read. The boundary day is inside the window, a Workday "30+" age floor is outside it, and an undated posting is kept — a board that stops publishing dates should render em dashes, not go quiet |

## 5 · Personal RAG analyst

| # | Gate | Threshold |
|---|------|-----------|
| A1 | Retrieval recall | **recall@5 ≥ 0.9** over a gold set of question→file pairs (currently 10/10) |
| A2 | Hybrid beats either half | hybrid recall ≥ lexical-only and ≥ semantic-only on the same gold set |
| A3 | "last month" is a filter, not a hint | an older note that matches more words is excluded from a windowed query |
| A4 | An empty window is admitted, not silently widened | falling back to the whole index adds a degradation |
| A5 | Citations are real | every cited path exists in the index |
| A6 | The index is process-stable **(regression)** | embeddings built in one process match a query embedded in another (crc32, not salted `hash()`) |
| A7 | Search-only degradation | with no LLM, passages are still returned and labelled search-only |
| A8 | Retrieval balances breadth and depth **(regression)** | a cross-document question gets one passage from each of `k - k/4` documents before any gets a second; the remaining slots then follow score with no per-document cap, so a question about a document that *is* a long list can retrieve several slices of it. `_spread` never returns more than k |
| A9 | Hyphenated compounds match their parts **(regression)** | "moving-average crossover" shared no token with a brief describing a "20-day moving average", so the retriever returned that backtest's cost table instead of its strategy |

## 6 · EDGAR comps engine

| # | Gate | Threshold |
|---|------|-----------|
| C1 | EBITDA is never approximated from a missing leg | no operating income or no D&A means a blank cell, not a number built from net income |
| C2 | Unknown debt yields no enterprise value **(regression)** | an EV that treats unknown debt as zero is a market cap wearing a different label |
| C3 | Medians are taken over the peers that have the metric **(regression)** | a blank is not a zero; averaging blanks in drags the median toward a multiple no peer trades at |
| C4 | A negative denominator blanks the multiple | a loss-making company is not trading at −8x EBITDA |
| C5 | A model-resolved peer set is checked against the SEC ticker file | a plausible ticker belonging to something else is harder to spot in a finished table than a missing one |

## 7 · Deal book

| # | Gate | Threshold |
|---|------|-----------|
| D1 | The other meanings of "acquisition" are refused | defence procurement, land banks, customer acquisition and job titles outnumber real M&A in an unfiltered feed |
| D2 | Newswire site furniture is stripped before any filter **(regression)** | PR Newswire renders its whole industry taxonomy into every page, so a keyword filter matched every sector on every release |
| D3 | A business-scale figure is not a purchase price **(regression)** | "$1 billion in annualized marketplace sales" was recorded as the price of four brands |
| D4 | A foreign-currency deal is labelled, not counted as dollars | A$2.8B is not $2.8B, and the size floor is not applied to it |
| D5 | Enrichment is spent only on fetchable links **(regression)** | ranked on parties-and-price alone, all 22 slots went to aggregator links whose bodies cannot be fetched, and 0 to the wires |
| D6 | Bank names survive the full stops inside them **(regression)** | "Goldman Sachs & Co. LLC" returned "LLC" under a sentence pattern bounded by `[^.]` |
| D7 | Your annotation survives every later sweep | the agent refreshes what it knows; `my_view`, `status` and `reviewed_at` are never written by a sweep |
| D8 | A watched advisor or sector overrides the size floor | the floor keeps tuck-ins out; the firm you are interviewing with is worth a page at any size |

## 6 · Performance

| # | Gate | Threshold |
|---|------|-----------|
| P1 | Index build | ≥ 200 chunks/second |
| P2 | Query latency | < 150 ms median over the gold set |
| P3 | Brief render | < 25 ms for a full research brief |
| P4 | EDGAR requests per ticker | ≤ 3 network fetches for a cold profile+fundamentals (was 9 before companyfacts) |
