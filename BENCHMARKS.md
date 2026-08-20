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
| S11 | An empty run does not destroy the day's findings **(regression)** | a second run of the same day that surfaces nothing keeps the earlier brief; the run that surfaced 104 roles was replaced by one that surfaced none, leaving them only in git where the analyst cannot see them |
| S10 | Ranking batches | 60 postings produce 3 model calls and 60 verdicts; one unusable batch costs only its own 25. A single completion for a hundred verdicts truncates mid-array and parses as nothing, so a raised cap would cost *every* verdict |

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

## 6 · Performance

| # | Gate | Threshold |
|---|------|-----------|
| P1 | Index build | ≥ 200 chunks/second |
| P2 | Query latency | < 150 ms median over the gold set |
| P3 | Brief render | < 25 ms for a full research brief |
| P4 | EDGAR requests per ticker | ≤ 3 network fetches for a cold profile+fundamentals (was 9 before companyfacts) |
