# agents_work

Five agents that do research work on a schedule, share one run log, and are
honest about what they could not reach.

| Agent | What it does | How it runs |
|---|---|---|
| `research` | Ticker or watchlist in; a one-page brief (thesis, risks, valuation context) out, committed to a git repo of research notes | on demand |
| `backtest` | A strategy idea in plain English → generated code → a sandboxed run → Sharpe, max drawdown, equity curve | on demand |
| `briefing` | Futures, macro, watchlist movers and today's earnings, before the bell | systemd timer, 08:00 Mon–Fri |
| `scout` | Quant / fintech / AI job boards swept nightly; only what is new since the last run | systemd timer, 03:30 nightly |
| `analyst` | Questions answered over your notes *and* the briefs the other four wrote, with citations | on demand |

```bash
agents doctor                              # what works right now
agents research NVDA AAPL
agents backtest "buy when 20d crosses above 50d, flat otherwise" --symbols SPY,QQQ
agents briefing
agents scout
agents ask "what did I conclude about NVDA last month"
agents status                              # recent runs
```

Dashboard: **http://192.168.1.149:8110** — read-only, one card per agent, the
run log, and the notes-repo commit history.

## The idea

Every one of these is mostly made of other people's systems: EDGAR, a price
lake, Google News, twenty ATS boards, an LLM endpoint. So the design question
is not "does it work" but "what does it do at 08:00 on the morning one of them
is down".

The answer everywhere is the same: **assemble facts first, let the model write
into the document last, and name every gap.** A brief is a data structure with
prose slotted into it, not prose with data mentioned in it. That ordering is
what makes an LLM outage cost you the analysis paragraphs instead of the
document, and it is why every agent can run with no network and no model at all
(gate X1 in [BENCHMARKS.md](BENCHMARKS.md)).

Concretely, a degraded run says so in three places that have to agree: a
`degraded: true` flag in the brief's frontmatter, a banner at the top of the
page, and the reasons stored on the row in the run log.

## What each agent is actually careful about

**research** — EDGAR is the only free financial source that is contractually
stable, and it is still a minefield. Two of its traps produce *confident wrong
numbers* rather than errors, so both are pinned by regression tests:

- Filers migrate between XBRL revenue tags and EDGAR keeps serving the
  abandoned one. NVDA stopped using `RevenueFromContractWithCustomer...` after
  FY2022; first-tag-wins reported FY2020 revenue in a 2026 brief — 20× too low
  and entirely plausible. Tags are now ranked by *which one yields a current
  TTM*.
- Most fiscal-year filers never tag Q4 on its own; it exists only as
  (FY − 9M year-to-date). A greedy scan jumps that hole and returns a
  fifteen-month "TTM". Missing quarters are reconstructed by subtraction, the
  four quarters are checked for contiguity, and a hole that cannot be filled
  yields *no number and a note* rather than a shorter window presented as a year.

Figures the model writes are then checked back against the dossier it was
given, and unmatched ones are listed under **Unverified figures** — derived or
invented, the brief cannot tell which, so it says that.

**backtest** — the model writes a *signal function only*. Lag, next-open fill
and costs live in the harness, so the classic LLM-backtest failure (trading on
the bar it decided on) is structurally impossible rather than something to
review for. The generated code runs with `--network none`, a read-only root, a
non-root user, `--cap-drop ALL`, a memory cap and a wall clock. Without Docker
it still runs, in an rlimited subprocess, and the brief says `isolation:
subprocess` so nobody mistakes it for the real thing.

**briefing** — every table renders whether or not its source answered; a dead
source is an `n/a` cell plus a named degradation, never a missing section.
Instruments are quoted in their own units, because given "US 10y yield: +0.92%"
a model opened a brief with "yields spiked 92 basis points" — the move was four.
The lede is the only model-written part of the page, so it is the only part that
can be wrong about a number the tables got right: its figures are checked
against the tape *and* the headline titles it was shown, and unsupported ones
are named beneath it. Checking against the tape alone would flag "disappointing
Walmart earnings" as invented when a Reuters headline says exactly that.

**scout** — the diff is the product. A scout that re-lists yesterday's ninety
postings stops being read. The diff runs in one transaction, so a crash cannot
half-remember a batch and silently drop those postings out of tomorrow's list
too, and identity is the ATS URL rather than the title, which recruiters edit
in place.

**analyst** — retrieval is hybrid (BM25 + a local hashed embedding) because
keyword search alone cannot answer "what did I *conclude*" about a note that
never uses the word, and time is a filter applied *before* ranking rather than
a phrase in the prompt. An answer is built from twelve passages, spread across
documents before any document gets a second slot: the corpus these agents write
is one document per subject, so a question about "the watchlist" is a question
about seven files at once. A narrower window covered six of eleven documents and
dropped a name out of an answer that claimed to cover all of them — and a
document that was never retrieved cannot be reported as missing, so an omission
reads exactly like an absence. Embeddings use `crc32`, not the builtin `hash()`, which
is salted per process — an index built by the timer would not have matched a
query typed at the CLI.

## Layout

```
agents_work/
  agents/       research · backtest · briefing · scout · analyst
  sources/      edgar · prices · news · jobs
  web/          read-only FastAPI dashboard
  brief.py      the document every agent emits
  grounding.py  checks model figures against the dossier
  netcache.py   cache-first, rate-limited, never-raises HTTP
  store.py      the run log and the seen-postings diff
  gitsink.py    commits briefs to git
sandbox/        the container the generated backtest code runs in
deploy/         systemd units + install.sh
tests/          219 tests, 59 of them benchmark gates
```

## Setup

```bash
uv venv --python 3.13 && uv pip install --python .venv -e ".[dev]"
cp .env.example .env          # then fill in AGENTS_LLM_API_KEY
docker build -t agents-backtest-sandbox:latest sandbox
./tests/run_all.sh
sudo ./deploy/install.sh
```

`agents doctor` prints which capabilities are live. Everything has a default
that works without a secret, so a missing key is a smaller answer, not a crash.

### The notes repo

Briefs are committed to `data/research-notes` on every run. Set
`AGENTS_GIT_REMOTE` and the same commits get pushed — the agent code does not
change, which is why the sink is a seam and not an inline `git` call inside the
research agent. Point it anywhere:

```bash
scripts/link-notes-remote.sh git@github.com:you/research-notes.git
GITEA_TOKEN=... scripts/link-gitea.sh <user> research-notes  # creates the repo too
```

A dead remote never loses a commit: the commit is already local, and the push
failure becomes a degradation on the run.

## Tests

`./tests/run_all.sh` — 219 tests, no network, no LLM, no real data directory.
[BENCHMARKS.md](BENCHMARKS.md) lists the 59 gates and the failure each one
exists to prevent; several were written after finding that failure here.
