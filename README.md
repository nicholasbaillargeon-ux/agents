# agents_work

Eight agents that do research work on a schedule, share one run log, and are
honest about what they could not reach.

| Agent | What it does | How it runs |
|---|---|---|
| `research` | Ticker or watchlist in; a one-page brief (thesis, risks, valuation context) out, committed to a git repo of research notes | on demand |
| `backtest` | A strategy idea in plain English → generated code → a sandboxed run → Sharpe, max drawdown, equity curve | on demand |
| `briefing` | The morning tape: the Treasury curve and the day's move in basis points, what fed funds futures price for the next three FOMC meetings, overnight M&A, futures, movers and today's earnings — plus one talking point, pushed to a phone | systemd timer, 06:30 Mon–Fri |
| `scout` | 84 quant, bank, broker, exchange, fintech, AI and enterprise-IT boards swept nightly; only what is new since the last run, with how long each has been open | systemd timer, every 3h at :30 |
| `comps` | A peer set in; a comparable-companies table out — market cap, an enterprise value bridged through debt, cash, preferred and minorities, and the multiples that follow, every cell traced to a filed XBRL fact | on demand, plus a weekly set |
| `dealbook` | M&A feeds watched for your filters (size, sector, or an advisor you are watching); a structured one-pager drafted into Postgres for each match, for you to annotate | systemd timer, 07:15 and 17:15 Mon–Fri |
| `ainews` | Nine AI feeds — the labs' own blogs, the trade press and one news query — folded into one row per event, ordered by what it changes, with a model's line on why each matters, and the top three pushed to a phone | systemd timer, 07:00 daily |
| `analyst` | Questions answered over your notes *and* the briefs the other seven wrote, with citations | on demand |

```bash
agents doctor                              # what works right now
agents research NVDA AAPL
agents backtest "buy when 20d crosses above 50d, flat otherwise" --symbols SPY,QQQ
agents briefing                            # the morning tape
agents comps --set advisory                # or: agents comps --peers "mid-cap asset managers"
agents comps --list-sets
agents deals                               # sweep the M&A feeds
agents deals --list                        # the book; ! = watched advisor, * = you have a view
agents deals --show 7
agents deals --note 7 "Multiple looks full versus the 2024 comp."
agents scout
agents ainews                              # the daily AI brief
agents ainews --no-llm --hours 48          # stories and ordering only, wider window
agents ask "what did I conclude about NVDA last month"
agents status                              # recent runs
```

Dashboard: **http://192.168.1.149:8110** — read-only, one card per agent, the
run log, and the notes-repo commit history. The deal book is at
**/deals**, rendered from Postgres with your annotations.

## The three newest, and what is actually hard about them

**The morning tape** is a section of the briefing rather than an agent of its
own, because the briefing already owns 06:30: a second agent covering the same
minute would duplicate the timer, the dashboard row and the notes commit, and
hand you two documents to read instead of one.

Its two non-obvious choices are both about not hardcoding a number that goes
stale. Fed expectations come from the 30-day fed funds futures strip rather
than a scraped FedWatch figure — ZQ settles to the month's average effective
rate, so `100 − price` *is* the market's priced path, and the front contract
doubles as a market-derived reading of where policy sits today, with no target
range written down anywhere to go wrong after the next meeting. And the FOMC
calendar is scraped from the Fed's own page, because a table of meeting dates
baked into source is correct for about a year and then silently points at the
past.

**The comps engine** is mostly a fight with XBRL. EDGAR is the best free
financial source there is and it is only nominally standardised: filers migrate
between tags and EDGAR serves the abandoned one forever, banks report no
operating income line, Microsoft splits depreciation from amortisation and tags
no combined figure, Meta files its share count per class. So every lookup is
ranked by *which tag yields a current figure*, alternatives are never summed
(adding `LongTermDebt` to `LongTermDebtCurrent` double-counts the maturity
wall), and a peer whose filings do not support a multiple gets an empty cell
and a footnote. One invented cell in a comps table is invisible and moves the
median everyone reads off it.

**The deal book** draws the line at the annotation. A sweep refreshes what the
agent knows about a deal and never writes `my_view`, `status` or `reviewed_at`
— a library of fifteen deals you have formed a view on is the product, and a
library of fifteen deals a model summarised is not.

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

## The AI brief, and why it is not just nine feeds printed

Two problems, and neither is fetching.

**One event arrives nine times.** The lab posts it, the trade press rewrites
it, and Google News indexes both. A list of items is not a list of stories, so
headlines are clustered on the tokens that identify an *event* — and the hard
part is that the lab's own name is the least informative token in a feed about
the lab. Two headlines sharing only "Anthropic" and "Claude" are two stories
about one company, which is how a warning about a product got merged into that
product's launch and left the page. Tokens carried by a tenth of the day's
headlines therefore stop counting as identity, measured per sweep rather than
written down, because which names are everywhere changes weekly. Every headline
in a cluster is printed under its story: clustering is a heuristic, and a wrong
merge should be a visible duplicate line rather than an event that quietly
vanished.

**"AI" is the most overloaded token in a news feed.** A parish task force, a
Coast Guard research hub and "3 AI Stocks to Buy" all match it, and on the
first live sweep they outnumbered the model releases. The gate is whole-word
matching — "ai" is inside Dubai, chair, said and email — plus a noise list
fitted to what actually came back, plus a short list of outlets whose entire
output is retail-investor content and whose headlines carry no noise word at
all.

Ordering is deterministic and the model never touches it: release and legal
language in the headline, a model family next to a version number, how many
independent outlets carried it, and whether the lab announced it itself. The
model writes a lede, one line per top story, and a watch list — and every
figure in that prose is checked back against the story list it was given.

The lede and the top three stories go to a phone, after the brief is written
and committed and never before: a notification about a brief that then failed
to persist is the only copy the reader ever saw. It carries **no links** — one
Google News redirect token is five hundred characters, and three of them would
truncate the notification before the news got into it — so the tap target is
the brief itself on the dashboard. Priority 3, where the morning tape is 4,
because a daily brief that buzzes like a market alert teaches you to silence
the topic they share.

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
in place. Coverage is seven ATS vendors, not three: banks, brokers, exchanges
and enterprise IT firms run on Workday, Oracle Recruiting and Eightfold, so a
Greenhouse/Lever/Ashby registry cannot see Cantor Fitzgerald or Nasdaq at all.
Every row carries how long the posting has been open, counted from the board's
own publication date — Greenhouse's `first_published` rather than `updated_at`,
which moves whenever a recruiter edits the requisition.

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
  agents/       research · backtest · briefing · scout · analyst · comps
                dealbook · ainews · tape
  sources/      edgar · prices · news · jobs · rates · deals · aifeeds
  web/          read-only FastAPI dashboard
  brief.py      the document every agent emits
  grounding.py  checks model figures against the dossier
  netcache.py   cache-first, rate-limited, never-raises HTTP
  store.py      the run log and the seen-postings diff
  gitsink.py    commits briefs to git
sandbox/        the container the generated backtest code runs in
deploy/         systemd units + install.sh
tests/          565 tests, 139 of them benchmark gates
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

`./tests/run_all.sh` — 565 tests, no network, no LLM, no real data directory.
[BENCHMARKS.md](BENCHMARKS.md) lists the 98 gates and the failure each one
exists to prevent; several were written after finding that failure here. The
runner also cross-references the two: a gate documented with no test behind it
fails the run, because it reads as covered and is not.
