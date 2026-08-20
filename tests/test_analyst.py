"""Personal RAG analyst: retrieval quality, time filtering, degradation."""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
from datetime import date

import pytest
from conftest import GOLD

from agents_work.agents import analyst
from agents_work.agents.analyst import (Hit, Index, ask, chunk_markdown, embed, time_window,
                                        tokenize)
from agents_work.llm import FakeLLM


@pytest.fixture
def index(cfg, vault) -> Index:
    idx = Index(cfg.data_dir / "analyst.db")
    idx.build([vault])
    yield idx
    idx.close()


def _recall_at_k(index: Index, k: int = 5, **kw) -> float:
    found = 0
    for question, expected in GOLD:
        hits = index.search(question, k=k, **kw)
        if any(h.path == expected for h in hits):
            found += 1
    return found / len(GOLD)


# -- retrieval quality -------------------------------------------------------

@pytest.mark.benchmark
def test_retrieval_recall_at_5(index):
    """A1: recall@5 >= 0.9 over the gold set."""
    recall = _recall_at_k(index, k=5)
    assert recall >= 0.9, f"recall@5 = {recall:.2f}"


@pytest.mark.benchmark
def test_hybrid_retrieval_beats_either_half_alone(index):
    """A2. Keyword search misses "what did I conclude" because the note never
    says "conclude"; a hashed embedder alone is too blunt to rank the rest."""
    hybrid = _recall_at_k(index, k=3, alpha=0.5)
    lexical = _recall_at_k(index, k=3, alpha=1.0)
    semantic = _recall_at_k(index, k=3, alpha=0.0)
    assert hybrid >= lexical and hybrid >= semantic, (
        f"hybrid {hybrid:.2f} lexical {lexical:.2f} semantic {semantic:.2f}")


def test_top_hit_is_usually_the_right_note(index):
    top1 = _recall_at_k(index, k=1)
    assert top1 >= 0.7, f"recall@1 = {top1:.2f}"


def test_search_returns_nothing_for_an_unrelated_question(index):
    assert index.search("zzzqqq unrelated aardvark", k=5) == [] or all(
        h.score < 0.5 for h in index.search("zzzqqq unrelated aardvark", k=5))


# -- time as a filter --------------------------------------------------------

@pytest.mark.benchmark
def test_last_month_excludes_the_older_note(index):
    """A3. Both NVDA notes match the words; only one is inside the window."""
    since, _ = time_window("what did I conclude about NVDA last month",
                           today=date(2026, 8, 20))
    assert since == "2026-07-21"
    paths = [h.path for h in index.search("what did I conclude about NVDA", k=5, since=since)]
    assert "2026-08-14-nvda.md" in paths
    assert "2026-07-02-nvda-doubts.md" not in paths


def test_unqualified_questions_have_no_window():
    assert time_window("what do I think about NVDA") == (None, None)


@pytest.mark.parametrize("phrase,expected_days", [
    ("last week", 7), ("yesterday", 1), ("last quarter", 92),
    ("last year", 365), ("recently", 21),
])
def test_time_windows_are_recognised(phrase, expected_days):
    since, _ = time_window(f"what changed {phrase}", today=date(2026, 8, 20))
    assert since == (date(2026, 8, 20) - __import__("datetime").timedelta(
        days=expected_days)).isoformat()


@pytest.mark.benchmark
def test_empty_window_falls_back_but_says_so(ctx, index):
    """A4. Silently widening a window is how a stale answer looks current."""
    answer = ask(ctx, "what did I conclude about banks yesterday", index=index,
                 today=date(2026, 8, 20))
    assert answer.hits
    assert any("window" in d for d in answer.degradations)


# -- answers -----------------------------------------------------------------

@pytest.mark.benchmark
def test_citations_point_at_indexed_files(ctx, index, vault):
    """A5."""
    answer = ask(ctx, "what did I conclude about NVDA", index=index)
    assert answer.citations
    for path in answer.citations:
        assert (vault / path).is_file()


@pytest.mark.benchmark
def test_search_only_mode_when_the_model_is_down(cfg, ctx, index):
    """A7. Retrieval worked; ship the passages rather than nothing."""
    ctx.llm = FakeLLM(cfg, available=False)
    answer = ask(ctx, "what did I conclude about NVDA", index=index)
    assert answer.search_only
    assert answer.hits
    assert "backlog" in answer.text
    assert any("search-only" in d for d in answer.degradations)


def test_answer_uses_the_model_when_available(ctx, index):
    ctx.llm.default_response = "You concluded the backlog is the thesis [2026-08-14-nvda.md]."
    answer = ask(ctx, "what did I conclude about NVDA", index=index)
    assert not answer.search_only
    assert "backlog is the thesis" in answer.text
    assert "Excerpts from the user's notes" in ctx.llm.prompts[0]


def test_no_hits_is_stated_plainly(ctx, index):
    answer = ask(ctx, "qqzz nonexistent topic wombat", index=index, k=0)
    assert answer.search_only
    assert "Nothing in the indexed notes" in answer.text


# -- index mechanics ---------------------------------------------------------

@pytest.mark.benchmark
def test_embeddings_are_stable_across_processes():
    """A6 (regression). Python's builtin hash() is salted per process, so an
    index built by the timer would not match a query typed at the CLI."""
    here = embed("datacenter backlog margin")
    out = subprocess.run(
        [sys.executable, "-c",
         "from agents_work.agents.analyst import embed;"
         "print(sum(embed('datacenter backlog margin')))"],
        capture_output=True, text=True, check=True)
    assert float(out.stdout) == pytest.approx(sum(here), abs=1e-6)


def test_build_reports_missing_roots(cfg, tmp_path):
    idx = Index(cfg.data_dir / "analyst.db")
    stats = idx.build([tmp_path / "does-not-exist"])
    idx.close()
    assert stats["chunks"] == 0
    assert stats["skipped_roots"]


def test_rebuild_replaces_rather_than_duplicates(cfg, vault):
    idx = Index(cfg.data_dir / "analyst.db")
    first = idx.build([vault])
    second = idx.build([vault])
    assert first["chunks"] == second["chunks"] == idx.count()
    idx.close()


def test_chunker_splits_on_headings_and_keeps_the_date(vault):
    path = vault / "2026-08-14-nvda.md"
    chunks = chunk_markdown(path, vault)
    assert chunks
    assert all(c.note_date == "2026-08-14" for c in chunks)
    assert all("---" not in c.content.split("\n")[0] for c in chunks)


def test_chunker_survives_an_unreadable_file(tmp_path):
    missing = tmp_path / "gone.md"
    assert chunk_markdown(missing, tmp_path) == []


def test_tokenizer_drops_stopwords_and_singletons():
    assert tokenize("What did I do about the NVDA margin?") == ["nvda", "margin"]


def test_run_records_and_writes_an_artifact(ctx, cfg):
    res = analyst.run(ctx, "what did I conclude about NVDA", reindex=True)
    assert res.ok
    assert res.artifact and res.artifact.is_file()
    assert res.data["hits"] > 0


# -- performance -------------------------------------------------------------

@pytest.mark.benchmark
def test_index_build_throughput(cfg, tmp_path):
    """P1: >= 200 chunks/second."""
    big = tmp_path / "big"
    big.mkdir()
    body = ("Some analysis of the position and the thesis behind it. " * 30)
    for i in range(400):
        (big / f"2026-0{i % 9 + 1}-01-note{i}.md").write_text(
            f"---\ndate: 2026-0{i % 9 + 1}-01\n---\n\n# Note {i}\n\n{body}\n")
    idx = Index(cfg.data_dir / "perf.db")
    t0 = time.perf_counter()
    stats = idx.build([big])
    elapsed = time.perf_counter() - t0
    rate = stats["chunks"] / elapsed
    idx.close()
    assert stats["chunks"] >= 400
    assert rate >= 200, f"{rate:.0f} chunks/s"


@pytest.mark.benchmark
def test_query_latency(index):
    """P2: median query under 150 ms."""
    timings = []
    for question, _ in GOLD:
        t0 = time.perf_counter()
        index.search(question, k=6)
        timings.append((time.perf_counter() - t0) * 1000)
    median = statistics.median(timings)
    assert median < 150, f"median {median:.0f} ms"


# -- A8: retrieval spreads across documents ----------------------------------

@pytest.mark.benchmark
def test_retrieval_spreads_across_documents(cfg, tmp_path):
    """A8 (regression). Asked to compare seven briefs, pure score order returned
    two chunks of one brief and a headline list, and omitted the only two names
    that carried the multiple the question was about — so the answer reported
    that the corpus contained one.

    Built on its own corpus: every document answers the question, and one of
    them answers it in several places, which is exactly when score order
    collapses onto a single file.
    """
    vault = tmp_path / "briefs"
    vault.mkdir()
    for name in ("nvda", "aapl", "msft", "qqq", "spy", "tlt"):
        sections = "\n\n".join(
            f"## {heading}\n\nThe valuation multiple for {name.upper()} against its "
            f"own growth rate, discussed at length under {heading.lower()}."
            for heading in ("Thesis", "Valuation context", "Risks"))
        (vault / f"2026-08-20-{name}.md").write_text(
            f"---\ndate: 2026-08-20\n---\n\n# {name.upper()}\n\n{sections}\n")

    idx = Index(cfg.data_dir / "spread.db")
    idx.build([vault])
    hits = idx.search("which name is most expensive relative to its own growth", k=8)
    idx.close()

    paths = [h.path for h in hits]
    assert len(paths) == 8
    assert len(set(paths)) == 6, (
        f"all six documents should be represented, got {len(set(paths))}: "
        f"{sorted(set(paths))}")


@pytest.mark.benchmark
def test_spread_prefers_breadth_then_score():
    """A8: k documents represented before any document gets a second slot."""
    from agents_work.agents.analyst import _spread
    hits = [Hit("a.md", "h1", "c", "", 0.90), Hit("a.md", "h2", "c", "", 0.85),
            Hit("a.md", "h3", "c", "", 0.80), Hit("b.md", "h1", "c", "", 0.70),
            Hit("c.md", "h1", "c", "", 0.60)]
    assert [h.path for h in _spread(hits, 3, per_file=2)] == ["a.md", "b.md", "c.md"]
    # slots left over after every document is represented go to the best chunks
    got = _spread(hits, 4, per_file=2)
    assert sorted(h.path for h in got) == ["a.md", "a.md", "b.md", "c.md"]
    assert [h.score for h in got] == sorted((h.score for h in got), reverse=True)


def test_spread_respects_the_per_file_cap():
    from agents_work.agents.analyst import _spread
    hits = [Hit("a.md", str(i), "c", "", 1.0 - i / 10) for i in range(6)]
    assert len(_spread(hits, 4, per_file=2)) == 4      # falls back when documents run out
    assert len({id(h) for h in _spread(hits, 4, per_file=2)}) == 4


def test_spread_handles_degenerate_inputs():
    from agents_work.agents.analyst import _spread
    assert _spread([], 5, 2) == []
    assert _spread([Hit("a.md", "", "c", "", 1.0)], 0, 2) == []
    one = [Hit("a.md", "", "c", "", 1.0), Hit("b.md", "", "c", "", 0.5)]
    assert len(_spread(one, 5, per_file=0)) == 2


@pytest.mark.benchmark
def test_a_single_document_question_still_gets_depth(cfg, tmp_path):
    """A8: breadth must not starve a question about one document. Asked what a
    backtest found, pure breadth returned one chunk of it and seven of
    everything else, so the answer could not describe the strategy."""
    vault = tmp_path / "corpus"
    (vault / "backtests").mkdir(parents=True)
    (vault / "briefs").mkdir()
    (vault / "backtests" / "2026-08-20-spy.md").write_text(
        "---\ndate: 2026-08-20\n---\n\n"
        "## Idea\n\nThe moving-average crossover strategy buys the 20-day over 50-day.\n\n"
        "## Results\n\nThe moving-average crossover produced a Sharpe of 0.64.\n\n"
        "## Cost drag\n\nThe moving-average crossover paid 4.8x turnover a year.\n\n"
        "## Method\n\nThe moving-average crossover fills at the next open.\n")
    for name in ("nvda", "aapl", "msft", "qqq", "spy", "tlt"):
        (vault / "briefs" / f"2026-08-20-{name}.md").write_text(
            f"---\ndate: 2026-08-20\n---\n\n## Thesis\n\n{name.upper()} valuation and "
            "growth, unrelated to any moving-average strategy.\n")

    idx = Index(cfg.data_dir / "depth.db")
    idx.build([vault])
    hits = idx.search("what did my moving-average crossover backtest find", k=8)
    idx.close()

    from_backtest = [h for h in hits if h.path.startswith("backtests/")]
    assert len(from_backtest) >= 2, (
        f"only {len(from_backtest)} of {len(hits)} chunks came from the document "
        f"the question named: {[h.path for h in hits]}")


def test_breadth_and_depth_split_is_proportional_to_k():
    from agents_work.agents.analyst import _spread
    hits = [Hit(f"{i}.md", "h", "c", "", 1.0 - i / 100) for i in range(20)]
    for k in (4, 6, 8, 12):
        got = _spread(hits, k, per_file=2)
        assert len(got) == k
        assert len({h.path for h in got}) >= k - max(1, k // 4)


@pytest.mark.benchmark
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 8, 12, 30])
def test_spread_never_returns_more_than_k(k):
    """A8 (regression). Breaking out of the breadth pass at exactly k slots fell
    through to the fill pass and returned k+1 passages, which quietly widens
    every prompt built from them."""
    from agents_work.agents.analyst import _spread
    hits = [Hit(f"{i % 4}.md", f"h{i}", "c", "", 1.0 - i / 100) for i in range(20)]
    got = _spread(hits, k, per_file=2)
    assert len(got) == min(k, len(hits))
    assert len({id(h) for h in got}) == len(got), "a chunk was returned twice"


@pytest.mark.benchmark
def test_hyphenated_compounds_match_their_parts(cfg, tmp_path):
    """A9 (regression). "moving-average crossover" shared not one token with a
    brief describing a "20-day moving average", so the retriever returned that
    backtest's cost table instead of its strategy."""
    assert set(tokenize("a moving-average crossover")) >= {"moving", "average"}
    assert "moving-average" in tokenize("a moving-average crossover")

    vault = tmp_path / "corpus"
    vault.mkdir()
    (vault / "2026-08-20-backtest.md").write_text(
        "---\ndate: 2026-08-20\n---\n\n"
        "## Idea\n\nBuy when the 20-day moving average crosses above the 50-day.\n\n"
        "## Cost drag\n\nSharpe gross 0.68 against net 0.64 at 4.8x turnover.\n")
    idx = Index(cfg.data_dir / "hyphen.db")
    idx.build([vault])
    top = idx.search("what did the moving-average crossover do", k=1)
    idx.close()
    assert top and top[0].heading == "Idea", (
        f"retrieved {top[0].heading if top else 'nothing'} instead of the strategy")


def test_tokenizer_does_not_duplicate_or_reorder():
    assert tokenize("moving-average moving-average") == ["moving-average", "moving", "average"]
    assert tokenize("c++ and python3.13") == ["c++", "python3.13"]
    assert tokenize("co-op") == ["co-op", "co", "op"]   # "co" is two letters, so kept


def test_the_retrieval_width_has_one_source_of_truth():
    """Three call sites default to it; drift between them is a silent behaviour
    change in whichever one the caller happens to use."""
    import inspect

    from agents_work import cli
    from agents_work.agents.analyst import DEFAULT_K

    assert DEFAULT_K == 12
    assert inspect.signature(analyst.ask).parameters["k"].default == DEFAULT_K
    assert inspect.signature(analyst.run).parameters["k"].default == DEFAULT_K
    assert inspect.signature(Index.search).parameters["k"].default == DEFAULT_K
    assert cli.analyst.DEFAULT_K == DEFAULT_K


def test_the_cli_passes_the_default_width_through(monkeypatch, cfg, capsys):
    from agents_work import cli
    from agents_work.agents.analyst import DEFAULT_K
    monkeypatch.setattr(cli, "load_config", lambda **kw: cfg)
    seen = {}
    monkeypatch.setattr(cli.analyst, "run", lambda ctx, q, **kw: seen.update(kw) or
                        type("R", (), {"agent": "analyst", "target": q, "ok": True,
                                       "summary": "", "artifact": None, "brief": None,
                                       "degradations": [], "error": None, "data": {}})())
    cli.main(["ask", "anything"])
    assert seen["k"] == DEFAULT_K
    seen.clear()
    cli.main(["ask", "anything", "--k", "3"])
    assert seen["k"] == 3
