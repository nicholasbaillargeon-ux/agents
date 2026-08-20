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
from agents_work.agents.analyst import Index, ask, chunk_markdown, embed, time_window, tokenize
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
