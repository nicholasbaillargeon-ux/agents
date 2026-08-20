"""Agent 5 — personal RAG analyst.

Indexes your notes *and* the briefs the other four agents write, then answers
questions over both with citations: "what did I conclude about NVDA last month"
returns the paragraph you actually wrote, with the file it came from.

Retrieval is hybrid — BM25 over terms plus cosine over a local hashing
embedding — because pure keyword search fails the question above (you did not
write the word "conclude") and a hashing embedder alone is not semantic enough
to carry it. Neither needs a network call or an embedding bill, which is what
keeps this runnable on a timer.

Time is a first-class filter, not a phrase in the prompt: "last month" narrows
the candidate set before ranking, so recent notes cannot be crowded out by an
older note that happens to share more words.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import sqlite3
import zlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from ..brief import Brief
from ..llm import LLMUnavailable
from ..store import Run, record
from .base import AgentResult, Context

log = logging.getLogger(__name__)

NAME = "analyst"
DIM = 512
SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id        INTEGER PRIMARY KEY,
    path      TEXT NOT NULL,
    heading   TEXT NOT NULL DEFAULT '',
    content   TEXT NOT NULL,
    mtime     REAL NOT NULL,
    note_date TEXT NOT NULL DEFAULT '',
    hash      TEXT NOT NULL,
    vector    BLOB NOT NULL,
    tokens    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_notes_path ON notes(path);
CREATE INDEX IF NOT EXISTS idx_notes_date ON notes(note_date);
"""

SYSTEM = (
    "You answer from the user's own notes and research briefs, which are given "
    "to you below. Rules:\n"
    "1. Answer only from the excerpts. If they do not contain the answer, say "
    "exactly what is missing — do not fill the gap from general knowledge.\n"
    "2. Cite the file path in square brackets after each claim, e.g. "
    "[briefs/2026-08-20-nvda.md].\n"
    "3. When the excerpts disagree or a view changed over time, say so and give "
    "both dates. That change is usually the answer the user is looking for.\n"
    "4. Be direct. No preamble, no summary of the question."
)

_TOKEN = re.compile(r"[a-z0-9][a-z0-9'+.#-]*")
_DATE_IN_NAME = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for", "on",
    "with", "as", "at", "by", "that", "this", "was", "were", "be", "are", "i",
    "my", "what", "did", "do", "about", "from", "we", "you", "have", "has",
}


def tokenize(text: str) -> list[str]:
    """Words, plus the parts of any hyphenated compound.

    A compound is indexed as itself *and* as its pieces, because writers are not
    consistent about the hyphen and the reader should not have to be. Asked
    about a "moving-average crossover", the retriever shared not one token with
    a brief describing a "20-day moving average" and returned the cost table
    instead of the strategy.
    """
    out: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        for form in (token, *(token.split("-") if "-" in token else ())):
            if form not in _STOP and len(form) > 1 and form not in out:
                out.append(form)
    return out


def embed(text: str, dim: int = DIM) -> list[float]:
    """Deterministic hashed bag-of-words, L2-normalised.

    crc32, not the builtin hash(): hash() is salted per process, so an index
    built in one run would not match a query in the next.
    """
    vec = [0.0] * dim
    for tok in tokenize(text):
        vec[zlib.crc32(tok.encode()) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def pack(vec: list[float]) -> bytes:
    import struct
    return struct.pack(f"{len(vec)}f", *vec)


def unpack(blob: bytes) -> list[float]:
    import struct
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Chunk:
    path: str
    heading: str
    content: str
    note_date: str = ""
    mtime: float = 0.0

    @property
    def hash(self) -> str:
        return hashlib.sha256(f"{self.path}\x00{self.heading}\x00{self.content}"
                              .encode()).hexdigest()[:32]


@dataclass
class Hit:
    path: str
    heading: str
    content: str
    note_date: str
    score: float
    lexical: float = 0.0
    semantic: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.path}" + (f" — {self.heading}" if self.heading else "")


def chunk_markdown(path: Path, root: Path, *, max_chars: int = 1600) -> list[Chunk]:
    """Split on headings, then on size. Frontmatter is stripped but its date is
    kept — that date is what makes "last month" answerable."""
    try:
        text = path.read_text(errors="replace")
    except OSError as e:
        log.warning("unreadable note %s: %s", path, e)
        return []
    rel = str(path.relative_to(root)) if _within(path, root) else path.name
    note_date = _note_date(text, path)
    mtime = path.stat().st_mtime
    text = _FRONTMATTER.sub("", text)

    chunks, heading, buf = [], "", []

    def flush():
        body = "\n".join(buf).strip()
        if len(body) < 25:
            return
        for i in range(0, len(body), max_chars):
            chunks.append(Chunk(rel, heading, body[i:i + max_chars], note_date, mtime))

    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            heading, buf = line.lstrip("#").strip(), []
        else:
            buf.append(line)
    flush()
    return chunks


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _note_date(text: str, path: Path) -> str:
    m = _FRONTMATTER.match(text)
    if m:
        d = re.search(r"^date:\s*\"?(\d{4}-\d{2}-\d{2})", m.group(1), re.M)
        if d:
            return d.group(1)
    m = _DATE_IN_NAME.search(path.name)
    if m:
        return m.group(0)
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d")
    except OSError:
        return ""


class Index:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, timeout=15)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def build(self, roots: list[Path], *, patterns=("*.md", "*.markdown")) -> dict:
        """Full rebuild. Cheap enough at personal-vault scale that incremental
        indexing would be complexity without a payoff."""
        stats = {"roots": 0, "files": 0, "chunks": 0, "skipped_roots": []}
        rows = []
        for root in roots:
            root = Path(root).expanduser()
            if not root.is_dir():
                stats["skipped_roots"].append(str(root))
                continue
            stats["roots"] += 1
            for pattern in patterns:
                for path in sorted(root.rglob(pattern)):
                    if any(part.startswith(".") for part in path.parts):
                        continue
                    chunks = chunk_markdown(path, root)
                    if chunks:
                        stats["files"] += 1
                    for c in chunks:
                        rows.append((c.path, c.heading, c.content, c.mtime, c.note_date,
                                     c.hash, pack(embed(f"{c.heading}\n{c.content}")),
                                     len(tokenize(c.content))))
        with self.conn:
            self.conn.execute("DELETE FROM notes")
            self.conn.executemany(
                "INSERT INTO notes(path,heading,content,mtime,note_date,hash,vector,tokens)"
                " VALUES(?,?,?,?,?,?,?,?)", rows)
        stats["chunks"] = len(rows)
        return stats

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]

    def search(self, query: str, *, k: int = 6, since: str | None = None,
               until: str | None = None, alpha: float = 0.5,
               per_file: int = 2) -> list[Hit]:
        """Hybrid BM25 + cosine, spread across documents.

        `alpha` weights lexical against semantic. `per_file` caps how many
        chunks one document may contribute before others get a turn.

        Breadth matters more than it looks. Asked to compare seven research
        briefs, pure score order returned two chunks of the same brief and a
        headline list, and left out the only two names that had the multiple the
        question was about — so the answer confidently reported that the corpus
        contained one. A cross-document question needs one good passage from
        many documents, not the best passages overall.
        """
        sql = "SELECT * FROM notes"
        clauses, args = [], []
        if since:
            clauses.append("note_date >= ?")
            args.append(since)
        if until:
            clauses.append("note_date <= ?")
            args.append(until)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        rows = self.conn.execute(sql, args).fetchall()
        if not rows:
            return []

        q_tokens = tokenize(query)
        q_vec = embed(query)
        docs = [tokenize(f"{r['heading']} {r['content']}") for r in rows]
        lexical = _bm25(q_tokens, docs)
        lex_max = max(lexical) or 1.0

        hits = []
        for row, doc_lex in zip(rows, lexical):
            sem = cosine(q_vec, unpack(row["vector"]))
            combined = alpha * (doc_lex / lex_max) + (1 - alpha) * sem
            hits.append(Hit(row["path"], row["heading"], row["content"],
                            row["note_date"], combined, doc_lex / lex_max, sem))
        hits.sort(key=lambda h: -h.score)
        return [h for h in _spread(hits, k, per_file) if h.score > 0]


def _spread(hits: list[Hit], k: int, per_file: int) -> list[Hit]:
    """Best chunk from each document first, then fill by score.

    Two passes rather than a per-file cap applied in one: the first guarantees
    that k documents are represented before any document gets a second slot,
    which is the property a comparison question needs.
    """
    if k <= 0 or not hits:
        return []
    if per_file <= 0:
        return hits[:k]

    # Breadth first, but not with every slot. A question about one document
    # ("what did my backtest find") is served badly by one chunk of it and seven
    # of everything else, and a question across documents is served badly by
    # three chunks of the same brief. Reserving a quarter of the slots for score
    # order lets the same retriever answer both shapes.
    # A quarter of the slots, rounded down: small k stays purely broad, since
    # with four slots there is no depth worth reserving.
    breadth = k - k // 4

    chosen: list[Hit] = []
    taken: set[int] = set()
    seen_paths: set[str] = set()
    for i, hit in enumerate(hits):
        if hit.path in seen_paths:
            continue
        seen_paths.add(hit.path)
        chosen.append(hit)
        taken.add(i)
        if len(chosen) >= breadth:
            break

    # The length check comes before the append, not after: breaking out of the
    # breadth pass at exactly k slots used to fall through to here and return
    # k + 1 passages.
    counts = Counter(h.path for h in chosen)
    for i, hit in enumerate(hits):
        if len(chosen) >= k:
            break
        if i in taken or counts[hit.path] >= per_file:
            continue
        chosen.append(hit)
        counts[hit.path] += 1
        taken.add(i)
    for i, hit in enumerate(hits):  # fewer documents than slots: fill with the rest
        if len(chosen) >= k:
            break
        if i not in taken:
            chosen.append(hit)
            taken.add(i)
    chosen.sort(key=lambda h: -h.score)
    return chosen


def _bm25(query: list[str], docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    if not docs:
        return []
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n or 1.0
    df = Counter()
    doc_counts = [Counter(d) for d in docs]
    for counts in doc_counts:
        for term in counts:
            df[term] += 1
    scores = []
    for counts, doc in zip(doc_counts, docs):
        s, dl = 0.0, len(doc) or 1
        for term in query:
            f = counts.get(term, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


# --- natural-language time windows -----------------------------------------

_WINDOWS = (
    (re.compile(r"\blast month\b", re.I), 30),
    (re.compile(r"\bpast month\b", re.I), 30),
    (re.compile(r"\blast week\b", re.I), 7),
    (re.compile(r"\bpast week\b", re.I), 7),
    (re.compile(r"\byesterday\b", re.I), 1),
    (re.compile(r"\blast quarter\b", re.I), 92),
    (re.compile(r"\blast year\b", re.I), 365),
    (re.compile(r"\brecently\b", re.I), 21),
)


def time_window(question: str, *, today: date | None = None) -> tuple[str | None, str | None]:
    """('2026-07-21', None) for 'last month'. (None, None) when unqualified."""
    today = today or datetime.now(timezone.utc).date()
    for pattern, days in _WINDOWS:
        if pattern.search(question):
            return (today - timedelta(days=days)).isoformat(), None
    return None, None


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    since: str | None = None
    search_only: bool = False
    degradations: list[str] = field(default_factory=list)

    @property
    def citations(self) -> list[str]:
        seen, out = set(), []
        for h in self.hits:
            if h.path not in seen:
                seen.add(h.path)
                out.append(h.path)
        return out


def ask(ctx: Context, question: str, *, k: int = 6, since: str | None = None,
        until: str | None = None, index: Index | None = None,
        today: date | None = None) -> Answer:
    own = index is None
    idx = index or Index(ctx.cfg.data_dir / "analyst.db")
    try:
        if idx.count() == 0:
            stats = idx.build(ctx.cfg.vault_roots)
            log.info("cold index built: %s", stats)
        auto_since, auto_until = time_window(question, today=today)
        since, until = since or auto_since, until or auto_until
        hits = idx.search(question, k=k, since=since, until=until)

        if not hits and since:
            # An empty window is a real answer, but a silently empty one is not.
            hits = idx.search(question, k=k)
            ans = Answer(question, "", hits, since=since)
            ans.degradations.append(
                f"nothing matched inside the {since}+ window; showing the whole index instead")
            since = None
        else:
            ans = Answer(question, "", hits, since=since)

        if not hits:
            ans.text = ("_Nothing in the indexed notes matches that. "
                        f"{idx.count()} chunks indexed from "
                        f"{len(ctx.cfg.vault_roots)} configured roots._")
            ans.search_only = True
            return ans

        excerpts = "\n\n---\n\n".join(
            f"[{h.path}]{(' — ' + h.heading) if h.heading else ''} (dated {h.note_date})\n"
            f"{h.content}" for h in hits)
        try:
            ans.text = ctx.llm.complete(
                f"Excerpts from the user's notes:\n\n{excerpts}\n\nQuestion: {question}",
                system=SYSTEM, max_tokens=1500)
        except LLMUnavailable as e:
            # The spec's search-only mode: retrieval still worked, so ship it.
            ans.search_only = True
            ans.degradations.append(f"search-only mode ({e})")
            ans.text = "\n\n".join(
                f"**{h.label}** _(dated {h.note_date}, score {h.score:.2f})_\n\n"
                f"> {h.content[:600].strip()}" for h in hits)
        return ans
    finally:
        if own:
            idx.close()


def run(ctx: Context, question: str, *, reindex: bool = False, k: int = 6,
        commit: bool = False, **kw) -> AgentResult:
    started = datetime.now(timezone.utc)
    res = AgentResult(agent=NAME, target=question[:80])
    idx = Index(ctx.cfg.data_dir / "analyst.db")
    try:
        if reindex or idx.count() == 0:
            stats = idx.build(ctx.cfg.vault_roots)
            res.data["index"] = stats
            for skipped in stats["skipped_roots"]:
                res.degrade(f"vault root missing: {skipped}")
        answer = ask(ctx, question, k=k, index=idx, **kw)
    except Exception as e:  # noqa: BLE001
        log.exception("analyst failed")
        res.ok, res.error = False, f"{type(e).__name__}: {e}"
        record(ctx.db, Run(agent=NAME, target=res.target, ok=False, error=res.error,
                           started_at=started.timestamp()))
        return res
    finally:
        idx.close()

    for d in ctx.base_degradations() + answer.degradations:
        res.degrade(d)

    brief = Brief(title=f"Q: {question[:80]}", agent=NAME, target=question[:60],
                  tags=["analyst", "rag"])
    for d in res.degradations:
        brief.degrade(d)
    brief.add("Answer" if not answer.search_only else "Matching passages (search-only)",
              answer.text)
    if answer.hits:
        brief.add("Sources", "\n".join(
            f"- `{h.label}` — dated {h.note_date} "
            f"(lexical {h.lexical:.2f}, semantic {h.semantic:.2f})" for h in answer.hits))
    if answer.since:
        brief.extra_meta["window_since"] = answer.since
    res.brief = brief
    res.artifact = brief.write(ctx.cfg.out_dir / NAME)
    res.data.update({"citations": answer.citations, "search_only": answer.search_only,
                     "hits": len(answer.hits)})
    res.summary = (f"{len(answer.hits)} passages"
                   f"{' (search-only)' if answer.search_only else ''}"
                   f"{f', since {answer.since}' if answer.since else ''}")
    record(ctx.db, Run(agent=NAME, target=res.target, ok=True, artifact=str(res.artifact),
                       summary=res.summary, degradations=res.degradations,
                       started_at=started.timestamp(),
                       duration_s=(datetime.now(timezone.utc) - started).total_seconds()))
    return res
