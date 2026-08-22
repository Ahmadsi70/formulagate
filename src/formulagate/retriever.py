"""Lightweight built-in retriever using SQLite FTS5 — zero extra dependencies.

Why this exists:
  Formulagate is a gate, not a search engine.  But every integrator needs to
  retrieve relevant documents *before* the gate can judge them.  Until now, the
  integrator had to bring their own retriever (Elasticsearch, Pinecone, …).
  This module provides a zero-dependency, zero-configuration retriever that
  works out of the box for small-to-medium corpora (up to ~1M documents) using
  only Python's stdlib ``sqlite3`` module.

  FTS5 provides BM25 ranking natively — no extra libraries, no model downloads,
  no GPU.  Indexing is instant (in-memory) or persistent (on-disk SQLite file).

Memory:
  RAM usage is proportional to the FTS5 index, roughly 10-30 MB per 100K
  documents.  For a corpus of 10K records the overhead is under 5 MB.

Usage:
    from formulagate.retriever import SQLiteRetriever

    retriever = SQLiteRetriever(my_documents)
    results = retriever.search("energy mass equivalence", top_k=5)
    # → [{"id": "paper-1", "text": "...", "formula": "E=mc^2", "score": 12.3}, …]

    # Then pass to Formulagate:
    gate.check(brief=..., draft=..., sources=results)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Interface that any document retriever must satisfy to plug into Formulagate.

    Implement this protocol to bring your own retriever (Elasticsearch, Pinecone,
    Weaviate, custom FAISS index, …).  The gate only calls ``search()`` — it
    never indexes or owns the retriever lifecycle.

    Usage:
        class MyRetriever:
            def search(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
                ...
            def __len__(self) -> int:
                return self._count

        gate = Formulagate(sources=docs, retriever=MyRetriever())
        gate.check(brief="E=mc^2", draft="...")
    """

    def search(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """Return the top-*k* documents ranked by relevance to ``query``."""
        ...

    def __len__(self) -> int:
        """Return the number of indexed documents."""
        ...


@dataclass(frozen=True)
class RetrievedDoc:
    """One search result from the built-in retriever."""

    id: str
    text: str = ""
    formula: str = ""
    score: float = 0.0
    meta: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "formula": self.formula,
            "score": self.score,
        }
        if self.meta:
            d["meta"] = dict(self.meta)
        return d


class SQLiteRetriever:
    """BM25-powered retriever backed by an in-memory or on-disk SQLite database.

    Args:
        documents: List of dicts with at least ``id``, ``text``, and optionally
            ``formula``, ``domain``, and any other metadata fields.
        db_path: Path to a persistent SQLite file.  When ``None`` (default),
            the database is in-memory only.
        cache_size_kb: SQLite ``cache_size`` pragma in KB.  Higher = faster
            but more RAM.  Default -8000 (8 MB negative = 8000 pages).

    Memory footprint:
        In-memory mode: index size ≈ 10-15% of raw text size.
        On-disk mode: RAM usage ≈ cache_size_kb + small overhead.
    """

    def __init__(
        self,
        documents: Sequence[Mapping[str, Any]] | None = None,
        *,
        db_path: str = ":memory:",
        cache_size_kb: int = -8000,
    ) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(f"PRAGMA cache_size = {cache_size_kb}")
        self._conn.execute("PRAGMA journal_mode = OFF")  # no WAL overhead
        self._conn.execute("PRAGMA synchronous = OFF")

        # FTS5 virtual table: content-less (we store the full row separately).
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5("
            "  id UNINDEXED,"
            "  text,"
            "  formula,"
            "  domain,"
            "  tokenize='porter unicode61'"
            ")"
        )

        # Shadow table for retrieving full records by id.
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS records ("
            "  id TEXT PRIMARY KEY,"
            "  text TEXT,"
            "  formula TEXT,"
            "  domain TEXT,"
            "  meta_json TEXT"
            ")"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_records_id ON records(id)"
        )

        self._count = 0
        if documents:
            self.index(documents)

    # ── indexing ──────────────────────────────────────────────────────────

    def index(self, documents: Sequence[Mapping[str, Any]]) -> None:
        """Add documents to the index.  Can be called multiple times."""
        import json

        cur = self._conn.cursor()
        for doc in documents:
            doc_id = str(doc.get("id", ""))
            if not doc_id:
                continue
            text = str(doc.get("text") or doc.get("english") or "")
            formula = str(doc.get("formula") or doc.get("math_formula") or "")
            domain = str(doc.get("domain") or doc.get("scientific_domain") or "")
            meta = {
                k: v
                for k, v in doc.items()
                if k not in ("id", "text", "english", "formula", "math_formula",
                              "domain", "scientific_domain")
            }

            # FTS5 insert (the text columns are tokenised and indexed).
            cur.execute(
                "INSERT OR REPLACE INTO docs(id, text, formula, domain) "
                "VALUES (?, ?, ?, ?)",
                (doc_id, text, formula, domain),
            )
            # Shadow table for full-record retrieval.
            cur.execute(
                "INSERT OR REPLACE INTO records(id, text, formula, domain, meta_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc_id, text, formula, domain, json.dumps(meta, ensure_ascii=False)),
            )
            self._count += 1

        self._conn.commit()

    # ── search ────────────────────────────────────────────────────────────

    def search(
        self, query: str, top_k: int = 5
    ) -> list[RetrievedDoc]:
        """Return the top-*k* documents ranked by BM25.

        Args:
            query: Free-text search query.
            top_k: Maximum number of results to return.

        Returns:
            List of ``RetrievedDoc`` sorted by descending BM25 score.
        """
        import json

        if not query or not query.strip():
            return []

        # Sanitise: FTS5 has a simple query syntax; we escape special chars.
        safe = query.replace('"', '""')
        # Use OR of individual terms so partial matches also score.
        terms = " OR ".join(w for w in safe.split() if len(w) >= 2)
        if not terms:
            return []

        # Negate BM25 so higher score = more relevant (conventional ordering).
        rows = self._conn.execute(
            "SELECT d.id, r.text, r.formula, r.domain, r.meta_json, "
            "       -bm25(docs, 0.0, 0.75, 1.0, 0.5) AS score "
            "FROM docs d "
            "JOIN records r ON r.id = d.id "
            "WHERE docs MATCH ? "
            "ORDER BY score DESC "
            "LIMIT ?",
            (terms, max(1, top_k)),
        ).fetchall()

        results: list[RetrievedDoc] = []
        for row in rows:
            meta = json.loads(row[4]) if row[4] else None
            results.append(RetrievedDoc(
                id=row[0],
                text=row[1],
                formula=row[2],
                score=float(row[5]) if row[5] is not None else 0.0,
                meta=meta,
            ))

        return results

    # ── convenience ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._count

    def close(self) -> None:
        """Release the SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "SQLiteRetriever":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()