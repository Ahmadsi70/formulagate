"""Tests for built-in SQLite FTS5 retriever."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from formulagate.retriever import RetrievedDoc, SQLiteRetriever

SAMPLE_DOCS = [
    {"id": "1", "text": "energy mass equivalence einstein special relativity",
     "formula": "E = m c^2", "domain": "physics", "year": 1905},
    {"id": "2", "text": "newton second law force equals mass times acceleration",
     "formula": "F = m a", "domain": "physics", "year": 1687},
    {"id": "3", "text": "quantum mechanics planck photon energy relation",
     "formula": "E = h \\nu", "domain": "physics", "year": 1900},
    {"id": "4", "text": "shakespeare hamlet tragedy drama literature",
     "formula": "", "domain": "literature", "year": 1600},
]


def test_empty_retriever() -> None:
    retriever = SQLiteRetriever()
    assert len(retriever) == 0
    results = retriever.search("energy")
    assert results == []


def test_index_and_search() -> None:
    retriever = SQLiteRetriever(SAMPLE_DOCS)
    assert len(retriever) == 4

    results = retriever.search("energy mass", top_k=3)
    assert len(results) >= 1
    assert results[0].id == "1"  # energy mass equivalence should be top


def test_search_returns_scores() -> None:
    retriever = SQLiteRetriever(SAMPLE_DOCS)
    results = retriever.search("energy")
    assert len(results) > 0
    # BM25 scores should be non-zero for relevant docs
    assert results[0].score > 0.0


def test_search_no_match() -> None:
    retriever = SQLiteRetriever(SAMPLE_DOCS)
    results = retriever.search("xyzwyzyx_nonexistent_term")
    assert results == []


def test_search_empty_query() -> None:
    retriever = SQLiteRetriever(SAMPLE_DOCS)
    assert retriever.search("") == []
    assert retriever.search("   ") == []


def test_retrieved_doc_serialization() -> None:
    doc = RetrievedDoc(id="1", text="hello", formula="x=y", score=12.5)
    d = doc.to_dict()
    assert d["id"] == "1"
    assert d["score"] == 12.5


def test_retrieved_doc_meta() -> None:
    doc = RetrievedDoc(id="1", text="hello", meta={"year": 2024})
    d = doc.to_dict()
    assert d["meta"]["year"] == 2024


def test_persistent_db() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "test.db")
        retriever = SQLiteRetriever(SAMPLE_DOCS, db_path=path)
        results = retriever.search("energy")
        assert len(results) > 0
        retriever.close()

        # Re-open and verify data persisted
        retriever2 = SQLiteRetriever(db_path=path)
        results2 = retriever2.search("energy")
        assert len(results2) > 0
        retriever2.close()


def test_incremental_indexing() -> None:
    retriever = SQLiteRetriever()
    retriever.index(SAMPLE_DOCS[:2])
    assert len(retriever) == 2
    results = retriever.search("energy")
    assert len(results) == 1

    retriever.index(SAMPLE_DOCS[2:])
    assert len(retriever) == 4
    results = retriever.search("energy")
    assert len(results) >= 2  # doc 1 and doc 3 both match


def test_context_manager() -> None:
    with SQLiteRetriever(SAMPLE_DOCS) as retriever:
        results = retriever.search("force")
        assert len(results) > 0


def test_search_ranks_by_relevance() -> None:
    retriever = SQLiteRetriever(SAMPLE_DOCS)
    results = retriever.search("energy mass equivalence")
    assert len(results) >= 2
    # Doc 1 (energy mass equivalence) should rank higher than others
    assert results[0].id == "1"
    # Scores should be descending (first is highest)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_docs_without_formula() -> None:
    retriever = SQLiteRetriever([{"id": "x", "text": "just text no formula"}])
    results = retriever.search("text")
    assert len(results) == 1
    assert results[0].formula == ""