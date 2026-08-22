"""Dense hybrid (FAISS) + Formulagate rerank + golden eval tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("rank_bm25")
pytest.importorskip("numpy")
pytest.importorskip("faiss")

from formulagate.dense import DenseHybridRetriever, HashEmbedder
from formulagate.metrics import evaluate_golden_cases, load_golden_cases
from formulagate.rag import run_scientific_rag
from formulagate.rerank import FormulagateReranker, RerankingRetriever


ROOT = Path(__file__).resolve().parents[1]


def _corpus() -> list[dict]:
    return json.loads(
        (ROOT / "data" / "sample_corpus.json").read_text(encoding="utf-8")
    )


def test_dense_hybrid_surfaces_mod_congruence() -> None:
    retriever = DenseHybridRetriever(_corpus(), embedder=HashEmbedder(dim=64))
    hits = retriever.retrieve(
        "integer modular arithmetic congruence residue class divisible",
        top_k=3,
    )
    assert any(h["id"] == "mod-congruence" for h in hits)


def test_formulagate_rerank_promotes_domain_relevant_hit() -> None:
    corpus = _corpus()
    pool = [
        next(r for r in corpus if r["id"] == "decoy-integral"),
        next(r for r in corpus if r["id"] == "mod-congruence"),
        next(r for r in corpus if r["id"] == "harmonic-bridge"),
    ]
    ranked = FormulagateReranker().rerank(
        brief="modular residue congruence divisible number theory",
        draft="n equiv 1 (mod 3)",
        candidates=pool,
        top_k=2,
    )
    assert ranked[0]["id"] == "mod-congruence"


def test_rerank_in_pipeline_generate() -> None:
    base = DenseHybridRetriever(_corpus(), embedder=HashEmbedder(dim=64))
    decision = run_scientific_rag(
        brief="modular residue congruence divisible number theory",
        draft="n equiv 1 (mod 3)",
        retriever=base,
        reranker=FormulagateReranker(),
        retrieve_pool=10,
        retrieve_k=3,
    )
    assert decision.action == "generate"
    assert decision.retrieved_ids[0] == "mod-congruence"


def test_reranking_retriever_set_context() -> None:
    base = DenseHybridRetriever(_corpus(), embedder=HashEmbedder(dim=64))
    wrapped = RerankingRetriever(base, FormulagateReranker(), pool_k=10)
    decision = run_scientific_rag(
        brief="modular residue congruence divisible number theory",
        draft="n equiv 1 (mod 3)",
        retriever=wrapped,
        retrieve_k=3,
    )
    assert decision.action == "generate"
    assert decision.retrieved_ids[0] == "mod-congruence"


def test_golden_eval_meets_ci_floor() -> None:
    cases = load_golden_cases(ROOT / "data" / "golden_cases.json")
    assert len(cases) >= 50
    report = evaluate_golden_cases(
        cases,
        corpus=_corpus(),
        make_retriever=lambda rows: DenseHybridRetriever(
            rows, embedder=HashEmbedder(dim=64)
        ),
        reranker=FormulagateReranker(),
        retrieve_k=3,
    )
    assert report.n >= 50
    assert report.accuracy >= 0.75
    assert report.abstain_recall == 1.0
