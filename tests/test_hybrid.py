"""Hybrid BM25+TF-IDF+RRF and abstain metric tests (option A)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("rank_bm25")

from formulagate.hybrid import HybridRrfRetriever, rrf_fuse
from formulagate.metrics import AbstainReport, evaluate_abstain_cases
from formulagate.rag import run_scientific_rag


def _corpus() -> list[dict]:
    path = Path(__file__).resolve().parents[1] / "data" / "sample_corpus.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_rrf_prefers_docs_ranked_high_in_both_lists() -> None:
    """RRF (k=60) boosts consensus hits over single-list noise."""

    fused = rrf_fuse(
        [["a", "noise", "b"], ["b", "a", "noise"]],
        k=60,
        top_k=2,
    )
    assert fused[0] in {"a", "b"}
    assert set(fused[:2]) == {"a", "b"}


def test_hybrid_retrieves_mod_congruence_for_number_theory_query() -> None:
    retriever = HybridRrfRetriever(_corpus())
    hits = retriever.retrieve("modular residue congruence divisible integers", top_k=3)
    assert hits[0]["id"] == "mod-congruence"


def test_hybrid_rag_generate_then_abstain_on_weak_analog() -> None:
    corpus = _corpus()
    hybrid = HybridRrfRetriever(corpus)
    ok = run_scientific_rag(
        brief="modular residue congruence divisible number theory",
        draft="n equiv 1 (mod 3); congruence classes",
        retriever=hybrid,
    )
    assert ok.action == "generate"

    weak = run_scientific_rag(
        brief="modular residue congruence divisible number theory",
        draft="use integral energy as analogy",
        retriever=HybridRrfRetriever(
            [r for r in corpus if r["id"] == "decoy-integral"]
        ),
    )
    assert weak.action == "abstain"


def test_abstain_metrics_on_labeled_scientific_cases() -> None:
    """Safety metric: weak analogs must abstain; true cites may generate."""

    corpus = _corpus()
    full = HybridRrfRetriever(corpus)
    decoy = HybridRrfRetriever([r for r in corpus if r["id"] == "decoy-integral"])

    generate_report = evaluate_abstain_cases(
        [
            {
                "brief": "modular residue congruence divisible number theory",
                "draft": "n equiv 1 (mod 3)",
                "expect_generate": True,
            },
            {
                "brief": "periodic oscillation frequency vibration period mechanics",
                "draft": "sinusoidal N(t) with period T",
                "expect_generate": True,
            },
        ],
        retriever=full,
    )
    abstain_report = evaluate_abstain_cases(
        [
            {
                "brief": "modular residue congruence divisible number theory",
                "draft": "integral energy analogy",
                "expect_generate": False,
            }
        ],
        retriever=decoy,
    )
    assert isinstance(generate_report, AbstainReport)
    assert generate_report.accuracy == 1.0
    assert abstain_report.abstain_recall == 1.0
    assert abstain_report.n == 1
