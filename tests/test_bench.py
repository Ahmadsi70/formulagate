"""Rerank benchmark tests (Formulagate vs CrossEncoder)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("rank_bm25")
pytest.importorskip("numpy")
pytest.importorskip("faiss")

from formulagate.bench import RerankBenchmark, benchmark_rerankers
from formulagate.dense import DenseHybridRetriever, HashEmbedder
from formulagate.metrics import load_golden_cases
from formulagate.rerank import FormulagateReranker

ROOT = Path(__file__).resolve().parents[1]


class _IdentityReranker:
    """Stub CE stand-in: keeps retrieval order (for offline CI)."""

    def rerank(self, *, brief, draft, candidates, top_k):
        return candidates[: max(1, top_k)]


def test_benchmark_reports_delta_against_stub_ce() -> None:
    corpus = json.loads(
        (ROOT / "data" / "sample_corpus.json").read_text(encoding="utf-8")
    )
    cases = load_golden_cases(ROOT / "data" / "golden_cases.json")[:12]
    result = benchmark_rerankers(
        cases,
        corpus=corpus,
        make_retriever=lambda rows: DenseHybridRetriever(
            rows, embedder=HashEmbedder(dim=64)
        ),
        cross_encoder=_IdentityReranker(),
        retrieve_k=3,
    )
    assert isinstance(result, RerankBenchmark)
    assert result.formulagate.n == 12
    assert result.cross_encoder.n == 12
    # Formulagate reranker should find at least some matches (not 0 accuracy).
    assert result.formulagate.accuracy > 0.5, (
        f"Formulagate accuracy {result.formulagate.accuracy:.3f} too low"
    )


@pytest.mark.dense_model
def test_cross_encoder_reranker_imports() -> None:
    pytest.importorskip("sentence_transformers")
    from formulagate.rerank import CrossEncoderReranker

    ce = CrossEncoderReranker()
    corpus = json.loads(
        (ROOT / "data" / "sample_corpus.json").read_text(encoding="utf-8")
    )
    pool = corpus[:3]
    out = ce.rerank(
        brief="modular congruence",
        draft="mod n",
        candidates=pool,
        top_k=2,
    )
    assert len(out) == 2
