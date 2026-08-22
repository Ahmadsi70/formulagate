"""Benchmark Formulagate rerank vs CrossEncoder on golden cases.

Why: quantify whether neural rerank improves top-id / abstain metrics enough
to justify latency versus the deterministic Formulagate score rerank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from formulagate.metrics import AbstainReport, evaluate_golden_cases
from formulagate.rag import Retriever
from formulagate.rerank import CrossEncoderReranker, FormulagateReranker, Reranker


@dataclass(frozen=True)
class RerankBenchmark:
    """Side-by-side golden metrics for two rerankers."""

    formulagate: AbstainReport
    cross_encoder: AbstainReport
    top_id_delta: float
    accuracy_delta: float


def benchmark_rerankers(
    cases: Sequence[dict[str, Any]],
    *,
    corpus: list[dict[str, Any]],
    make_retriever: Callable[[list[dict[str, Any]]], Retriever],
    cross_encoder: Reranker | None = None,
    retrieve_k: int = 3,
) -> RerankBenchmark:
    """Run the same golden set with Formulagate vs CrossEncoder rerank."""

    fg = FormulagateReranker()
    ce = cross_encoder or CrossEncoderReranker()
    report_fg = evaluate_golden_cases(
        cases,
        corpus=corpus,
        make_retriever=make_retriever,
        retrieve_k=retrieve_k,
        reranker=fg,
    )
    report_ce = evaluate_golden_cases(
        cases,
        corpus=corpus,
        make_retriever=make_retriever,
        retrieve_k=retrieve_k,
        reranker=ce,
    )
    fg_top = (
        report_fg.top_id_hits / report_fg.top_id_checked
        if report_fg.top_id_checked
        else 0.0
    )
    ce_top = (
        report_ce.top_id_hits / report_ce.top_id_checked
        if report_ce.top_id_checked
        else 0.0
    )
    return RerankBenchmark(
        formulagate=report_fg,
        cross_encoder=report_ce,
        top_id_delta=ce_top - fg_top,
        accuracy_delta=report_ce.accuracy - report_fg.accuracy,
    )
