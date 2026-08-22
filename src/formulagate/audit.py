"""Build JSON audit payloads for RAG gate decisions.

Why: product demos and compliance need an auditable trail of retrieve→rerank→judge.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from formulagate.calibration import get_calibration, get_multi_calibration
from formulagate.rag import RagDecision


def build_rag_audit(
    *,
    brief: str,
    draft: str,
    decision: RagDecision,
    corpus_path: str,
    embedder: str,
    reranker: str,
) -> dict[str, Any]:
    """Serialize a full RAG decision for CLI ``--audit`` / JSON output."""

    entries = [
        {
            "record_id": e.record_id,
            "score": e.score,
            "math_relevance": e.math_relevance,
            "formula_excerpt": e.formula_excerpt,
            "scientific_domain": e.scientific_domain,
            "meta": e.meta,
        }
        for e in decision.gate.result.entries
    ]
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": decision.action,
        "ok": decision.ok,
        "detail": decision.gate.detail,
        "confidence": decision.gate.confidence,
        "threshold": decision.gate.threshold,
        "combined_score": decision.gate.combined_score,
        "draft_consistency": decision.gate.draft_consistency,
        "physics": decision.gate.physics,
        "confidence_model": decision.gate.model,
        "calibration": get_calibration().to_dict(),
        "calibration_multi": (
            get_multi_calibration().to_dict() if get_multi_calibration() else None
        ),
        "domain": decision.gate.result.domain,
        "brief": brief,
        "draft": draft,
        "corpus_path": corpus_path,
        "embedder": embedder,
        "reranker": reranker,
        "retrieved_ids": decision.retrieved_ids,
        "entries": entries,
        "formula": "score = overlap + 2*rel",
        "pipeline": "retrieve -> rerank -> formulagate judge",
    }
