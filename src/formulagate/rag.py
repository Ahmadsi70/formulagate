"""Scientific RAG sketch: retrieve → Formulagate judge → generate/abstain.

Why: 2026 RAG best practice separates high-recall search from an evidence
gate. Formulagate is the deterministic Judge; the Retriever is swappable
(vector/hybrid in production, lexical stub here — no new deps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from formulagate.domain import classify_domain
from formulagate.gate import (
    DiscoveryResult,
    GateEvaluation,
    LinkageEntry,
    evaluate_gate_signals,
)
from formulagate.scoring import keywords, score_record


class Retriever(Protocol):
    """High-recall candidate generator (vector/hybrid/BM25 in production)."""

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return corpus rows most relevant to ``query`` (usually the brief)."""


@dataclass(frozen=True)
class RagDecision:
    """Auditable RAG outcome after retrieval + Formulagate."""

    ok: bool
    action: str  # "generate" | "abstain"
    retrieved_ids: list[str]
    gate: GateEvaluation


class LexicalStubRetriever:
    """Stand-in for embedding recall: ranks rows by keyword overlap on text.

    Why: lets the pipeline be tested without FAISS/Chroma while preserving
    the Retrieve→Judge contract for a real vector backend later.
    """

    def __init__(self, corpus: list[dict[str, Any]]) -> None:
        self._corpus = list(corpus)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        keys = keywords(query)
        scored: list[tuple[int, dict[str, Any]]] = []
        for row in self._corpus:
            blob = " ".join(
                str(row.get(k) or "")
                for k in ("english", "math_formula", "formula", "title", "text")
            ).lower()
            hit = sum(1 for k in keys if k in blob)
            if hit > 0:
                scored.append((hit, row))
        scored.sort(key=lambda x: (-x[0], str(x[1].get("id", ""))))
        return [row for _, row in scored[: max(1, top_k)]]


def _discover_from_candidates(
    *,
    brief: str,
    draft: str,
    candidates: list[dict[str, Any]],
    top_k: int,
) -> DiscoveryResult:
    """Rank retriever hits with Formulagate scoring (in-memory corpus slice)."""

    domain = classify_domain(brief)
    keys = keywords(brief + " " + draft)
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for row in candidates:
        score, rel = score_record(keys, row, domain)
        if score > 0:
            scored.append((score, rel, row))
    scored.sort(key=lambda x: (-x[0], -x[1], str(x[2].get("id", ""))))

    entries: list[LinkageEntry] = []
    for score, rel, row in scored[: max(1, top_k)]:
        mf = str(row.get("math_formula") or row.get("formula") or "")
        excerpt = mf[:280] + ("…" if len(mf) > 280 else "")
        meta = {
            k: row[k]
            for k in ("surah", "ayah", "title", "source")
            if k in row
        }
        entries.append(
            LinkageEntry(
                record_id=str(row.get("id")),
                score=score,
                math_relevance=rel,
                formula_excerpt=excerpt,
                scientific_domain=(
                    str(row["scientific_domain"])
                    if row.get("scientific_domain") is not None
                    else None
                ),
                meta=meta,
            )
        )
    return DiscoveryResult(
        domain=domain,
        corpus_path="rag://candidates",
        entries=entries,
    )


def run_scientific_rag(
    *,
    brief: str,
    draft: str,
    retriever: Retriever,
    retrieve_k: int = 5,
    judge_top_k: int = 3,
    reranker: Any | None = None,
    retrieve_pool: int | None = None,
    use_physics: bool = True,
) -> RagDecision:
    """Retrieve → optional rerank → Formulagate-judge before generation.

    Why: retrieval alone is not evidence sufficiency — accept only when the
    domain-relevant formula clears τ; otherwise abstain (no hallucinated cite).
    """

    set_ctx = getattr(retriever, "set_context", None)
    if callable(set_ctx):
        set_ctx(brief=brief, draft=draft)

    pool_k = retrieve_pool if retrieve_pool is not None else (
        max(retrieve_k, 20) if reranker is not None else retrieve_k
    )
    candidates = retriever.retrieve(brief, top_k=pool_k)
    if reranker is not None:
        candidates = reranker.rerank(
            brief=brief,
            draft=draft,
            candidates=candidates,
            top_k=retrieve_k,
        )
    retrieved_ids = [str(r.get("id", "")) for r in candidates]
    result = _discover_from_candidates(
        brief=brief,
        draft=draft,
        candidates=candidates,
        top_k=judge_top_k,
    )
    ok, detail, signals = evaluate_gate_signals(
        result, brief=brief, draft=draft, use_semantic=True, use_physics=use_physics
    )
    gate = GateEvaluation(
        ok=ok,
        detail=detail,
        result=result,
        confidence=signals.confidence,
        threshold=signals.threshold,
        combined_score=signals.combined_score,
        draft_consistency=signals.draft_consistency,
        physics=signals.physics.to_dict() if signals.physics else None,
        model=signals.model,
    )
    action = "generate" if ok else "abstain"
    return RagDecision(
        ok=ok,
        action=action,
        retrieved_ids=retrieved_ids,
        gate=gate,
    )
