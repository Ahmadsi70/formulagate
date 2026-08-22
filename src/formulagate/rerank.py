"""Post-retrieval rerankers for scientific RAG.

Why: hybrid recall optimizes coverage; rerank optimizes precision before the
Formulagate evidence gate (Search → Rerank → Judge).
"""

from __future__ import annotations

from typing import Any, Protocol

from formulagate.domain import classify_domain
from formulagate.rag import Retriever
from formulagate.scoring import keywords, score_record


class Reranker(Protocol):
    """Reorder retrieved candidates using brief+draft evidence signals."""

    def rerank(
        self,
        *,
        brief: str,
        draft: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        ...


class FormulagateReranker:
    """Rerank by Formulagate ``score = overlap + 2*rel`` (no extra model)."""

    def rerank(
        self,
        *,
        brief: str,
        draft: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        domain = classify_domain(brief)
        keys = keywords(brief + " " + draft)
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for row in candidates:
            score, rel = score_record(keys, row, domain)
            scored.append((score, rel, row))
        scored.sort(key=lambda x: (-x[0], -x[1], str(x[2].get("id", ""))))
        # Keep zeros at end but still return up to top_k for auditability.
        return [row for _, _, row in scored[: max(1, top_k)]]


class CrossEncoderReranker:
    """Optional neural rerank (``formulagate[dense]`` + cross-encoder model)."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "CrossEncoderReranker requires: pip install 'formulagate[dense]'"
            ) from exc
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        *,
        brief: str,
        draft: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        query = f"{brief} {draft}"
        pairs = []
        for row in candidates:
            blob = " ".join(
                str(row.get(k) or "")
                for k in ("english", "math_formula", "formula", "title", "text")
            )
            pairs.append([query, blob])
        scores = self._model.predict(pairs)
        order = sorted(
            range(len(candidates)),
            key=lambda i: (-float(scores[i]), str(candidates[i].get("id", ""))),
        )
        return [candidates[i] for i in order[: max(1, top_k)]]


class RerankingRetriever:
    """Retrieve a wide pool, then Formulagate/CE-rerank to ``top_k``.

    ``brief``/``draft`` for rerank are taken from the last ``set_context`` call
    or fall back to the retrieve ``query`` for both fields.
    """

    def __init__(
        self,
        base: Retriever,
        reranker: Reranker,
        *,
        pool_k: int = 20,
    ) -> None:
        self._base = base
        self._reranker = reranker
        self._pool_k = max(1, pool_k)
        self._brief: str | None = None
        self._draft: str | None = None

    def set_context(self, *, brief: str, draft: str) -> None:
        """Bind brief/draft used by the reranker on the next retrieve."""

        self._brief = brief
        self._draft = draft

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        pool = self._base.retrieve(query, top_k=max(top_k, self._pool_k))
        brief = self._brief if self._brief is not None else query
        draft = self._draft if self._draft is not None else query
        return self._reranker.rerank(
            brief=brief,
            draft=draft,
            candidates=pool,
            top_k=top_k,
        )
