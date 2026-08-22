"""Lightweight hybrid retrieval: BM25 + TF-IDF cosine + RRF fusion.

Why: 2026 hybrid RAG raises recall without embedding-model downloads; Formulagate
remains the evidence Judge. Dense side is pure-Python TF-IDF (no numpy/FAISS).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from formulagate.scoring import keywords

RRF_K_DEFAULT = 60
_TOKEN = re.compile(r"[a-zA-Z]{2,}|\\\\[a-zA-Z]+|[0-9]+")


def _row_blob(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(k) or "")
        for k in ("english", "math_formula", "formula", "title", "text", "scientific_domain")
    )


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def rrf_fuse(
    ranked_id_lists: list[list[str]],
    *,
    k: int = RRF_K_DEFAULT,
    top_k: int = 5,
) -> list[str]:
    """Reciprocal Rank Fusion over rank lists (1-based ranks).

    Why: fuses incompatible BM25/TF-IDF score scales via ranks only.
    """

    scores: dict[str, float] = {}
    for ranked in ranked_id_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.keys(), key=lambda d: (-scores[d], d))
    return ordered[: max(1, top_k)]


class _TfidfIndex:
    """In-memory TF-IDF over formula corpus blobs (stdlib only)."""

    def __init__(self, docs: list[list[str]]) -> None:
        self._docs = docs
        n = len(docs)
        df: Counter[str] = Counter()
        for toks in docs:
            df.update(set(toks))
        self._idf = {
            t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()
        }
        self._vectors = [self._tfidf(toks) for toks in docs]

    def _tfidf(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        length = max(1, len(toks))
        vec = {t: (c / length) * self._idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def top_indices(self, query: str, top_k: int) -> list[int]:
        q = self._tfidf(_tokenize(query))
        scored: list[tuple[float, int]] = []
        for i, doc in enumerate(self._vectors):
            sim = sum(q.get(t, 0.0) * w for t, w in doc.items())
            if sim > 0:
                scored.append((sim, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [i for _, i in scored[: max(1, top_k)]]


class HybridRrfRetriever:
    """BM25 + TF-IDF recall fused with RRF (optional ``rank_bm25`` extra)."""

    def __init__(
        self,
        corpus: list[dict[str, Any]],
        *,
        rrf_k: int = RRF_K_DEFAULT,
        candidate_pool: int = 20,
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HybridRrfRetriever requires optional extra: pip install 'formulagate[rag]'"
            ) from exc

        self._corpus = list(corpus)
        self._rrf_k = rrf_k
        self._pool = max(1, candidate_pool)
        self._blobs = [_row_blob(r) for r in self._corpus]
        tokenized = [_tokenize(b) for b in self._blobs]
        # BM25Okapi rejects empty token lists — use placeholder token.
        safe = [toks if toks else ["_empty"] for toks in tokenized]
        self._bm25 = BM25Okapi(safe)
        self._tfidf = _TfidfIndex(safe)
        self._ids = [str(r.get("id", i)) for i, r in enumerate(self._corpus)]

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Hybrid recall for ``query`` (typically the scientific brief)."""

        pool = max(top_k, self._pool)
        q_toks = _tokenize(query) or list(keywords(query)) or ["_empty"]
        bm25_scores = self._bm25.get_scores(q_toks)
        bm25_order = sorted(
            range(len(self._corpus)),
            key=lambda i: (-float(bm25_scores[i]), self._ids[i]),
        )[:pool]
        tfidf_order = self._tfidf.top_indices(query, pool)
        bm25_ids = [self._ids[i] for i in bm25_order]
        tfidf_ids = [self._ids[i] for i in tfidf_order]
        fused_ids = rrf_fuse([bm25_ids, tfidf_ids], k=self._rrf_k, top_k=top_k)
        by_id = {self._ids[i]: self._corpus[i] for i in range(len(self._corpus))}
        return [by_id[i] for i in fused_ids if i in by_id]
