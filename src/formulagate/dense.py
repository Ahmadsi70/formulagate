"""Dense hybrid retrieval: BM25 + FAISS embeddings + RRF.

Why: semantic recall (paraphrase) complements BM25 exact tokens; Formulagate
still judges evidence. Tests use HashEmbedder; production uses SentenceTransformer.
"""

from __future__ import annotations

from typing import Any, Protocol

from formulagate.hybrid import RRF_K_DEFAULT, _row_blob, _tokenize, rrf_fuse
from formulagate.scoring import keywords


class Embedder(Protocol):
    """Maps texts to L2-normalized float vectors for FAISS IP search."""

    def encode(self, texts: list[str]) -> Any:
        """Return ``(n, dim)`` float32 array (normalized rows)."""


class HashEmbedder:
    """Deterministic bag-hash embedder for CI (no model download)."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> Any:
        import numpy as np

        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _tokenize(text) or ["_empty"]:
                h = hash(tok) % self.dim
                mat[i, h] += 1.0
            n = float(np.linalg.norm(mat[i])) or 1.0
            mat[i] /= n
        return mat


class SentenceTransformerEmbedder:
    """Production MiniLM embedder (``formulagate[dense]``)."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pip install 'formulagate[dense]' for SentenceTransformerEmbedder"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> Any:
        import numpy as np

        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


class DenseHybridRetriever:
    """BM25 + FAISS dense recall fused with RRF.

    Default embedder is SentenceTransformer (MiniLM) for semantic recall.
    For CI/testing without model download, pass HashEmbedder explicitly.
    """

    def __init__(
        self,
        corpus: list[dict[str, Any]],
        *,
        embedder: Embedder | None = None,
        rrf_k: int = RRF_K_DEFAULT,
        candidate_pool: int = 20,
        batch_size: int = 64,
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi
            import faiss
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "DenseHybridRetriever requires: pip install 'formulagate[dense]'"
            ) from exc

        self._np = np
        self._faiss = faiss
        self._corpus = list(corpus)
        self._rrf_k = rrf_k
        self._pool = max(1, candidate_pool)
        self._ids = [str(r.get("id", i)) for i, r in enumerate(self._corpus)]
        self._blobs = [_row_blob(r) for r in self._corpus]
        tokenized = [_tokenize(b) for b in self._blobs]
        safe = [toks if toks else ["_empty"] for toks in tokenized]
        self._bm25 = BM25Okapi(safe)
        self._embedder = embedder or SentenceTransformerEmbedder()

        # Batch-encode for large corpora
        blobs = self._blobs
        if hasattr(self._embedder, '_model') or len(blobs) > 500:
            # Use batching for SentenceTransformer
            emb_list = []
            for i in range(0, len(blobs), batch_size):
                batch = blobs[i:i + batch_size]
                emb_list.append(self._embedder.encode(batch))
            emb = np.concatenate(emb_list, axis=0)
        else:
            emb = self._embedder.encode(blobs)

        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[0] != len(self._corpus):
            raise ValueError("embedder.encode must return (n_docs, dim) array")
        # Ensure L2-normalized for IndexFlatIP ~= cosine.
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        self._index = faiss.IndexFlatIP(emb.shape[1])
        self._index.add(emb)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Hybrid BM25+dense recall for the scientific brief."""

        pool = max(top_k, self._pool)
        q_toks = _tokenize(query) or list(keywords(query)) or ["_empty"]
        bm25_scores = self._bm25.get_scores(q_toks)
        bm25_order = sorted(
            range(len(self._corpus)),
            key=lambda i: (-float(bm25_scores[i]), self._ids[i]),
        )[:pool]
        q = self._embedder.encode([query])
        q = self._np.asarray(q, dtype=self._np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        nrm = float(self._np.linalg.norm(q[0])) or 1.0
        q = q / nrm
        _scores, idxs = self._index.search(q, min(pool, len(self._corpus)))
        dense_order = [int(i) for i in idxs[0] if int(i) >= 0]
        bm25_ids = [self._ids[i] for i in bm25_order]
        dense_ids = [self._ids[i] for i in dense_order]
        fused = rrf_fuse([bm25_ids, dense_ids], k=self._rrf_k, top_k=top_k)
        by_id = {self._ids[i]: self._corpus[i] for i in range(len(self._corpus))}
        return [by_id[i] for i in fused if i in by_id]
