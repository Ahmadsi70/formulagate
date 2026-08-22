"""Semantic entropy — detect hallucinations via meaning-level uncertainty.

Why this exists:
  Token-level uncertainty (perplexity, sequence probability) is a weak
  hallucination signal because a model can express the same correct answer
  in many different wordings, each with different token probabilities.
  Semantic entropy (Kuhn et al., Nature 2024) solves this by clustering
  outputs by **meaning** before computing entropy: high semantic entropy
  means the model is uncertain about the answer's meaning itself, which is
  a strong indicator of potential hallucination.

  This module provides two paths:

  1.  **Bidirectional entailment clustering** — the original Nature method.
      Uses a Natural Language Inference (NLI) model to check whether two
      sampled answers entail each other.  If A ⇒ B and B ⇒ A, they are
      semantically equivalent and fall into the same cluster.

  2.  **Jaccard overlap clustering** (fallback, no model needed) — clusters
      outputs by token overlap.  Less accurate but dependency-free.

  The output is a ``SemanticEntropy`` dataclass whose ``entropy`` field is
  the primary signal: higher = more uncertain = more likely hallucinated.

References:
  - Kuhn et al. (Nature 2024): "Detecting hallucinations in large language
    models using semantic entropy"
  - Semantic Entropy Probes (2024): hidden states encode this signal
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class SemanticEntropy:
    """Semantic uncertainty measurement over multiple sampled outputs.

    Attributes:
        entropy: Semantic entropy in nats (0 = completely certain, higher = uncertain).
        normalized_entropy: Entropy divided by log(N) so it's in [0, 1].
        n_samples: Number of outputs evaluated.
        n_clusters: Number of distinct meaning clusters found.
        clusters: List of (representative_text, count) for each cluster.
        method: ``"nli"`` or ``"jaccard"`` — which clustering method was used.
    """

    entropy: float = 0.0
    normalized_entropy: float = 0.0
    n_samples: int = 0
    n_clusters: int = 0
    clusters: tuple[tuple[str, int], ...] = ()
    method: str = "jaccard"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entropy": self.entropy,
            "normalized_entropy": self.normalized_entropy,
            "n_samples": self.n_samples,
            "n_clusters": self.n_clusters,
            "clusters": [
                {"text": text[:120], "count": count} for text, count in self.clusters
            ],
            "method": self.method,
        }

    @property
    def is_uncertain(self) -> bool:
        """Convenience: normalized entropy > 0.5 suggests high uncertainty."""
        return self.normalized_entropy > 0.5

    @property
    def confidence(self) -> float:
        """1 − normalized_entropy: higher = more certain."""
        return max(0.0, 1.0 - self.normalized_entropy)


# ─── Jaccard overlap clustering (no model needed) ────────────────────────────


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity."""
    if not a or not b:
        return 0.0
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def jaccard_cluster(samples: Sequence[str], threshold: float = 0.30) -> dict[int, list[str]]:
    """Cluster samples by token overlap.

    Two samples are in the same cluster if their Jaccard similarity ≥ threshold.

    Args:
        samples: List of text outputs to cluster.
        threshold: Jaccard similarity threshold (0.30 default).

    Returns:
        Dict mapping cluster_id → list of sample texts.
    """
    if not samples:
        return {}

    clusters: dict[int, list[str]] = {}
    cluster_ids: list[int] = []

    for text in samples:
        assigned = False
        for cid in clusters:
            representative = clusters[cid][0]
            if _jaccard(text, representative) >= threshold:
                clusters[cid].append(text)
                cluster_ids.append(cid)
                assigned = True
                break
        if not assigned:
            new_id = len(clusters)
            clusters[new_id] = [text]
            cluster_ids.append(new_id)

    return clusters


# ─── NLI-based clustering (semantic entailment) ──────────────────────────────


def nli_cluster(
    samples: Sequence[str],
    *,
    nli_fn: Callable[[str, str], float | None] | None = None,
) -> dict[int, list[str]]:
    """Cluster samples by bidirectional entailment.

    Two samples A and B are semantically equivalent if A entails B AND B entails A.
    The ``nli_fn`` should return a score in [0, 1] where higher means stronger
    entailment.  A threshold of 0.5 is used for the entailment decision.

    Args:
        samples: List of text outputs to cluster.
        nli_fn: ``(premise, hypothesis) -> entailment_score``.  When None, falls
            back to Jaccard clustering (so the caller never gets a hard error).

    Returns:
        Dict mapping cluster_id → list of sample texts.
    """
    if nli_fn is None or not samples:
        return jaccard_cluster(samples)

    clusters: dict[int, list[str]] = {}
    for text in samples:
        assigned = False
        for cid in clusters:
            representative = clusters[cid][0]
            try:
                fwd = nli_fn(representative, text)
                bwd = nli_fn(text, representative)
            except Exception:
                continue
            if fwd is not None and bwd is not None and fwd >= 0.5 and bwd >= 0.5:
                clusters[cid].append(text)
                assigned = True
                break
        if not assigned:
            new_id = len(clusters)
            clusters[new_id] = [text]

    return clusters


# ─── Entropy computation ─────────────────────────────────────────────────────


def _entropy_from_counts(counts: list[int], n_total: int) -> float:
    """Shannon entropy in nats from cluster counts."""
    if n_total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / n_total
            h -= p * math.log(p)
    return h


def compute_semantic_entropy(
    samples: Sequence[str],
    *,
    method: str = "jaccard",
    nli_fn: Callable[[str, str], float | None] | None = None,
    jaccard_threshold: float = 0.30,
) -> SemanticEntropy:
    """Compute semantic entropy over multiple sampled outputs.

    Args:
        samples: Multiple LLM outputs for the same input (≥ 2 recommended).
        method: ``"jaccard"`` (default) or ``"nli"``.
        nli_fn: Entailment function for the ``"nli"`` method.
        jaccard_threshold: Similarity threshold for Jaccard clustering.

    Returns:
        ``SemanticEntropy`` with entropy, normalized entropy, and cluster info.
    """
    if len(samples) < 2:
        return SemanticEntropy(
            entropy=0.0,
            normalized_entropy=0.0,
            n_samples=len(samples),
            n_clusters=min(1, len(samples)),
            clusters=tuple((s, 1) for s in samples),
            method=method,
        )

    if method == "nli" and nli_fn is not None:
        clusters = nli_cluster(samples, nli_fn=nli_fn)
    else:
        clusters = jaccard_cluster(samples, threshold=jaccard_threshold)

    counts = [len(v) for v in clusters.values()]
    n_total = sum(counts)
    n_clusters = len(clusters)

    entropy = _entropy_from_counts(counts, n_total)
    max_entropy = math.log(n_total) if n_total > 0 else 0.0
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0

    # Build cluster summaries: (representative text, count)
    cluster_summaries = tuple(
        (texts[0][:200], len(texts)) for texts in clusters.values()
    )

    return SemanticEntropy(
        entropy=float(entropy),
        normalized_entropy=float(normalized),
        n_samples=n_total,
        n_clusters=n_clusters,
        clusters=cluster_summaries,
        method=method,
    )