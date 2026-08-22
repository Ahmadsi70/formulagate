"""Grounding sensitivity — detect hallucinations by measuring evidence dependence.

Why this exists:
  GASP (Grounding-Aware Sensitivity by Perturbation, Bouke 2026) showed that a
  sentence truly grounded in retrieved evidence will experience a sharp drop in
  likelihood when that evidence is removed.  A hallucinated sentence is by
  definition independent of the evidence, so its likelihood barely changes.

  This module provides a training‑free grounding check that works at two levels:

  1.  **Token‑level** (when log‑probabilities are available): measures the
      Jensen‑Shannon divergence between the token distribution with and without
      the evidence context.  This is the original GASP signal.

  2.  **Semantic‑level** (fallback, no log‑probs needed): measures the change in
      semantic similarity between the draft and two contexts — one with the
      evidence, one with a neutral/generic prompt.  A large similarity drop
      indicates grounding.

  The output is a ``GroundingScore`` that feeds directly into the gate's physics
  signals, making it a fifth orthogonal feature alongside dimension check,
  equivalence, symbol overlap, and candidate‑has‑formula.

References:
  - GASP (Bouke 2026): "Detecting Hallucinations in RAG through Grounding-Aware
    Sensitivity by Perturbation"
  - SelfCheckGPT (Manakul et al. 2023): sampling‑based consistency
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class GroundingScore:
    """How strongly a draft depends on its supposed evidence.

    All values are in [0, 1] unless noted.  Higher = more grounded.
    """

    #: Primary score: semantic similarity drop when evidence is removed.
    #: 1.0 = draft is only explainable with the evidence (strongly grounded).
    #: 0.0 = draft is equally similar with or without evidence (not grounded).
    sensitivity: float = 0.0

    #: Semantic similarity between draft and evidence.
    sim_with_evidence: float = 0.0

    #: Semantic similarity between draft and neutral context.
    sim_without_evidence: float = 0.0

    #: Whether this score came from log‑probs (True) or semantic fallback (False).
    from_logprobs: bool = False

    #: Number of tokens compared (only populated when from_logprobs=True).
    n_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensitivity": self.sensitivity,
            "sim_with_evidence": self.sim_with_evidence,
            "sim_without_evidence": self.sim_without_evidence,
            "from_logprobs": self.from_logprobs,
            "n_tokens": self.n_tokens,
        }

    @property
    def is_grounded(self) -> bool:
        """Convenience threshold: sensitivity > 0.15 suggests grounding."""
        return self.sensitivity > 0.15


# ─── Semantic fallback (no log‑probs needed) ─────────────────────────────────


def _jaccard_similarity(a: str, b: str) -> float:
    """Token‑level Jaccard similarity between two strings.

    Fast, deterministic, dependency‑free.  Suitable as a fallback when no
    embedding model is available.
    """
    if not a or not b:
        return 0.0
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _overlap_coefficient(a: str, b: str) -> float:
    """Overlap coefficient: |A ∩ B| / min(|A|, |B|).

    This is more sensitive than Jaccard for short drafts against long evidence
    passages.  A draft of 5 words that all appear in a 200‑word passage should
    score 1.0, not 0.025.
    """
    if not a or not b:
        return 0.0
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))


def semantic_grounding(
    draft: str,
    evidence: str,
    *,
    sim_fn: Callable[[str, str], float] | None = None,
) -> GroundingScore:
    """Compute grounding sensitivity from semantic similarity alone.

    The "neutral context" is an empty string — a draft that is equally similar
    to nothing as it is to the evidence is, by definition, not grounded.

    Args:
        draft: The LLM output being judged.
        evidence: The retrieved passage that supposedly supports the draft.
        sim_fn: Similarity function ``(text_a, text_b) -> float``.  Defaults to
            ``_overlap_coefficient`` (fast, dependency‑free).  Pass an embedding
            cosine‑similarity function for higher accuracy.

    Returns:
        ``GroundingScore`` with ``from_logprobs=False``.
    """
    sim = sim_fn or _overlap_coefficient

    sim_with = sim(draft, evidence) if evidence else 0.0
    sim_without = sim(draft, "")  # always 0.0 with overlap-based sim

    # For semantic similarity, the grounding sensitivity is the difference.
    sensitivity = max(0.0, sim_with - sim_without)

    return GroundingScore(
        sensitivity=float(sensitivity),
        sim_with_evidence=float(sim_with),
        sim_without_evidence=float(sim_without),
        from_logprobs=False,
    )


# ─── Token‑level grounding (GASP‑style, needs log‑probs) ────────────────────


def token_level_grounding(
    logprobs_with: Sequence[dict[str, float]] | None,
    logprobs_without: Sequence[dict[str, float]] | None,
    *,
    top_k: int = 20,
) -> GroundingScore:
    """GASP‑style grounding check from token log‑probability distributions.

    For each token position, computes the Jensen‑Shannon divergence between the
    top‑k token distribution with and without evidence.  The mean JS divergence
    across all positions is the grounding sensitivity score.

    Args:
        logprobs_with: Per‑token log‑prob dicts when evidence IS in the prompt.
        logprobs_without: Per‑token log‑prob dicts when evidence is REMOVED.
        top_k: How many top tokens to keep per position.

    Returns:
        ``GroundingScore`` with ``from_logprobs=True``.
    """
    if not logprobs_with or not logprobs_without:
        return GroundingScore(from_logprobs=True)

    n = min(len(logprobs_with), len(logprobs_without))
    if n == 0:
        return GroundingScore(from_logprobs=True)

    js_divergences: list[float] = []
    for i in range(n):
        dist_with = _top_k_probs(logprobs_with[i], top_k)
        dist_without = _top_k_probs(logprobs_without[i], top_k)
        if not dist_with or not dist_without:
            continue
        js = _jensen_shannon_divergence(dist_with, dist_without)
        js_divergences.append(js)

    if not js_divergences:
        return GroundingScore(from_logprobs=True, n_tokens=0)

    # Mean JS divergence across all token positions.
    mean_js = sum(js_divergences) / len(js_divergences)

    # Normalise: JS ∈ [0, ln(2)], so divide by ln(2) to get [0, 1].
    sensitivity = mean_js / math.log(2)

    return GroundingScore(
        sensitivity=float(min(1.0, sensitivity)),
        sim_with_evidence=0.0,   # not applicable for log‑prob path
        sim_without_evidence=0.0,
        from_logprobs=True,
        n_tokens=len(js_divergences),
    )


def _top_k_probs(logprob_dict: dict[str, float], k: int) -> dict[str, float]:
    """Keep the top‑k tokens by probability and renormalise."""
    if not logprob_dict:
        return {}
    probs = {tok: math.exp(lp) for tok, lp in logprob_dict.items()}
    top = dict(Counter(probs).most_common(k))
    total = sum(top.values())
    if total <= 0:
        return {}
    return {tok: v / total for tok, v in top.items()}


def _jensen_shannon_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen‑Shannon divergence between two discrete distributions."""
    all_keys = set(p) | set(q)
    js = 0.0
    for key in all_keys:
        pv = p.get(key, 0.0)
        qv = q.get(key, 0.0)
        m = (pv + qv) / 2.0
        if m > 0:
            if pv > 0:
                js += pv * math.log(pv / m)
            if qv > 0:
                js += qv * math.log(qv / m)
    return js / 2.0


# ─── High‑level grounding check (auto‑selects the best available path) ──────


def check_grounding(
    draft: str,
    evidence: str,
    *,
    logprobs_with: Sequence[dict[str, float]] | None = None,
    logprobs_without: Sequence[dict[str, float]] | None = None,
    sim_fn: Callable[[str, str], float] | None = None,
) -> GroundingScore:
    """Compute grounding sensitivity, preferring log‑probs when available.

    Args:
        draft: LLM output to evaluate.
        evidence: Retrieved passage that supposedly supports the draft.
        logprobs_with: Token log‑probs with evidence (optional).
        logprobs_without: Token log‑probs without evidence (optional).
        sim_fn: Semantic similarity function for the fallback path.

    Returns:
        ``GroundingScore`` — the sensitivity field is the primary signal.
    """
    if logprobs_with and logprobs_without:
        return token_level_grounding(logprobs_with, logprobs_without)

    return semantic_grounding(draft, evidence, sim_fn=sim_fn)