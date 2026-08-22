"""Semantic Gate v0.5 — embedding evidence check with calibrated confidence.

Why:
  Lexical overlap accepts weak analogies ("QFT renormalisation" answered with
  "2+2=4") whenever the vocabulary happens to align. Embeddings measure whether
  the draft is actually about the brief, and whether the retrieved formula
  actually answers it.

Algorithm:
  1. draft_consistency   = cos(embed(brief), embed(draft))
  2. candidate_relevance = cos(embed((brief+draft)/2), embed(candidate))
  3. combined_score      = lexical_norm * 0.3 + candidate_relevance * 0.7
  4. confidence          = sigmoid(a * combined_score + b)      [calibration]
  5. accept              ⟺ confidence >= threshold              [calibration]

Steps 4–5 live in :mod:`formulagate.calibration`; every parameter they use is
learned from labeled data, so the accept/abstain boundary is auditable instead
of hand-tuned. Names are re-exported here for backward compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from formulagate.calibration import (
    LEXICAL_SCORE_MAX,
    CalibrationError,
    CalibrationParams,
    CalibrationReport,
    brier_score,
    calibrate_confidence,
    evaluate_calibration,
    expected_calibration_error,
    find_optimal_threshold,
    fit_calibration,
    get_calibration,
    load_calibration,
    save_calibration,
    set_calibration,
    sigmoid,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CalibrationError",
    "CalibrationParams",
    "CalibrationReport",
    "SemanticVerdict",
    "brier_score",
    "calibrate_confidence",
    "evaluate_calibration",
    "evaluate_semantic_gate",
    "expected_calibration_error",
    "find_optimal_threshold",
    "fit_calibration",
    "get_calibration",
    "load_calibration",
    "save_calibration",
    "set_calibration",
    "sigmoid",
]

# ─── Evidence thresholds (structural, not probabilistic) ──────────────────────

DRAFT_CONSISTENCY_HARD_THRESHOLD = 0.12
DRAFT_CONSISTENCY_SOFT_THRESHOLD = 0.20
LEXICAL_WEIGHT = 0.3
SEMANTIC_WEIGHT = 0.7

# ─── Shared model ─────────────────────────────────────────────────────────────

_MODEL = None
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_model():
    """Lazy-load SentenceTransformer; ``None`` ⇒ callers use lexical fallback."""

    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer

            _MODEL = SentenceTransformer(_MODEL_NAME)
            logger.info("SemanticGate: loaded %s", _MODEL_NAME)
        except ImportError:
            logger.warning("sentence-transformers not available, semantic gate disabled")
            return None
        except Exception as exc:  # noqa: BLE001 — model load must never abort the gate
            logger.warning("Failed to load ST: %s", exc)
            return None
    return _MODEL


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two L2-normalised vectors."""

    return float(np.dot(a, b))


# ─── Verdict ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SemanticVerdict:
    """Gate outcome plus every Information Point behind it (audit surface)."""

    ok: bool
    action: str  # "generate" | "abstain"
    detail: str
    draft_consistency: float
    candidate_relevance: float
    lexical_score: float
    combined_score: float
    confidence: float = 0.5
    threshold_used: float = 0.50


def _verdict(
    *,
    ok: bool,
    detail: str,
    draft_consistency: float,
    candidate_relevance: float,
    lexical_score: float,
    combined: float,
    confidence: float,
    threshold: float,
) -> SemanticVerdict:
    return SemanticVerdict(
        ok=ok,
        action="generate" if ok else "abstain",
        detail=detail,
        draft_consistency=draft_consistency,
        candidate_relevance=candidate_relevance,
        lexical_score=lexical_score,
        combined_score=combined,
        confidence=confidence,
        threshold_used=threshold,
    )


def evaluate_semantic_gate(
    *,
    brief: str,
    draft: str,
    candidate_text: str,
    lexical_score: float = 0.0,
    lexical_max: float = LEXICAL_SCORE_MAX,
    calibration: CalibrationParams | None = None,
) -> SemanticVerdict:
    """Decide whether a retrieved candidate is sufficient evidence.

    Args:
        brief: Scientific question or topic.
        draft: Proposed answer (possibly a weak analogy).
        candidate_text: Retrieved formula text.
        lexical_score: ``overlap + 2*rel`` from the lexical gate.
        lexical_max: Normalisation ceiling for ``lexical_score``.
        calibration: Override for the process-wide calibration.

    Returns:
        SemanticVerdict — decision, calibrated confidence, component scores.
    """

    model = _get_model()
    cal = calibration or get_calibration()
    lex_norm = min(float(lexical_score) / max(lexical_max, 1.0), 1.0)

    # ── Lexical fallback: no embeddings available ────────────────────────
    if model is None:
        confidence = calibrate_confidence(lex_norm, cal)
        ok = confidence >= cal.threshold
        detail = (
            "gate_pass (lexical fallback)"
            if ok
            else f"confidence {confidence:.2f} < {cal.threshold:.2f} (lexical fallback)"
        )
        return _verdict(
            ok=ok, detail=detail, draft_consistency=0.5, candidate_relevance=lex_norm,
            lexical_score=lexical_score, combined=lex_norm,
            confidence=confidence, threshold=cal.threshold,
        )

    embs = model.encode(
        [brief[:512], draft[:512], candidate_text[:512]],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    brief_emb = np.asarray(embs[0], dtype=np.float32)
    draft_emb = np.asarray(embs[1], dtype=np.float32)
    cand_emb = np.asarray(embs[2], dtype=np.float32)

    draft_consistency = _cosine_sim(brief_emb, draft_emb)

    query_emb = (brief_emb + draft_emb) / 2.0
    query_emb = query_emb / (float(np.linalg.norm(query_emb)) or 1.0)
    candidate_relevance = _cosine_sim(query_emb, cand_emb)

    combined = lex_norm * LEXICAL_WEIGHT + candidate_relevance * SEMANTIC_WEIGHT
    confidence = calibrate_confidence(combined, cal)

    # Hard fail: the draft is about a different topic than the brief, so no
    # amount of candidate relevance can make the citation honest.
    if draft_consistency < DRAFT_CONSISTENCY_HARD_THRESHOLD:
        return _verdict(
            ok=False,
            detail=(
                f"draft consistency {draft_consistency:.2f} < "
                f"{DRAFT_CONSISTENCY_HARD_THRESHOLD} (unrelated topic)"
            ),
            draft_consistency=draft_consistency, candidate_relevance=candidate_relevance,
            lexical_score=lexical_score, combined=combined,
            confidence=confidence, threshold=cal.threshold,
        )

    threshold = (
        cal.strict_threshold
        if draft_consistency < DRAFT_CONSISTENCY_SOFT_THRESHOLD
        else cal.threshold
    )

    if confidence < threshold:
        return _verdict(
            ok=False,
            detail=(
                f"confidence {confidence:.2f} < {threshold:.2f} "
                f"(dc={draft_consistency:.2f}, combined={combined:.2f})"
            ),
            draft_consistency=draft_consistency, candidate_relevance=candidate_relevance,
            lexical_score=lexical_score, combined=combined,
            confidence=confidence, threshold=threshold,
        )

    return _verdict(
        ok=True, detail="gate_pass (semantic)",
        draft_consistency=draft_consistency, candidate_relevance=candidate_relevance,
        lexical_score=lexical_score, combined=combined,
        confidence=confidence, threshold=threshold,
    )
