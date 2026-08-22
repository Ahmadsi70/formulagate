"""Discover top linkages and evaluate accept threshold τ.

Why: productize the IR gate — LLM/draft text is accepted only when a
domain-relevant formula/document clears the score floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from formulagate.calibration import (
    LEXICAL_SCORE_MAX,
    calibrate_confidence,
    get_calibration,
    get_multi_calibration,
)
from formulagate.corpus import load_corpus
from formulagate.domain import Domain, classify_domain
from formulagate.physics_signals import PhysicsSignals, physics_signals
from formulagate.scoring import keywords, score_record

# Thresholds from the reference algorithm.
MIN_SCORE_GENERAL = 2
MIN_SCORE_DOMAIN = 3
MIN_MATH_RELEVANCE = 1


@dataclass(frozen=True)
class LinkageEntry:
    """One ranked formula/document hit."""

    record_id: str
    score: int
    math_relevance: int
    formula_excerpt: str
    scientific_domain: str | None
    full_candidate_text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiscoveryResult:
    """Full discovery payload for auditing."""

    domain: Domain
    corpus_path: str
    entries: list[LinkageEntry]
    error: str | None = None


@dataclass(frozen=True)
class GateSignals:
    """Calibrated evidence readout behind an accept/reject decision.

    Kept separate from the boolean so audits can answer "how sure was it?"
    even when the verdict itself is unchanged.
    """

    confidence: float | None = None
    threshold: float | None = None
    combined_score: float | None = None
    draft_consistency: float | None = None
    semantic: bool = False
    physics: PhysicsSignals | None = None
    model: str = "platt"


@dataclass(frozen=True)
class GateEvaluation:
    """Accept/reject decision plus its calibrated confidence."""

    ok: bool
    detail: str
    result: DiscoveryResult
    confidence: float | None = None
    threshold: float | None = None
    combined_score: float | None = None
    draft_consistency: float | None = None
    physics: dict[str, float] | None = None
    model: str = "platt"


def discover_linkages(
    *,
    brief: str,
    candidate_text: str,
    corpus_path: Path,
    top_k: int = 3,
) -> DiscoveryResult:
    """Rank corpus rows by score = overlap + 2*rel against brief+candidate."""

    path = Path(corpus_path)
    domain = classify_domain(brief)
    if not path.is_file():
        return DiscoveryResult(
            domain=domain,
            corpus_path=str(path),
            entries=[],
            error=f"corpus not found: {path}",
        )
    return rank_records(
        brief=brief,
        candidate_text=candidate_text,
        rows=load_corpus(path),
        top_k=top_k,
        corpus_path=str(path.resolve()),
    )


def rank_records(
    *,
    brief: str,
    candidate_text: str,
    rows: list[dict[str, Any]],
    top_k: int = 3,
    corpus_path: str = "<memory>",
    use_ml_domain: bool | None = None,
) -> DiscoveryResult:
    """Same ranking as :func:`discover_linkages`, over rows already in memory.

    Exists so an SDK caller can pass records straight from their own store
    instead of being forced to serialise a corpus file first. ``use_ml_domain``
    is forwarded to :func:`classify_domain`; pass ``False`` to guarantee that no
    embedding model is loaded.
    """

    domain = classify_domain(brief, use_ml=use_ml_domain)
    keys = keywords(brief + " " + candidate_text)
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for row in rows:
        score, rel = score_record(keys, row, domain)
        if score > 0:
            scored.append((score, rel, row))
    scored.sort(key=lambda x: (-x[0], -x[1], str(x[2].get("id", ""))))

    entries: list[LinkageEntry] = []
    for score, rel, row in scored[: max(1, top_k)]:
        mf = str(row.get("math_formula") or row.get("formula") or "")
        excerpt = mf[:280] + ("…" if len(mf) > 280 else "")
        # Build the full candidate text (same structure used during training)
        # Include all available text fields so physics_signals has maximum context
        full_text = " ".join(
            str(row.get(k) or "")
            for k in ("math_formula", "formula", "english", "text", "title", "abstract", "context")
        )
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
                full_candidate_text=full_text,
                meta=meta,
            )
        )
    return DiscoveryResult(domain=domain, corpus_path=corpus_path, entries=entries)


def _lexical_signals(score: int) -> GateSignals:
    """Report the calibrated confidence implied by a lexical score alone."""

    cal = get_calibration()
    combined = min(float(score) / LEXICAL_SCORE_MAX, 1.0)
    return GateSignals(
        confidence=calibrate_confidence(combined, cal),
        threshold=cal.threshold,
        combined_score=combined,
        semantic=False,
    )


def _fuse(signals: GateSignals, physics: PhysicsSignals) -> GateSignals:
    """Recompute confidence from the fused model when one is installed.

    Without a fused model nothing changes — the physics features are still
    reported, they simply do not move the probability.

    The fused model was trained on semantic-combined scores (range [0, 1]) via
    ``bench_real.py --embedder minilm``.  When the gate runs in the default
    (lexical, no embeddings) path the ``combined_score`` is ``score/15``, which
    has a very different distribution.  We rescale to approximate the training
    range so the fused weights remain meaningful: ``combined / max_observed``
    where ``max_observed`` is clamped to ``[0.01, 1.0]`` to avoid division by
    zero and to honour the [0,1] contract the model expects.
    """

    model = get_multi_calibration()
    expected = ("score",) + PhysicsSignals.FEATURE_NAMES
    if model is None or tuple(model.feature_names) != expected:
        return signals

    raw_score = float(signals.combined_score or 0.0)
    if not signals.semantic and raw_score > 0:
        # Normalise lexical scores to approximate the semantic-score distribution
        # that the fused model was trained on.  Lexical scores are coarser (integer
        # counts / LEXICAL_SCORE_MAX) and cluster around 0.3–0.5; semantic scores
        # use cosine-similarity and spread across the full [0,1] range.
        # We apply an adaptive rescaling: stretch the distribution while preserving
        # rank order (monotonic), with a ceiling at 1.0.
        norm = min(raw_score * 1.4, 1.0)  # expand lexical range
        # Second pass: if score is very low (<0.25), compress toward zero
        # so the fused model doesn't see inflated low-confidence scores.
        fused_score = norm * norm if raw_score < 0.25 else norm
    else:
        fused_score = raw_score

    features = [fused_score, *physics.as_features()]
    return replace(
        signals,
        confidence=model.confidence(features),
        threshold=model.threshold,
        model="fused",
    )


def evaluate_gate_signals(
    result: DiscoveryResult,
    *,
    min_score_general: int = MIN_SCORE_GENERAL,
    min_score_domain: int = MIN_SCORE_DOMAIN,
    min_math_relevance: int = MIN_MATH_RELEVANCE,
    brief: str = "",
    draft: str = "",
    use_semantic: bool = False,
    use_physics: bool = True,
    conformal: "ConformalCalibration | None" = None,
    semantic_entropy: "SemanticEntropy | None" = None,
) -> tuple[bool, str, GateSignals]:
    """Apply acceptance thresholds τ and return the calibrated evidence too.

    With ``use_semantic=True`` the decision is taken by the embedding gate,
    whose accept boundary is the *learned* confidence threshold. The lexical
    path keeps its integer thresholds (stable, dependency-free) and reports
    confidence as an advisory signal.

    ``use_physics`` adds a deterministic veto in front of both paths: a draft
    whose own equation fails dimensional analysis is rejected no matter how well
    it reads. Similarity cannot see that error — it is the one class of
    hallucination that algebra refutes outright.

    ``conformal``, when provided, applies a risk-controlled threshold on top of
    the calibrated confidence.  The gate may pass the lexical/semantic bar but
    still be rejected if its confidence is below the conformal threshold τ.

    ``semantic_entropy``, when provided, adjusts confidence downward when the
    model shows high uncertainty about the meaning of its output.  A draft with
    normalized entropy > 0.5 has its confidence multiplied by (1 − entropy),
    making it harder to pass the threshold.
    """

    if result.error:
        return False, result.error, GateSignals()
    if not result.entries:
        return False, "gate: no domain-relevant formula/document matched", GateSignals()
    best = result.entries[0]
    if not best.formula_excerpt.strip():
        return False, "gate: top entry missing formula excerpt", GateSignals()

    # Use full_candidate_text for physics_signals (matches training distribution).
    # Fall back to formula_excerpt + brief when full text is unavailable
    # (backward compat with cached DiscoveryResult).
    candidate_for_physics = best.full_candidate_text or f"{best.formula_excerpt} {brief}"

    # Symbol grounding: resolve symbols from the draft's context to their
    # physical dimensions, feeding overrides to the dimensional analysis layer.
    # This lifts classic equations (F = G m1 m2 / r^2, E = h nu) from
    # "unknown" to "consistent" by grounding ambiguous single-letter symbols
    # (c, h, G, e, ...) via the 4-layer resolver (doc, domain, ontology, table).
    physics_overrides = None
    if use_physics and draft:
        try:
            from formulagate.symbol_grounding import ground_symbols
            grounding = ground_symbols(draft, context_before="")
            physics_overrides = grounding.as_overrides() or None
        except Exception:
            physics_overrides = None  # never let grounding crash the gate

    physics = (
        physics_signals(draft, candidate_for_physics, overrides=physics_overrides, domain=result.domain)
        if use_physics and draft
        else PhysicsSignals()
    )
    if physics.dimension_ok < 0:
        return (
            False,
            "gate: draft equation fails dimensional analysis (physically impossible)",
            GateSignals(physics=physics),
        )

    # ── Physics constraints veto ─────────────────────────────────────────
    if use_physics and draft:
        try:
            from formulagate.physics_constraints import check_physics_constraints

            constraints = check_physics_constraints(draft)
            if constraints.n_inconsistent > 0:
                return (
                    False,
                    f"gate: draft violates physics constraint: "
                    + "; ".join(v.detail for v in constraints.verdicts if v.is_reject),
                    GateSignals(physics=physics),
                )
        except Exception:
            pass  # Degrade gracefully — never let constraint checks crash the gate.

    # ── Semantic gate (embeddings + calibrated threshold) ────────────────
    if use_semantic and brief and draft:
        try:
            from formulagate.semantic_gate import evaluate_semantic_gate

            semantic = evaluate_semantic_gate(
                brief=brief,
                draft=draft,
                candidate_text=best.full_candidate_text or best.formula_excerpt,
                lexical_score=float(best.score),
            )
            signals = _fuse(
                GateSignals(
                    confidence=semantic.confidence,
                    threshold=semantic.threshold_used,
                    combined_score=semantic.combined_score,
                    draft_consistency=semantic.draft_consistency,
                    semantic=True,
                    physics=physics,
                ),
                physics,
            )
            if signals.model == "fused":
                accept = signals.confidence >= signals.threshold
                detail = "gate_pass (fused)" if accept else "gate: fused confidence below θ"
                return accept, detail, signals
            if semantic.action == "abstain":
                return False, semantic.detail, signals
            return True, "gate_pass (semantic)", signals
        except Exception:  # noqa: BLE001 — degrade to the lexical gate, never abort
            pass

    # ── Lexical gate (original) ──────────────────────────────────────────
    signals = _fuse(replace(_lexical_signals(best.score), physics=physics), physics)
    min_score = min_score_general if result.domain == "general" else min_score_domain
    if best.score < min_score:
        return (
            False,
            f"gate: top score {best.score} < minimum {min_score} for domain={result.domain}",
            signals,
        )
    if result.domain == "mathematics" and best.math_relevance < min_math_relevance:
        return (
            False,
            f"gate: math_relevance {best.math_relevance} < {min_math_relevance} (weak analog)",
            signals,
        )
    # Physics also requires domain relevance (like mathematics)
    if result.domain == "physics" and best.math_relevance < min_math_relevance:
        return (
            False,
            f"gate: physics_relevance {best.math_relevance} < {min_math_relevance} (weak analog)",
            signals,
        )
    # Pharmacology requires domain relevance
    if result.domain == "pharmacology" and best.math_relevance < min_math_relevance:
        return (
            False,
            f"gate: pharma_relevance {best.math_relevance} < {min_math_relevance} (weak analog)",
            signals,
        )
    if signals.model == "fused" and signals.confidence < signals.threshold:
        # Installing a fused model means asking it to decide; reporting a
        # sub-threshold confidence next to an accept would be incoherent.
        return False, "gate: fused confidence below θ", signals

    # ── Semantic entropy modifier ────────────────────────────────────────
    if semantic_entropy is not None and signals.confidence is not None:
        if semantic_entropy.is_uncertain:
            # High semantic uncertainty → penalize confidence.
            adjusted = signals.confidence * (1.0 - semantic_entropy.normalized_entropy)
            signals = replace(signals, confidence=adjusted)
            if adjusted < (signals.threshold or 0.5):
                return (
                    False,
                    f"gate: semantic entropy too high "
                    f"(entropy={semantic_entropy.normalized_entropy:.3f}, "
                    f"confidence adjusted {signals.confidence:.3f}→{adjusted:.3f})",
                    signals,
                )

    if conformal is not None and signals.confidence is not None:
        if not conformal.accepts(signals.confidence):
            return (
                False,
                f"gate: confidence {signals.confidence:.4f} below conformal "
                f"threshold {conformal.threshold:.4f} (α={conformal.alpha:.1%})",
                signals,
            )
    return True, "gate_pass", signals


def evaluate_gate(
    result: DiscoveryResult,
    *,
    min_score_general: int = MIN_SCORE_GENERAL,
    min_score_domain: int = MIN_SCORE_DOMAIN,
    min_math_relevance: int = MIN_MATH_RELEVANCE,
    brief: str = "",
    draft: str = "",
    use_semantic: bool = False,
) -> tuple[bool, str]:
    """Boolean view of :func:`evaluate_gate_signals` (stable public contract)."""

    ok, detail, _ = evaluate_gate_signals(
        result,
        min_score_general=min_score_general,
        min_score_domain=min_score_domain,
        min_math_relevance=min_math_relevance,
        brief=brief,
        draft=draft,
        use_semantic=use_semantic,
    )
    return ok, detail


def run_gate(
    *,
    brief: str,
    candidate_text: str,
    corpus_path: Path,
    top_k: int = 3,
    use_semantic: bool = False,
    use_physics: bool = True,
) -> GateEvaluation:
    """Discover + evaluate in one call."""

    result = discover_linkages(
        brief=brief,
        candidate_text=candidate_text,
        corpus_path=corpus_path,
        top_k=top_k,
    )
    ok, detail, signals = evaluate_gate_signals(
        result,
        brief=brief,
        draft=candidate_text,
        use_semantic=use_semantic,
        use_physics=use_physics,
    )
    return GateEvaluation(
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
