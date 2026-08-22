"""Physics evidence the gate can act on, extracted deterministically.

The measured ceiling of the similarity score on live arXiv data was AUC 0.711:
lexical overlap and embeddings cannot tell "cites the right paper" from "sounds
like the right paper". These features are orthogonal to that score because
they are computed by algebra, not by resemblance:

    dimension_ok           +1 provable physics, −1 provably impossible, 0 unknown
    equivalence            +1 the draft's relation is in the record, −1 it is
                              contradicted over shared symbols, 0 unknown
    symbol_overlap         Jaccard over formula symbols (not words)
    candidate_has_formula  whether the record contains any parsable equation
    grounding_sensitivity  how much the draft's wording depends on the evidence
                              (GASP-style: 1.0 = strongly grounded, 0.0 = not)

Cost control matters: every rung is bounded (span count, pair count) so a single
adversarial abstract cannot stall the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from formulagate.dimensions import check_dimensions
from formulagate.equivalence import check_equivalence
from formulagate.formula_extract import extract_formulas

MAX_DRAFT_FORMULAS = 3
MAX_CANDIDATE_FORMULAS = 6
MAX_PAIR_COMPARISONS = 9


@dataclass(frozen=True)
class PhysicsSignals:
    """Fixed-width, deterministic feature bundle. All defaults mean "no evidence"."""

    FEATURE_NAMES = (
        "dimension_ok",
        "equivalence",
        "symbol_overlap",
        "candidate_has_formula",
        "grounding_sensitivity",
        "constraints_score",
    )

    dimension_ok: float = 0.0
    equivalence: float = 0.0
    symbol_overlap: float = 0.0
    candidate_has_formula: float = 0.0
    grounding_sensitivity: float = 0.0
    constraints_score: float = 0.0

    def as_features(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in self.FEATURE_NAMES)

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.FEATURE_NAMES}

    @property
    def is_hard_reject(self) -> bool:
        """Physics that cannot be true regardless of how well it reads."""

        return self.dimension_ok < 0 or self.equivalence < 0


def _dimension_signal(draft_formulas, overrides=None, domain=None) -> float:
    """−1 dominates: one provably impossible equation poisons the whole draft.

    Two-tier check — cheap first, deep when needed (plus SMT verification on
    all consistent results):

    1. Exact walker (``check_dimensions``): table-lookup, instantaneous. Catches
       obvious mismatches like ``E = m c^3`` where every symbol is in the table.
       An ``inconsistent`` result here is a hard reject.
    2. SMT solver (``check_dimensions_smt``): fires on both:
       a. Unknown results from the exact walker — Z3 can prove inconsistency
          even with floating unknowns.
       b. Consistent results from the exact walker — Z3 double-checks for
          subtle multi-term errors the walker's table-lookup may miss.
       The reject is only reported when the proof rests on at least one
       *certain* symbol (fundamental constant or prose-stated override) —
       ``is_dimension_anchored`` is the honesty gate.

    Returns:
        -1.0 — provably impossible (hard reject)
         0.0 — no decision
         1.0 — proven consistent by at least one formula
    """

    from formulagate.dimensions import check_dimensions
    from formulagate.verify_smt import check_dimensions_smt, is_dimension_anchored

    saw_consistent = False

    for f in draft_formulas:
        if not f.is_usable:
            continue

        # Tier 1: fast exact walker.
        exact = check_dimensions(f, overrides=overrides, domain=domain)
        if exact.status == "inconsistent":
            return -1.0
        if exact.status == "consistent":
            # Tier 1.5: SMT double-check for equations the walker classifies
            # as consistent.  The walker is fast but may miss inconsistencies
            # involving multi-term equations or symbols with partial grounding.
            # SMT runs here as a verification — it costs more but catches
            # subtle dimensional errors that the walker's table-lookup misses.
            smt = check_dimensions_smt(f, require_grounded=True, overrides=overrides)
            if smt.status == "inconsistent":
                anchored, _ = is_dimension_anchored(f, overrides=overrides)
                if anchored:
                    return -1.0
            saw_consistent = True
            continue

        # Tier 2: SMT solver for equations the walker cannot decide (multi-term,
        # symbolic, under-constrained). Runs only when exact != consistent.
        smt = check_dimensions_smt(f, require_grounded=True, overrides=overrides)
        if smt.status == "inconsistent":
            anchored, _ = is_dimension_anchored(f, overrides=overrides)
            if anchored:
                return -1.0

    if saw_consistent:
        return 1.0
    return 0.0


def _equivalence_signal(draft_formulas, candidate_formulas) -> float:
    """+1 as soon as one pair matches; −1 only if every shared-symbol pair differs."""

    compared = 0
    saw_difference = False

    for draft in draft_formulas:
        for candidate in candidate_formulas:
            if compared >= MAX_PAIR_COMPARISONS:
                break
            if not (set(draft.symbols) & set(candidate.symbols)):
                continue
            compared += 1
            verdict = check_equivalence(draft, candidate)
            if verdict.status == "equivalent":
                return 1.0
            if verdict.status == "different":
                saw_difference = True

    if saw_difference:
        return -1.0
    return 0.0


def _symbol_overlap(draft_formulas, candidate_formulas) -> float:
    left = {s for f in draft_formulas for s in f.symbols}
    right = {s for f in candidate_formulas for s in f.symbols}
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def physics_signals(draft: str, candidate_text: str, overrides: dict | None = None, domain: str | None = None) -> PhysicsSignals:
    """Compute the physics feature bundle for one (draft, record) pair.

    Never raises: any failure inside the symbolic stack degrades to a neutral
    feature, so the gate falls back to its similarity score instead of breaking.

    Args:
        draft: The LLM output text being judged for hallucination.
        candidate_text: The retrieved corpus entry that supposedly supports the draft.
        overrides: Optional symbol→dimension mappings from symbol grounding
            (4-layer resolver: doc, domain, ontology, table). These pin
            ambiguous symbols (c, G, e, h, ...) to their physical dimensions,
            lifting classic equations from "unknown" to "consistent".
    """

    try:
        draft_formulas = extract_formulas(draft or "", limit=MAX_DRAFT_FORMULAS)
        candidate_formulas = extract_formulas(candidate_text or "", limit=MAX_CANDIDATE_FORMULAS)
    except Exception:
        return PhysicsSignals()

    # Compute grounding sensitivity (GASP-style semantic fallback).
    grounding = 0.0
    try:
        from formulagate.grounding import semantic_grounding

        gs = semantic_grounding(draft or "", candidate_text or "")
        grounding = gs.sensitivity
    except Exception:
        grounding = 0.0

    if not draft_formulas and not candidate_formulas:
        return PhysicsSignals(grounding_sensitivity=grounding)

    # Compute physics constraints score from the draft's own formulas.
    constraints = 0.5  # neutral default
    try:
        from formulagate.physics_constraints import check_physics_constraints

        if draft_formulas:
            report = check_physics_constraints(draft_formulas[0].latex)
            constraints = report.score
    except Exception:
        constraints = 0.5

    try:
        return PhysicsSignals(
            dimension_ok=_dimension_signal(draft_formulas, overrides=overrides, domain=domain),
            equivalence=_equivalence_signal(draft_formulas, candidate_formulas),
            symbol_overlap=_symbol_overlap(draft_formulas, candidate_formulas),
            candidate_has_formula=1.0 if candidate_formulas else 0.0,
            grounding_sensitivity=grounding,
            constraints_score=constraints,
        )
    except Exception:
        return PhysicsSignals()
