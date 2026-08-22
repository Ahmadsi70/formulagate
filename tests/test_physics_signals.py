"""Deterministic physics features handed to the gate."""

from __future__ import annotations

import pytest

from formulagate.physics_signals import PhysicsSignals, physics_signals

CANDIDATE = r"We revisit the mass-energy relation $E = m c^2$ in curved space."


def test_valid_draft_formula_reports_dimensional_support() -> None:
    signals = physics_signals(r"The relation $E = m c^2$ holds.", CANDIDATE)
    assert signals.dimension_ok == 1.0


def test_dimensionally_broken_draft_is_flagged_regardless_of_wording() -> None:
    """The wording is perfect physics prose; only the algebra betrays it."""

    signals = physics_signals(r"By relativity the energy is $E = m c^3$.", CANDIDATE)
    assert signals.dimension_ok == -1.0


def test_prose_without_formulas_stays_neutral() -> None:
    signals = physics_signals("The energy grows with mass.", "A paper about energy.")
    assert signals.dimension_ok == 0.0
    assert signals.equivalence == 0.0
    assert signals.symbol_overlap == 0.0
    assert signals.candidate_has_formula == 0.0


def test_matching_formula_is_proven_equivalent() -> None:
    signals = physics_signals(r"Hence $m c^2 = E$.", CANDIDATE)
    assert signals.equivalence == 1.0
    assert signals.candidate_has_formula == 1.0


def test_conflicting_formula_over_the_same_symbols_is_refuted() -> None:
    signals = physics_signals(r"Hence $E = m c^4$.", CANDIDATE)
    assert signals.equivalence == -1.0


def test_symbol_overlap_is_a_bounded_jaccard() -> None:
    same = physics_signals(r"$E = m c^2$", CANDIDATE)
    disjoint = physics_signals(r"$F = \rho a$", CANDIDATE)
    assert same.symbol_overlap == pytest.approx(1.0)
    assert 0.0 <= disjoint.symbol_overlap < 1.0


def test_features_are_a_fixed_width_numeric_vector() -> None:
    features = physics_signals(r"$E = m c^2$", CANDIDATE).as_features()
    assert len(features) == len(PhysicsSignals.FEATURE_NAMES)
    assert all(isinstance(value, float) for value in features)


def test_signals_are_deterministic() -> None:
    first = physics_signals(r"$E = m c^2$", CANDIDATE)
    second = physics_signals(r"$E = m c^2$", CANDIDATE)
    assert first == second


def test_garbage_latex_never_raises() -> None:
    signals = physics_signals(r"$\bad{ latex$", r"$\also{ broken$")
    assert signals == PhysicsSignals()


def test_serialises_for_the_audit_log() -> None:
    payload = physics_signals(r"$E = m c^2$", CANDIDATE).to_dict()
    assert payload["dimension_ok"] == 1.0
    assert payload["equivalence"] == 1.0


def test_smt_tier_catches_deep_inconsistency_on_certain_constant() -> None:
    # ``q = ε₀r + ε₀t`` — q is not in the table, so the exact walker returns
    # "unknown".  SMT proves the two RHS terms disagree (M⁻¹L⁻²T⁴I² vs
    # M⁻¹L⁻³T⁵I²) and the anchor is ε₀ (~CERTAIN_CONSTANTS) → the veto fires.
    signals = physics_signals(r"$q = \epsilon_0 r + \epsilon_0 t$", "")
    assert signals.dimension_ok == -1.0


def test_smt_without_certain_constant_abstains() -> None:
    # ``q = m r + m t`` — same structure but m is NOT a certain constant.
    # The exact walker says "unknown" (q not in table).  SMT proves unsat
    # (M·L vs M·T) but the anchor is m (guessable table entry) → veto
    # abstains.  A reject here would overclaim.
    signals = physics_signals(r"$q = m r + m t$", "")
    assert signals.dimension_ok == 0.0  # unknown — not rejected
