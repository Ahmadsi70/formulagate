"""Equivalence: proving two formulas are the same, instead of guessing."""

from __future__ import annotations

from formulagate.equivalence import (
    EquivalenceVerdict,
    check_equivalence,
    numeric_spot_check,
)
from formulagate.formula_extract import canonicalize


def _verdict(a: str, b: str) -> EquivalenceVerdict:
    return check_equivalence(canonicalize(a), canonicalize(b))


# ─── Cheap path: structural hash ──────────────────────────────────────────────


def test_identical_structure_short_circuits_on_the_hash() -> None:
    verdict = _verdict("E = m c^2", "m c^2 = E")
    assert verdict.status == "equivalent"
    assert verdict.method == "hash"


def test_rearrangement_matches_via_hash() -> None:
    assert _verdict("F = m a", "F - m a = 0").status == "equivalent"


# ─── Symbolic path: same equation up to a nonzero factor ──────────────────────


def test_scaled_equation_is_still_the_same_equation() -> None:
    """``2E = 2mc²`` states exactly the physics of ``E = mc²``."""

    verdict = _verdict("E = m c^2", "2 E = 2 m c^2")
    assert verdict.status == "equivalent"
    assert verdict.method == "symbolic"


def test_equation_multiplied_by_a_symbolic_factor_is_the_same_equation() -> None:
    """``S = A/4G`` and ``4GS = A`` have identical solution sets; the canonical
    forms differ by the nonzero factor ``1/4G``, which must not read as a
    contradiction."""

    verdict = _verdict(r"S = \frac{A}{4 G}", "4 G S = A")
    assert verdict.status == "equivalent"
    assert verdict.method == "symbolic"


def test_algebraic_rewrite_is_recognised() -> None:
    verdict = _verdict(r"E = \frac{1}{2} m v^2", r"2 E = m v^2")
    assert verdict.status == "equivalent"


def test_two_unrelated_monomials_are_not_equivalent() -> None:
    """Any monomial divides any other monomial, so "equal up to a factor" is
    vacuous there — ``ħω`` is not ``k_B T`` just because the ratio is a product."""

    verdict = check_equivalence(canonicalize(r"E = \hbar \omega"), canonicalize("E = k T"))
    assert verdict.status != "equivalent"
    assert check_equivalence(
        canonicalize(r"\hbar \omega"), canonicalize("k T")
    ).status != "equivalent"


def test_different_physics_is_reported_as_different() -> None:
    verdict = _verdict("E = m c^2", "E = m c^3")
    assert verdict.status == "different"


def test_disjoint_symbols_are_different() -> None:
    assert _verdict("E = m c^2", "F = m a").status == "different"


# ─── Numeric spot-check ───────────────────────────────────────────────────────


def test_numeric_spot_check_confirms_a_true_identity() -> None:
    left = canonicalize(r"y = \sin(x)^2 + \cos(x)^2")
    right = canonicalize("y = 1")
    assert numeric_spot_check(left, right) is True


def test_numeric_spot_check_refutes_a_false_identity() -> None:
    assert numeric_spot_check(canonicalize("y = x^2"), canonicalize("y = x^3")) is False


def test_numeric_spot_check_is_deterministic() -> None:
    a, b = canonicalize("y = x^2"), canonicalize("y = x^2 + 1")
    assert numeric_spot_check(a, b) == numeric_spot_check(a, b)


def test_numeric_spot_check_reports_unknown_when_it_cannot_evaluate() -> None:
    assert numeric_spot_check(canonicalize(r"\bad{"), canonicalize("y = x")) is None


# ─── Degradation ──────────────────────────────────────────────────────────────


def test_unparsable_input_is_unknown_not_different() -> None:
    verdict = _verdict(r"\bad{", "E = m c^2")
    assert verdict.status == "unknown"


def test_verdict_serialises_for_the_audit_log() -> None:
    payload = _verdict("E = m c^2", "m c^2 = E").to_dict()
    assert payload["status"] == "equivalent"
    assert payload["method"] == "hash"
