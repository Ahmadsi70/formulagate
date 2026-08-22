"""Dimensional analysis: the deterministic physics check no LLM can fake."""

from __future__ import annotations

import pytest

from formulagate.dimensions import (
    DIMENSIONLESS,
    Dimension,
    DimensionVerdict,
    check_dimensions,
    dimension_of_symbol,
)
from formulagate.formula_extract import canonicalize


def _status(latex: str, **kwargs) -> str:
    return check_dimensions(canonicalize(latex), **kwargs).status


# ─── Dimension algebra ────────────────────────────────────────────────────────


def test_dimension_arithmetic_follows_exponent_rules() -> None:
    length = Dimension(length=1)
    time = Dimension(time=1)
    velocity = length / time
    assert velocity == Dimension(length=1, time=-1)
    assert (velocity**2) == Dimension(length=2, time=-2)
    assert (length / length) == DIMENSIONLESS
    assert DIMENSIONLESS.is_dimensionless


def test_dimension_renders_readable_physics() -> None:
    assert str(Dimension(mass=1, length=2, time=-2)) == "M L^2 T^-2"
    assert str(DIMENSIONLESS) == "1"


def test_known_constants_carry_their_si_dimensions() -> None:
    assert dimension_of_symbol("c") == Dimension(length=1, time=-1)
    assert dimension_of_symbol("hbar") == Dimension(mass=1, length=2, time=-1)
    assert dimension_of_symbol("k_B") == Dimension(mass=1, length=2, time=-2, temperature=-1)
    assert dimension_of_symbol("zzz") is None


def test_ambiguous_symbols_report_unknown_instead_of_guessing() -> None:
    """``T`` is temperature in thermodynamics and time in mechanics; guessing
    would produce false rejects, which are far worse than an abstention."""

    assert dimension_of_symbol("T") is None
    assert dimension_of_symbol("P") is None


# ─── Verdicts on real physics ─────────────────────────────────────────────────


def test_mass_energy_relation_is_consistent() -> None:
    verdict = check_dimensions(canonicalize("E = m c^2"))
    assert verdict.status == "consistent"
    assert verdict.dimension == Dimension(mass=1, length=2, time=-2)


def test_wrong_exponent_is_rejected_deterministically() -> None:
    verdict = check_dimensions(canonicalize("E = m c^3"))
    assert verdict.status == "inconsistent"
    assert "M L^2 T^-2" in verdict.detail


def test_newton_second_law_is_consistent() -> None:
    assert _status("F = m a") == "consistent"


def test_force_equals_momentum_is_rejected() -> None:
    assert _status("F = m v") == "inconsistent"


def test_kinematics_with_multiple_terms_is_consistent() -> None:
    assert _status("x = v t + a t^2") == "consistent"


def test_transcendental_of_a_dimensional_argument_is_rejected() -> None:
    """``\\sin(t)`` is meaningless: only pure numbers may enter a series."""

    assert _status(r"y = \sin(t)") == "inconsistent"


def test_unknown_symbol_yields_unknown_not_a_reject() -> None:
    # \Xi is absent from the table; "zzz" would not do — LaTeX reads it as z^3.
    verdict = check_dimensions(canonicalize(r"E = m \Xi"))
    assert verdict.status == "unknown"
    assert "Xi" in verdict.detail


def test_overrides_resolve_ambiguity_from_context() -> None:
    """Domain context can pin an ambiguous symbol; nothing else changes."""

    thermal = {"T": Dimension(temperature=1)}
    assert _status(r"E = k_B T", overrides=thermal) == "consistent"
    assert _status(r"E = k_B T^2", overrides=thermal) == "inconsistent"


def test_natural_units_are_not_treated_as_an_error() -> None:
    """``p = 1`` is natural units (or an index), not broken physics.

    Measured on 902 live arXiv drafts: rejecting "quantity = number" produced
    half of all false vetoes, so it must resolve to unknown."""

    assert _status("p = 1") == "unknown"


def test_user_defined_functions_do_not_trigger_a_rejection() -> None:
    """``j(z)`` and ``f(R)`` are maps with unknown codomain — unlike ``sin``,
    they say nothing about the dimension of their argument."""

    assert _status("j(z) = 1") == "unknown"
    assert _status(r"y = \sin(t)") == "inconsistent"


def test_unparsable_formula_is_unknown() -> None:
    verdict = check_dimensions(canonicalize(r"\bad{"))
    assert verdict.status == "unknown"
    assert isinstance(verdict, DimensionVerdict)


def test_check_is_deterministic() -> None:
    first = check_dimensions(canonicalize("E = m c^2"))
    second = check_dimensions(canonicalize("E = m c^2"))
    assert (first.status, first.detail) == (second.status, second.detail)


@pytest.mark.parametrize(
    "latex,expected",
    [
        ("p = m v", "consistent"),
        ("p = m v^2", "inconsistent"),
        (r"\lambda = c / \omega", "consistent"),
        (r"\lambda = c \omega", "inconsistent"),
    ],
)
def test_parametrised_physics_battery(latex: str, expected: str) -> None:
    assert _status(latex) == expected
