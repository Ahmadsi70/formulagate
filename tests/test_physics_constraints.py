"""Tests for physics constraint checks."""

from __future__ import annotations

from formulagate.physics_constraints import (
    ConstraintVerdict,
    ConstraintsReport,
    check_classical_limit,
    check_energy_conservation,
    check_momentum_conservation,
    check_physics_constraints,
    check_time_reversal_symmetry,
)


def test_constraint_verdict_serialization() -> None:
    v = ConstraintVerdict("energy", "consistent", "all good")
    d = v.to_dict()
    assert d["constraint"] == "energy"
    assert d["status"] == "consistent"


def test_constraint_verdict_is_reject() -> None:
    assert ConstraintVerdict("x", "inconsistent").is_reject is True
    assert ConstraintVerdict("x", "consistent").is_reject is False
    assert ConstraintVerdict("x", "unknown").is_reject is False


def test_energy_conservation_empty() -> None:
    v = check_energy_conservation("")
    assert v.status == "unknown"


def test_energy_conservation_equation_form() -> None:
    v = check_energy_conservation(r"E_{total} = E_{kin} + E_{pot}")
    assert v.status == "consistent"


def test_energy_conservation_trivial() -> None:
    v = check_energy_conservation("E = E")
    assert v.status == "consistent"


def test_momentum_conservation_empty() -> None:
    v = check_momentum_conservation("")
    assert v.status == "unknown"


def test_momentum_conservation_newton_third() -> None:
    v = check_momentum_conservation(r"F_{12} = -F_{21}")
    assert v.status == "consistent"


def test_momentum_conservation_sum_form() -> None:
    v = check_momentum_conservation(r"\sum p_i = \sum p_f")
    assert v.status == "consistent"


def test_classical_limit_empty() -> None:
    v = check_classical_limit("")
    assert v.status == "unknown"


def test_classical_limit_non_relativistic() -> None:
    # F = ma — no speed of light
    v = check_classical_limit("F = m a")
    assert v.status == "consistent"


def test_classical_limit_rest_energy() -> None:
    v = check_classical_limit("E = m c^2")
    assert v.status == "consistent"


def test_classical_limit_lorentz() -> None:
    v = check_classical_limit(r"\frac{1}{\sqrt{1 - v^2/c^2}}")
    assert v.status == "consistent"


def test_time_reversal_empty() -> None:
    v = check_time_reversal_symmetry("")
    assert v.status == "unknown"


def test_time_reversal_with_velocity() -> None:
    v = check_time_reversal_symmetry(r"p = m v")
    assert v.status == "consistent"


def test_check_physics_constraints_all() -> None:
    report = check_physics_constraints("F = m a")
    assert report.n_consistent >= 0
    assert report.n_inconsistent == 0
    assert report.ok is True
    assert 0.0 <= report.score <= 1.0


def test_check_physics_constraints_selected() -> None:
    report = check_physics_constraints(
        "E = m c^2",
        checks=("energy_conservation", "classical_limit"),
    )
    assert len(report.verdicts) == 2
    assert report.ok is True


def test_constraints_report_serialization() -> None:
    report = check_physics_constraints("F = m a")
    d = report.to_dict()
    assert "verdicts" in d
    assert "ok" in d
    assert "score" in d


def test_constraints_report_score_all_consistent() -> None:
    report = ConstraintsReport(
        verdicts=(
            ConstraintVerdict("a", "consistent"),
            ConstraintVerdict("b", "consistent"),
        ),
        n_consistent=2,
        n_inconsistent=0,
        n_unknown=0,
    )
    assert report.score == 1.0


def test_constraints_report_score_all_inconsistent() -> None:
    report = ConstraintsReport(
        verdicts=(ConstraintVerdict("a", "inconsistent"),),
        n_consistent=0,
        n_inconsistent=1,
        n_unknown=0,
    )
    assert report.score == 0.0


def test_constraints_report_score_neutral() -> None:
    report = ConstraintsReport(
        verdicts=(ConstraintVerdict("a", "unknown"),),
        n_consistent=0,
        n_inconsistent=0,
        n_unknown=1,
    )
    assert report.score == 0.5  # neutral when nothing decidable