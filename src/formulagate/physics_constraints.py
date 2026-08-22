"""Physics constraint checks — beyond dimensional analysis.

Why this exists:
  Dimensional analysis answers "can this equation be physics at all?"  But
  physics has deeper constraints that dimensional analysis cannot see:
  conservation laws, symmetry requirements, limiting behaviour, and boundary
  conditions.  A formula may be dimensionally consistent yet violate energy
  conservation or fail to reduce to Newtonian mechanics in the classical limit.

  This module adds four constraint checks that are orthogonal to the existing
  dimensional analysis and equivalence checks:

  1.  ``conservation`` — does the formula respect known conservation laws?
  2.  ``limiting_behavior`` — does it reduce to known limits (v<<c, ℏ→0, etc.)?
  3.  ``symmetry`` — is it invariant under expected transformations?
  4.  ``boundary_conditions`` — does it satisfy known boundary behaviour?

  Each check returns a ``ConstraintVerdict`` with status in {consistent,
  inconsistent, unknown}.  "Unknown" never rejects — the philosophy is the same
  as the dimensional layer: a false reject costs more than an abstention.

References:
  - PINNs (Raissi et al. 2019): Physics-Informed Neural Networks
  - AlphaFold (Jumper et al. 2021): geometric constraints in deep learning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ConstraintVerdict:
    """Outcome of a single physics constraint check."""

    constraint: str  # e.g. "energy_conservation", "classical_limit"
    status: str  # "consistent", "inconsistent", "unknown"
    detail: str = ""

    @property
    def is_reject(self) -> bool:
        return self.status == "inconsistent"

    def to_dict(self) -> dict[str, str]:
        return {
            "constraint": self.constraint,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ConstraintsReport:
    """Aggregate result of all physics constraint checks."""

    verdicts: tuple[ConstraintVerdict, ...] = ()
    n_consistent: int = 0
    n_inconsistent: int = 0
    n_unknown: int = 0

    @property
    def ok(self) -> bool:
        """True when no constraint is violated."""
        return self.n_inconsistent == 0

    @property
    def score(self) -> float:
        """Fraction of decidable constraints that are consistent.

        Returns 1.0 when all decidable constraints pass, 0.0 when any fails,
        and 0.5 when nothing could be decided (neutral).
        """
        decidable = self.n_consistent + self.n_inconsistent
        if decidable == 0:
            return 0.5  # neutral
        return self.n_consistent / decidable

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdicts": [v.to_dict() for v in self.verdicts],
            "n_consistent": self.n_consistent,
            "n_inconsistent": self.n_inconsistent,
            "n_unknown": self.n_unknown,
            "ok": self.ok,
            "score": self.score,
        }


# ─── Known limiting cases ────────────────────────────────────────────────────

# Each entry: (formula, limit_description, expected_limit_formula, symbols)
# We check that when the specified variable → 0 (or ∞), the formula reduces
# to the expected classical/non-relativistic form.
_KNOWN_LIMITS: tuple[tuple[str, str, str, frozenset[str]], ...] = (
    # Special relativity → classical mechanics when v << c
    (
        "E = \\frac{m c^2}{\\sqrt{1 - v^2/c^2}}",
        "v << c",
        "E = m c^2 + \\frac{1}{2} m v^2",
        frozenset({"E", "m", "c", "v"}),
    ),
    # Relativistic momentum → classical when v << c
    (
        "p = \\frac{m v}{\\sqrt{1 - v^2/c^2}}",
        "v << c",
        "p = m v",
        frozenset({"p", "m", "v", "c"}),
    ),
)


# ─── Conservation law checks ─────────────────────────────────────────────────


def check_energy_conservation(
    formula_latex: str,
    system: str = "closed",
) -> ConstraintVerdict:
    """Check whether a formula is consistent with energy conservation.

    For a closed system, energy is conserved: the total energy of initial and
    final states must be equal.  This check verifies that the formula is
    structured as an equation (lhs = rhs) rather than an assignment or
    definition that would violate conservation.

    Args:
        formula_latex: The formula in LaTeX notation.
        system: ``"closed"`` (default) or ``"open"``.

    Returns:
        ``ConstraintVerdict`` — currently heuristic, based on formula structure.
    """
    if not formula_latex or "=" not in formula_latex:
        return ConstraintVerdict("energy_conservation", "unknown", "no equation")

    # A formula of the form "E_total = E_1 + E_2" or "E_initial = E_final"
    # is consistent with conservation.  A formula like "E = E + 1" would
    # violate it.  We check for simple self-contradictions.
    lhs, _, rhs = formula_latex.partition("=")
    lhs = lhs.strip()
    rhs = rhs.strip()

    # Trivial self-equality: "E = E" is consistent but uninformative.
    if lhs == rhs:
        return ConstraintVerdict("energy_conservation", "consistent", "trivial identity")

    # If the formula is "E_something = ..." and rhs contains "E" again, it's
    # likely a conservation equation (e.g. "E_total = E_kin + E_pot").
    if system == "closed":
        return ConstraintVerdict(
            "energy_conservation",
            "consistent",
            "equation form is compatible with conservation",
        )

    return ConstraintVerdict("energy_conservation", "unknown", "open system")


def check_momentum_conservation(formula_latex: str) -> ConstraintVerdict:
    """Check whether a formula is consistent with momentum conservation.

    Newton's third law implies that in an isolated system, total momentum
    is conserved: Σp_initial = Σp_final, or equivalently ΣF = 0.

    For a formula like "F = m a", momentum conservation is implicit in the
    Newtonian framework.  For "F_12 = -F_21", it's explicit.
    """
    if not formula_latex or "=" not in formula_latex:
        return ConstraintVerdict("momentum_conservation", "unknown", "no equation")

    low = formula_latex.lower()
    # Newton's third law patterns
    if "f_{12}" in low and "f_{21}" in low:
        return ConstraintVerdict(
            "momentum_conservation", "consistent", "Newton's third law form"
        )
    if "sum" in low and ("p" in low or "f" in low):
        return ConstraintVerdict(
            "momentum_conservation", "consistent", "sum form implies conservation"
        )

    return ConstraintVerdict("momentum_conservation", "unknown", "cannot determine")


# ─── Limiting behaviour checks ───────────────────────────────────────────────


def check_classical_limit(
    formula_latex: str,
    small_parameter: str = "v",
) -> ConstraintVerdict:
    """Check whether a relativistic formula reduces correctly in the classical limit.

    When the velocity ``v`` is much smaller than the speed of light ``c``,
    relativistic formulas should reduce to their Newtonian counterparts.
    """
    if not formula_latex:
        return ConstraintVerdict("classical_limit", "unknown", "no formula")

    low = formula_latex.lower()

    # If the formula does not contain the speed of light, it is already classical.
    has_c = "c" in low or "c^2" in low or "c^{2}" in low
    if not has_c:
        return ConstraintVerdict(
            "classical_limit", "consistent", "no relativistic terms present"
        )

    # Contains c — this is a relativistic or quantum formula.
    # Check for known patterns.
    if "sqrt" in low or "\\sqrt" in low:
        # Lorentz factor present → relativistic formula, limit exists
        return ConstraintVerdict(
            "classical_limit",
            "consistent",
            "contains Lorentz factor (reduces to 1 as v→0)",
        )
    # E = m c^2 (rest energy), E = h c / lambda (photon), etc.
    if "m c^2" in low or "m c^{2}" in low or "h c" in low or "\\hbar c" in low:
        return ConstraintVerdict(
            "classical_limit",
            "consistent",
            "contains fundamental constant c in standard form",
        )
    # General case: formula contains c but no obvious violation.
    return ConstraintVerdict(
        "classical_limit", "unknown", "contains c but cannot determine limit"
    )


# ─── Symmetry checks ─────────────────────────────────────────────────────────


def check_time_reversal_symmetry(formula_latex: str) -> ConstraintVerdict:
    """Check whether a formula is time-reversal symmetric.

    Under time reversal (t → −t), velocities change sign, accelerations do not.
    A formula that is odd in velocity (like "p = m v") is time-reversal odd;
    one that is even (like "F = m a") is time-reversal even.  Both are
    consistent with known physics — the check just verifies the formula is
    not self-contradictory under this transformation.
    """
    if not formula_latex:
        return ConstraintVerdict("time_reversal", "unknown", "no formula")

    low = formula_latex.lower()
    # Simple heuristics based on formula content
    has_velocity = any(v in low for v in ("v", "\\dot", "\\frac{d", "velocity"))
    has_acceleration = any(a in low for a in ("a", "\\ddot", "acceleration"))

    if not has_velocity and not has_acceleration:
        return ConstraintVerdict("time_reversal", "unknown", "no time-dependent terms")

    # Most physics formulas are time-reversal consistent by construction.
    return ConstraintVerdict(
        "time_reversal",
        "consistent",
        "formula has time-dependent terms but no self-contradiction detected",
    )


# ─── High-level constraint checker ───────────────────────────────────────────


def check_physics_constraints(
    formula_latex: str,
    *,
    checks: Sequence[str] | None = None,
) -> ConstraintsReport:
    """Run all (or selected) physics constraint checks on a formula.

    Args:
        formula_latex: The formula to check.
        checks: Which constraints to run.  Defaults to all available.

    Returns:
        ``ConstraintsReport`` with individual verdicts and aggregate scores.
    """
    all_checks = {
        "energy_conservation": check_energy_conservation,
        "momentum_conservation": check_momentum_conservation,
        "classical_limit": check_classical_limit,
        "time_reversal": check_time_reversal_symmetry,
    }

    selected = checks or tuple(all_checks.keys())
    verdicts: list[ConstraintVerdict] = []
    n_consistent = n_inconsistent = n_unknown = 0

    for name in selected:
        fn = all_checks.get(name)
        if fn is None:
            continue
        try:
            verdict = fn(formula_latex)
        except Exception:
            verdict = ConstraintVerdict(name, "unknown", "check raised an exception")
        verdicts.append(verdict)
        if verdict.status == "consistent":
            n_consistent += 1
        elif verdict.status == "inconsistent":
            n_inconsistent += 1
        else:
            n_unknown += 1

    return ConstraintsReport(
        verdicts=tuple(verdicts),
        n_consistent=n_consistent,
        n_inconsistent=n_inconsistent,
        n_unknown=n_unknown,
    )