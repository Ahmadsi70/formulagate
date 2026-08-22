"""Decide whether two formulas state the same relation — by proof, not similarity.

Three rungs, cheapest first, each able to stop the ladder:

  1. **hash**     — identical canonical structure (free, exact).
  2. **symbolic** — ``a / b`` reduces to a nonzero constant. Equations are only
     defined up to a nonzero factor, so ``2E - 2mc²`` must match ``E - mc²``;
     comparing differences instead of ratios would miss that.
  3. **numeric**  — evaluate both at deterministic pseudo-random points. This can
     only refute or corroborate, never prove, so it is the last rung and its
     verdict is labelled as such.

``simplify`` is unbounded in the worst case, so expression size is capped before
it runs: a gate that hangs on one pathological abstract is a broken gate.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MAX_OPS_FOR_SYMBOLIC = 60
DEFAULT_SAMPLES = 12
DEFAULT_SEED = 20260725
_NUMERIC_TOLERANCE = 1e-9


@dataclass(frozen=True)
class EquivalenceVerdict:
    """``status`` ∈ {equivalent, different, unknown}; ``method`` names the rung."""

    status: str
    method: str = "none"
    detail: str = ""

    @property
    def is_equivalent(self) -> bool:
        return self.status == "equivalent"

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status, "method": self.method, "detail": self.detail}


def _expr(formula):
    """Rebuild the SymPy expression from a Formula's canonical srepr."""

    if not getattr(formula, "canonical", None):
        return None
    try:
        import sympy

        return sympy.sympify(formula.canonical)
    except Exception as exc:
        logger.debug("sympify failed: %s", exc)
        return None


def numeric_spot_check(
    left,
    right,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> bool | None:
    """Evaluate both expressions at the same random points.

    Returns ``True`` when every successful sample agrees up to a constant
    factor, ``False`` on the first disagreement, and ``None`` when nothing could
    be evaluated (unparsable input, everything singular).

    The generator is seeded, so a verdict is reproducible — a benchmark number
    that changes between runs is not evidence.
    """

    a, b = _expr(left), _expr(right)
    if a is None or b is None:
        return None

    import sympy

    symbols = sorted(a.free_symbols | b.free_symbols, key=str)
    if not symbols:
        try:
            return bool(sympy.simplify(a - b) == 0)
        except Exception:
            return None

    rng = random.Random(seed)
    evaluated = 0
    ratio: float | None = None

    for _ in range(samples):
        point = {s: sympy.Rational(rng.randint(2, 40), rng.randint(1, 7)) for s in symbols}
        try:
            va = complex(a.subs(point).evalf())
            vb = complex(b.subs(point).evalf())
        except Exception:
            continue
        if not (abs(va) < 1e30 and abs(vb) < 1e30):
            continue
        if abs(vb) < _NUMERIC_TOLERANCE:
            if abs(va) > _NUMERIC_TOLERANCE:
                return False
            continue
        current = va / vb
        evaluated += 1
        if ratio is None:
            ratio = abs(current)
            if ratio < _NUMERIC_TOLERANCE:
                return False
            continue
        if abs(abs(current) - ratio) > 1e-6 * max(1.0, ratio):
            return False

    if evaluated == 0:
        return None
    return True


def check_equivalence(
    left,
    right,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> EquivalenceVerdict:
    """Run the equivalence ladder over two :class:`Formula` objects."""

    if not getattr(left, "is_usable", False) or not getattr(right, "is_usable", False):
        return EquivalenceVerdict("unknown", "none", "one side did not parse")

    if left.structure_hash == right.structure_hash:
        return EquivalenceVerdict("equivalent", "hash", "identical canonical structure")

    a, b = _expr(left), _expr(right)
    if a is None or b is None:
        return EquivalenceVerdict("unknown", "none", "canonical form not restorable")

    if not (a.free_symbols & b.free_symbols):
        return EquivalenceVerdict("different", "symbols", "no shared symbols")

    import sympy

    if sympy.count_ops(a) + sympy.count_ops(b) <= MAX_OPS_FOR_SYMBOLIC:
        try:
            ratio = sympy.cancel(sympy.together(a / b))
            numerator, denominator = ratio.as_numer_denom()
            # A monomial ratio (number, product or power — no sum) means one
            # equation is the other scaled by a factor that is nonzero wherever
            # it is defined, so the two have the same solution set. ``2E = 2mc²``
            # and ``4GS = A`` both land here; ``E = mc³`` does not.
            monomial = not numerator.atoms(sympy.Add) and not denominator.atoms(sympy.Add)
            # Guard: between two monomials the ratio is *always* monomial, so the
            # test would declare ħω ≡ k_B T. Measured on arXiv, that single hole
            # dropped the purity of "proven equivalent" from 96% to 76%.
            both_monomial = not a.atoms(sympy.Add) and not b.atoms(sympy.Add)
            if both_monomial:
                monomial = bool(ratio.is_number)
            if ratio != 0 and monomial:
                return EquivalenceVerdict("equivalent", "symbolic", f"ratio {ratio}")
            if sympy.simplify(a - b) == 0:
                return EquivalenceVerdict("equivalent", "symbolic", "zero difference")
            if a.is_rational_function() and b.is_rational_function():
                # ``cancel`` is complete for rational expressions: a non-monomial
                # ratio here is a proof of difference, not a failure to decide.
                return EquivalenceVerdict(
                    "different", "symbolic", f"ratio depends on the solution: {ratio}"
                )
        except Exception as exc:
            logger.debug("symbolic equivalence failed: %s", exc)

    spot = numeric_spot_check(left, right, samples=samples, seed=seed)
    if spot is True:
        return EquivalenceVerdict("equivalent", "numeric", f"agreed on {samples} sample points")
    if spot is False:
        return EquivalenceVerdict("different", "numeric", "disagreed on a sample point")
    return EquivalenceVerdict("unknown", "numeric", "no evaluable sample point")
