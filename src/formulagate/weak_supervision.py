"""Weak supervision: learn symbol dimensions from equation consensus.

When a symbol appears in multiple anchored equations within one paper and
those equations agree on its dimension, we can infer the dimension without
explicit prose definition.  The SMT solver is the teacher.
"""

from __future__ import annotations

from collections import defaultdict

from formulagate.dimensions import Dimension, canonical_symbol_name
from formulagate.formula_extract import canonicalize
from formulagate.verify_smt import SmtVerdict, check_dimensions_smt, is_dimension_anchored


def _try_infer_single(sym_name: str, formula, grounded: dict, ov: dict, certain):
    """Try to infer the dimension of one symbol from one equation.

    Strategy: temporarily ground the symbol with every plausible dimension
    and check if the equation becomes consistent.  If exactly one dimension
    makes it consistent, that is the learned dimension.
    """
    # Plausible dimension candidates: try the 7 base dimensions.
    candidates = [
        Dimension(),
        Dimension(mass=1),
        Dimension(length=1),
        Dimension(time=1),
        Dimension(mass=1, length=2, time=-2),  # energy
        Dimension(mass=1, length=1, time=-2),  # force
        Dimension(length=1, time=-1),  # velocity
        Dimension(length=1, time=-2),  # acceleration
        Dimension(length=-1),  # inverse length
        Dimension(length=-2),  # inverse length^2
        Dimension(length=2),  # area
        Dimension(time=-1),  # frequency
        Dimension(mass=1, time=-2, length=-1),  # pressure
    ]
    inferred = None
    for cand in candidates:
        test_ov = dict(ov)
        test_ov[sym_name] = cand
        v = check_dimensions_smt(formula, overrides=test_ov, certain_symbols=certain)
        if v.status == "consistent":
            if inferred is not None and inferred != cand:
                return None  # ambiguous
            inferred = cand
    return inferred


def learn_paper_symbols(equations, groundings):
    """Infer dimensions for symbols appearing in multiple anchored equations.

    Returns:
        Dict: canonical_symbol_name -> (dimension, confidence_int)
    """
    rows = []
    for eq, grd in zip(equations, groundings):
        f = canonicalize(eq.latex)
        if not f.is_usable:
            continue
        ctx = [g for g in grd.grounded if g.source in ("context", "constant")]
        if not ctx:
            continue
        ov = {g.symbol: g.dimension for g in ctx if g.dimension}
        certain = [g.symbol for g in ctx]
        anchored, _ = is_dimension_anchored(f, overrides=ov, certain_symbols=certain)
        if not anchored:
            continue
        grounded = {g.symbol: g.dimension for g in grd.grounded if g.dimension}
        rows.append((f, grounded, certain, ov))

    if len(rows) < 2:
        return {}

    # Collect equations per ungrounded symbol
    sym_eqs = defaultdict(list)
    for i, (f, grounded, certain, ov) in enumerate(rows):
        import sympy
        try:
            expr = sympy.sympify(f.canonical)
        except Exception:
            continue
        for s in expr.free_symbols:
            name = canonical_symbol_name(str(s))
            if name not in grounded and name not in {"pi", "e", "i"}:
                sym_eqs[name].append(i)

    learned = {}
    for sym_name, eq_indices in sym_eqs.items():
        if len(eq_indices) < 2:
            continue
        if len(sym_name) < 2 and "_" not in sym_name:
            continue  # skip single-letter unqualified symbols

        dims = []
        for idx in eq_indices[:4]:
            f, grounded, certain, ov = rows[idx]
            d = _try_infer_single(sym_name, f, grounded, ov, certain)
            if d is not None:
                dims.append(d)

        if dims and all(d == dims[0] for d in dims):
            learned[sym_name] = (dims[0], len(dims))

    return learned
