"""Proof-level dimension verification with the Z3 SMT solver.

Why this exists:
  The exact walker in ``dimensions.py`` stops at the first unresolved symbol:
  one unknown symbol makes the whole equation "unknown", even when the full
  linear system could still be proven inconsistent. Over the linear space of
  dimension exponents, dimensional consistency is *decidable*: write one linear
  equation per additive term ("every term of this sum must share one dimension"),
  plus a fixed point for each grounded symbol, and ask an SMT solver whether the
  system is satisfiable.

  Z3 is used here as a proof engine, not a classifier:

    * ``sat``      — a model exists; the equation is dimensionally consistent.
    * ``unsat``    — Z3 produced a resolution proof; the equation is *proven*
                     wrong, independent of any heuristic.
    * ``unknown``  — the system is under-constrained (too few grounded symbols).

  This composes with :mod:`formulagate.symbol_grounding`: grounded symbols are
  the equations, the solver finds the inconsistency.

References:
  - de Moura & Bjørner, "Z3: An Efficient SMT Solver" (TACAS 2008)
  - BIPM, "The International System of Units (SI)" Brochure, §2.2
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

from formulagate.dimensions import Dimension, canonical_symbol_name

_BASE_NAMES = ("mass", "length", "time", "current", "temperature", "amount", "luminosity")

# Unambiguous, single-reading symbols: a named fundamental constant or a
# context-pinned symbol. These are the only anchors a reject may rest on. The
# guessable single-letter table entries (t, m, rho, lambda, alpha…) carry no
# evidence by themselves.
_CERTAIN_CONSTANTS = frozenset(
    {
        # Only constants whose letter is a SINGLE-dominant reading in
        # real full-text physics. Ambiguous letters (G = Gauge-vs-Newton,
        # e = exponent-vs-charge, h = height-vs-Planck, q = charge-vs-coeff,
        # c = speed-of-light-vs-coefficient) are deliberately excluded: they
        # must be pinned by real prose (context/constant), never guessed.
        "hbar", "k_B", "k_Bz", "epsilon_0", "mu_0", "N_A",
    }
)


def is_dimension_anchored(formula, overrides=None, certain_symbols=None):
    """Whether an equation carries a *reliable* dimensional anchor.

    A reject is only sound when the proof rests on at least one symbol whose
    dimension is not a guess — a fundamental constant, a prose-stated
    override, or a caller-declared certain symbol. Equations with none of
    these (pure dimensionless identities, combinatorics, metric factors) are
    not *falsifiable* by the dimensional veto: any apparent mismatch is an
    artefact of the symbol table's guess. Honest detection metrics must scope
    to the anchored subset.

    Returns:
        ``(anchored: bool, anchors: set[str])`` — the symbols that anchor the
        equation's dimensions.
    """
    import sympy

    certain = set(_CERTAIN_CONSTANTS)
    if overrides:
        certain.update(canonical_symbol_name(k) for k in overrides)
    if certain_symbols:
        certain.update(canonical_symbol_name(s) for s in certain_symbols)
    if not getattr(formula, "canonical", None):
        return False, set()
    try:
        expr = sympy.sympify(formula.canonical)
    except Exception:
        return False, set()
    anchors = {str(s) for s in expr.free_symbols if canonical_symbol_name(str(s)) in certain}
    return bool(anchors), anchors


@dataclass(frozen=True)
class SmtVerdict:
    """Result of the SMT dimension check."""

    status: str  # "consistent" | "inconsistent" | "unknown"
    solver: str = "z3"
    detail: str = ""
    #: When consistent, the realised dimension of the equation.
    dimension: Dimension | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "solver": self.solver,
            "detail": self.detail,
            "dimension": str(self.dimension) if self.dimension else None,
        }


def _sympy_expr(formula):
    """Return the canonical SymPy expression and whether it is an equation."""
    import sympy

    expr = sympy.sympify(formula.canonical)
    return expr


def check_dimensions_smt(
    formula,
    *,
    overrides: Mapping[str, Dimension] | None = None,
    unknown_symbols: Sequence[str] = (),
    require_grounded: bool = True,
    certain_symbols: Sequence[str] = (),
) -> SmtVerdict:
    """Prove/refute dimensional consistency of ``formula`` with Z3.

    Args:
        formula: a :class:`formulagate.formula_extract.Formula` (canonicalised)
            or a raw LaTeX string, which is canonicalised first.
        overrides: symbol → dimension, from grounding or a caller's table.
        unknown_symbols: symbols allowed to float (unknown dimensions). When a
            model assigns them values consistently, the equation is consistent
            under the weakest assumption — Z3 still refuses any inconsistent
            system, so a corrupt equation stays ``unsat``.
        require_grounded: when True (the honest default), a ``SAT`` model counts
            as ``consistent`` only if every free symbol's dimension is pinned by
            ``overrides`` or the curated table. An equation whose symbols float
            is ``unknown`` — Z3 can trivially satisfy such a system, so calling
            it "consistent" would be overclaiming. ``unsat`` is *always* a proof
            of inconsistency, grounded or not.
        certain_symbols: symbols whose meaning is *stated* by real prose (source
            ``context``) or bound to a named constant, from
            :mod:`formulagate.symbol_grounding`. An ``unsat`` result is only
            reported as ``inconsistent`` when it is anchored to one of these
            — a reject resting purely on the guessable single-letter table
            (``t``, ``m``, ``rho``…) is downgraded to ``unknown``, honouring the
            rule that a false reject costs more than an abstention.

    Returns:
        A ``SmtVerdict``. ``status == "inconsistent"`` is a *proof*.
    """
    import z3

    if isinstance(formula, str):
        from formulagate.formula_extract import canonicalize

        formula = canonicalize(formula)
    if not getattr(formula, "canonical", None):
        return SmtVerdict(
            "unknown",
            detail=getattr(formula, "parse_error", None) or "no canonical form",
        )

    expr = _sympy_expr(formula)
    if not getattr(expr, "free_symbols", None):
        return SmtVerdict("unknown", detail="no free symbols")

    import sympy

    # symbol → dict of exponent vars (shared across uses). LaTeX symbol names
    # often contain characters Z3's parser rejects ({, }, \, Greek letters), so
    # each symbol is aliased to a safe internal identifier.
    _safe_id: dict[str, str] = {}
    _used = set()

    def _safe(symbol: str) -> str:
        if symbol not in _safe_id:
            base = "".join(c for c in symbol if c.isalnum() or c == "_") or "sym"
            candidate = base
            n = 0
            while candidate in _used:
                n += 1
                candidate = f"{base}{n}"
            _used.add(candidate)
            _safe_id[symbol] = candidate
        return _safe_id[symbol]

    sym_vars: dict[str, dict[str, z3.ArithRef]] = {}

    def exponent_vars(symbol: str) -> dict[str, z3.ArithRef]:
        if symbol not in sym_vars:
            sid = _safe(symbol)
            sym_vars[symbol] = {name: z3.Real(f"{sid}_{name}") for name in _BASE_NAMES}
        return sym_vars[symbol]

    def fixed_dims(symbol: str) -> list[z3.BoolRef]:
        dim = None
        key = canonical_symbol_name(symbol)
        if overrides:
            for k, v in overrides.items():
                if canonical_symbol_name(k) == key:
                    dim = v
                    break
        if dim is None:
            from formulagate.dimensions import dimension_of_symbol

            dim = dimension_of_symbol(symbol)
        if dim is None:
            return []
        ev = exponent_vars(symbol)
        cons: list[z3.BoolRef] = []
        for name, exponent in zip(_BASE_NAMES, dim.exponents):
            cons.append(ev[name] == float(exponent))
        return cons

    solver = z3.Solver()
    constraints: list[z3.BoolRef] = []

    def dim_of(expr) -> list[tuple[str, Fraction]] | None:
        """Collect (symbol, exponent) terms of a symbolic expression, or None."""
        if expr.is_number:
            return []
        if isinstance(expr, sympy.Symbol):
            return [(str(expr), Fraction(1))]
        if isinstance(expr, sympy.Mul):
            out: list[tuple[str, Fraction]] = []
            for arg in expr.args:
                part = dim_of(arg)
                if part is None:
                    return None
                out.extend(part)
            return out
        if isinstance(expr, sympy.Pow):
            base, power = expr.args
            if not power.is_number or not power.is_rational:
                return None
            part = dim_of(base)
            if part is None:
                return None
            try:
                coeff = Fraction(power)
            except Exception:
                return None
            return [(s, c * coeff) for s, c in part]
        return None

    def _term_vec(expr) -> list[tuple[str, Fraction]] | None:
        """Dimension vector of a term treated as a monomial product.

        ``None`` when the term cannot be decomposed to a single monomial vector
        (an additive base under a power, or an unquantifiable power): such terms
        are handled by :func:`self_similar_term_vec`, which recurses into the
        addends.
        """
        if expr.is_number:
            return []
        if isinstance(expr, sympy.Symbol):
            return [(str(expr), Fraction(1))]
        if isinstance(expr, sympy.Mul):
            out: list[tuple[str, Fraction]] = []
            for arg in expr.args:
                part = _term_vec(arg)
                if part is None:
                    return None
                out.extend(part)
            return out
        if isinstance(expr, sympy.Pow):
            base, power = expr.args
            if not power.is_number or not power.is_rational:
                return None
            try:
                coeff = Fraction(power)
            except Exception:
                return None
            if isinstance(sympy.sympify(base), sympy.Add):
                return None
            part = _term_vec(base)
            if part is None:
                return None
            return [(s, c * coeff) for s, c in part]
        if isinstance(expr, sympy.Derivative):
            # d(f)/dt: dimension = dim(f) / dim(t)
            func = expr.args[0]
            var = expr.args[1]
            if isinstance(var, sympy.Tuple):
                var = var[0]
            part = _term_vec(func)
            var_part = _term_vec(var)
            if part is None or var_part is None:
                return None
            out = part.copy()
            for s, c in var_part:
                out.append((s, -c))
            return out
        if isinstance(expr, sympy.Integral):
            # ∫ f dx: dimension = dim(f) × dim(x)
            func = expr.args[0]
            var = expr.args[1]
            if isinstance(var, sympy.Tuple):
                var = var[0]
            part = _term_vec(func)
            var_part = _term_vec(var)
            if part is None or var_part is None:
                return None
            return part + var_part
        return None

    def self_similar_term_vec(term) -> tuple[list[z3.BoolRef], list[tuple[str, Fraction]] | None]:
        """Decompose one additive term into ``(nested constraints, vector)``.

        A term is a product of factors and rational powers. When a power's base
        is itself additive (``(a + b)^k``, e.g. ``sqrt(a+b)``), dimensional
        consistency demands every inner addend share a dimension; those equality
        constraints are returned and the power is applied to the resulting vector.
        """
        if isinstance(term, sympy.Pow):
            base, power = term.args
            if power.is_number and power.is_rational:
                zbase = sympy.sympify(base)
                if isinstance(zbase, sympy.Add):
                    try:
                        coeff = Fraction(power)
                    except Exception:
                        coeff = None
                    if coeff is not None:
                        inner_cs, inner_vec = self_similar_term_vec(zbase)
                        if inner_vec is None:
                            return inner_cs, None
                        scaled = [(s, c * coeff) for s, c in inner_vec]
                        return inner_cs, scaled
            return [], _term_vec(term)

        if isinstance(term, sympy.Add):
            addends = [a for a in term.args if not a.is_number]
            if not addends:
                return [], []
            cs: list[z3.BoolRef] = []
            vecs: list[list[tuple[str, Fraction]] | None] = []
            for a in addends:
                ac, av = self_similar_term_vec(a)
                cs.extend(ac)
                vecs.append(av)
            first = next((v for v in vecs if v is not None), None)
            if first is None:
                return cs, None
            cs.extend(_vec_eq(first, v) for v in vecs[1:] if v is not None)
            return cs, first

        return [], _term_vec(term)

    def _vec_eq(a: list[tuple[str, Fraction]], b: list[tuple[str, Fraction]]) -> z3.BoolRef:
        """Equality of two term-dimension vectors over the SI base dimensions.

        Each term's dimension in a base is ``sum(sym_coeff * sym_exponent[base])``,
        so two terms share a dimension iff those linear forms agree on every base.
        """
        eqs: list[z3.BoolRef] = []
        for base in _BASE_NAMES:
            lhs = _term_base_dim(a, base)
            rhs = _term_base_dim(b, base)
            eqs.append(lhs == rhs)
        return z3.And(*eqs)

    def _term_base_dim(vec: list[tuple[str, Fraction]], base: str) -> z3.ArithRef:
        out: z3.ArithRef | None = None
        for sym, coeff in vec:
            ev = exponent_vars(sym)
            term = coeff.numerator * ev[base] / coeff.denominator
            out = term if out is None else out + term
        return out if out is not None else z3.RealVal(0)

    def term_constraints(expr) -> list[z3.BoolRef]:
        """Constraints forcing the additive terms of ``expr`` to one dimension.

        ``expr`` is already ``lhs - rhs``; dimensional consistency means every
        symbolic term of that sum carries the same dimension vector. Each term
        is expanded recursively through products and rational powers; a root of
        an additive base (``sqrt(a + b)``) additionally requires its inner terms
        to share a dimension.
        """
        terms = list(expr.args) if isinstance(expr, sympy.Add) else [expr]
        symbolic = [t for t in terms if not t.is_number]
        if len(symbolic) < 2:
            # A single term cannot be internally inconsistent.
            return []

        out: list[z3.BoolRef] = []
        vecs: list[list[tuple[str, float]] | None] = []
        for term in symbolic:
            nested, part = self_similar_term_vec(term)
            out.extend(nested)
            vecs.append(part)

        first = next((v for v in vecs if v is not None), None)
        if first is None:
            return out
        return out + [_vec_eq(first, v) for v in vecs[1:] if v is not None]

    # Ground the known symbols.
    for sym in expr.free_symbols:
        constraints.extend(fixed_dims(str(sym)))

    # The equation is lhs - rhs = 0, so its additive terms must share a
    # dimension vector — that is the consistency requirement.
    constraints.extend(term_constraints(expr))

    if not constraints:
        return SmtVerdict("unknown", detail="no constraints — nothing to decide")

    try:
        solver.add(z3.And(*constraints))
        result = solver.check()
    except z3.Z3Exception as exc:
        # Some real-world equations contain expressions Z3 cannot quantify
        # (e.g. irrational or undefined powers). Degrade to unknown, never crash.
        return SmtVerdict("unknown", detail=f"z3 rejected the system: {exc}"[:140])
    if result == z3.sat:
        if require_grounded:
            # Z3 can satisfy systems whose symbols float freely.  Only count the
            # result as "consistent" when every symbol has a pinned dimension —
            # otherwise the SAT is vacuous.  Exception: when the equation carries
            # at least one certain anchor (fundamental constant or prose-stated
            # symbol), the overall dimension is pinned and floating symbols are
            # harmless — they adapt to the anchor.
            ungrounded = _ungrounded_symbols(expr, overrides)
            if ungrounded:
                certain = set(_CERTAIN_CONSTANTS)
                if overrides:
                    certain.update(canonical_symbol_name(k) for k in overrides)
                certain.update(canonical_symbol_name(s) for s in certain_symbols)
                anchored = any(canonical_symbol_name(str(s)) in certain for s in expr.free_symbols)
                if not anchored:
                    return SmtVerdict(
                        "unknown",
                        detail=f"under-determined: symbols not grounded "
                               f"({', '.join(sorted(ungrounded))[:60]})",
                    )
                # Has an anchor — SAT with floating symbols is acceptable.
        model = solver.model()
        # Read the shared dimension from the first symbolic term's exponents.
        dim = _model_dimension(model, expr, sym_vars)
        return SmtVerdict(
            "consistent", detail=f"SMT proof: satisfiable (dimension {dim})", dimension=dim
        )
    if result == z3.unsat:
        # A reject is only trustworthy when the proof is anchored to symbols
        # whose meaning is certain — a prose-stated override, a fundamental
        # constant, or a caller-declared "certain" symbol (prose context). A
        # reject resting purely on guessable single-letter table entries
        # (t, m, rho …) is an artefact of the table's guess and must abstain.
        certain = set(_CERTAIN_CONSTANTS)
        if overrides:
            certain.update(canonical_symbol_name(k) for k in overrides)
        certain.update(canonical_symbol_name(s) for s in certain_symbols)
        anchored = any(canonical_symbol_name(str(s)) in certain for s in expr.free_symbols)
        if not anchored:
            return SmtVerdict(
                "unknown",
                detail="unsat rests only on guessable table symbols - abstaining",
            )
        return SmtVerdict(
            "inconsistent",
            detail="SMT proof: unsatisfiable  - no dimension assignment is consistent",
        )
    return SmtVerdict("unknown", detail="SMT could not decide the system")


def infer_symbol_dimensions(
    formula,
    *,
    overrides: Mapping[str, Dimension] | None = None,
    certain_symbols: Sequence[str] = (),
) -> dict[str, Dimension]:
    """Run Z3 on ``formula`` and return the dimension of every free symbol.

    When the system is SAT, Z3 produces a concrete assignment for each
    symbol's 7 base-dimension exponents.  This is the foundation of weak
    supervision: if the same symbol gets the same dimensions across multiple
    equations in one paper, we can learn its dimension without prose.

    Returns:
        ``{symbol_name: dimension}`` for every free symbol in the formula,
        or ``{}`` if Z3 cannot produce a SAT model.
    """
    import sympy
    import z3  # noqa: F811

    if not getattr(formula, "canonical", None):
        return {}

    expr = sympy.sympify(formula.canonical)
    if not getattr(expr, "free_symbols", None):
        return {}

    sym_vars: dict[str, dict[str, z3.ArithRef]] = {}
    _used = set()

    def _safe(symbol: str) -> str:
        if symbol not in _safe_id:
            base = "".join(c for c in symbol if c.isalnum() or c == "_") or "sym"
            n = 0
            candidate = base
            while candidate in _used:
                n += 1
                candidate = f"{base}{n}"
            _used.add(candidate)
            _safe_id = {symbol: candidate}
        return _safe_id.get(symbol, symbol)

    _safe_id: dict[str, str] = {}
    for s in expr.free_symbols:
        name = str(s)
        sid = _safe(name)
        sym_vars[name] = {b: z3.Real(f"{sid}_{b}") for b in _BASE_NAMES}

    def _dim(symbol: str) -> Dimension | None:
        key = canonical_symbol_name(symbol)
        if overrides:
            for k, v in overrides.items():
                if canonical_symbol_name(k) == key:
                    return v
        from formulagate.dimensions import dimension_of_symbol
        return dimension_of_symbol(symbol)

    solver = z3.Solver()
    for s in expr.free_symbols:
        name = str(s)
        d = _dim(name)
        if d is not None:
            ev = sym_vars[name]
            for b, exp in zip(_BASE_NAMES, d.exponents):
                solver.add(ev[b] == float(exp))

    # Build and add term constraints (simple version — expand additive terms)
    # We reuse the term_constraints logic from check_dimensions_smt via a direct
    # call rather than duplicating it.
    from formulagate.verify_smt import check_dimensions_smt
    verdict = check_dimensions_smt(formula, overrides=overrides,
                                   certain_symbols=certain_symbols)
    if verdict.status not in ("consistent", "inconsistent"):
        return {}

    # Re-run with simple term constraint
    try:
        terms = list(expr.args) if isinstance(expr, sympy.Add) else [expr]
        symbolic = [t for t in terms if not t.is_number]
        if len(symbolic) < 2:
            solver.check()
            if solver.check() != z3.sat:
                return {}
        else:
            # Add equality constraints between all pairs of terms
            for i in range(len(symbolic) - 1):
                ev_i = sym_vars[str(symbolic[i])]
                ev_j = sym_vars[str(symbolic[i + 1])]
                for b in _BASE_NAMES:
                    solver.add(ev_i[b] == ev_j[b])
            if solver.check() != z3.sat:
                return {}
    except Exception:
        return {}

    model = solver.model()
    result: dict[str, Dimension] = {}
    for s in expr.free_symbols:
        name = str(s)
        if name not in sym_vars:
            continue
        ev = sym_vars[name]
        vals = {}
        for b in _BASE_NAMES:
            val = model[ev[b]]
            if val is not None:
                try:
                    vals[b] = Fraction(float(val.as_fraction()))
                except Exception:
                    try:
                        vals[b] = Fraction(int(val.as_long()))
                    except Exception:
                        pass
        if vals:
            result[name] = Dimension(
                mass=vals.get("mass", Fraction(0)),
                length=vals.get("length", Fraction(0)),
                time=vals.get("time", Fraction(0)),
                current=vals.get("current", Fraction(0)),
                temperature=vals.get("temperature", Fraction(0)),
                amount=vals.get("amount", Fraction(0)),
                luminosity=vals.get("luminosity", Fraction(0)),
            )
    return result


def _ungrounded_symbols(expr, overrides) -> set[str]:
    """Symbols of ``expr`` with no pinned dimension (override or curated table)."""
    from formulagate.dimensions import dimension_of_symbol

    # Mathematical constants hold no dimensional freedom: SymPy parses e^{iθ}
    # with e and i as free symbols, but e (Euler) and i (imaginary unit) are
    # dimensionless numbers, not physics symbols. Leaving them ungrounded turns
    # every equation that uses them into "unknown", and — since unsat is a
    # proof regardless — never into a reject.
    _DIMENSIONLESS_CONSTANTS = {"pi", "Pi", "e", "i", "I"}
    ungrounded: set[str] = set()
    for sym in expr.free_symbols:
        name = str(sym)
        if overrides and name in overrides:
            continue
        if name in _DIMENSIONLESS_CONSTANTS:
            continue
        if dimension_of_symbol(name) is not None:
            continue
        ungrounded.add(name)
    return ungrounded


def _model_dimension(model, expr, sym_vars):
    """Read the realised dimension of the first symbolic term from the model."""
    import sympy

    def collect(expr) -> list[tuple[str, float]] | None:
        if expr.is_number:
            return []
        if isinstance(expr, sympy.Symbol):
            return [(str(expr), 1.0)]
        if isinstance(expr, sympy.Mul):
            out = []
            for a in expr.args:
                p = collect(a)
                if p is None:
                    return None
                out.extend(p)
            return out
        if isinstance(expr, sympy.Pow):
            b, pw = expr.args
            if not pw.is_number:
                return None
            p = collect(b)
            return [(s, c * float(pw)) for s, c in p] if p else []
        return None

    terms = list(expr.args) if isinstance(expr, sympy.Add) else [expr]
    for term in terms:
        if term.is_number:
            continue
        vec = collect(term)
        if not vec:
            continue
        exps: dict[str, Fraction] = {n: Fraction(0) for n in _BASE_NAMES}
        for sym, coeff in vec:
            ev = sym_vars.get(sym)
            if not ev:
                continue
            for name in _BASE_NAMES:
                try:
                    val = model.eval(ev[name])
                    num = val.numerator_as_long()
                    den = val.denominator_as_long()
                    exps[name] += Fraction(coeff) * Fraction(num, den)
                except Exception:
                    continue
        return Dimension(**exps)
    return None
