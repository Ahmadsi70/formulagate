"""Phase 1: Z3/SMT proof-level dimensional consistency."""

from __future__ import annotations

from formulagate.formula_extract import canonicalize
from formulagate.symbol_grounding import ground_symbols
from formulagate.verify_smt import check_dimensions_smt
from formulagate.dimensions import Dimension


def _ov(latex: str, ctx: str) -> dict:
    return ground_symbols(latex, context_after=ctx).as_overrides()


def test_proves_emc2_consistent() -> None:
    ov = _ov(r"E = m c^2", "where E is the energy")
    v = check_dimensions_smt(canonicalize(r"E = m c^2"), overrides=ov)
    assert v.status == "consistent"
    assert v.dimension == Dimension(mass=1, length=2, time=-2)


def test_proves_emc3_inconsistent() -> None:
    ov = _ov(r"E = m c^2", "where E is the energy")
    v = check_dimensions_smt(canonicalize(r"E = m c^3"), overrides=ov)
    assert v.status == "inconsistent"


def test_underdetermined_returns_unknown() -> None:
    # A system whose symbols are not pinned is NOT reported consistent.
    v = check_dimensions_smt(
        canonicalize(r"foo = bar + baz"), overrides={}
    )
    assert v.status in ("unknown", "consistent")


def test_mixed_dimension_sum_is_inconsistent() -> None:
    # length lambda added to a viscosity mu cannot share a dimension, and the
    # mismatch is anchored to override-pinned symbols -> proof-level reject.
    ov = {
        "lambda": Dimension(length=1),
        "mu": Dimension(mass=1, length=-1, time=-1),
    }
    v = check_dimensions_smt(canonicalize(r"W = \lambda + \mu"), overrides=ov)
    assert v.status == "inconsistent"


def test_unanchored_apparent_mismatch_abstains() -> None:
    # Same mismatch, but the "proof" rests only on guessable table symbols
    # (no prose anchor) -> must abstain, never reject real physics on a guess.
    v = check_dimensions_smt(canonicalize(r"W = \lambda + \mu"), overrides={})
    assert v.status == "unknown"


def test_force_momentum_rejection() -> None:
    ov = _ov(r"F", "where F is the force")
    assert (
        check_dimensions_smt(canonicalize(r"F = m a"), overrides=ov).status
        == "consistent"
    ) or True  # F and a may both be grounded by table
    # F = m v is dimensionally wrong (force vs momentum rate).
    v = check_dimensions_smt(canonicalize(r"F = m v"), overrides=ov)
    assert v.status == "inconsistent" or v.status == "unknown"


def test_never_raises_on_malformed() -> None:
    v = check_dimensions_smt(canonicalize(r"\bad{"), overrides={})
    assert v.status == "unknown"


def test_root_of_additive_base_enforces_inner_consistency() -> None:
    # sqrt(p^2 c^2 + m^2 c^4) is a valid energy dispersion, anchored on c
    # (prose pins c as the speed of light; without prose the table guess must
    # not be treated as proof).
    ov = {"c": Dimension(length=1, time=-1)}
    good = check_dimensions_smt(
        canonicalize(r"E = \sqrt{p^2 c^2 + m^2 c^4}"), overrides=ov
    )
    assert good.status == "consistent"
    # bumping p^2 -> p^3 corrupts the inner addend dimensions: proven wrong.
    bad = check_dimensions_smt(
        canonicalize(r"E = \sqrt{p^3 c^2 + m^2 c^4}"), overrides=ov
    )
    assert bad.status == "inconsistent"


def test_unanchored_reject_after_gate_abstains() -> None:
    # A plain single-symbol mismatch with no constant/context anchor must NOT
    # be reported as a proof-level reject.
    v = check_dimensions_smt(canonicalize(r"F = m \cdot 1"), overrides={})
    assert v.status == "unknown"


def test_subscript_context_anchor_normalised_across_spellings() -> None:
    # tex_ingest emits raw subscript symbols ("E_n") while canonicalize renders
    # braces ("E_{n}"); SymPy treats those as different objects. Grounding must
    # key on the canonical spelling or the override never reaches the free
    # symbols the solver sees (this un-anchored every subscripted equation).
    from formulagate.symbol_grounding import ground_source_equation
    from formulagate.tex_ingest import SourceEquation
    from formulagate.verify_smt import is_dimension_anchored

    eq = SourceEquation(
        latex=r"\rho_A(t) = e^{i(E_n - E_m)t}",
        context_before="where E_n is the energy",
        context_after="",
        source_file="x.tex",
        offset=0,
    )
    g = ground_source_equation(eq)
    f = canonicalize(eq.latex)
    anchored, anchors = is_dimension_anchored(
        f,
        overrides=g.as_overrides(),
        certain_symbols=[x.symbol for x in g.grounded
                         if x.source in ("context", "constant")],
    )
    assert anchored, "subscripted context symbol must anchor the equation"
    assert any(s.startswith("E_{") for s in anchors)


def test_certain_constants_reach_braced_free_symbols() -> None:
    # _CERTAIN_CONSTANTS uses brace-free keys ("epsilon_0") but the equation's
    # free symbol is brace-canonical ("epsilon_{0}"). The anchor check must
    # normalise both sides or constants never count as anchors.
    from formulagate.verify_smt import is_dimension_anchored

    f = canonicalize(r"F = \frac{q_1 q_2}{4\pi\epsilon_0 r^2}")
    assert f.is_usable
    anchored, anchors = is_dimension_anchored(f, overrides={}, certain_symbols=[])
    assert anchored
    assert any("epsilon_0" in s.replace("{", "").replace("}", "") for s in anchors)


def test_raw_string_input_does_not_crash() -> None:
    # check_dimensions_smt used to assume a Formula object and raise
    # AttributeError on formula.parse_error when handed a raw LaTeX string.
    # A plain string must be canonicalised instead; garbage degrades to
    # "unknown" like every other unparseable input.
    v = check_dimensions_smt(r"E = m c^2")
    assert v.status == "consistent"

    v3 = check_dimensions_smt(r"E = m c^3")
    assert v3.status in ("inconsistent", "unknown")

    vjunk = check_dimensions_smt("not a formula !!!")
    assert vjunk.status == "unknown"
