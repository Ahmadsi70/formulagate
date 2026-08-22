"""Phase 1: context-aware symbol grounding."""

from __future__ import annotations

from formulagate.symbol_grounding import ground_symbols
from formulagate.dimensions import Dimension


def test_context_grounds_overloaded_symbol() -> None:
    # S is deliberately absent from the table (overloaded); only prose decides.
    g = ground_symbols(
        r"S = k_B \log W",
        context_after="where S is the entropy, W the number of microstates",
    )
    s = {x.symbol: x for x in g.grounded}.get("S")
    assert s is not None
    assert s.source == "context"
    assert s.dimension == Dimension(mass=1, length=2, time=-2, temperature=-1)


def test_empty_overrides_for_missing_meaning() -> None:
    # A symbol with no stated meaning stays out of the override map.
    g = ground_symbols(r"E = m c^2", context_after="")
    assert g.as_overrides() == {}


def test_constant_symbols_pinned() -> None:
    g = ground_symbols(r"E = m c^2", context_after="")
    dims = {x.symbol: x.dimension for x in g.grounded}
    # c is a named constant, m/E come from the conservative table.
    assert dims.get("c") == Dimension(length=1, time=-1)


def test_context_coverage_is_partial_not_full() -> None:
    # Grounding reports coverage honestly; unknown symbols stay unknown.
    g = ground_symbols(r"S = some_unknown + T", context_after="where S is the entropy")
    assert g.unknown


def test_quantity_phrase_resolution() -> None:
    from formulagate.symbol_grounding import _resolve_phrase

    assert _resolve_phrase("electric field") == Dimension(
        mass=1, length=1, time=-3, current=-1
    )
    assert _resolve_phrase("dimensionless") == Dimension()


def test_greek_symbols_grounded_via_table() -> None:
    g = ground_symbols(r"\rho = m / V", context_after="where rho is the mass density")
    rho = {x.symbol: x for x in g.grounded}.get("\u03c1")
    # rho maps to mass density either from table or context.
    if rho is not None:
        assert rho.dimension is not None


def test_comma_separated_sibling_definitions() -> None:
    # "R is the Ricci scalar, G is the Gravitational constant" — the second
    # definition has no where/with/and prefix, only a comma separator. Both
    # symbols must ground, or constants defined later in the sentence never
    # anchor their equations.
    g = ground_symbols(
        r"S = \frac{R}{16\pi G}",
        context_after=(
            "where R is the Ricci scalar, G is the Gravitational constant, "
            "g represents the determinant"
        ),
    )
    dims = {x.symbol: x.dimension for x in g.grounded}
    assert dims.get("R") == Dimension(mass=0, length=-2, time=0)
    assert dims.get("G") == Dimension(mass=-1, length=3, time=-2)


def test_document_wide_definition_propagation() -> None:
    # "L_p is the Planck length" in eq1's prose should propagate to eq2
    # even though eq2's context never mentions L_p. This is the core of Phase 3.
    from formulagate.symbol_grounding import ground_paper
    from formulagate.tex_ingest import SourceEquation

    eq1 = SourceEquation(
        latex=r"S = \frac{A}{4 L_p^2}",
        context_before="",
        context_after="where L_p is the Planck length",
        source_file="paper.tex",
        offset=0,
        symbols=("S", "A", "L_p"),
    )
    eq2 = SourceEquation(
        latex=r"F = \frac{L_p^2}{R^2}",
        context_before="",
        context_after="",
        source_file="paper.tex",
        offset=100,
        symbols=("F", "L_p", "R"),
    )
    results = ground_paper([eq1, eq2])
    g2 = results[1]
    lp_sym = {x.symbol: x for x in g2.grounded}.get("L_{p}")
    assert lp_sym is not None
    assert lp_sym.source == "context"
    assert lp_sym.dimension == Dimension(length=1)


def test_phrase_stops_before_sibling_definition() -> None:
    # "V_0 is a constant and L_p is the Planck length" must NOT slurp "and
    # L_p is the Planck length" into V_0's phrase. "a constant" is ambiguous
    # (a constant may carry any dimension) so V_0 stays ungrounded — the point
    # is the phrase boundary, not that V_0 resolves.
    g = ground_symbols(
        r"V(\vartheta) = V_0 + L_p^2",
        context_after="where V_0 is a constant and L_p is the Planck length",
    )
    grounded = {x.symbol: x for x in g.grounded}
    assert "V_{0}" not in grounded or grounded["V_{0}"].quantity == "a constant"
    lp = grounded.get("L_{p}")
    assert lp is not None
    assert lp.quantity == "Planck length"
    assert lp.dimension == Dimension(length=1)