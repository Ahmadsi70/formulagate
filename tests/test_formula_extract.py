"""Formula extraction and canonicalisation: text → equations the gate can reason about."""

from __future__ import annotations

import pytest

from formulagate.formula_extract import (
    Formula,
    canonicalize,
    extract_formulas,
    extract_latex_spans,
    structural_match,
)


# ─── Span extraction (no symbolic engine involved) ────────────────────────────


def test_extracts_inline_and_display_math() -> None:
    text = r"We find $E = mc^2$ and, in the strong regime, \[ S = \frac{A}{4G} \]."
    assert extract_latex_spans(text) == ["E = mc^2", r"S = \frac{A}{4G}"]


def test_extracts_equation_environments() -> None:
    text = r"\begin{equation} F = m a \end{equation}"
    assert extract_latex_spans(text) == ["F = m a"]


def test_ignores_currency_and_prose() -> None:
    assert extract_latex_spans("The grant is $5 million and covers 3 years.") == []


def test_span_extraction_is_length_bounded() -> None:
    """Runaway spans are parser bombs — a 4000-char 'formula' is never a formula."""

    text = "$" + "x + " * 2000 + "y$"
    assert extract_latex_spans(text) == []


# ─── Canonicalisation ─────────────────────────────────────────────────────────


def test_canonicalize_returns_structural_hash_for_equation() -> None:
    formula = canonicalize("E = m c^2")
    assert formula.is_equation
    assert formula.parse_error is None
    assert formula.structure_hash
    assert set(formula.symbols) == {"E", "m", "c"}


def test_algebraically_equal_equations_share_a_hash() -> None:
    """The point of canonicalisation: surface form must stop mattering."""

    a = canonicalize("E = m c^2")
    b = canonicalize("m c^2 = E")
    c = canonicalize(r"E = c^{2} m")
    assert a.structure_hash == b.structure_hash == c.structure_hash
    assert structural_match(a, b)


def test_different_physics_gets_a_different_hash() -> None:
    assert canonicalize("E = m c^2").structure_hash != canonicalize("E = m c^3").structure_hash


def test_rearranged_equation_matches_original() -> None:
    assert structural_match(canonicalize("F = m a"), canonicalize("F - m a = 0"))


def test_unparsable_latex_degrades_instead_of_raising() -> None:
    formula = canonicalize(r"\this{is not} \valid[ latex")
    assert formula.parse_error is not None
    assert formula.structure_hash is None
    assert not structural_match(formula, canonicalize("E = m c^2"))


def test_canonicalize_is_deterministic() -> None:
    first = canonicalize(r"S = \frac{A}{4 G}")
    second = canonicalize(r"S = \frac{A}{4 G}")
    assert first.structure_hash == second.structure_hash


def test_structural_match_never_matches_two_failures() -> None:
    """Two unparsable strings are not evidence of agreement."""

    broken = canonicalize(r"\nonsense{")
    assert not structural_match(broken, canonicalize(r"\other{"))


# ─── Whole-text extraction ────────────────────────────────────────────────────


def test_extract_formulas_keeps_only_usable_equations() -> None:
    text = r"Background text. $E = m c^2$ and $x$ and \[ \bad{latex \]"
    formulas = extract_formulas(text)
    assert [f.latex for f in formulas] == ["E = m c^2"]


def test_extract_formulas_respects_limit() -> None:
    text = "$a = b$ $c = d$ $e = f$"
    assert len(extract_formulas(text, limit=2)) == 2


def test_formula_is_hashable_and_frozen() -> None:
    formula = canonicalize("E = m c^2")
    assert isinstance(formula, Formula)
    with pytest.raises(Exception):
        formula.latex = "x"  # type: ignore[misc]
