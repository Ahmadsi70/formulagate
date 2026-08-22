"""Phase 1: full-text equation extraction from real LaTeX sources."""

from __future__ import annotations

from formulagate.tex_ingest import extract_equations, _split_aligned_terms


_SAMPLE = r"""
\documentclass{article}
\begin{document}
The energy of the system is given by
\begin{equation}
E = m c^2
\label{eq:emc}
\end{equation}
where $E$ is the total energy.

The momentum and position satisfy
\begin{align}
p &= m v \\
x &= v t \nonumber
\end{align}
for a free particle.
\end{document}
"""


def test_extract_single_equation_with_context() -> None:
    eqs = extract_equations(_SAMPLE)
    eqs = [e for e in eqs if "emc" not in e.latex and e.latex.startswith("E")]
    assert any("E =" in e.latex for e in eqs)
    body = [e for e in eqs if e.latex.startswith("E")]
    assert body
    assert "energy" in (body[0].context_after or body[0].context_before)


def test_align_preserves_context_and_removes_nonumber() -> None:
    eqs = extract_equations(_SAMPLE)
    align = [e for e in eqs if e.env == "align"]
    assert align, "align block should yield equations"
    assert all("\\nonumber" not in e.latex for e in align)


def test_empty_source_returns_nothing() -> None:
    assert extract_equations("") == []
    assert extract_equations(None) == []


def test_expressions_report_symbols() -> None:
    eqs = extract_equations(_SAMPLE)
    momentum = [e for e in eqs if e.latex.startswith("p")]
    assert momentum and "m" in momentum[0].symbols and "v" in momentum[0].symbols


def test_comment_lines_are_not_equations() -> None:
    src = "% only a comment\n\\begin{equation}\nE = m c^2\n\\end{equation}"
    eqs = extract_equations(src)
    assert any("[2pt]" not in e.latex for e in eqs)