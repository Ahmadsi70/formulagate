"""Extract complete equations and their prose context from real LaTeX sources.

Why this exists:
  The dimensional veto needs *complete equations* (two comparable terms) to
  decide anything. Abstracts only quote fragments. The honest source of
  complete equations is the paper's own LaTeX source, which ``scripts/
  fetch_arxiv_sources.py`` downloads. This module turns that source into
  structured ``SourceEquation`` records:

    - the equation body (LaTeX, de-labelled, single-equation)
    - the symbol set
    - the surrounding prose (for context-aware symbol grounding)
    - the source file and character offset (for provenance)

  Design rules:
    * Never crashes on malformed LaTeX — a broken environment degrades to
      nothing, and the caller sees fewer equations, not an exception.
    * ``align``/``eqnarray`` blocks are split on ``\\\\`` into individual
      equations, but only at depth zero (not inside ``pmatrix``/``cases``).
    * ``\\text{...}`` is kept as a hint for grounding, not parsed as math.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Equation-like environments whose body we capture (with or without *).
_ENV = re.compile(
    r"\\begin\{((?:equation|displaymath|align|eqnarray|gather|multline)\*?)\}(.*?)\\end\{\1\}",
    re.DOTALL,
)
# Nested alignment/matrix environments that must not split on \\.
_NESTED_ENV = re.compile(r"\\begin\{([a-zA-Z*]+)\}")
# Strips \label{...}, \nonumber, \notag.
_NOISE = re.compile(r"\\label\s*\{[^}]*\}|\\nonumber|\\notag")
# Inline math markers so we can talk about symbols in prose.
_INLINE = re.compile(r"(?<!\\)\$([^$]+?)(?<!\\)\$")
# Comment start (a % that is not escaped).
_COMMENT = re.compile(r"(?<!\\)%[^\n]*")

_DEPTHLESS_SPLIT = re.compile(r"\\\\")
_NESTED = frozenset(
    {"pmatrix", "bmatrix", "vmatrix", "Vmatrix", "array", "cases", "aligned",
     "alignedat", "split", "smallmatrix", "matrix", "bsmallmatrix", "dcases",
     "rcases", "subarray"}
)


@dataclass(frozen=True)
class SourceEquation:
    """One complete equation lifted from a real LaTeX source."""

    latex: str
    context_before: str
    context_after: str
    source_file: str
    offset: int
    env: str = "equation"
    symbols: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "latex": self.latex,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "source_file": self.source_file,
            "offset": self.offset,
            "env": self.env,
            "symbols": list(self.symbols),
        }


def _strip_comments(source: str) -> str:
    return _COMMENT.sub("", source)


def _contains_noise(match: re.Match[str]) -> bool:
    """A line that is only comments/marks is not a real equation."""
    body = match.group(2)
    return bool(re.fullmatch(r"[\s\\*%&]+", body))


def _extract_prose_symbols(context: str) -> tuple[str, ...]:
    """Collect the LaTeX symbol tokens that appear in a prose passage.

    Falls back to bare alphanumeric tokens in the body itself (equation
    environments have no ``$...$`` around each symbol).
    """
    out: list[str] = []
    for match in _INLINE.finditer(context):
        body = match.group(1).strip()
        for token in _BARE_SYMBOL.findall(body):
            out.append(token)
    # Single-letter and subscripted tokens are the useful ground: multi-letter
    # words like "tr" or "exp" are functions, not physical symbols.
    filtered = [t for t in out if len(t) == 1 or "_" in t or t in _KNOWN_NAMED]
    return tuple(dict.fromkeys(filtered))


_BARE_SYMBOL = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_]*\b")

# Multi-letter LaTeX tokens that name physical quantities, not functions.
_KNOWN_NAMED = {
    "hbar", "ell", "lambda", "Lambda", "alpha", "beta", "gamma", "Gamma",
    "delta", "Delta", "epsilon", "zeta", "eta", "theta", "Theta", "iota",
    "kappa", "mu", "nu", "xi", "pi", "Pi", "rho", "sigma", "Sigma", "tau",
    "upsilon", "phi", "Phi", "chi", "psi", "Psi", "omega", "Omega",
    "Re", "Im", "Pr", "Re", "Nu", "Ra", "Pe", "Ma", "Fr", "St",
}


def _split_aligned_terms(body: str) -> list[str]:
    """Split an align-style body into individual equations at depth zero."""
    terms: list[str] = []
    current: list[str] = []
    depth = 0
    stack: list[str] = []

    for token in _DEPTHLESS_SPLIT.split(body):
        current.append(token)
    joined = body
    # Find \\ markers that sit at depth zero by scanning nested environments.
    open_pat = re.compile(r"\\begin\{([a-zA-Z*]+)\}")
    close_pat = re.compile(r"\\end\{([a-zA-Z*]+)\}")

    depth = 0
    nested: list[str] = []
    last = 0
    for m in _DEPTHLESS_SPLIT.finditer(body):
        segment = body[last : m.start()]
        last = m.end()
        # Update depth from segment.
        for op in open_pat.finditer(segment):
            name = op.group(1)
            nested.append(name)
            if name in _NESTED:
                depth += 1
        for cl in close_pat.finditer(segment):
            if nested:
                name = nested.pop()
                if name in _NESTED:
                    depth -= 1
        if depth == 0 and segment.strip():
            terms.append(segment.strip())
    if body[last:].strip():
        terms.append(body[last:].strip())
    return terms


def extract_equations(source: str, source_file: str = "main.tex") -> list[SourceEquation]:
    """Extract complete equations from a LaTeX source with prose context.

    Args:
        source: Raw LaTeX text of a real paper.
        source_file: File name for provenance.

    Returns:
        A list of ``SourceEquation`` records, each a single complete equation
        plus its surrounding prose context.
    """
    if not source:
        return []
    clean = _strip_comments(source)
    out: list[SourceEquation] = []
    seen: set[str] = set()

    for match in _ENV.finditer(clean):
        if _contains_noise(match):
            continue
        env = match.group(1).rstrip("*")
        body = _NOISE.sub("", match.group(2)).strip()
        if not body:
            continue
        start = match.start()

        before = clean[max(0, start - 350) : start]
        after = clean[match.end() : match.end() + 350]

        if env in ("align", "eqnarray", "gather", "multline"):
            equations = _split_aligned_terms(body)
            # First segment may be empty if the block starts with \\.
            for eq in equations:
                _push_equation(out, seen, eq, before, after, source_file, start, env)
        else:
            _push_equation(out, seen, body, before, after, source_file, start, env)

    return out


def _extract_body_symbols(body: str) -> tuple[str, ...]:
    """Candidate symbol names inside an equation body.

    Keeps single letters, letters with subscripts, and known multi-letter
    physical symbols; drops function words (``tr``, ``exp``, ``det``, ``sin``)
    and LaTeX commands like ``frac`` or ``right``.
    """
    _FUNCTION_WORDS = frozenset(
        """tr Trace det exp log ln sin cos tan cot sec csc min max arg Re Im Pr
        sum prod int lim right left frac sqrt over text quad qquad phantom
        operatorname big small middle lvert rvert langle rangle lbrace rbrace
        mathcal mathbb mathrm mathbf begin end nonumber label ref eqref cite
        pm mp times div partial nabla grad cdot cdots ldots vdots ddots""".split()
    )
    tokens = _BARE_SYMBOL.findall(body)
    out: list[str] = []
    for token in tokens:
        if token in _FUNCTION_WORDS:
            continue
        if token in _KNOWN_NAMED or token in _FUNCTION_WORDS:
            if token in _KNOWN_NAMED:
                out.append(token)
            continue
        if len(token) == 1 and token.isalpha():
            out.append(token)
            continue
        # subscripted or greek-style: e.g. x_i, E_0, X_IJ
        if "_" in token or len(token) <= 4:
            out.append(token)
    return tuple(dict.fromkeys(out))


def _push_equation(
    out: list[SourceEquation],
    seen: set[str],
    body: str,
    before: str,
    after: str,
    source_file: str,
    offset: int,
    env: str,
) -> None:
    """Append one equation unless it is empty, only-ampersand, or a duplicate."""
    body = body.strip().strip("&").strip()
    body = " ".join(body.split())
    if not body or len(body) < 3:
        return
    if body in seen:
        return
    seen.add(body)
    symbols = _extract_prose_symbols(body)
    if not symbols:
        symbols = _extract_body_symbols(body)
    out.append(
        SourceEquation(
            latex=body,
            context_before=" ".join(before.split()),
            context_after=" ".join(after.split()),
            source_file=source_file,
            offset=offset,
            env=env,
            symbols=symbols,
        )
    )


def extract_from_tex_file(path: Path, source_file: str | None = None) -> list[SourceEquation]:
    """Convenience: read a ``.tex`` file and extract its equations."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return extract_equations(text, source_file or path.name)
