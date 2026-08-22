"""Document-wide definition extraction from LaTeX source text.

Symbols defined in an introduction paragraph never reach equations 20 pages
later via the 350-char context window.  This module scans the *entire* paper
for symbol-definition patterns and returns a table that ``ground_paper``
feeds as ``doc_defs``.
"""

from __future__ import annotations

import re

from formulagate.dimensions import Dimension, canonical_symbol_name
from formulagate.symbol_grounding import _resolve_phrase

# Pattern: "$X$ is the <quantity>" — inline math delimiters with LaTeX dollars.
_DEF_PATTERN = re.compile(
    r"(?:let\s+|define\s+|we\s+(?:define|denote|set)\s+)?"
    r"(\$[A-Za-z][A-Za-z0-9_]*\$)\s+"
    r"(?:is|denotes|denote|represents|means|being|be)\s+"
    r"(?:the\s+|a\s+)?([A-Za-z][A-Za-z _-]{3,80}?)\s*"
    r"(?=[.,;)]|$)",
    re.IGNORECASE,
)

# Pattern: "where X is the ..." or "let X denote the ..."
_BARE_DEF = re.compile(
    r"(?:where|with|here|let|define)\s+"
    r"([A-Za-z][A-Za-z0-9_]*)\s+"
    r"(?:is|denotes|denote|represents|means|being|be)\b\s+"
    r"(?:the\s+|a\s+)?([A-Za-z][A-Za-z _-]{3,80}?)\s*"
    r"(?=[.,;)]|$)",
    re.IGNORECASE,
)

# Pattern: "X is the Y" anywhere — most general, but weaker signal.
_BARE_ANY = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]*)\s+"
    r"(?:is|denotes|denote|represents|means)\b\s+"
    r"(?:the\s+|a\s+)?([A-Za-z][A-Za-z _-]{3,80}?)\s*"
    r"(?=[.,;)]|$)",
    re.IGNORECASE,
)

# Sibling: ", G the gravitational constant" or "; X is the entropy"
_SIBLING_DEF = re.compile(
    r"[,;]\s*(?:and\s+)?"
    r"(\$?[A-Za-z][A-Za-z0-9_]*\$?)\s+"
    r"(?:is|denotes|represents|means|denote)\b\s+"
    r"(?:the\s+|a\s+)?([A-Za-z][A-Za-z _-]{3,80}?)\s*"
    r"(?=[.,;)]|$)",
    re.IGNORECASE,
)


def extract_paper_defs(source_text: str) -> dict[str, tuple[str, Dimension]]:
    """Extract symbol definitions from the full paper text.

    Returns:
        Dict mapping canonical symbol name to (quantity_phrase, dimension).
    """
    result: dict[str, tuple[str, Dimension]] = {}

    for pattern in (_DEF_PATTERN, _BARE_DEF, _BARE_ANY, _SIBLING_DEF):
        for m in pattern.finditer(source_text):
            sym = m.group(1).strip("$")
            phrase = m.group(2).strip().lower().rstrip(" .,;)")
            dim = _resolve_phrase(phrase)
            if dim is None:
                continue
            if sym.lower() in {"this", "that", "it", "its", "they", "them",
                               "which", "what", "when", "how", "why", "not",
                               "but", "and", "for", "with", "from", "into"}:
                continue
            canon = canonical_symbol_name(sym)
            # Only multi-letter or subscripted symbols; bare single-letters
            # change meaning across sections (s = entropy vs spacetime).
            is_subscripted = "_" in canon
            base = canon.split("_")[0]
            if (len(base) >= 2 or is_subscripted) and canon not in result:
                result[canon] = (phrase, dim)

    return result
