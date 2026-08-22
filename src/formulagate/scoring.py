"""Exact scoring formula for formula-gate retrieval.

Why: deterministic, auditable accept/reject without embeddings:

    score = keyword_overlap(keys, record_blob) + SCORE_WEIGHT_REL * rel(formula, domain)

where ``rel`` counts domain-specific markers inside ``math_formula``.
"""

from __future__ import annotations

import re
from typing import Any

from formulagate.domain import Domain

# Canonical weight from the reference algorithm.
SCORE_WEIGHT_REL = 2

_STOP = frozenset(
    "the and for with that this from have will your must into when then than only also "
    "not are was were been being does should would could about after before under over "
    "each all any such other more most some than into through during while where which "
    "their there these those".split()
)

MATH_FORMULA_MARKERS = (
    # --- Original undergrad markers ---
    "mod",
    "equiv",
    "pmod",
    "gcd",
    "remainder",
    "mathbb{z}",
    "frac{",
    "sum_",
    "binom",
    "phi(",
    # --- PhD expansion: advanced math symbols ---
    "otimes",
    "oplus",
    "wedge",
    "vee",
    "forall",
    "exists",
    "mapsto",
    "longrightarrow",
    "hookrightarrow",
    "twoheadrightarrow",
    "cong",
    "simeq",
    "approx",
    "propto",
    "sim",
    "doteq",
    "overset",
    "underset",
    "xrightarrow",
    "xleftarrow",
    "mathcal",
    "mathfrak",
    "mathbb",
    "mathscr",
    "mathrm",
    "operatorname",
    "overline",
    "underline",
    "widehat",
    "widetilde",
    "bar",
    "vec",
    "dot",
    "ddot",
    "partial",
    "nabla",
    "int",
    "iint",
    "iiint",
    "oint",
    "prod",
    "coprod",
    "bigcup",
    "bigcap",
    "bigvee",
    "bigwedge",
    "bigoplus",
    "bigotimes",
    "lim",
    "liminf",
    "limsup",
    "inf",
    "sup",
    "max",
    "min",
    "det",
    "tr",
    "dim",
    "ker",
    "coker",
    "im",
    "hom",
    "ext",
    "tor",
    "spec",
    "proj",
    "colim",
    "injlim",
    "projlim",
    "varliminf",
    "varlimsup",
    "stackrel",
    "substack",
    "text",
    "mbox",
    "mathrm{d}",
    "mathrm{e}",
    "mathrm{i}",
    "langle",
    "rangle",
    "lVert",
    "rVert",
    "lvert",
    "rvert",
    "left",
    "right",
    "middle",
    "big",
    "Big",
    "bigg",
    "Bigg",
    "binom",
    "tfrac",
    "dfrac",
    "cfrac",
    "sfrac",
    "mathrlap",
    "mathllap",
    "smash",
    "phantom",
    "vphantom",
    "hphantom",
    "mathbin",
    "mathrel",
    "mathop",
    "mathord",
    "mathopen",
    "mathclose",
    "mathpunct",
    "sqrt",
    "root",
    "displaystyle",
    "textstyle",
    "scriptstyle",
    "scriptscriptstyle",
    # LaTeX from PhD formulas
    "hookrightarrow",
    "twoheadrightarrow",
    "relbar",
    "joinrel",
    "xrightarrow",
    "xleftarrow",
    "xleftrightarrow",
    "xmapsto",
    "xhookrightarrow",
    "xhookleftarrow",
    "xleftharpoondown",
    "xrightharpoondown",
)

PHYSICS_FORMULA_MARKERS = (
    # --- Basic equation structure ---
    "=",
    "^",
    "+",
    "-",
    "*",
    "/",
    # --- Original undergrad markers ---
    "frac{",
    "int",
    "nabla",
    "sigma",
    "omega",
    "cdot",
    "vec",
    "frac{d",
    # --- PhD expansion: quantum / field theory ---
    "hbar",
    "dagger",
    "braket",
    "ket",
    "bra",
    "langle",
    "rangle",
    "mathcal{L}",
    "mathcal{H}",
    "mathcal{O}",
    "mathcal{D}",
    "mathcal{Z}",
    "mathcal{T}",
    "mathcal{F}",
    "mathcal{M}",
    "mathcal{N}",
    "mathcal{S}",
    "mathcal{P}",
    "mathcal{R}",
    "mathcal{C}",
    "mathcal{G}",
    "mathcal{A}",
    "mathcal{W}",
    "mathcal{E}",
    "mathcal{B}",
    "mathcal{J}",
    "mathcal{K}",
    "mathcal{U}",
    "mathcal{V}",
    "mathcal{Q}",
    "hat",
    "bar",
    "tilde",
    "vec",
    "dot",
    "ddot",
    "partial",
    "delta",
    "Delta",
    "nabla",
    "square",
    "Box",
    "gamma",
    "Gamma",
    "lambda",
    "Lambda",
    "psi",
    "Psi",
    "phi",
    "Phi",
    "varphi",
    "epsilon",
    "varepsilon",
    "rho",
    "varrho",
    "theta",
    "Theta",
    "vartheta",
    "alpha",
    "beta",
    "eta",
    "mu",
    "nu",
    "tau",
    "xi",
    "Xi",
    "pi",
    "Pi",
    "varpi",
    "chi",
    "zeta",
    "Zeta",
    "kappa",
    "varkappa",
    "otimes",
    "oplus",
    "wedge",
    "star",
    "ast",
    "times",
    "cdot",
    "bullet",
    "circ",
    "tr",
    "det",
    "exp",
    "log",
    "ln",
    "sin",
    "cos",
    "tan",
    "cot",
    "sec",
    "csc",
    "sinh",
    "cosh",
    "tanh",
    "arcsin",
    "arccos",
    "arctan",
    "operatorname",
    "mathrm",
    "mathbf",
    "mathcal",
    "mathfrak",
    "mathbb",
    "mathscr",
    # --- PhD expansion: advanced physics notation ---
    "sum_",
    "prod_",
    "int_",
    "oint_",
    "iint_",
    "iiint_",
    "lim_",
    "inf",
    "sup",
    "left(",
    "right)",
    "left[",
    "right]",
    "left{",
    "right}",
    "left|",
    "right|",
    "left.",
    "right.",
    "big(",
    "big[",
    "big{",
    "Big(",
    "Big[",
    "bigg(",
    "bigg[",
    "Bigg(",
    "Bigg[",
    "quad",
    "qquad",
    "nonumber",
    "tag",
    "label",
    "ref",
    "eqref",
    "cite",
    "text",
    "mbox",
    "displaystyle",
    "textstyle",
    # More advanced physics
    "rangle",
    "langle",
    "ket",
    "bra",
    "braket",
    "ev",
    "order",
    "commutator",
    "anticommutator",
    "trace",
    "Tr",
    "diag",
    "rank",
    "sgn",
    "residue",
    "Res",
    "Pf",
    "erf",
    "erfc",
    "Gamma",
    "Beta",
    "zeta",
    "Li",
    "Ei",
    "Si",
    "Ci",
    "sinc",
    "J_",
    "Y_",
    "H_",
    "I_",
    "K_",
    "P_",
    "delta",
    "Theta",
    "varepsilon",
    "varkappa",
    "varpi",
    "varrho",
    "varsigma",
    "varphi",
    "ell",
    "wp",
    "Re",
    "Im",
    "aleph",
    "beth",
    "gimel",
    "daleth",
    "infty",
    "propto",
    "sim",
    "simeq",
    "approx",
    "equiv",
    "cong",
    "neq",
    "le",
    "ge",
    "ll",
    "gg",
    "lll",
    "ggg",
    "prec",
    "succ",
    "preceq",
    "succeq",
    "subset",
    "supset",
    "subseteq",
    "supseteq",
    "in",
    "ni",
    "notin",
    "subsetneq",
    "supsetneq",
    "subseteqq",
    "supseteqq",
    "cap",
    "cup",
    "setminus",
    "smallsetminus",
    "times",
    "otimes",
    "oslash",
    "odot",
    "circledcirc",
    "boxplus",
    "boxtimes",
    "boxdot",
    "mp",
    "pm",
    "div",
    "ast",
    "star",
    "circ",
    "bullet",
    "diamond",
    "triangle",
    "triangledown",
    "triangleleft",
    "triangleright",
    "bigtriangleup",
    "bigtriangledown",
    "oplus",
    "ominus",
    "oslash",
    "odot",
    "bigcirc",
    "dagger",
    "ddagger",
    "amalg",
    "wr",
    "cdot",
    "centerdot",
    "circ",
    "bullet",
    "bigcirc",
    "setminus",
    "smallsetminus",
)

_RECORD_TEXT_KEYS = ("english", "math_formula", "scientific_domain", "arabic", "title", "text")


def keywords(text: str) -> set[str]:
    """Extract tokens length >= 4, lowercased, minus English stopwords."""

    return {
        w.lower()
        for w in re.findall(r"[a-zA-Z]{4,}", text)
        if w.lower() not in _STOP
    }


# Fallback markers for simple equations that lack fancy LaTeX (e.g. "E = m c^2").
# Without these, basic algebraic physics formulas get hard-zeroed and become
# invisible to the gate even when they are perfectly valid physics.
_BASIC_EQUATION_STRUCTURE = frozenset(
    ("=", "^", "+", "-", "*", "/", "_", "{", "}", "(", ")", "[", "]")
)

# A non-trivial equation must have at least this many basic structure markers
# (like =, +, ^, etc.) to benefit from the fallback.  Simple strings like
# "x=1+1" that happen to contain basic markers are still filtered out because
# they lack sufficient structure.
_MIN_STRUCTURE_MARKERS = 2


def formula_domain_relevance(math_formula: str, domain: Domain) -> int:
    """Count domain markers inside a formula string (``rel``)."""

    low = math_formula.lower()
    if domain == "mathematics":
        rel = sum(1 for m in MATH_FORMULA_MARKERS if m in low)
        if rel == 0:
            # Fallback: count basic equation structure markers ( =, ^, +, … ).
            # Requires at least _MIN_STRUCTURE_MARKERS to avoid giving positive
            # rel to trivial strings like "x=1+1" or "abcdef".
            basic = sum(1 for m in _BASIC_EQUATION_STRUCTURE if m in low)
            if basic >= _MIN_STRUCTURE_MARKERS:
                rel = basic
        return rel
    if domain == "physics":
        rel = sum(1 for m in PHYSICS_FORMULA_MARKERS if m in low)
        if rel == 0:
            basic = sum(1 for m in _BASIC_EQUATION_STRUCTURE if m in low)
            if basic >= _MIN_STRUCTURE_MARKERS:
                rel = basic
        return rel
    if domain == "pharmacology":
        rel = sum(1 for m in PHYSICS_FORMULA_MARKERS if m in low)
        if rel == 0:
            basic = sum(1 for m in _BASIC_EQUATION_STRUCTURE if m in low)
            if basic >= _MIN_STRUCTURE_MARKERS:
                rel = basic
        return rel
    # General domain: count formula markers plus basic structure, with a
    # minimum symbol-count threshold to filter trivial strings.  This catches
    # cross-domain content (chemistry, biology, economics) that uses math
    # notation without being strictly math or physics.
    if len(math_formula) < 6:
        return 0
    marker_count = sum(1 for m in MATH_FORMULA_MARKERS if m in low)
    marker_count += sum(1 for m in PHYSICS_FORMULA_MARKERS if m in low)
    if marker_count > 0:
        return marker_count
    # Fallback: count basic equation structure markers with a stricter minimum.
    basic = sum(1 for m in _BASIC_EQUATION_STRUCTURE if m in low)
    if basic >= _MIN_STRUCTURE_MARKERS + 1:  # stricter: need >=3 structure markers
        return basic
    # Symbol overlap: count distinct variable letters as evidence of formula content.
    vars_found = _count_distinct_variables(low)
    if vars_found >= 3:
        return max(1, vars_found // 2)
    return 0


def _count_distinct_variables(formula_lower: str) -> int:
    """Count distinct letters that could be physics/math variables.

    Removes LaTeX commands first, then counts each unique letter that appears
    as a standalone variable.  Adjacent letters (like ``PV`` meaning P×V) are
    counted individually because in physics notation that is implicit
    multiplication of single-letter quantities.
    """
    import re
    # Remove LaTeX commands (backslash + letters) so \int, \frac, etc. are skipped.
    cleaned = re.sub(r"\\[a-zA-Z]+", " ", formula_lower)
    # Collect every alphabetic character that is not part of a known multi-letter
    # word (like "sin", "log", "pH").  We keep it simple: count every letter.
    letters = set(c for c in cleaned if c.isalpha())
    return len(letters)


def score_record(
    keys: set[str],
    row: dict[str, Any],
    domain: Domain,
) -> tuple[int, int]:
    """Return ``(score, rel)`` for one corpus row.

    Mathematics rows with ``rel == 0`` are hard-zeroed (weak analog rejected).
    """

    blob = " ".join(str(row.get(k) or "") for k in _RECORD_TEXT_KEYS).lower()
    mf = str(row.get("math_formula") or row.get("formula") or "")
    if not mf or len(mf) < 6:
        return 0, 0
    rel = formula_domain_relevance(mf, domain)
    if domain == "mathematics" and rel == 0:
        return 0, 0
    # Physics also hard-zeroes when rel=0 (like mathematics)
    # This prevents pure-keyword matches without domain formula evidence
    if domain == "physics" and rel == 0:
        return 0, 0
    # Pharmacology: require at least equation-structure evidence
    if domain == "pharmacology" and rel == 0:
        return 0, 0
    overlap = sum(1 for k in keys if k in blob)
    # Reject pure marker/length hits with zero lexical support (esp. general τ=2).
    if overlap == 0:
        return 0, rel
    score = overlap + SCORE_WEIGHT_REL * rel
    return score, rel
