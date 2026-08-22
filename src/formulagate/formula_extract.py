"""Turn prose into equations the gate can reason about structurally.

Why this module exists: until now a "formula" was a text blob, so two records
stating the same physics in different notation looked unrelated, and a record
whose ``math_formula`` field held a problem statement looked like mathematics.
Canonicalising to a SymPy expression replaces that string similarity with a
structural identity — ``E = mc^2``, ``mc^2 = E`` and ``E - mc^2 = 0`` collapse to
one hash.

SymPy's LaTeX parser rejects a large share of real arXiv markup, so every entry
point degrades to ``parse_error`` instead of raising: a gate that crashes on a
malformed abstract is worse than one that abstains.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

# A span longer than this is prose with dollar signs in it, or a parser bomb:
# SymPy's ANTLR grammar is superlinear on pathological input.
MAX_SPAN_CHARS = 200
MIN_SPAN_CHARS = 3
DEFAULT_FORMULA_LIMIT = 8

_INLINE = re.compile(r"(?<!\\)\$([^$]+?)(?<!\\)\$")
_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_PAREN = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
_ENVIRONMENT = re.compile(
    r"\\begin\{(?:equation|align|eqnarray|gather)\*?\}(.+?)\\end\{(?:equation|align|eqnarray|gather)\*?\}",
    re.DOTALL,
)

# "$5 million", "$3.2B" — currency, not mathematics.
_CURRENCY = re.compile(r"^\s*\d[\d,.]*\s*(?:million|billion|trillion|[kKmMbB])?\s*$")

# Plain-text equations without LaTeX delimiters: "E = m c^2", "F = ma", etc.
# Captures "VAR = …" broadly; ``_trim_rhs`` then strips trailing prose tokens.
_PLAINTEXT_EQ = re.compile(
    r"\b([a-zA-Z])\s*=\s*(.+?)(?=\s+[a-z]{3,}\b|[.;,!]\s|$)"
)


@dataclass(frozen=True)
class Formula:
    """One extracted equation plus whatever the symbolic engine could prove.

    ``structure_hash`` is ``None`` whenever parsing failed; downstream code must
    treat that as "unknown", never as "no match".
    """

    latex: str
    canonical: str | None = None
    symbols: tuple[str, ...] = ()
    structure_hash: str | None = None
    is_equation: bool = False
    parse_error: str | None = None

    @property
    def is_usable(self) -> bool:
        return self.structure_hash is not None


# ─── Span extraction (pure text, no symbolic engine) ──────────────────────────


def extract_latex_spans(text: str, limit: int | None = None) -> list[str]:
    """Collect LaTeX math spans in the order they appear.

    Environments are matched first and removed so their inner ``$...$`` is not
    reported twice. Plain-text equations (``E = m c^2`` without delimiters) are
    also collected as a fallback so the physics layer can see them.
    """

    if not text:
        return []

    spans: list[tuple[int, str]] = []
    remaining = text

    for pattern in (_ENVIRONMENT, _DISPLAY, _PAREN):
        for match in pattern.finditer(remaining):
            spans.append((match.start(), match.group(1)))
        remaining = pattern.sub(" ", remaining)

    for match in _INLINE.finditer(remaining):
        spans.append((match.start(), match.group(1)))

    out: list[str] = []
    for _, raw in sorted(spans, key=lambda item: item[0]):
        body = " ".join(raw.split())
        if not (MIN_SPAN_CHARS <= len(body) <= MAX_SPAN_CHARS):
            continue
        if _CURRENCY.match(body):
            continue
        out.append(body)
        if limit is not None and len(out) >= limit:
            break

    # ── Plain-text fallback ────────────────────────────────────────────────
    # If no delimited spans were found, scan for bare equations so the physics
    # layer is not blind to formulas written without $…$ or \[…\].
    if not out:
        out = _extract_plaintext_equations(text, limit=limit)

    return out


def _extract_plaintext_equations(text: str, limit: int | None = None) -> list[str]:
    """Find bare ``E = m c^2``-style equations in prose.

    Only fires when no LaTeX-delimited spans were found, so it never duplicates
    a formula that was already captured inside ``$…$``.
    """
    if not text:
        return []

    out: list[str] = []
    for match in _PLAINTEXT_EQ.finditer(text):
        var = match.group(1)
        rhs = match.group(2).strip()
        body = f"{var} = {_trim_rhs(rhs)}"
        # Drop trivial equations: "x = 10" or "x = 5 meters" carry no structure.
        if not _is_plausible_equation(body):
            continue
        if not (MIN_SPAN_CHARS <= len(body) <= MAX_SPAN_CHARS):
            continue
        out.append(body)
        if limit is not None and len(out) >= limit:
            break
    return out


def _trim_rhs(rhs: str) -> str:
    """Trim trailing prose words from a plain-text equation RHS.

    ``"m c^2 is the famous equation"`` → ``"m c^2"``.
    """
    # Strip trailing punctuation that the regex may have captured.
    rhs = rhs.rstrip(".,;:!?")
    tokens = rhs.split()
    # Walk backwards, keeping only math-like tokens.
    kept: list[str] = []
    for token in reversed(tokens):
        if _is_math_token(token):
            kept.append(token)
        else:
            # Stop at the first non-math token (prose word).
            break
    kept.reverse()
    if kept:
        return " ".join(kept)
    # Fallback: if nothing looked like math, return at most the first two tokens
    # (avoids returning empty when the RHS is "m c^2 is..." but "is" trips first).
    return " ".join(tokens[:2]) if tokens else ""


def _is_math_token(token: str) -> bool:
    """True when a token looks like a variable, number, operator, or unit."""
    if not token:
        return False
    # Single letter (variable): m, c, v, t, E, F, etc.
    if len(token) == 1 and token.isalpha():
        return True
    # Pure number or number with exponent: 2, 1/2, 3.14, c^2, 10^-3
    stripped = token.lstrip("^+-")
    if stripped and all(c.isdigit() or c in "./" for c in stripped):
        return True
    # Operator-only tokens: ^2, +, *, /, (), []
    if all(c in "^+*/()-[]{}" or c.isdigit() for c in token):
        return True
    # Known multi-char symbols: hbar, alpha, etc.
    if token in ("hbar", "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
                 "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi",
                 "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi",
                 "chi", "psi", "omega", "nabla", "partial", "int", "sum", "prod"):
        return True
    # Multi-char token with subscript/superscript/parentheses: c^2, v_0, m_earth
    # Must contain at least one digit or operator — pure-alpha is a prose word.
    if len(token) >= 2 and token[0].isalpha():
        has_non_alpha = any(c.isdigit() or c in "^_()[]{}" for c in token[1:])
        if has_non_alpha:
            return True
    return False


def _is_plausible_equation(body: str) -> bool:
    """Heuristic: a plausible equation has at least one variable on each side
    or a non-trivial operator (power, fraction, product)."""
    stripped = body.replace(" ", "")
    if "=" not in stripped:
        return False
    lhs, rhs = stripped.split("=", 1)
    # Must have at least one letter on each side (or an operator on RHS).
    has_lhs_var = any(c.isalpha() for c in lhs)
    has_rhs_structure = (
        any(c.isalpha() for c in rhs)
        or any(op in rhs for op in ("^", "+", "-", "*", "/", "(", "{"))
    )
    return has_lhs_var and has_rhs_structure


# ─── Canonicalisation ─────────────────────────────────────────────────────────


_BRACKET_PAIRS = {"{": "}", "[": "]", "(": ")"}
_MAX_BARE_SYMBOLS = 6

# LaTeX font-switch directives that are typesetting, not mathematics. SymPy
# would otherwise parse them as phantom symbols (rm, mathrm, bf, cal, sc, …).
_FONT_COMMANDS = (
    "rm", "it", "bf", "tt", "sc", "sf", "sl", "cal", "mit",
    "mathrm", "mathbf", "mathit", "mathsf", "mathtt", "mathbb",
    "mathcal", "mathfrak", "mathscr", "boldsymbol", "bm",
)

# Map LaTeX Greek-letter commands to single Unicode characters so SymPy's
# parser treats them as individual symbols instead of products of Latin
# letters.  Without this, ``\\eta`` becomes ``e × t × a`` and dimensional
# analysis breaks on every formula containing a Greek letter.
_GREEK_UNICODE: dict[str, str] = {
    # Lowercase
    "\\alpha": "α", "\\beta": "β", "\\gamma": "γ", "\\delta": "δ",
    "\\epsilon": "ε", "\\varepsilon": "ε", "\\zeta": "ζ", "\\eta": "η",
    "\\theta": "θ", "\\vartheta": "ϑ", "\\iota": "ι", "\\kappa": "κ",
    "\\varkappa": "ϰ", "\\lambda": "λ", "\\mu": "μ", "\\nu": "ν",
    "\\xi": "ξ", "\\pi": "π", "\\varpi": "ϖ", "\\rho": "ρ",
    "\\varrho": "ϱ", "\\sigma": "σ", "\\varsigma": "ς", "\\tau": "τ",
    "\\upsilon": "υ", "\\phi": "φ", "\\varphi": "ϕ", "\\chi": "χ",
    "\\psi": "ψ", "\\omega": "ω",
    # Uppercase
    "\\Gamma": "Γ", "\\Delta": "Δ", "\\Theta": "Θ", "\\Lambda": "Λ",
    "\\Xi": "Ξ", "\\Pi": "Π", "\\Sigma": "Σ", "\\Upsilon": "Υ",
    "\\Phi": "Φ", "\\Psi": "Ψ", "\\Omega": "Ω",
    # Common physics symbols (not Greek but multi-letter LaTeX)
    "\\hbar": "ħ", "\\ell": "ℓ", "\\partial": "∂",
}

# ── Pharmaceutical multi-letter → single-char mapping ────────────────────
# SymPy's LaTeX parser treats any multi-letter sequence in math mode as a
# product of individual letters: "CL" → C × L, "AUC" → A × U × C.  This
# breaks dimensional analysis on all pharmacokinetic / pharmacodynamic
# formulas.  The same approach used for Greek letters (Unicode remapping)
# is applied here: each common pharma abbreviation is replaced with a
# single unique character before parsing, and restored after.
#
# Mapping uses characters from the Mathematical Alphanumeric Symbols block
# (U+1D400–U+1D7FF) and other safe Unicode ranges that SymPy treats as
# single symbols.

_PHARMA_TO_UNICODE: dict[str, str] = {
    # Pharmacokinetic core symbols
    "\\mathrm{CL}": "ℂ",    # clearance — U+2102 DOUBLE-STRUCK CAPITAL C
    "CL":          "ℂ",
    "\\mathrm{AUC}": "𝔄",   # area under curve — U+1D504
    "AUC":         "𝔄",
    "\\mathrm{Vd}": "𝔙",   # volume of distribution — U+1D519
    "Vd":          "𝔙",
    "\\mathrm{Vss}": "𝕍",  # steady-state volume — U+1D54D
    "Vss":         "𝕍",
    "\\mathrm{Dose}": "𝔻", # drug dose — U+1D53B
    "Dose":        "𝔻",
    "\\mathrm{C0}": "ℭ₀",  # initial concentration
    "C0":          "ℭ₀",
    "\\mathrm{Cp}": "ℭₚ",  # plasma concentration
    "Cp":          "ℭₚ",
    "\\mathrm{Css}": "ℭₛ", # steady-state concentration
    "Css":         "ℭₛ",
    "\\mathrm{Cmax}": "ℭₓ",# peak concentration
    "Cmax":        "ℭₓ",
    "\\mathrm{Cmin}": "ℭₙ",# trough concentration
    "Cmin":        "ℭₙ",
    "\\mathrm{Cav}": "ℭₐ", # average concentration
    "Cav":         "ℭₐ",
    "\\mathrm{AUMC}": "𝔘", # area under moment curve
    "AUMC":        "𝔘",
    "\\mathrm{MRT}": "𝕄",  # mean residence time
    "MRT":         "𝕄",
    "\\mathrm{F}": "𝔽",    # bioavailability
    "F":           "𝔽",
    "\\mathrm{Ke}": "𝕂ₑ",  # elimination rate constant
    "Ke":          "𝕂ₑ",
    "\\mathrm{Ka}": "𝕂ₐ",  # absorption rate constant
    "Ka":          "𝕂ₐ",
    # Enzyme kinetics
    "\\mathrm{Vmax}": "𝕍ₘ", # maximum velocity
    "Vmax":        "𝕍ₘ",
    "\\mathrm{Km}": "𝕂ₘ",   # Michaelis constant
    "Km":          "𝕂ₘ",
    "\\mathrm{Ki}": "𝕂ᵢ",   # inhibition constant
    "Ki":          "𝕂ᵢ",
    "\\mathrm{Kd}": "𝕂ₔ",   # dissociation constant
    "Kd":          "𝕂ₔ",
    "\\mathrm{EC50}": "𝔼₅₀",# half-maximal effective concentration
    "EC50":        "𝔼₅₀",
    "\\mathrm{IC50}": "𝕀₅₀",# half-maximal inhibitory concentration
    "IC50":        "𝕀₅₀",
    "\\mathrm{ED50}": "𝔻₅₀",# median effective dose
    "ED50":        "𝔻₅₀",
    "\\mathrm{TD50}": "𝕋₅₀",# median toxic dose
    "TD50":        "𝕋₅₀",
    # Physiologically-based PK
    "\\mathrm{Qh}": "ℚₕ",   # hepatic blood flow
    "Qh":          "ℚₕ",
    "\\mathrm{CLh}": "ℂₕ",  # hepatic clearance
    "CLh":         "ℂₕ",
    "\\mathrm{CLr}": "ℂᵣ",  # renal clearance
    "CLr":         "ℂᵣ",
    "\\mathrm{CLint}": "ℂᵢ",# intrinsic clearance
    "CLint":       "ℂᵢ",
    "\\mathrm{fu}": "𝕗",    # fraction unbound
    "fu":          "𝕗",
    "\\mathrm{BW}": "𝔹𝕎",   # body weight
    "BW":          "𝔹𝕎",
    "\\mathrm{MW}": "𝕄𝕎",   # molecular weight
    "MW":          "𝕄𝕎",
    # PBPK / Tissue
    "\\mathrm{Kp}": "𝕂ₚ",   # tissue:plasma partition coeff
    "Kp":          "𝕂ₚ",
    "\\mathrm{Qt}": "ℚₜ",   # tissue blood flow
    "Qt":          "ℚₜ",
    "\\mathrm{Vt}": "𝕍ₜ",   # tissue volume
    "Vt":          "𝕍ₜ",
    "\\mathrm{Ct}": "ℭₜ",   # tissue concentration
    "Ct":          "ℭₜ",
    "\\mathrm{Ca}": "ℭₐₐ",  # arterial concentration
    "Ca":          "ℭₐₐ",
    # Permeability / Transport
    "\\mathrm{Papp}": "ℙₐ", # apparent permeability
    "Papp":        "ℙₐ",
    "\\mathrm{TI}": "𝕋𝕀",   # therapeutic index
    "TI":          "𝕋𝕀",
    # Dosing
    "\\mathrm{LD}": "𝕃𝔻",   # loading dose
    "LD":          "𝕃𝔻",
    "\\mathrm{MD}": "𝕄𝔻",   # maintenance dose
    "MD":          "𝕄𝔻",
    # GFR
    "\\mathrm{GFR}": "𝔾",   # glomerular filtration rate
    "GFR":         "𝔾",
}

# Reverse mapping for symbol name restoration after SymPy parsing.
_UNICODE_TO_PHARMA: dict[str, str] = {v: k for k, v in _PHARMA_TO_UNICODE.items()}
# Only keep the plain-text keys (not the \mathrm{} versions) for reverse mapping.
for k in list(_UNICODE_TO_PHARMA):
    if "\\mathrm" in _UNICODE_TO_PHARMA[k]:
        continue
    _UNICODE_TO_PHARMA[k] = k.replace("\\mathrm{", "").replace("}", "")


def _preprocess_latex(body: str) -> str:
    """Strip LaTeX font-switch directives (typesetting, not mathematics).

    SymPy's ``parse_latex`` reads ``\\rm``, ``\\mathrm``, ``\\mathbf`` … as
    phantom symbols (``rm``, ``mathrm``, ``bf``) that poison symbol grounding
    and the dimensional veto, so they are removed before parsing: ``\\mathrm{X}``
    becomes ``X`` and a bare ``\\rm X`` becomes ``X``.
    """
    for cmd in _FONT_COMMANDS:
        body = re.sub(rf"\\{cmd}\s*\{{([^}}]*)\}}", r"\1", body)
    for cmd in _FONT_COMMANDS:
        body = re.sub(rf"\\{cmd}\s+", "", body)
        body = re.sub(rf"\\{cmd}(?![A-Za-z])", "", body)
    # Textual labels in superscripts/subscripts (e.g. \Psi^{\rm trial}) are
    # prose, not symbols; SymPy misparses them as letter products that would
    # poison the veto. Drop multi-letter all-alpha groups.
    body = re.sub(r"\^\{[A-Za-z]+\}", "", body)
    body = re.sub(r"\_\{[A-Za-z]+\}", "", body)
    # Plain-text integer fractions ("1/2 m v^2") are read by SymPy's LaTeX
    # parser as \frac{1}{2 m v^2} — the denominator swallows the following
    # product. Rewrite "N/M" to an explicit \frac{N}{M} so "1/2 m v^2" means
    # (1/2)*m*v^2 (kinetic energy), not 1/(2 m v^2).
    body = re.sub(r"(\d+)\s*/\s*(\d+)", r"\\frac{\1}{\2}", body)
    return body


def _is_balanced(span: str) -> bool:
    """Reject spans with unmatched delimiters.

    SymPy's LaTeX grammar is permissive enough to read ``\\bad{latex`` as a
    product of single-letter symbols, so a successful parse is not proof that
    the span was mathematics. Delimiter balance is the cheapest deterministic
    filter that separates a broken macro from a real formula.
    """

    stack: list[str] = []
    escaped = False
    for char in span:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
        elif char in _BRACKET_PAIRS:
            stack.append(_BRACKET_PAIRS[char])
        elif char in _BRACKET_PAIRS.values():
            if not stack or stack.pop() != char:
                return False
    return not stack


def _is_spelled_out_prose(expr) -> bool:
    """A long product of single-letter symbols is a word, not a formula."""

    import sympy

    if not isinstance(expr, sympy.Mul):
        return False
    letters = [s for s in expr.free_symbols if len(str(s)) == 1]
    return len(letters) > _MAX_BARE_SYMBOLS and len(letters) == len(expr.free_symbols)


def _sign_normalised(expr):
    """Fix the arbitrary sign of ``lhs - rhs`` so both directions agree.

    ``E - mc²`` and ``mc² - E`` describe the same equation; without this the two
    orderings would hash differently and structural matching would be useless.
    """

    import sympy

    negated = -expr
    return min((expr, negated), key=lambda e: sympy.srepr(e))


# ─── Pharmaceutical multi-letter symbol support ──────────────────────────────
# SymPy's LaTeX parser treats any multi-letter sequence as a product of
# individual letters: "CL" → C×L, "AUC" → A×U×C.  This makes dimensional
# analysis blind to pharmacokinetic/pharmacodynamic formulas.  The fix:
# detect known pharma abbreviations and parse them through sympy.sympify()
# with pre-declared multi-letter Symbol objects.

#: Recognized pharmaceutical multi-letter abbreviation patterns.  Detected
#: by their characteristic form: 2-4 uppercase letters optionally followed by
#: digits or subscripts (CL, AUC, Vd, EC50, Cmax, t1/2, K_m, etc.).
_PHARMA_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z]{1,3}\b"
    r"|[A-Z][a-z]{2,4}\b"
    r"|[A-Z0-9][a-z]?\b"
    r")"
)

_PHARMA_KNOWN: set[str] = {
    "CL", "AUC", "Vd", "Vss", "Dose", "C0", "Cp", "Css", "Cmax", "Cmin",
    "Cav", "AUMC", "MRT", "F", "Ke", "Ka", "K", "Vmax", "Km", "Ki", "Kd",
    "EC50", "IC50", "ED50", "TD50", "Qh", "CLh", "CLr", "CLint", "fu",
    "BW", "MW", "Kp", "Qt", "Vt", "Ct", "Ca", "Papp", "TI", "LD", "MD",
    "GFR", "t_{1/2}", "C_{max}", "C_{min}", "C_{0}", "C_{ss}", "C_{av}",
    "C_{p}", "C_{t}", "C_{a}", "t_{max}", "k_{a}", "k_{e}", "K_{m}",
    "K_{i}", "K_{d}", "K_{p}", "Q_{h}", "V_{d}", "V_{ss}", "V_{t}",
}

_PHARMA_TO_SYMPY: dict[str, str] = {
    "t_{1/2}": "t_half", "C_{max}": "Cmax", "C_{min}": "Cmin",
    "C_{0}": "C0", "C_{ss}": "Css", "C_{av}": "Cav",
    "C_{p}": "Cp", "C_{t}": "Ct", "C_{a}": "Ca",
    "t_{max}": "tmax", "k_{a}": "ka", "k_{e}": "ke",
    "K_{m}": "Km", "K_{i}": "Ki", "K_{d}": "Kd",
    "K_{p}": "Kp", "Q_{h}": "Qh", "V_{d}": "Vd",
    "V_{ss}": "Vss", "V_{t}": "Vt",
    "t_half": "t_half",  # already normalized
}

# Reverse: valid Python identifier → display name
_SYMPY_TO_PHARMA: dict[str, str] = {v: k for k, v in _PHARMA_TO_SYMPY.items()}


def _normalize_pharma_name(name: str) -> str:
    """Convert a pharma symbol name to a valid Python identifier."""
    return _PHARMA_TO_SYMPY.get(name, name.replace("{", "").replace("}", "").replace("/", "_"))


def _detect_pharma_symbols(body: str) -> set[str]:
    """Return the set of known pharma abbreviations found in ``body``.

    Strips LaTeX wrappers then performs a simple substring search against
    the known pharma symbol set.  Returns symbols only when >= 2 are found
    (avoids false activation from single-letter overlaps like "C" or "K").
    """
    stripped = re.sub(r"\\mathrm\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", body)
    stripped = re.sub(r"\\operatorname\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", stripped)
    stripped = re.sub(r"\\text\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", stripped)
    stripped = stripped.replace(" ", "")

    found: set[str] = set()
    # Check each known symbol and also subscripted variants
    for sym_name in _PHARMA_KNOWN:
        if sym_name in stripped:
            found.add(sym_name)
        # Also check for LaTeX subscript forms: t_{1/2}, C_{max}, k_{a}, etc.
        if "_{" in sym_name:
            continue
        subscript_forms = [f"{sym_name}_{{1/2}}", f"{sym_name}_{{max}}",
                           f"{sym_name}_{{min}}", f"{sym_name}_{{ss}}",
                           f"{sym_name}_{{0}}", f"{sym_name}_{{av}}"]
        for sf in subscript_forms:
            if sf in stripped:
                found.add(sf.replace("_{", "_").replace("}", ""))  # normalize

    if len(found) >= 2:
        return found
    return set()


# Pre-compiled regex for stripping LaTeX font commands from pharma formulas.
# Match: \mathrm{...}, \operatorname{...}, \text{...}
_RE_MATH_RM = re.compile(r"\\mathrm\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_RE_MATH_OP = re.compile(r"\\operatorname\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_RE_MATH_TX = re.compile(r"\\text\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _parse_pharma(body: str, pharma_symbols: set[str]) -> Any | None:
    """Parse a pharma formula by direct sympy expression construction.

    Bypasses LaTeX parsing entirely — strips font commands, expands \\frac
    to (...)/(...), converts \\cdot to *, then uses sympy.sympify() with
    pre-declared multi-letter Symbol objects.
    """
    import sympy

    # Normalise the formula string
    s = body.replace("\n", " ")
    s = _RE_MATH_RM.sub(r"\1", s)
    s = _RE_MATH_OP.sub(r"\1", s)
    s = _RE_MATH_TX.sub(r"\1", s)
    s = _expand_frac(s)
    s = s.replace("\\cdot", "*")
    s = s.replace("^", "**")
    s = "".join(s.split())
    # Strip concentration brackets: [S] -> S, [I] -> I (pharma notation)
    s = s.replace("[", "").replace("]", "")

    if "=" not in s:
        return None

    lhs_str, rhs_str = s.split("=", 1)

    local: dict[str, Any] = {"sympy": sympy}
    sym_map: dict[str, str] = {}
    for name in pharma_symbols:
        if name in ("sympy", "I", "E", "N", "S", "O", "C", "Q", "D", "pi", "e"):
            continue
        py_name = _normalize_pharma_name(name)
        local[py_name] = sympy.Symbol(name)
        sym_map[name] = py_name
    for name in list(pharma_symbols):
        alt = _PHARMA_TO_SYMPY.get(name)
        if alt and alt not in local:
            local[alt] = sympy.Symbol(name)
            sym_map[name] = alt
    # Also create symbols for any remaining single-letter identifiers
    # in the expression (like S for substrate, I for inhibitor in pharma)
    extra = re.findall(r'\b([A-Za-z])\b', lhs_str + ' ' + rhs_str)
    for c in set(extra):
        if c not in local and c not in ('e',):
            local[c] = sympy.Symbol(c)

    # Replace pharma names in the expression string with valid Python identifiers
    for orig, py_name in sym_map.items():
        if orig != py_name:
            lhs_str = lhs_str.replace(orig, py_name)
            rhs_str = rhs_str.replace(orig, py_name)

    try:
        lhs = sympy.sympify(lhs_str, locals=local)
        rhs = sympy.sympify(rhs_str, locals=local)
        return lhs - rhs
    except Exception:
        return None


def _expand_frac(s: str) -> str:
    """Replace \\frac{num}{den} with (num)/(den) using bracket counting.

    Handles nested \\frac and braces.  Returns the string with all \\frac
    expanded to (...)/(...).
    """
    result: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        if s[i:i+5] == "\\frac" and i+5 < n and s[i+5] == "{":
            # Find the numerator: match brace pairs starting at i+5
            num_start = i + 6  # after "{"
            num_end = _find_matching_brace(s, num_start - 1)
            if num_end < n and num_end+1 < n and s[num_end+1] == "{":
                den_start = num_end + 2
                den_end = _find_matching_brace(s, den_start - 1)
                num_inner = s[num_start:num_end]
                den_inner = s[den_start:den_end]
                # Recursively expand nested \\frac in numerator and denominator
                result.append("(")
                result.append(_expand_frac(num_inner))
                result.append(")/(")
                result.append(_expand_frac(den_inner))
                result.append(")")
                i = den_end + 1
                continue
        result.append(s[i])
        i += 1
    return "".join(result)


def _find_matching_brace(s: str, open_pos: int) -> int:
    """Find the position of the closing brace matching the brace at open_pos."""
    depth = 0
    for i in range(open_pos, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(s)


@lru_cache(maxsize=4096)
def canonicalize(latex: str) -> Formula:
    """Parse LaTeX and reduce it to a canonical, hashable structure.

    Equations become ``lhs - rhs`` in expanded, sign-normalised form; bare
    expressions are expanded as-is. The hash is over ``srepr`` (SymPy's exact
    structural repr), not over printed output, so cosmetic printing changes
    cannot silently change identities.

    Never raises: unparsable input comes back with ``parse_error`` set.
    Cached: the ground layer calls this on every equation, and the benchmark
    runs canonicalize once directly and once via ground_source_equation — the
    duplicate call was making full-text runs 3× slower.
    """

    body = " ".join((latex or "").split())
    if not body:
        return Formula(latex=latex or "", parse_error="empty")
    if len(body) > MAX_SPAN_CHARS:
        return Formula(latex=body, parse_error="too long")
    if not _is_balanced(body):
        return Formula(latex=body, parse_error="unbalanced delimiters")

    try:
        import sympy
        from sympy.parsing.latex import parse_latex

        # Try pharma path first: multi-letter symbols (CL, AUC, Vd, etc.).
        pharma_symbols = _detect_pharma_symbols(body)
        if pharma_symbols:
            pharma_expr = _parse_pharma(body, pharma_symbols)
            if pharma_expr is not None:
                expr = pharma_expr
                is_equation = True
                expr = sympy.expand(sympy.cancel(expr))
                if _is_spelled_out_prose(expr):
                    return Formula(latex=body, parse_error="prose parsed as symbol product")
                if expr == 0:
                    return Formula(latex=body, is_equation=True, parse_error="trivial identity")
                expr = _sign_normalised(expr)
                canonical = sympy.srepr(expr)
                symbols = tuple(sorted(str(s) for s in expr.free_symbols))
                return Formula(
                    latex=body,
                    canonical=canonical,
                    symbols=symbols,
                    structure_hash=hashlib.sha1(canonical.encode("utf-8")).hexdigest(),
                    is_equation=True,
                )

        parsed = parse_latex(_preprocess_latex(body))
        is_equation = isinstance(parsed, sympy.Eq)
        expr = parsed.lhs - parsed.rhs if is_equation else parsed
        expr = sympy.expand(sympy.cancel(expr))
        if _is_spelled_out_prose(expr):
            return Formula(latex=body, parse_error="prose parsed as symbol product")
        if expr == 0 and is_equation:
            return Formula(latex=body, is_equation=True, parse_error="trivial identity")
        expr = _sign_normalised(expr)
        canonical = sympy.srepr(expr)
        symbols = tuple(sorted(str(s) for s in expr.free_symbols))
    except Exception as exc:
        logger.debug("canonicalize failed for %r: %s", body[:60], exc)
        return Formula(latex=body, parse_error=f"{type(exc).__name__}: {exc}"[:200])

    return Formula(
        latex=body,
        canonical=canonical,
        symbols=symbols,
        structure_hash=hashlib.sha1(canonical.encode("utf-8")).hexdigest(),
        is_equation=is_equation,
    )


def structural_match(left: Formula, right: Formula) -> bool:
    """True only when both sides parsed and describe the same structure."""

    return bool(left.structure_hash) and left.structure_hash == right.structure_hash


def extract_formulas(text: str, limit: int = DEFAULT_FORMULA_LIMIT) -> list[Formula]:
    """Extract and canonicalise every usable equation in ``text``.

    Spans that fail to parse are dropped rather than returned: callers want
    evidence, and a failed parse is the absence of evidence.
    """

    out: list[Formula] = []
    for span in extract_latex_spans(text):
        formula = canonicalize(span)
        if formula.is_usable:
            out.append(formula)
        if len(out) >= limit:
            break
    return out

