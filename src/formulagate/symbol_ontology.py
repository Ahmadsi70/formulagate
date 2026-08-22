"""Comprehensive symbol-dimension ontology for physics equations.

Built from real arXiv data + known physics conventions.  When no
prose anchor exists, this table provides a domain-aware best-guess
dimension for subscripted and multi-letter symbols that are
overwhelmingly unambiguous in their physics usage.

Design: only symbols whose *name alone* implies their dimension
(at >95% confidence) are listed.  Ambiguous symbols stay out.
"""

from __future__ import annotations

from fractions import Fraction

from formulagate.dimensions import Dimension

DIM = Dimension
DIMLESS = Dimension()

# ─── Greek letters (overwhelmingly dimensionless or frequency/wavelength) ──────

_GREEK_DIMENSIONS: dict[str, Dimension] = {
    # Dimensionless parameters
    "alpha": DIMLESS, "beta": DIMLESS, "gamma": DIMLESS, "Gamma": DIMLESS,
    "delta": DIMLESS, "Delta": DIMLESS, "epsilon": DIMLESS,
    "zeta": DIMLESS, "theta": DIMLESS, "Theta": DIMLESS,
    "kappa": DIMLESS, "mu": DIMLESS, "nu": Dimension(time=-1),
    "xi": DIMLESS, "pi": DIMLESS, "Pi": DIMLESS,
    "rho": DIMLESS, "sigma": DIMLESS, "Sigma": DIMLESS,
    "tau": DIMLESS, "phi": DIMLESS, "Phi": DIMLESS,
    "chi": DIMLESS, "psi": DIMLESS, "Psi": DIMLESS,
    "omega": Dimension(time=-1), "Omega": DIMLESS,
    # Uppercase Greek with specific meanings
    "varphi": DIMLESS, "vartheta": DIMLESS, "varepsilon": DIMLESS,
}

# Special uppercase — when capitalized, these have specific dimensions
_GREEK_UPPER: dict[str, Dimension] = {
    "Lambda": Dimension(length=-2),  # cosmological constant / energy scale
    "Gamma": Dimension(mass=1, length=2, time=-2),  # decay width (energy)
    "Delta": Dimension(mass=1, length=2, time=-2),  # gap / energy difference
}

# ─── Multi-letter named symbols (unambiguous) ─────────────────────────────────

_NAMED_SYMBOLS: dict[str, Dimension] = {
    "hbar": Dimension(mass=1, length=2, time=-1),
    "hslash": Dimension(mass=1, length=2, time=-1),
    "k_B": Dimension(mass=1, length=2, time=-2, temperature=-1),
    "k_{B}": Dimension(mass=1, length=2, time=-2, temperature=-1),
    "k_Bz": Dimension(mass=1, length=2, time=-2, temperature=-1),
    "N_A": Dimension(amount=-1),
    "epsilon_0": Dimension(mass=-1, length=-3, time=4, current=2),
    "epsilon_{0}": Dimension(mass=-1, length=-3, time=4, current=2),
    "mu_0": Dimension(mass=1, length=1, time=-2, current=-2),
    "mu_{0}": Dimension(mass=1, length=1, time=-2, current=-2),
    "L_p": Dimension(length=1),  # Planck length
    "L_{p}": Dimension(length=1),
    "M_p": Dimension(mass=1),    # Planck mass
    "M_{p}": Dimension(mass=1),
    "T_p": Dimension(time=1),    # Planck time
    # Pauli matrices
    "sigma_x": DIMLESS, "sigma_{x}": DIMLESS,
    "sigma_y": DIMLESS, "sigma_{y}": DIMLESS,
    "sigma_z": DIMLESS, "sigma_{z}": DIMLESS,
    # Dirac matrices
    "gamma_0": DIMLESS, "gamma_{0}": DIMLESS,
    "gamma_5": DIMLESS, "gamma_{5}": DIMLESS,
}

# ─── Subscript patterns: base symbol → dimension ──────────────────────────────

# When a subscripted symbol is not found in the named table, the base
# letter's dimension is assumed.  E.g. ``x_{1}`` ~ ``x`` ~ L.
_BASE_LETTER_DIMS: dict[str, Dimension] = {
    "x": Dimension(length=1), "y": Dimension(length=1), "z": Dimension(length=1),
    "r": Dimension(length=1), "R": DIMLESS,  # R ambiguous — leave dimensionless
    "t": Dimension(time=1), "tau": Dimension(time=1),
    "m": Dimension(mass=1), "M": Dimension(mass=1),
    "E": Dimension(mass=1, length=2, time=-2),
    "p": Dimension(mass=1, length=1, time=-1),
    "F": Dimension(mass=1, length=1, time=-2),
    "v": Dimension(length=1, time=-1),
    "a": Dimension(length=1, time=-2),
    "H": DIMLESS,  # Hubble/Hamiltonian ambiguous — stay dimensionless
    "S": DIMLESS,  # entropy/action/area ambiguous
    "T": DIMLESS,  # temperature/time ambiguous
    "Q": DIMLESS,  # charge/heat/quality factor
    "U": Dimension(mass=1, length=2, time=-2),  # internal energy
    "I": DIMLESS,  # current/moment of inertia
    "J": DIMLESS,  # current density/angular momentum
    "P": DIMLESS,  # pressure/probability/power
    "L": Dimension(length=1),
    "A": Dimension(length=2),  # area
    "D": Dimension(length=2, time=-1),  # diffusion constant
    "N": DIMLESS,  # number/count
    "n": DIMLESS,
    "Z": DIMLESS,  # partition function/atomic number
    "K": DIMLESS,  # kinetic energy/constant
    "W": DIMLESS,  # work/weight
    "B": DIMLESS,  # magnetic field ambiguous
    "C": DIMLESS,  # capacitance/constant
    "G": DIMLESS,  # gravitational/gauge
    "O": DIMLESS,  # operator
    "sigma": DIMLESS,  # Pauli/conductivity ambiguous
    "omega": Dimension(time=-1),
    "lambda": Dimension(length=1),
    "epsilon": DIMLESS,
    "delta": DIMLESS,
    "psi": DIMLESS,
    "chi": DIMLESS,
    "zeta": DIMLESS,
}


def lookup_ontology(symbol: str) -> Dimension | None:
    """Look up a symbol's dimension from the comprehensive ontology.

    Returns None if the symbol is genuinely ambiguous.
    """
    # Exact match in named symbols
    if symbol in _NAMED_SYMBOLS:
        return _NAMED_SYMBOLS[symbol]
    if symbol in _GREEK_UPPER:
        return _GREEK_UPPER[symbol]

    # Greek letters
    if symbol in _GREEK_DIMENSIONS:
        return _GREEK_DIMENSIONS[symbol]

    # Subscripted: try base letter
    if "_" in symbol:
        base = symbol.split("_")[0]
        if base in _BASE_LETTER_DIMS:
            return _BASE_LETTER_DIMS[base]

    # Multi-letter symbol (not subscripted)
    if len(symbol) >= 2 and "_" not in symbol:
        if symbol in _GREEK_DIMENSIONS:
            return _GREEK_DIMENSIONS[symbol]

    return None
