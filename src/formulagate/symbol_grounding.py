"""Context-aware symbol grounding: map equation symbols to physical quantities.

Why this exists:
  Dimensional analysis fails on most real equations because the symbol table
  does not know what a symbol means (``E`` is energy, but also electric field,
  and ``S`` is entropy, action, or area). Real LaTeX sources resolve this: the
  prose around an equation almost always *says* what each symbol is —

      "... where $E$ is the total energy of the system ..."
      "... with $S$ the entropy ..."

  This module reads that real prose and grounds symbols to physical quantities
  (dimensions) before the dimensional layer runs. It is deterministic: a
  symbol whose meaning is not stated is left unknown, never guessed.

  Three resolution layers, cheapest first:

    1. ``SYMBOL_DIMENSIONS`` — the curated table (already in ``dimensions``).
    2. Grounding patterns — "X is the energy", "X denotes ...", "X = ... energy".
    3. ``CONSTANT_NAMES`` — symbols bound to fundamental constants.

  Output is a ``SymbolGrounding`` map from symbol name to
  :class:`formulagate.dimensions.Dimension`, empty where nothing could be
  grounded. Nothing here invents symbols or equations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Mapping

from formulagate.dimensions import Dimension, DIMENSIONLESS, canonical_symbol_name
from formulagate.tex_ingest import _BARE_SYMBOL

# Phrase that names the quantity right after a symbol: "X is the energy",
# "X denotes the electric field", "X is the wave function". The phrase is
# stopped at a punctuation comma/semicolon (which separates sibling
# definitions: "R is the Ricci scalar, G is the Gravitational constant") or at
# a new definition verb ("and L_p is the Planck length"), never at a bare
# "and" that continues the phrase ("constant and of order unity").
_DEFINITION_VERB = r"(?:is|denotes|stands for|represents|means|being)"
_CONNECTIVE = r"(?:\s+(?:where|with|here|for|denotes|being)\b|[,.;:]|" \
              r"\s+(?:and|but)\s+[A-Za-z][A-Za-z0-9_]*\s+" + _DEFINITION_VERB + r")"
_PHRASE_CAPTURE = r"([a-zA-Z][a-zA-Z _-]*?)"
_GROUNDING = re.compile(
    r"(?:where|with|and|here|for)\s+"
    r"\$?\s*([A-Za-z][A-Za-z0-9_]*)\s*\$?\s+"
    r"(?:is|denotes|stands for|represents|means|being)\s+"
    r"the?\s+" + _PHRASE_CAPTURE + r"(?=" + _CONNECTIVE + r"|$)",
    re.IGNORECASE,
)
# "where X is the ..." variant where the article is absent.
_GROUNDING2 = re.compile(
    r"(?:where|with|here)\s+\$?\s*([A-Za-z][A-Za-z0-9_]*)\s*\$?\s+"
    r"is\s+(?:the\s+)?" + _PHRASE_CAPTURE + r"(?=" + _CONNECTIVE + r"|$)",
    re.IGNORECASE,
)
# Sibling definitions after a separator: ", G is the Gravitational constant"
# or "; X is the entropy". The where/with/and prefix is absent, so this pass
# fires on any "X is the <quantity>" that follows a comma, semicolon, or period.
_GROUNDING3 = re.compile(
    r"[,;.]\s+\$?\s*([A-Za-z][A-Za-z0-9_]*)\s*\$?\s+"
    r"is\s+(?:the\s+)?" + _PHRASE_CAPTURE + r"(?=" + _CONNECTIVE + r"|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GroundedSymbol:
    """One symbol resolved to a physical quantity (and its dimension)."""

    symbol: str
    quantity: str
    dimension: Dimension | None
    source: str  # "table" | "context" | "constant"


@dataclass(frozen=True)
class SymbolGrounding:
    """Resolution of every symbol in one equation."""

    equation: str
    symbols: tuple[str, ...] = ()
    grounded: tuple[GroundedSymbol, ...] = ()
    unknown: tuple[str, ...] = ()

    @property
    def coverage(self) -> float:
        total = len(self.symbols)
        if not total:
            return 0.0
        return len(self.grounded) / total

    def as_overrides(self) -> dict[str, Dimension]:
        """Dimensions to pass as ``overrides`` to ``check_dimensions``.

        Deliberately conservative — the core design rule is that a false reject
        costs more than an abstention. Only symbols whose meaning is *stated*
        by the prose (source == "context") or fixed by a named constant are
        emitted. The conservative ``SYMBOL_DIMENSIONS`` table in
        :mod:`formulagate.dimensions` already handles the safe standard letters;
        nothing ambiguous is guessed here, so the veto never rejects real
        physics on a guessed dimension.
        """
        return {
            canonical_symbol_name(g.symbol): g.dimension
            for g in self.grounded
            if g.dimension is not None
            and (g.source == "context" or g.source == "constant")
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "equation": self.equation,
            "symbols": list(self.symbols),
            "grounded": [
                {
                    "symbol": g.symbol,
                    "quantity": g.quantity,
                    "dimension": str(g.dimension) if g.dimension else None,
                    "source": g.source,
                }
                for g in self.grounded
            ],
            "unknown": list(self.unknown),
            "coverage": self.coverage,
        }


# ─── Physical quantity → dimension lexicon (curated, not generated) ──────────

def _D(**kw: int | float) -> Dimension:
    return Dimension(
        mass=Fraction(kw.get("mass", 0)),
        length=Fraction(kw.get("length", 0)),
        time=Fraction(kw.get("time", 0)),
        current=Fraction(kw.get("current", 0)),
        temperature=Fraction(kw.get("temperature", 0)),
        amount=Fraction(kw.get("amount", 0)),
        luminosity=Fraction(kw.get("luminosity", 0)),
    )


# Every entry is a real SI dimension of a named physical quantity.
QUANTITY_DIMENSIONS: dict[str, Dimension] = {
    # mechanics
    "mass": _D(mass=1), "energy": _D(mass=1, length=2, time=-2),
    "kinetic energy": _D(mass=1, length=2, time=-2),
    "potential energy": _D(mass=1, length=2, time=-2),
    "total energy": _D(mass=1, length=2, time=-2),
    "rest energy": _D(mass=1, length=2, time=-2),
    "rest mass": _D(mass=1),
    "momentum": _D(mass=1, length=1, time=-1),
    "velocity": _D(length=1, time=-1), "speed": _D(length=1, time=-1),
    "acceleration": _D(length=1, time=-2), "force": _D(mass=1, length=1, time=-2),
    "torque": _D(mass=1, length=2, time=-2), "length": _D(length=1),
    "distance": _D(length=1), "position": _D(length=1), "radius": _D(length=1),
    "width": _D(length=1), "height": _D(length=1), "depth": _D(length=1),
    "wavelength": _D(length=1), "time": _D(time=1), "period": _D(time=1),
    "lifetime": _D(time=1), "duration": _D(time=1), "area": _D(length=2),
    "cross section": _D(length=2), "cross-section": _D(length=2),
    "volume": _D(length=3), "density": _D(mass=1, length=-3),
    "mass density": _D(mass=1, length=-3),
    "frequency": _D(time=-1), "angular frequency": _D(time=-1),
    "power": _D(mass=1, length=2, time=-3),
    "pressure": _D(mass=1, length=-1, time=-2),
    "stress": _D(mass=1, length=-1, time=-2),
    "work": _D(mass=1, length=2, time=-2),
    "action": _D(mass=1, length=2, time=-1),
    "angular momentum": _D(mass=1, length=2, time=-1),
    "moment of inertia": _D(mass=1, length=2),
    "impulse": _D(mass=1, length=1, time=-1),
    "strain": DIMENSIONLESS, "stress tensor": _D(mass=1, length=-1, time=-2),
    # electromagnetism
    "charge": _D(current=1, time=1), "electric charge": _D(current=1, time=1),
    "current": _D(current=1), "electric current": _D(current=1),
    "voltage": _D(mass=1, length=2, time=-3, current=-1),
    "potential": _D(mass=1, length=2, time=-3, current=-1),
    "electric potential": _D(mass=1, length=2, time=-3, current=-1),
    "electromotive force": _D(mass=1, length=2, time=-3, current=-1),
    "resistance": _D(mass=1, length=2, time=-3, current=-2),
    "capacitance": _D(mass=-1, length=-2, time=4, current=2),
    "inductance": _D(mass=1, length=2, time=-2, current=-2),
    "magnetic field": _D(mass=1, time=-2, current=-1),
    "electric field": _D(mass=1, length=1, time=-3, current=-1),
    "magnetic flux": _D(mass=1, length=2, time=-2, current=-1),
    "conductivity": _D(mass=-1, length=-3, time=3, current=2),
    "resistivity": _D(mass=1, length=3, time=-3, current=-2),
    "permittivity": _D(mass=-1, length=-3, time=4, current=2),
    "permeability": _D(mass=1, length=1, time=-2, current=-2),
    "current density": _D(length=-2, current=1),
    "charge density": _D(length=-3, time=1, current=1),
    "polarization": _D(length=-2, time=1, current=1),
    "magnetization": _D(length=-1, current=1),
    "energy density": _D(mass=1, length=-1, time=-2),
    "power density": _D(mass=1, time=-3),
    # thermodynamics
    "temperature": _D(temperature=1), "entropy": _D(mass=1, length=2, time=-2, temperature=-1),
    "heat capacity": _D(mass=1, length=2, time=-2, temperature=-1),
    "specific heat": _D(length=2, time=-2, temperature=-1),
    "heat": _D(mass=1, length=2, time=-2),
    "enthalpy": _D(mass=1, length=2, time=-2),
    "free energy": _D(mass=1, length=2, time=-2),
    "internal energy": _D(mass=1, length=2, time=-2),
    "chemical potential": _D(mass=1, length=2, time=-2),
    "thermal conductivity": _D(mass=1, length=1, time=-3, temperature=-1),
    "viscosity": _D(mass=1, length=-1, time=-1),
    "surface tension": _D(mass=1, time=-2),
    "diffusivity": _D(length=2, time=-1),
    "thermodynamic temperature": _D(temperature=1),
    # quantum / relativity
    "wavefunction": DIMENSIONLESS, "wave function": DIMENSIONLESS,
    "hamiltonian": _D(mass=1, length=2, time=-2),
    "planck constant": _D(mass=1, length=2, time=-1),
    "reduced planck constant": _D(mass=1, length=2, time=-1),
    "planck length": _D(length=1), "planck mass": _D(mass=1),
    "planck time": _D(time=1), "planck energy": _D(mass=1, length=2, time=-2),
    "planck temperature": _D(temperature=1),
    "boltzmann constant": _D(mass=1, length=2, time=-2, temperature=-1),
    "gravitational constant": _D(mass=-1, length=3, time=-2),
    "speed of light": _D(length=1, time=-1),
    "lorentz factor": DIMENSIONLESS,
    "decay rate": _D(time=-1), "decay constant": _D(time=-1),
    "width": _D(mass=1, length=2, time=-2),  # particle decay width (energy)
    "cross-section": _D(length=2),
    "flux": _D(time=-1, length=-2), "particle flux": _D(time=-1, length=-2),
    "number density": _D(length=-3),
    "number": DIMENSIONLESS,  # a count ("number of free parameters", "number of events")
    "curvature": _D(length=-2), "ricci scalar": _D(length=-2),
    "cosmological constant": _D(length=-2),
    # fluids
    "reynolds number": DIMENSIONLESS, "mach number": DIMENSIONLESS,
    "prandtl number": DIMENSIONLESS, "nusselt number": DIMENSIONLESS,
    "volume flow": _D(length=3, time=-1), "flow rate": _D(length=3, time=-1),
    # optics
    "intensity": _D(mass=1, time=-3), "irradiance": _D(mass=1, time=-3),
    "refractive index": DIMENSIONLESS,
    "focal length": _D(length=1), "focal distance": _D(length=1),
    # dimensionless
    "probability": DIMENSIONLESS, "angle": DIMENSIONLESS, "solid angle": DIMENSIONLESS,
    "coupling constant": DIMENSIONLESS, "fine structure constant": DIMENSIONLESS,
    "phase": DIMENSIONLESS, "phase shift": DIMENSIONLESS,
    "efficiency": DIMENSIONLESS, "fraction": DIMENSIONLESS, "ratio": DIMENSIONLESS,
    "amplitude": DIMENSIONLESS, "coefficient": DIMENSIONLESS, "parameter": DIMENSIONLESS,
    "index": DIMENSIONLESS, "quantum number": DIMENSIONLESS,
    "spin": DIMENSIONLESS, "charge number": DIMENSIONLESS,
    "dimensionless": DIMENSIONLESS,
    # ── Pharmaceutical / Pharmacokinetic quantities ──────────────────
    # Clearance: volume of plasma cleared per unit time.
    "clearance": _D(length=3, time=-1),
    "plasma clearance": _D(length=3, time=-1),
    "hepatic clearance": _D(length=3, time=-1),
    "renal clearance": _D(length=3, time=-1),
    "total clearance": _D(length=3, time=-1),
    "oral clearance": _D(length=3, time=-1),
    "intrinsic clearance": _D(length=3, time=-1),
    # Volume of distribution: apparent volume.
    "volume of distribution": _D(length=3),
    "apparent volume": _D(length=3),
    "volume": _D(length=3),
    # Half-life: time.
    "half life": _D(time=1),
    "elimination half life": _D(time=1),
    "terminal half life": _D(time=1),
    "half-life": _D(time=1),
    # Area Under Curve: concentration × time.
    "area under curve": _D(mass=1, length=-3, time=1),
    "area under the curve": _D(mass=1, length=-3, time=1),
    "auc": _D(mass=1, length=-3, time=1),
    # AUMC: concentration × time².
    "area under moment curve": _D(mass=1, length=-3, time=2),
    "aumc": _D(mass=1, length=-3, time=2),
    # Mean Residence Time.
    "mean residence time": _D(time=1),
    "mrt": _D(time=1),
    # Concentration (mass/volume).
    "plasma concentration": _D(mass=1, length=-3),
    "drug concentration": _D(mass=1, length=-3),
    "steady state concentration": _D(mass=1, length=-3),
    "peak concentration": _D(mass=1, length=-3),
    "trough concentration": _D(mass=1, length=-3),
    "initial concentration": _D(mass=1, length=-3),
    # Dose: mass of drug.
    "dose": _D(mass=1),
    "loading dose": _D(mass=1),
    "maintenance dose": _D(mass=1),
    "oral dose": _D(mass=1),
    "intravenous dose": _D(mass=1),
    # Bioavailability: dimensionless fraction.
    "bioavailability": DIMENSIONLESS,
    "absolute bioavailability": DIMENSIONLESS,
    "relative bioavailability": DIMENSIONLESS,
    "oral bioavailability": DIMENSIONLESS,
    # Extraction ratio: dimensionless.
    "extraction ratio": DIMENSIONLESS,
    "hepatic extraction": DIMENSIONLESS,
    # Rate of dissolution.
    "dissolution rate": _D(mass=1, time=-1),
    "dissolution": _D(mass=1, time=-1),
    # Permeability.
    "permeability": _D(length=1, time=-1),
    "apparent permeability": _D(length=1, time=-1),
    # Enzyme kinetics.
    "reaction rate": _D(mass=1, length=-3, time=-1),
    "maximum velocity": _D(mass=1, length=-3, time=-1),
    "vmax": _D(mass=1, length=-3, time=-1),
    "enzyme velocity": _D(mass=1, length=-3, time=-1),
    "michaelis constant": _D(mass=1, length=-3),
    "km": _D(mass=1, length=-3),
    "inhibition constant": _D(mass=1, length=-3),
    "ki": _D(mass=1, length=-3),
    "ki_value": _D(mass=1, length=-3),
    "dissociation constant": _D(mass=1, length=-3),
    "kd": _D(mass=1, length=-3),
    "half maximal concentration": _D(mass=1, length=-3),
    "ec50": _D(mass=1, length=-3),
    "ic50": _D(mass=1, length=-3),
    "inhibitory concentration": _D(mass=1, length=-3),
    # Diffusion coefficient.
    "diffusion coefficient": _D(length=2, time=-1),
    "diffusion": _D(length=2, time=-1),
    # Partition coefficient: dimensionless.
    "partition coefficient": DIMENSIONLESS,
    "log p": DIMENSIONLESS,
    "log d": DIMENSIONLESS,
    "octanol water coefficient": DIMENSIONLESS,
    "lipophilicity": DIMENSIONLESS,
    # Absorption rate.
    "absorption rate": _D(time=-1),
    "absorption constant": _D(time=-1),
    "ka": _D(time=-1),
    "elimination rate": _D(time=-1),
    "elimination constant": _D(time=-1),
    "ke": _D(time=-1),
    # Blood flow.
    "blood flow": _D(length=3, time=-1),
    "hepatic flow": _D(length=3, time=-1),
    "flow rate": _D(length=3, time=-1),
    "qh": _D(length=3, time=-1),
    # Protein binding: dimensionless.
    "fraction unbound": DIMENSIONLESS,
    "protein binding": DIMENSIONLESS,
    "fu": DIMENSIONLESS,
    # pH/pKa: dimensionless.
    "ph": DIMENSIONLESS,
    "pka": DIMENSIONLESS,
    "acid dissociation": DIMENSIONLESS,
    # Flux: mass per area per time.
    "flux": _D(mass=1, length=-2, time=-1),
    "membrane flux": _D(mass=1, length=-2, time=-1),
    # Solubility.
    "solubility": _D(mass=1, length=-3),
    "saturated solubility": _D(mass=1, length=-3),
    # Therapeutic index.
    "therapeutic index": DIMENSIONLESS,
    "therapeutic window": _D(mass=1, length=-3),
    # Toxic dose / effective dose.
    "toxic dose": _D(mass=1),
    "effective dose": _D(mass=1),
    "td50": _D(mass=1),
    "ed50": _D(mass=1),
    # Accumulation ratio: dimensionless.
    "accumulation ratio": DIMENSIONLESS,
    "accumulation index": DIMENSIONLESS,
    # Membrane thickness.
    "membrane thickness": _D(length=1),
    "diffusion layer thickness": _D(length=1),
    # Surface area.
    "surface area": _D(length=2),
    "membrane area": _D(length=2),
    "effective area": _D(length=2),
    # Allometric scaling: dimensionless exponent.
    "allometric exponent": DIMENSIONLESS,
    "body weight": _D(mass=1),
}

# Fundamental-constant aliases resolvable by name.
CONSTANT_NAMES: dict[str, Dimension] = {
    "speed of light": _D(length=1, time=-1),
    "speedof light": _D(length=1, time=-1),
    "planck constant": _D(mass=1, length=2, time=-1),
    "reduced planck constant": _D(mass=1, length=2, time=-1),
    "boltzmann constant": _D(mass=1, length=2, time=-2, temperature=-1),
    "gravitational constant": _D(mass=-1, length=3, time=-2),
    "elementary charge": _D(current=1, time=1),
    "avogadro number": _D(amount=-1),
    "permittivity of free space": _D(mass=-1, length=-3, time=4, current=2),
    "permeability of free space": _D(mass=1, length=1, time=-2, current=-2),
}

# Constants by their canonical free-symbol spelling (SymPy braces subscripts).
# Layer 3 matches a symbol against ``name`` with the braces intact, so the
# canonical forms must be registered alongside the name aliases above.
CONSTANT_NAMES.update(
    {
        "epsilon_{0}": _D(mass=-1, length=-3, time=4, current=2),
        "mu_{0}": _D(mass=1, length=1, time=-2, current=-2),
        "k_{B}": _D(mass=1, length=2, time=-2, temperature=-1),
        "N_{A}": _D(amount=-1),
        "hbar": _D(mass=1, length=2, time=-1),
    }
)

# Symbols conventionally bound to a dimension even before prose says so.
# Only unambiguous, standard physics usage — overloaded symbols are excluded.
TABLE_FALLBACK: dict[str, Dimension] = {
    "c": _D(length=1, time=-1),
    "hbar": _D(mass=1, length=2, time=-1), "ℏ": _D(mass=1, length=2, time=-1),
    "ħ": _D(mass=1, length=2, time=-1),
    "k_B": _D(mass=1, length=2, time=-2, temperature=-1),
    "k_Bz": _D(mass=1, length=2, time=-2, temperature=-1),
    "m": _D(mass=1), "M": _D(mass=1),
    "v": _D(length=1, time=-1), "u": _D(length=1, time=-1),
    "a": _D(length=1, time=-2),
    "t": _D(time=1), "x": _D(length=1), "y": _D(length=1), "z": _D(length=1),
    "r": _D(length=1), "l": _D(length=1), "L": _D(length=1),
    "d": _D(length=1), "lambda": _D(length=1),
    "λ": _D(length=1), "omega": _D(time=-1), "ω": _D(time=-1),
    "nu": _D(time=-1), "ν": _D(time=-1), "f": _D(time=-1),
    "F": _D(mass=1, length=1, time=-2), "E": _D(mass=1, length=2, time=-2),
    "p": _D(mass=1, length=1, time=-1),
    "n": DIMENSIONLESS, "N": DIMENSIONLESS,
}

# Same entries under their brace-canonical spellings: SymPy renders ``epsilon_0``
# as the free symbol ``epsilon_{0}``, and ``dimension_of_symbol`` is not the
# only lookup path here (Layer 3 iterates this map directly). Without these the
# constants never anchored and grounded coverage stayed low.
_TABLE_FALLBACK_BRACED: dict[str, Dimension] = {
    "k_{B}": _D(mass=1, length=2, time=-2, temperature=-1),
    "k_{Bz}": _D(mass=1, length=2, time=-2, temperature=-1),
    "epsilon_{0}": _D(mass=-1, length=-3, time=4, current=2),
    "mu_{0}": _D(mass=1, length=1, time=-2, current=-2),
}
# Pharmaceutical symbol abbreviations — common multi-letter PK/PD variables.
PHARMA_CONSTANTS: dict[str, Dimension] = {
    "CL": _D(length=3, time=-1),     # clearance
    "CLh": _D(length=3, time=-1),    # hepatic clearance
    "CLr": _D(length=3, time=-1),    # renal clearance
    "CLint": _D(length=3, time=-1),  # intrinsic clearance
    "Vd": _D(length=3),              # volume of distribution
    "Vss": _D(length=3),             # steady-state volume
    "AUC": _D(mass=1, length=-3, time=1),
    "AUMC": _D(mass=1, length=-3, time=2),
    "MRT": _D(time=1),
    "Cmax": _D(mass=1, length=-3),
    "Cmin": _D(mass=1, length=-3),
    "Cav": _D(mass=1, length=-3),
    "Css": _D(mass=1, length=-3),
    "C0": _D(mass=1, length=-3),
    "Cp": _D(mass=1, length=-3),
    "F": DIMENSIONLESS,              # bioavailability
    "EC50": _D(mass=1, length=-3),
    "ED50": _D(mass=1),
    "IC50": _D(mass=1, length=-3),
    "Ki": _D(mass=1, length=-3),
    "Km": _D(mass=1, length=-3),
    "Kd": _D(mass=1, length=-3),
    "Vmax": _D(mass=1, length=-3, time=-1),
    "Ke": _D(time=-1),               # elimination constant
    "Ka": _D(time=-1),               # absorption constant
    "k": _D(time=-1),                # elimination rate
    "fu": DIMENSIONLESS,             # fraction unbound
    "fe": DIMENSIONLESS,             # fraction excreted
    "Qh": _D(length=3, time=-1),     # hepatic blood flow
    "Q": _D(length=3, time=-1),      # blood flow
    "BW": _D(mass=1),                # body weight
    "MW": _D(mass=1),                # molecular weight
    "C": _D(mass=1, length=-3),      # concentration
    "Cs": _D(mass=1, length=-3),     # saturated concentration
    "Papp": _D(length=1, time=-1),   # apparent permeability
    "TI": DIMENSIONLESS,             # therapeutic index
}
CONSTANT_NAMES.update(PHARMA_CONSTANTS)
TABLE_FALLBACK.update(PHARMA_CONSTANTS)

# Normalise description phrases so "electric field" and "the electric field"
# resolve to the same entry.
_PHRASE_CLEAN = re.compile(r"[^a-z ]+")
_QUANTITY_LOOKUP = {k: v for k, v in QUANTITY_DIMENSIONS.items()}
# Index by head word so "total energy" matches "energy" — but only words that
# are *unambiguous* (appear in exactly one phrase). "planck" belongs to
# "planck constant" (action) yet "Planck length" must resolve to a length;
# "constant" appears in eight phrases with different dimensions. A word shared
# by multiple phrases with different dimensions must never be resolved by its
# head alone, or prose grounding invents anchors with the wrong dimensions.
_HEAD_LOOKUP: dict[str, Dimension] = {}
_HEAD_DIMENSIONS: dict[str, set[str]] = {}
for _phrase, _dim in QUANTITY_DIMENSIONS.items():
    for _word in _phrase.split():
        _HEAD_LOOKUP.setdefault(_word, _dim)
        _HEAD_DIMENSIONS.setdefault(_word, set()).add(str(_dim))
for _word, _dims in _HEAD_DIMENSIONS.items():
    if len(_dims) > 1:
        _HEAD_LOOKUP.pop(_word, None)


def _resolve_phrase(phrase: str) -> Dimension | None:
    """Map a quantity phrase to a dimension, or None."""
    key = _PHRASE_CLEAN.sub("", phrase.lower().strip())
    if not key:
        return None
    hit = _QUANTITY_LOOKUP.get(key)
    if hit is not None:
        return hit
    hit = CONSTANT_NAMES.get(key)
    if hit is not None:
        return hit
    # Longest-prefix match on multi-word phrases.
    words = key.split()
    for n in range(len(words), 0, -1):
        head = " ".join(words[:n])
        if head in _QUANTITY_LOOKUP:
            return _QUANTITY_LOOKUP[head]
    # Single word match.
    for word in words:
        if word in _HEAD_LOOKUP:
            return _HEAD_LOOKUP[word]
    return None


def ground_symbols(
    equation: str,
    *,
    context_before: str = "",
    context_after: str = "",
    symbols: tuple[str, ...] | None = None,
    doc_defs: dict[str, tuple[str, Dimension]] | None = None,
    domain_table: dict[str, Dimension] | None = None,
) -> SymbolGrounding:
    """Resolve the symbols of one equation using context and the curated table.

    Args:
        equation: The equation LaTeX.
        context_before/context_after: Prose around the equation (real text).
        symbols: Optional precomputed symbol set; computed from ``equation``
            when omitted (using the canonicalisation symbol extractor).
        doc_defs: document-wide symbol definitions (Layer 0 — definition
            propagation from other equations in the same paper).
        domain_table: domain-specific symbol overrides (Layer 0.5 — more
            aggressive than the generic table because the paper's field is
            known, but still a guess, not a certain anchor).

    Returns:
        A ``SymbolGrounding`` where every resolvable symbol is mapped.
    """
    from formulagate.formula_extract import canonicalize
    from formulagate.dimensions import canonical_symbol_name

    # Symbol names come in two spellings that SymPy treats as *different*
    # objects: the tex_ingest extractor emits raw LaTeX ("E_n"), while
    # canonicalize() renders subscripts with braces ("E_{n}"). Grounding must
    # key on the canonical spelling, or overrides never reach the free symbols
    # the SMT solver sees — every anchored equation silently became unknown.
    canonical = canonicalize(equation)
    canonical_symbols = canonical.symbols if canonical.is_usable else ()

    def canonical_key(raw: str) -> str:
        if raw in canonical_symbols:
            return raw
        key = canonical_symbol_name(raw)
        for cs in canonical_symbols:
            if canonical_symbol_name(cs) == key:
                return cs
        return raw

    if symbols is None:
        symbols = canonical_symbols if canonical.is_usable else ()
        if not symbols:
            symbols = tuple(dict.fromkeys(_BARE_SYMBOL.findall(equation)))

    text = " ".join([context_before or "", context_after or ""])
    grounded: dict[str, tuple[str, Dimension | None, str]] = {}

    # Layer 0: document-wide definitions (propagated from other equations).
    if doc_defs:
        for sym in symbols:
            key = canonical_key(sym)
            canon = canonical_symbol_name(key)
            if canon in doc_defs:
                quantity, dim = doc_defs[canon]
                grounded[key] = (quantity, dim, "context")

    # Layer 0.5: domain-specific table.  More aggressive than the generic table
    # because we know the paper's field, but still a guess — source="domain"
    # keeps it out of the anchor set.
    if domain_table:
        for sym in symbols:
            key = canonical_key(sym)
            if key in grounded:
                continue
            canon = canonical_symbol_name(key)
            if canon in domain_table:
                grounded[key] = (canon, domain_table[canon], "domain")

    # Layer 0.7: symbol-name ontology.  Symbols whose *name alone* implies a
    # specific dimension (>95% confidence across arXiv data).  Stronger than
    # the generic table but weaker than prose anchors — source="ontology".
    from formulagate.symbol_ontology import lookup_ontology
    for sym in symbols:
        key = canonical_key(sym)
        if key in grounded:
            continue
        dim = lookup_ontology(key)
        if dim is not None:
            grounded[key] = (key, dim, "ontology")

    # Layer 1: table.
    for sym in symbols:
        from formulagate.dimensions import dimension_of_symbol

        key = canonical_key(sym)
        if key in grounded:
            continue  # Layer 0 already pinned this symbol
        dim = dimension_of_symbol(key)
        if dim is not None:
            grounded[key] = (sym, dim, "table")
            continue
        dim = TABLE_FALLBACK.get(canonical_symbol_name(key))
        if dim is not None:
            grounded[key] = (sym, dim, "table")

    # Layer 2: prose grounding — "X is the energy".
    # Blocklist: common English words that are not physics symbols.
    _PROSE_BLOCKLIST = frozenset({
        "this", "that", "these", "those", "it", "its", "they", "them", "their",
        "which", "what", "where", "when", "how", "why", "who", "whom", "whose",
        "there", "here", "now", "then", "thus", "hence", "also", "too", "very",
        "just", "only", "even", "still", "yet", "not", "no", "nor", "but", "and",
        "for", "with", "from", "into", "onto", "upon", "about", "over", "under",
    })
    for pattern in (_GROUNDING, _GROUNDING2, _GROUNDING3):
        for match in pattern.finditer(text):
            sym = match.group(1)
            if sym.lower() in _PROSE_BLOCKLIST:
                continue  # skip common English words
            phrase = match.group(2).strip()
            dim = _resolve_phrase(phrase)
            if dim is None:
                continue
            key = canonical_key(sym)
            # Prose is the strongest evidence: it states the symbol's meaning,
            # so it upgrades (or pins) the entry as context even when the table
            # already guessed a value. Without this, prose-stated constants
            # (e.g. "where c is the speed of light") never count as anchors.
            grounded[key] = (phrase, dim, "context")
            break

    # Layer 3: names of fundamental constants. A symbol only matches when it is
    # a *whole word* of the constant's name (or the name itself, compacted) —
    # "nt" must never match the "nt" inside "planck constant", nor "rm" the "rm"
    # inside "permittivity of free space". Substring matching fabricated 959
    # constant groundings that made the SMT veto reject real equations.
    for sym in list(symbols):
        key = canonical_key(sym)
        if key in grounded or len(sym) < 2:
            continue
        for name, dim in CONSTANT_NAMES.items():
            compact = name.replace(" ", "")
            words = name.lower().split()
            if sym == compact or any(sym.lower() == w for w in words):
                grounded[key] = (name, dim, "constant")
                break

    unknown = tuple(s for s in symbols if canonical_key(s) not in grounded)
    grounded_list = tuple(
        GroundedSymbol(symbol=s, quantity=q, dimension=d, source=src)
        for s, (q, d, src) in grounded.items()
    )
    return SymbolGrounding(
        equation=equation,
        symbols=tuple(symbols),
        grounded=grounded_list,
        unknown=unknown,
    )


def ground_source_equation(
    equation,
    *,
    context_before: str | None = None,
    context_after: str | None = None,
    doc_defs: dict[str, tuple[str, Dimension]] | None = None,
    domain_table: dict[str, Dimension] | None = None,
) -> SymbolGrounding:
    """Convenience: ground a :class:`tex_ingest.SourceEquation` record.

    Args:
        doc_defs: document-wide symbol definitions collected from other
            equations in the same paper (Layer 0 — definition propagation).
        domain_table: domain-specific symbol overrides (Layer 0.5).
    """
    return ground_symbols(
        equation.latex,
        context_before=context_before or equation.context_before,
        context_after=context_after or equation.context_after,
        symbols=equation.symbols,
        doc_defs=doc_defs,
        domain_table=domain_table,
    )


def ground_paper(equations) -> list[SymbolGrounding]:
    """Ground all equations in one paper with document-wide definition propagation.

    Two-pass strategy:
      Pass 1: collect context-anchored symbols from all equations into a
                document-wide table (``doc_defs``).  Also classify the paper's
                subdomain from the accumulated context text so domain-specific
                tables can apply.
      Pass 2: re-ground every equation, feeding ``doc_defs`` as Layer 0 and
                the domain table as Layer 0.5.

    This widens anchored coverage without inventing symbols — every propagated
    definition was stated by real prose somewhere in the paper.

    Returns:
        A list of ``SymbolGrounding`` records, one per equation, in the same
        order as the input.
    """
    from formulagate.dimensions import canonical_symbol_name
    from formulagate.domain_tables import get_domain_table
    from formulagate.paper_defs import extract_paper_defs
    from formulagate.weak_supervision import learn_paper_symbols

    # Collect all context for domain classification + doc_defs + paper defs.
    all_text = ""
    doc_defs: dict[str, tuple[str, Dimension]] = {}
    for eq in equations:
        g = ground_source_equation(eq)
        all_text += (eq.context_before or "") + " " + (eq.context_after or "") + " "
        for sym in g.grounded:
            if sym.source in ("context", "constant") and sym.dimension is not None:
                canon = canonical_symbol_name(sym.symbol)
                is_subscripted = "_" in canon
                base = canon.split("_")[0]
                if (len(base) >= 2 or is_subscripted) and canon not in doc_defs:
                    doc_defs[canon] = (sym.quantity, sym.dimension)

    # Merge paper-wide definition extraction (full text, not just context window).
    paper_defs = extract_paper_defs(all_text)
    for canon, (quantity, dim) in paper_defs.items():
        if canon not in doc_defs:
            doc_defs[canon] = (quantity, dim)

    domain_table = get_domain_table(all_text)

    # Pass 1 groundings (used for weak supervision)
    pass1_groundings = [ground_source_equation(eq) for eq in equations]

    # Weak supervision: learn symbols from equation consensus
    learned = learn_paper_symbols(equations, pass1_groundings)
    for canon, (dim, confidence) in learned.items():
        if canon not in doc_defs:
            doc_defs[canon] = (f"learned (c={confidence})", dim)

    # Pass 2: re-ground with document-wide definitions + domain table
    return [ground_source_equation(eq, doc_defs=doc_defs, domain_table=domain_table) for eq in equations]
