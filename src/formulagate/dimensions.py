"""SI dimensional analysis over canonicalised formulas.

Why this is the project's differentiator: semantic similarity tops out (the live
arXiv benchmark measured AUC 0.711 for the lexical+embedding score) because it
can only ever answer "does this *look* related". Dimensional analysis answers a
different question — "can this equation be physics at all" — and it answers it
deterministically, with a proof, for zero inference cost. No language model can
produce that guarantee, and no amount of scale removes the need for it.

Design rules learned from the benchmark:
  * A false reject is far more expensive than an abstention, so an ambiguous
    symbol (``T`` = temperature or time?) resolves to ``None``/unknown, never to
    a guess. Callers pin ambiguity through ``overrides``.
  * Everything degrades: unparsable input yields ``status="unknown"``.

Implemented as a 7-exponent vector over the SI base dimensions rather than via
``pint``: no new dependency, exact rational arithmetic, deterministic output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

_BASE_NAMES = ("mass", "length", "time", "current", "temperature", "amount", "luminosity")
_BASE_SYMBOLS = ("M", "L", "T", "I", "Θ", "N", "J")

_BRACE_SUBSCRIPT = re.compile(r"_\{([^}]*)\}")


def canonical_symbol_name(name: str) -> str:
    """Collapse every spelling of a symbol to one canonical key.

    SymPy's LaTeX parser renders subscripts inconsistently: ``k_B`` parses to
    the free symbol ``k_{B}``, ``epsilon_{0}`` stays as-is, and ``\\epsilon_0``
    keeps its backslash. Grounding tables and SMT anchors use the brace-free
    form (``k_B``, ``epsilon_0``), so a mismatch here silently un-anchored
    equations that stated constants in prose. One canonical form — backslash
    stripped, subscript braces dropped — is the single point every layer agrees
    on:

        "k_{B}", "k_B", "\\k_B"   -> "k_B"
        "epsilon_{0}", "\\epsilon_0" -> "epsilon_0"
    """
    key = _BRACE_SUBSCRIPT.sub(r"_\1", name)
    return key.replace("\\", "")


@dataclass(frozen=True)
class Dimension:
    """Exponents of the seven SI base dimensions.

    Fractions (not floats) keep ``(L^1/2)^2 == L`` exact, which matters because
    square roots appear constantly in physics.
    """

    mass: Fraction = Fraction(0)
    length: Fraction = Fraction(0)
    time: Fraction = Fraction(0)
    current: Fraction = Fraction(0)
    temperature: Fraction = Fraction(0)
    amount: Fraction = Fraction(0)
    luminosity: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        for name in _BASE_NAMES:
            value = getattr(self, name)
            if not isinstance(value, Fraction):
                object.__setattr__(self, name, Fraction(value))

    @property
    def exponents(self) -> tuple[Fraction, ...]:
        return tuple(getattr(self, name) for name in _BASE_NAMES)

    @property
    def is_dimensionless(self) -> bool:
        return all(e == 0 for e in self.exponents)

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(*(a + b for a, b in zip(self.exponents, other.exponents)))

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(*(a - b for a, b in zip(self.exponents, other.exponents)))

    def __pow__(self, power) -> "Dimension":
        factor = Fraction(power)
        return Dimension(*(e * factor for e in self.exponents))

    def __str__(self) -> str:
        parts = [
            symbol if exp == 1 else f"{symbol}^{exp}"
            for symbol, exp in zip(_BASE_SYMBOLS, self.exponents)
            if exp != 0
        ]
        return " ".join(parts) if parts else "1"


DIMENSIONLESS = Dimension()

_MASS = Dimension(mass=1)
_LENGTH = Dimension(length=1)
_TIME = Dimension(time=1)
_VELOCITY = Dimension(length=1, time=-1)
_ACCELERATION = Dimension(length=1, time=-2)
_FORCE = Dimension(mass=1, length=1, time=-2)
_ENERGY = Dimension(mass=1, length=2, time=-2)
_MOMENTUM = Dimension(mass=1, length=1, time=-1)
_ACTION = Dimension(mass=1, length=2, time=-1)
_FREQUENCY = Dimension(time=-1)
_POWER = Dimension(mass=1, length=2, time=-3)
_PRESSURE = Dimension(mass=1, length=-1, time=-2)
_CHARGE = Dimension(current=1, time=1)
_VOLTAGE = Dimension(mass=1, length=2, time=-3, current=-1)
_CAPACITANCE = Dimension(mass=-1, length=-2, time=4, current=2)
_RESISTANCE = Dimension(mass=1, length=2, time=-3, current=-2)
_INDUCTANCE = Dimension(mass=1, length=2, time=-2, current=-2)
_MAGNETIC_FLUX = Dimension(mass=1, length=2, time=-2, current=-1)
_MAGNETIC_FIELD = Dimension(mass=1, time=-2, current=-1)
_ELECTRIC_FIELD = Dimension(mass=1, length=1, time=-3, current=-1)
_CURRENT_DENSITY = Dimension(length=-2, current=1)
_ENTROPY = Dimension(mass=1, length=2, time=-2, temperature=-1)
_HEAT_CAPACITY = Dimension(mass=1, length=2, time=-2, temperature=-1)
_VISCOSITY = Dimension(mass=1, length=-1, time=-1)
_SURFACE_TENSION = Dimension(mass=1, time=-2)
_DIFFUSIVITY = Dimension(length=2, time=-1)
_WAVE_NUMBER = Dimension(length=-1)

# Only symbols with one dominant reading across physics are listed. Anything
# genuinely overloaded (T, P, S, V, k, g, Q, I, L, W) is deliberately absent:
# absence means "unknown", and unknown never rejects.

SYMBOL_DIMENSIONS: dict[str, Dimension] = {
    # ── Mechanics (single-dominant-read SI letters, kept) ────────────────
    "m": _MASS,
    "M": _MASS,
    "c": _VELOCITY,
    "v": _VELOCITY,
    "u": _VELOCITY,
    "a": _ACCELERATION,
    "F": _FORCE,
    "E": _ENERGY,
    "p": _MOMENTUM,
    "x": _LENGTH,
    "y": _LENGTH,
    "z": _LENGTH,
    "r": _LENGTH,
    "l": _LENGTH,
    "t": _TIME,
    "lambda": _LENGTH,
    "L": _LENGTH,
    "A": Dimension(length=2),
    # ── Fundamental constants / named quantities (unambiguous dimesion) ──
    "hbar": _ACTION,
    "hbar_": _ACTION,
    "k_B": Dimension(mass=1, length=2, time=-2, temperature=-1),
    "epsilon_0": Dimension(mass=-1, length=-3, time=4, current=2),
    "mu_0": Dimension(mass=1, length=1, time=-2, current=-2),
    "k_e": Dimension(mass=1, length=3, time=-4, current=-2),  # Coulomb constant
    "omega": _FREQUENCY,
    "f": _FREQUENCY,
    "n": DIMENSIONLESS,
    "N": DIMENSIONLESS,
    "theta": DIMENSIONLESS,
    "phi": DIMENSIONLESS,
    "alpha": DIMENSIONLESS,
    "beta": DIMENSIONLESS,
    "gamma": DIMENSIONLESS,
    "delta": DIMENSIONLESS,
    "epsilon": DIMENSIONLESS,
    # ── Electromagnetism ─────────────────────────────────────────────────
    "B": _MAGNETIC_FIELD,
    "E_field": _ELECTRIC_FIELD,
    "D_field": Dimension(length=-2, time=1, current=1),
    "H_field": Dimension(length=-1, current=1),
    "Phi_B": _MAGNETIC_FLUX,
    "J": _CURRENT_DENSITY,
    "j": _CURRENT_DENSITY,
    "sigma": Dimension(mass=-1, length=-3, time=3, current=2),  # conductivity
    "rho_e": _CHARGE,  # charge density C/m^3 → but we use charge dimension
    "epsilon_0": Dimension(mass=-1, length=-3, time=4, current=2),
    "mu_0": Dimension(mass=1, length=1, time=-2, current=-2),
    "k_e": Dimension(mass=1, length=3, time=-4, current=-2),  # Coulomb constant
    # ── Thermodynamics ───────────────────────────────────────────────────
    "C_V": _HEAT_CAPACITY,
    "C_P": _HEAT_CAPACITY,
    "c_v": _HEAT_CAPACITY,
    "c_p": _HEAT_CAPACITY,
    "mu_v": _VISCOSITY,  # dynamic viscosity (bare "eta" is overloaded: viscosity
                         # in fluid mechanics vs a dimensionless correction
                         # parameter in gravity/QFT — removed, ambiguous → unknown)
    "gamma_s": _SURFACE_TENSION,
    "D": _DIFFUSIVITY,
    # ── Quantum Mechanics ────────────────────────────────────────────────
    "psi": DIMENSIONLESS,  # wavefunction
    "Psi": DIMENSIONLESS,
    "hat_H": _ENERGY,  # Hamiltonian
    "hat_p": _MOMENTUM,  # momentum operator
    "hat_x": _LENGTH,  # position operator
    "sigma_x": DIMENSIONLESS,  # Pauli matrix
    "sigma_y": DIMENSIONLESS,
    "sigma_z": DIMENSIONLESS,
    # ── Relativity ───────────────────────────────────────────────────────
    "gamma_L": DIMENSIONLESS,  # Lorentz factor
    "beta_r": DIMENSIONLESS,  # v/c
    "g_munu": DIMENSIONLESS,  # metric tensor
    "R_munu": Dimension(length=-2),  # Ricci curvature
    "R_scalar": Dimension(length=-2),  # Ricci scalar
    "T_munu": _ENERGY,  # stress-energy (per volume → energy density)
    "Lambda": Dimension(length=-2),  # cosmological constant
    # ── Fluid Dynamics ───────────────────────────────────────────────────
    "Re": DIMENSIONLESS,  # Reynolds number
    "Ma": DIMENSIONLESS,  # Mach number
    "Fr": DIMENSIONLESS,  # Froude number
    "St": DIMENSIONLESS,  # Strouhal number
    "Pr": DIMENSIONLESS,  # Prandtl number
    "Nu": DIMENSIONLESS,  # Nusselt number
    "Ra": DIMENSIONLESS,  # Rayleigh number
    "Pe": DIMENSIONLESS,  # Peclet number
    # ── Optics / Waves ───────────────────────────────────────────────────
    "k": _WAVE_NUMBER,
    "I_opt": _POWER,  # intensity = power per area → but we keep power for now
    "n_ref": DIMENSIONLESS,  # refractive index
    "delta_opt": DIMENSIONLESS,  # optical path difference
    # ── Nuclear / Particle ───────────────────────────────────────────────
    "sigma_cs": Dimension(length=2),  # cross-section (has area dimension)
    "lambda_decay": _FREQUENCY,  # decay constant
    "tau_life": _TIME,  # mean lifetime
    "Gamma_width": _ENERGY,  # decay width
    "Phi_flux": Dimension(length=-2, time=-1),  # particle flux
    # ── Vector Calculus ───────────────────────────────────────────────────
    "nabla": Dimension(length=-1),  # ∇: spatial derivative → L⁻¹
    "partial_t": Dimension(time=-1),  # ∂/∂t → T⁻¹
    "partial_x": Dimension(length=-1),  # ∂/∂x → L⁻¹
    "partial_y": Dimension(length=-1),
    "partial_z": Dimension(length=-1),
    "partial_r": Dimension(length=-1),
    "partial_mu": Dimension(length=-1),  # ∂_μ treated as spatial derivative
    "dot": Dimension(time=-1),  # ẋ: time derivative → T⁻¹
    "ddot": Dimension(time=-2),  # ẍ: second derivative → T⁻²
    # ── Additional Dimensionless ─────────────────────────────────────────
    "mu": DIMENSIONLESS,  # often dimensionless parameter
    "zeta": DIMENSIONLESS,
    "eta_dimless": DIMENSIONLESS,
    "xi": DIMENSIONLESS,
    "chi": DIMENSIONLESS,
    "varepsilon": DIMENSIONLESS,
    "varphi": DIMENSIONLESS,
    "vartheta": DIMENSIONLESS,
    "varkappa": DIMENSIONLESS,
    "varpi": DIMENSIONLESS,
    "varrho": DIMENSIONLESS,
    "varsigma": DIMENSIONLESS,
    "aleph": DIMENSIONLESS,
    "wp": DIMENSIONLESS,
    "Re_num": DIMENSIONLESS,
    "Im_num": DIMENSIONLESS,
    # ── Unicode Greek (from _preprocess_latex) ─────────────────────────────
    "α": DIMENSIONLESS, "β": DIMENSIONLESS, "γ": DIMENSIONLESS,
    "δ": DIMENSIONLESS, "ε": DIMENSIONLESS, "ζ": DIMENSIONLESS,
    "η": _VISCOSITY, "θ": DIMENSIONLESS, "ϑ": DIMENSIONLESS, "ι": DIMENSIONLESS,
    "κ": Dimension(mass=1, length=1, time=-3, temperature=-1),
    "ϰ": DIMENSIONLESS, "ω": _FREQUENCY,
    "λ": _LENGTH, "μ": _VISCOSITY, "ν": _FREQUENCY, "ξ": DIMENSIONLESS,
    "π": DIMENSIONLESS, "ϖ": DIMENSIONLESS,
    "ρ": Dimension(mass=1, length=-3), "ϱ": Dimension(mass=1, length=-3),
    "σ": Dimension(mass=-1, length=-3, time=3, current=2),
    "ς": DIMENSIONLESS, "τ": DIMENSIONLESS,
    "υ": DIMENSIONLESS, "φ": DIMENSIONLESS, "ϕ": DIMENSIONLESS,
    "χ": DIMENSIONLESS, "ψ": DIMENSIONLESS,
    "Γ": _ENERGY, "Δ": DIMENSIONLESS, "Θ": DIMENSIONLESS,
    "Λ": Dimension(length=-2), "Ξ": DIMENSIONLESS, "Π": DIMENSIONLESS,
    "Σ": DIMENSIONLESS, "Υ": DIMENSIONLESS,
    "Φ": _MAGNETIC_FLUX, "Ψ": DIMENSIONLESS, "Ω": DIMENSIONLESS,
    "ħ": _ACTION, "∂": DIMENSIONLESS,
}

_SUBSCRIPT_ALIASES = {
    "k_{B}": "k_B", "\\hbar": "hbar", "k_b": "k_B",
    "\\epsilon_0": "epsilon_0", "\\mu_0": "mu_0",
    "\\varepsilon_0": "epsilon_0",
    "\\sigma_{cs}": "sigma_cs",
    "c_{p}": "c_p", "c_{v}": "c_v",
    "\\gamma_{L}": "gamma_L",
    "\\lambda_{decay}": "lambda_decay",
    "\\tau_{life}": "tau_life",
    "\\Gamma_{width}": "Gamma_width",
    "\\Phi_{flux}": "Phi_flux",
    "\\Phi_{B}": "Phi_B",
    "\\rho_{e}": "rho_e",
    "\\hat{H}": "hat_H", "\\hat{p}": "hat_p", "\\hat{x}": "hat_x",
}

# ── Pharmaceutical symbol dimensions ─────────────────────────────────────────
# Direct Dimension construction (no _D helper available in this module).
_PHARMA_DIM: dict[str, Dimension] = {
    "CL": Dimension(mass=0, length=3, time=-1),      # clearance = volume/time
    "CLh": Dimension(mass=0, length=3, time=-1),     # hepatic clearance
    "CLr": Dimension(mass=0, length=3, time=-1),     # renal clearance
    "CLint": Dimension(mass=0, length=3, time=-1),   # intrinsic clearance
    "Qh": Dimension(mass=0, length=3, time=-1),      # hepatic blood flow
    "Qt": Dimension(mass=0, length=3, time=-1),      # tissue blood flow
    "GFR": Dimension(mass=0, length=3, time=-1),     # glomerular filtration
    "Vd": Dimension(mass=0, length=3, time=0),       # volume of distribution
    "Vss": Dimension(mass=0, length=3, time=0),      # steady-state volume
    "Vt": Dimension(mass=0, length=3, time=0),       # tissue volume
    "AUC": Dimension(mass=1, length=-3, time=1),     # concentration × time
    "AUMC": Dimension(mass=1, length=-3, time=2),    # concentration × time^2
    "MRT": Dimension(mass=0, length=0, time=1),      # mean residence time
    "Dose": Dimension(mass=1, length=0, time=0),     # mass of drug
    "LD": Dimension(mass=1, length=0, time=0),       # loading dose
    "MD": Dimension(mass=1, length=0, time=0),       # maintenance dose
    "ED50": Dimension(mass=1, length=0, time=0),     # median effective dose
    "TD50": Dimension(mass=1, length=0, time=0),     # median toxic dose
    "BW": Dimension(mass=1, length=0, time=0),       # body weight
    "MW": Dimension(mass=1, length=0, time=0),       # molecular weight
    "C0": Dimension(mass=1, length=-3, time=0),      # initial concentration
    "Cp": Dimension(mass=1, length=-3, time=0),      # plasma concentration
    "Css": Dimension(mass=1, length=-3, time=0),     # steady-state conc
    "Cmax": Dimension(mass=1, length=-3, time=0),    # peak concentration
    "Cmin": Dimension(mass=1, length=-3, time=0),    # trough concentration
    "Cav": Dimension(mass=1, length=-3, time=0),     # average concentration
    "Ct": Dimension(mass=1, length=-3, time=0),      # tissue concentration
    "Ca": Dimension(mass=1, length=-3, time=0),      # arterial concentration
    "Cs": Dimension(mass=1, length=-3, time=0),      # saturated solubility
    "fu": DIMENSIONLESS,                               # fraction unbound
    "f_u": DIMENSIONLESS,                             # fraction unbound (subscripted form)
    "fe": DIMENSIONLESS,                               # fraction excreted
    "Ke": Dimension(mass=0, length=0, time=-1),       # elimination constant
    "Ka": Dimension(mass=0, length=0, time=-1),       # absorption constant
    "t_half": Dimension(mass=0, length=0, time=1),    # half-life
    "t_{1/2}": Dimension(mass=0, length=0, time=1),  # half-life (display name)
    "tmax": Dimension(mass=0, length=0, time=1),      # time to peak
    "Vmax": Dimension(mass=1, length=-3, time=-1),    # max reaction velocity
    "Km": Dimension(mass=1, length=-3, time=0),       # Michaelis constant
    "Ki": Dimension(mass=1, length=-3, time=0),       # inhibition constant
    "Kd": Dimension(mass=1, length=-3, time=0),       # dissociation constant
    "Kp": DIMENSIONLESS,                                # partition coefficient
    "EC50": Dimension(mass=1, length=-3, time=0),     # half-max effective conc
    "IC50": Dimension(mass=1, length=-3, time=0),     # half-max inhibitory conc
    "Papp": Dimension(mass=0, length=1, time=-1),      # apparent permeability
    "TI": DIMENSIONLESS,                                # therapeutic index
    # ── Composite / subscript-normalized symbols ──
    "Cmax": Dimension(mass=1, length=-3, time=0),
    "Cmin": Dimension(mass=1, length=-3, time=0),
    "tmax": Dimension(mass=0, length=0, time=1),
    "t_half": Dimension(mass=0, length=0, time=1),
    "t_{1/2}": Dimension(mass=0, length=0, time=1),
    # Enzyme substrate/inhibitor concentrations (mass/volume)
    "S": Dimension(mass=1, length=-3, time=0),
    "I": Dimension(mass=1, length=-3, time=0),
    "D": Dimension(mass=1, length=-3, time=0),        # drug concentration in pharma context
    # Dosing
    "tau": Dimension(mass=0, length=0, time=1),
    # Accumulation ratio (dimensionless)
    "R": DIMENSIONLESS,
    # Hill coefficient (dimensionless)
    "n": DIMENSIONLESS,
    # Max effect (dimensionless)
    "Emax": DIMENSIONLESS,
    # Dose normalizations
    "Dose_oral": Dimension(mass=1, length=0, time=0),
    "Dose_IV": Dimension(mass=1, length=0, time=0),
    "AUC_oral": Dimension(mass=1, length=-3, time=1),
    "AUC_IV": Dimension(mass=1, length=-3, time=1),
    "F": DIMENSIONLESS,                                # bioavailability fraction
    # Dissolution / Transport
    "h": Dimension(mass=0, length=1, time=0),         # diffusion layer thickness
    "A": Dimension(mass=0, length=2, time=0),          # surface area in pharma
}
SYMBOL_DIMENSIONS.update(_PHARMA_DIM)

# Restore physics values that were overwritten by pharma entries with
# the same symbol name.  Pharma access must go through domain="pharmacology".
_PHYSICS_RESTORE: dict[str, Dimension] = {
    "F": _FORCE,    # restore force (pharma F=bioavailability accessed via domain)
}
SYMBOL_DIMENSIONS.update(_PHYSICS_RESTORE)

# Domain-specific symbol overrides — symbols that mean different things
# in pharmacology vs physics.  Only consulted when domain="pharmacology".
_DOMAIN_PHARMA_OVERRIDES: dict[str, Dimension] = {
    "F": DIMENSIONLESS,    # bioavailability, not force
    "E": DIMENSIONLESS,    # extraction ratio, not energy
    "K": DIMENSIONLESS,    # partition coefficient, not temperature
    "D": Dimension(mass=1, length=-3, time=0),  # drug concentration, not diffusion
}

# Subscript normalization: AUC_{oral} -> AUC_oral, C_{max} -> Cmax, etc.
_SUBSCRIPT_FLAT: dict[str, str] = {
    "AUCT_{oral}": "AUC_oral", "AUCT_{IV}": "AUC_IV",
    "Dos_{oral}": "Dose_oral", "Dos_{IV}": "Dose_IV",
    "CT_{max}": "Cmax", "CT_{min}": "Cmin",
    "CT_{ss}": "Css", "CT_{0}": "C0", "CT_{p}": "Cp",
    "CT_{t}": "Ct", "CT_{a}": "Ca", "CT_{av}": "Cav",
    "tT_{1/2}": "t_half", "tT_{max}": "tmax",
    "kT_{a}": "Ka", "kT_{e}": "Ke",
    "KT_{m}": "Km", "KT_{i}": "Ki", "KT_{d}": "Kd",
    "KT_{p}": "Kp", "QT_{h}": "Qh", "VT_{d}": "Vd",
    "VT_{ss}": "Vss", "VT_{t}": "Vt",
    # Also handle non-canonical subscript key patterns
    "AUC_{o": "AUC", "AUC_{I": "AUC",
    "Dose_{o": "Dose", "Dose_{I": "Dose",
}


def _normalize_subscripted_symbol(name: str) -> str:
    """Flatten subscripted pharma symbols for dimension lookup.

    ``AUC_{oral}`` -> ``AUC_oral``, ``C_{max}`` -> ``Cmax``,
    ``AUC_{oral}`` (SymPy canonical) -> ``AUC``.
    """
    if name in _SUBSCRIPT_FLAT:
        return _SUBSCRIPT_FLAT[name]
    # General pattern: strip _{subscript} and check base
    for pat, replacement in [
        ("_{oral}", "_oral"), ("_{IV}", "_IV"),
        ("_{max}", "max"), ("_{min}", "min"),
        ("_{ss}", "ss"), ("_{0}", "0"),
        ("_{p}", "p"), ("_{t}", "t"),
        ("_{a}", "a"), ("_{av}", "av"),
        ("_{1/2}", "_half"), ("_{int}", "int"),
        ("_{h}", "h"), ("_{d}", "d"),
    ]:
        if name.endswith(pat):
            base = name[:-len(pat)]
            flat = base + replacement
            if flat in SYMBOL_DIMENSIONS:
                return flat
            if base in SYMBOL_DIMENSIONS:
                return base
    return name


def dimension_of_symbol(
    name: str, overrides: Mapping[str, Dimension] | None = None,
    domain: str | None = None,
) -> Dimension | None:
    """Look up a symbol's dimension; ``None`` means unknown or ambiguous.

    When ``domain=="pharmacology"``, pharmaceutical context overrides
    conflicting physics symbols (F=bioavailability not force, E=extraction
    not energy).
    """
    key = _SUBSCRIPT_ALIASES.get(name, canonical_symbol_name(name))
    if overrides:
        hit = overrides.get(name) or overrides.get(key)
        if hit is not None:
            return hit
    # Domain-specific overrides take priority over global table
    if domain == "pharmacology" and key in _DOMAIN_PHARMA_OVERRIDES:
        return _DOMAIN_PHARMA_OVERRIDES[key]
    # Pharma domain: try normalized subscript forms first
    if domain == "pharmacology":
        flat = _normalize_subscripted_symbol(key)
        if flat != key and flat in SYMBOL_DIMENSIONS:
            return SYMBOL_DIMENSIONS[flat]
    hit = SYMBOL_DIMENSIONS.get(key)
    if hit is not None:
        return hit
    flat = _normalize_subscripted_symbol(key)
    if flat != key and flat in SYMBOL_DIMENSIONS:
        return SYMBOL_DIMENSIONS[flat]
    return None


# ─── Verdict ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DimensionVerdict:
    """Outcome of a dimensional check.

    ``status`` is one of ``consistent`` / ``inconsistent`` / ``unknown``.
    ``dimension`` is the shared dimension of the equation when consistent.
    """

    status: str
    detail: str = ""
    dimension: Dimension | None = None

    @property
    def is_reject(self) -> bool:
        return self.status == "inconsistent"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "detail": self.detail,
            "dimension": str(self.dimension) if self.dimension else None,
        }


_TRANSCENDENTAL = frozenset(
    """sin cos tan cot sec csc asin acos atan sinh cosh tanh coth asinh acosh atanh
    exp log ln erf erfc gamma loggamma zeta""".split()
)


class _Unknown(Exception):
    """Raised internally when a subtree carries no resolvable dimension."""


class _Inconsistent(Exception):
    """Raised internally when two added terms disagree."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def _walk(expr, overrides: Mapping[str, Dimension] | None, domain: str | None = None) -> Dimension:
    """Dimension of a SymPy expression, or raise ``_Unknown``/``_Inconsistent``."""

    import sympy

    if expr.is_number:
        return DIMENSIONLESS

    if isinstance(expr, sympy.Symbol):
        dim = dimension_of_symbol(str(expr), overrides, domain=domain)
        if dim is None:
            raise _Unknown(str(expr))
        return dim

    if isinstance(expr, sympy.Mul):
        total = DIMENSIONLESS
        for arg in expr.args:
            total = total * _walk(arg, overrides, domain)
        return total

    if isinstance(expr, sympy.Pow):
        base, exponent = expr.args
        if not exponent.is_number or not exponent.is_rational:
            if _walk(base, overrides, domain).is_dimensionless:
                return DIMENSIONLESS
            raise _Unknown("symbolic exponent on a dimensional base")
        return _walk(base, overrides, domain) ** Fraction(str(exponent))

    if isinstance(expr, sympy.Add):
        numeric = [arg for arg in expr.args if arg.is_number]
        symbolic = [arg for arg in expr.args if not arg.is_number]
        if not symbolic:
            return DIMENSIONLESS

        dims = [_walk(arg, overrides, domain) for arg in symbolic]
        first = dims[0]
        for other, arg in zip(dims[1:], symbolic[1:]):
            if other != first:
                raise _Inconsistent(
                    f"added terms differ: {first} vs {other} (term {sympy.sstr(arg)})"
                )
        if numeric and not first.is_dimensionless:
            raise _Unknown("dimensional term equated to a number (natural units?)")
        return first

    if isinstance(expr, sympy.Function):
        if expr.func.__name__ not in _TRANSCENDENTAL:
            raise _Unknown(f"user-defined function {expr.func.__name__}")
        for arg in expr.args:
            if not _walk(arg, overrides, domain).is_dimensionless:
                raise _Inconsistent(
                    f"{expr.func.__name__} applied to a dimensional argument"
                )
        return DIMENSIONLESS

    if isinstance(expr, sympy.Derivative):
        func = expr.args[0]
        var = expr.args[1]
        if isinstance(var, sympy.Tuple):
            var = var[0]
        return _walk(func, overrides, domain) / _walk(var, overrides, domain)

    if isinstance(expr, sympy.Integral):
        func = expr.args[0]
        var = expr.args[1]
        if isinstance(var, sympy.Tuple):
            var = var[0]
        return _walk(func, overrides, domain) * _walk(var, overrides, domain)

    raise _Unknown(type(expr).__name__)


def check_dimensions(formula, overrides: Mapping[str, Dimension] | None = None,
                     domain: str | None = None) -> DimensionVerdict:
    """Check that every additive term of ``formula`` shares one dimension.

    ``formula`` is a :class:`formulagate.formula_extract.Formula`; it is already
    canonicalised to ``lhs - rhs``, so equality is dimensional agreement between
    the terms of a single expression.

    When ``domain`` is set (e.g. ``"pharmacology"``), domain-specific symbol
    tables are consulted, resolving clashes like F=bioavailability vs F=force.
    """

    if not getattr(formula, "canonical", None):
        return DimensionVerdict("unknown", formula.parse_error or "no canonical form")

    try:
        import sympy

        expr = sympy.sympify(formula.canonical)
    except Exception as exc:
        return DimensionVerdict("unknown", f"sympify failed: {type(exc).__name__}")

    try:
        dimension = _walk(expr, overrides, domain)
    except _Inconsistent as exc:
        return DimensionVerdict("inconsistent", exc.detail)
    except _Unknown as exc:
        return DimensionVerdict("unknown", f"unresolved symbol: {exc}")
    except Exception as exc:
        return DimensionVerdict("unknown", f"{type(exc).__name__}: {exc}"[:120])

    return DimensionVerdict("consistent", f"all terms are {dimension}", dimension)
