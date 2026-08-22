"""Domain-specific symbol tables — aggressive grounding when the field is known.

Why this exists:
  The generic table must be conservative: ``R`` could be radius (L) or Ricci
  scalar (L⁻²), so it stays out.  But when we *know* the paper is about
  cosmology, ``R`` is overwhelmingly the Ricci scalar — and leaving it
  ungrounded needlessly bloats the unknown count.

  This module classifies a LaTeX source into a physics subdomain, then
  provides a table where overloaded symbols get their dominant-domain
  reading.  The table sits at Layer 0.5 in ``symbol_grounding`` (after
  document-wide definitions, before the generic conservative table).

Sub-domains recognised:
  cosmology_gr     — GR, cosmology, black holes, gravitational waves
  fluid_dynamics   — fluid mechanics, hydrodynamics, magnetohydrodynamics
  quantum_field    — QFT, particle physics, standard model
  condensed_matter — condensed matter / solid-state physics
  thermodynamics  — thermodynamics, statistical mechanics
  electromagnetism — electrodynamics, optics, plasma physics
  nuclear          — nuclear / hadronic physics
  general          — any other physics (falls back to the generic table)
"""

from __future__ import annotations

from fractions import Fraction

from formulagate.dimensions import Dimension

DIMENSIONLESS = Dimension()


# ─── Sub-domain classification ────────────────────────────────────────────────

SubDomain = str

# Marker sets — multi-word keys are space-delimited
_MARKERS: dict[SubDomain, tuple[str, ...]] = {
    "cosmology_gr": (
        "cosmolog", "dark energy", "dark matter", "einstein", "friedmann",
        "robertson", "walker", "flrw", "schwarzschild", "kerr", "black hole",
        "gravitational wave", "general relativ", "ricci", "riemann tensor",
        "expansion factor", "redshift", "hubble", "inflaton", "planck scale",
        "planck mass", "planck length", "hawking", "penrose", "carter",
        "event horizon", "singularity", "wormhole", "graviton", "metric tensor",
        "cosmic microwave", "cmb", "baryon acoustic", "weak lensing",
        "large scale structure", "de sitter", "anti-de sitter", "ads/cft",
        "holographic", "entropic gravity", "modified gravity", "mond",
        "brans-dicke", "einstein-hilbert", "null geodesic", "geodesic",
        "warp drive", "brane", "extra dimension", "compactification",
        "scale factor", "conformal time", "causal", "light cone",
    ),
    "fluid_dynamics": (
        "fluid", "hydrodynam", "viscous", "viscosity", "reynolds",
        "navier-stokes", "turbulent", "turbulence", "laminar", "boundary layer",
        "convection", "buoyancy", "magnetohydrodynam", "mhd", "plasma flow",
        "aerodynam", "drag", "lift", "wing", "airfoil", "pipe flow",
        "channel flow", "mixing", "stirring", "vortex", "vorticity",
        "compressible flow", "incompressible", "shock wave", "supersonic",
        "subsonic", "mach number", "prandtl", "nusselt", "rayleigh number",
        "froude", "strouhal", "peclet", "schmidt number",
        "shear stress", "surface tension", "capillary",
        "porous", "darcy", "permeability",
    ),
    "quantum_field": (
        "quantum field", "qft", "qed", "qcd", "electroweak", "standard model",
        "feynman", "lagrangian", "path integral", "propagator", "vertex",
        "renormaliz", "wilson loop", "gauge theory", "yang-mills",
        "spontaneous symmetry", "higgs", "goldstone", "fermion", "boson",
        "quark", "gluon", "lepton", "neutrino", "cern", "lhc", "collider",
        "parton", "hadron", "meson", "baryon", "scattering amplitude",
        "s-matrix", "cross section", "decay width", "branching ratio",
        "chiral", "anomal", "instanton", "monopole", "soliton",
        "supersymmetr", "supergravity", "grand unified", "gut",
        "majorana", "weyl", "dirac", "klein-gordon", "proca",
        "ghost", "brst", "ward identity", "slavnov-taylor",
        "loop correction", "counterterm", "running coupling",
        "beta function", "asymptotic freedom", "confinement",
        "deep inelastic", "parton distribution", "pdf",
    ),
    "condensed_matter": (
        "condensed matter", "solid state", "superconduct", "superflu",
        "topological insulator", "weyl semimetal", "graphene",
        "quantum hall", "fractional quantum", "spintronic",
        "fermi liquid", "luttinger", "mott insulator", "hubbard",
        "density functional", "dft", "band structure", "bloch",
        "wannier", "tight-binding", "hopping", "phonon", "magnon",
        "exciton", "polaron", "plasmon", "spin wave",
        "magnetization", "ferromagnetic", "antiferromagnetic",
        "paramagnetic", "curie", "neel", "ising", "heisenberg model",
        "critical exponent", "phase transition", "order parameter",
        "landau", "ginzburg", "mean field", "bcs", "cooper pair",
        "josephson", "squid", "andreev", "majorana zero mode",
        "quantum dot", "nanowire", "2d material", "transition metal",
    ),
    "thermodynamics": (
        "thermodynam", "statistical mechanic", "entropy", "free energy",
        "partition function", "gibbs", "boltzmann", "maxwell relation",
        "carnot", "heat engine", "temperature", "heat capacity",
        "specific heat", "thermal conductivity", "thermal expansion",
        "isothermal", "adiabatic", "isobaric", "isochoric",
        "clausius", "clapeyron", "van der waals", "virial",
        "equation of state", "critical point", "triple point",
        "phase diagram", "latent heat", "enthalpy", "caloric",
        "brownian", "fluctuation-dissipation", "langevin",
        "fokker-planck", "master equation", "ergodic",
    ),
    "electromagnetism": (
        "electromagnet", "electrodynam", "maxwell", "electrostat",
        "magnetostat", "induction", "antenna", "waveguide",
        "radiation", "dipole", "multipole", "polarization",
        "dielectric", "permittivity", "permeability", "conductivity",
        "resistivity", "impedance", "capacitance", "inductance",
        "ohm", "kirchhoff", "circuit", "transmission line",
        "optical", "photonics", "laser", "refractive index",
        "diffraction", "interference", "plasmon", "metamaterial",
        "electromagnetic wave", "radio frequency", "microwave",
        "thz", "terahertz", "antenna array",
    ),
    "nuclear": (
        "nuclear", "nucleon", "nucleus", "isotope", "radioactive",
        "decay chain", "fission", "fusion", "alpha decay", "beta decay",
        "gamma ray", "neutron", "proton", "binding energy",
        "magic number", "shell model", "liquid drop", "bethe-weizsäcker",
        "r-process", "s-process", "nucleosynthesis", "reactor",
        "hadron", "qgp", "quark-gluon plasma", "heavy ion",
        "strangelet", "hyperon", "chiral effective",
        "neutron star", "nuclear astrophys",
    ),
}


def classify_subdomain(text: str) -> SubDomain:
    """Classify a LaTeX source text into a physics sub-domain."""
    import re

    lower = text.lower()
    scores: dict[SubDomain, int] = {}

    for domain, markers in _MARKERS.items():
        score = 0
        for marker in markers:
            if " " in marker:
                if marker in lower:
                    score += 3  # multi-word markers are stronger
            else:
                score += lower.count(marker)
        if score > 0:
            scores[domain] = score

    if not scores:
        return "general"

    return max(scores, key=lambda k: scores[k])


# ─── Domain-specific symbol tables ────────────────────────────────────────────


def _D(mass=0, length=0, time=0, current=0, temperature=0, amount=0, luminosity=0):
    _val = (mass, length, time, current, temperature, amount, luminosity)
    _args = (Fraction(x) for x in _val)
    return Dimension(*_args)


_DOMAIN_TABLES: dict[SubDomain, dict[str, Dimension]] = {
    "cosmology_gr": {
        "R": _D(length=-2),                 # Ricci scalar
        "a": DIMENSIONLESS,                 # scale factor
        "tau": _D(time=1),                  # conformal time
        "Phi": _D(length=2, time=-2),       # gravitational potential
        "Psi": _D(length=2, time=-2),       # gravitational potential
        "G": _D(mass=-1, length=3, time=-2),  # gravitational constant
        "kappa": _D(mass=-1, length=-1, time=2),  # 8πG/c⁴ coupling
        "Lambda": _D(length=-2),            # cosmological constant
        "g": _D(length=1, time=-2),         # gravitational acceleration
        "H": _D(time=-1),                   # Hubble parameter
        "Omega": DIMENSIONLESS,             # density parameter
    },
    "fluid_dynamics": {
        "R": _D(length=1),                  # radius / characteristic length
        "a": _D(length=1, time=-2),         # acceleration
        "nu": _D(length=2, time=-1),        # kinematic viscosity
        "eta": _D(mass=1, length=-1, time=-1),  # dynamic viscosity
        "kappa": _D(mass=1, length=1, time=-3, temperature=-1),  # thermal conductivity
    },
    "quantum_field": {
        "hbar": _D(mass=1, length=2, time=-1),  # Planck constant (action)
        "psi": DIMENSIONLESS,               # wavefunction
        "g": DIMENSIONLESS,                 # coupling constant
        "Gamma": _D(mass=1, length=2, time=-2),  # decay width (energy)
    },
    "thermodynamics": {
        "S": _D(mass=1, length=2, time=-2, temperature=-1),  # entropy
        "T": _D(temperature=1),             # temperature
        "k_B": _D(mass=1, length=2, time=-2, temperature=-1),  # Boltzmann
        "C_V": _D(mass=1, length=2, time=-2, temperature=-1),  # heat capacity
        "c_p": _D(mass=1, length=2, time=-2, temperature=-1),
        "c_v": _D(mass=1, length=2, time=-2, temperature=-1),
        "U": _D(mass=1, length=2, time=-2),  # internal energy
        "F": _D(mass=1, length=2, time=-2),  # free energy
        "H": _D(mass=1, length=2, time=-2),  # enthalpy
        "P": _D(mass=1, length=-1, time=-2),  # pressure
        "V": _D(length=3),                    # volume
        "n": _D(amount=1),                    # amount of substance (moles)
        "R": _D(mass=1, length=2, time=-2, temperature=-1, amount=-1),  # gas constant
        "N_A": _D(amount=-1),                 # Avogadro constant
    },
    "electromagnetism": {
        "B": _D(mass=1, time=-2, current=-1),   # magnetic field
        "E_field": _D(mass=1, length=1, time=-3, current=-1),
        "sigma": _D(mass=-1, length=-3, time=3, current=2),  # conductivity
    },
    "condensed_matter": {
        "psi": DIMENSIONLESS,               # order parameter / wavefunction
        "Delta": _D(mass=1, length=2, time=-2),  # gap (energy)
        "T_c": _D(temperature=1),           # critical temperature
        "J": _D(mass=1, length=2, time=-2),  # exchange coupling (energy)
    },
    "nuclear": {
        "sigma": _D(length=2),              # cross-section
        "Gamma": _D(mass=1, length=2, time=-2),  # decay width
        "Q": _D(mass=1, length=2, time=-2),  # reaction Q-value (energy)
    },
}


def get_domain_table(text: str) -> dict[str, Dimension]:
    """Return domain-specific symbol overrides for a LaTeX source."""
    sub = classify_subdomain(text)
    return _DOMAIN_TABLES.get(sub, {})
