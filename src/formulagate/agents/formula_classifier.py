"""Agent 5: Formula Classifier & Tagger.

Tags every formula in the corpus with rich metadata:
- Domain + Subdomain (physics taxonomy)
- Tags / keywords for search
- Difficulty level (undergrad, grad, PhD, research)
- Real-world applications (2-5 per formula)
- Industries where it's used (2-5 per formula)
- Related formulas (cross-references)

This agent is purely metadata — it doesn't discover new formulas,
it enriches existing ones with structured information.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from formulagate.agents.graph import BaseAgent, FormulaDiscovery

logger = logging.getLogger(__name__)

# ─── Physics Taxonomy ─────────────────────────────────────────────────────────

PHYSICS_TAXONOMY: dict[str, list[str]] = {
    "Classical Mechanics": [
        "Kinematics", "Dynamics", "Newton's Laws", "Work and Energy",
        "Momentum and Collisions", "Rotational Motion", "Gravitation",
        "Oscillations", "Waves", "Fluid Mechanics",
    ],
    "Electromagnetism": [
        "Electrostatics", "Magnetostatics", "Electrodynamics",
        "Maxwell's Equations", "Circuits", "Electromagnetic Waves",
        "Optics", "Photonics",
    ],
    "Thermodynamics": [
        "Laws of Thermodynamics", "Heat Transfer", "Kinetic Theory",
        "Statistical Mechanics", "Phase Transitions",
    ],
    "Quantum Mechanics": [
        "Wave Mechanics", "Operator Algebra", "Measurement Theory",
        "Perturbation Theory", "Scattering Theory", "Quantum Foundations",
    ],
    "Quantum Field Theory": [
        "QED", "QCD", "Electroweak Theory", "Yang-Mills Theory",
        "Renormalization", "Path Integrals", "Supersymmetry",
    ],
    "Quantum Computing": [
        "Qubit Theory", "Quantum Gates", "Quantum Algorithms",
        "Quantum Error Correction", "Quantum Cryptography",
    ],
    "Quantum Information": [
        "Entanglement Measures", "Open Quantum Systems",
        "Quantum Channels", "Quantum Thermodynamics",
    ],
    "Condensed Matter": [
        "Superconductivity", "Magnetism", "Semiconductors",
        "Topological Phases", "Strongly Correlated Systems",
    ],
    "Relativity": [
        "Special Relativity", "General Relativity", "Cosmology",
        "Black Holes", "Gravitational Waves",
    ],
    "Nuclear Physics": [
        "Nuclear Structure", "Radioactive Decay", "Nuclear Reactions",
        "Fission", "Fusion", "Particle Physics",
    ],
}

INDUSTRY_MAP: dict[str, list[str]] = {
    "Quantum Computing": ["Computing Hardware", "Cryptography", "Finance", "AI/ML"],
    "Semiconductor": ["Electronics", "Computing", "Telecommunications", "Solar Energy"],
    "Energy": ["Nuclear Power", "Solar Energy", "Battery Technology", "Fusion Research"],
    "Healthcare": ["Medical Imaging", "Radiation Therapy", "Drug Discovery", "Diagnostics"],
    "Materials Science": ["Nanotechnology", "Metallurgy", "Polymers", "Smart Materials"],
    "Telecommunications": ["Fiber Optics", "Wireless", "Satellite", "5G/6G"],
    "Aerospace": ["Satellite", "Propulsion", "Navigation", "Space Exploration"],
    "Defense": ["Radar", "Stealth", "Nuclear", "Directed Energy"],
    "Research": ["Particle Accelerators", "Telescopes", "Laboratories", "Universities"],
    "Transportation": ["Electric Vehicles", "Maglev", "Aviation", "Maritime"],
    "Manufacturing": ["Robotics", "Automation", "3D Printing", "Quality Control"],
    "Environmental": ["Climate Modeling", "Pollution Control", "Renewable Energy"],
}

# ─── Agent Implementation ─────────────────────────────────────────────────────


class FormulaClassifier(BaseAgent):
    """Classifies and tags formulas with rich metadata.

    Works on any list of discoveries (from other agents or corpus).
    Adds: domain, subdomain, tags, difficulty, applications, industries.

    Args:
        corpus_path: Corpus directory
        use_ollama: Use Ollama for intelligent tagging
        ollama_model: Ollama model
        openrouter_key: OpenRouter API key
    """

    def __init__(
        self,
        corpus_path: Path | None = None,
        use_ollama: bool = False,
        ollama_model: str = "qwen2.5:1.5b",
        openrouter_key: str | None = None,
    ):
        super().__init__("classifier", corpus_path)
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.openrouter_key = openrouter_key

    async def run(self, state) -> list[FormulaDiscovery]:
        """Classify all formulas from other agents' discoveries."""
        self.log("Classifying formulas...")
        discoveries = state.discoveries if state and hasattr(state, "discoveries") else []
        self.log(f"  Input: {len(discoveries)} formulas to classify")

        enriched: list[FormulaDiscovery] = []
        for discovery in discoveries:
            enrichment = self._classify_formula(
                discovery.formula,
                discovery.brief,
                discovery.domain,
            )
            # Store enrichment on the discovery
            discovery._enrichment = enrichment
            enriched.append(discovery)

            tags = enrichment.get("tags", [])
            apps = enrichment.get("applications", [])
            self.log(f"  🏷️ {discovery.formula[:40]}… → {enrichment.get('subdomain', '?')} [{', '.join(apps[:2])}]")

        self.log(f"Done: {len(enriched)} formulas classified")
        return enriched

    def _classify_formula(self, formula: str, brief: str, domain_hint: str) -> dict:
        """Classify a single formula with metadata.

        Uses regex + taxonomy matching (fast, no LLM needed for basic classification).
        For advanced tagging, Ollama/OpenRouter can be used.
        """
        formula_lower = formula.lower()
        brief_lower = brief.lower()

        # ── Determine domain ─────────────────────────────────────────────
        domain = self._match_domain(formula_lower, brief_lower, domain_hint)

        # ── Determine subdomain ──────────────────────────────────────────
        subdomain = self._match_subdomain(formula_lower, brief_lower, domain)

        # ── Extract tags ─────────────────────────────────────────────────
        tags = self._extract_tags(formula_lower, brief_lower)

        # ── Determine difficulty ─────────────────────────────────────────
        difficulty = self._estimate_difficulty(formula)

        # ── Find applications ────────────────────────────────────────────
        applications = self._find_applications(domain, subdomain, tags)

        # ── Find industries ──────────────────────────────────────────────
        industries = self._find_industries(domain, subdomain, applications)

        # ── Find related formulas ────────────────────────────────────────
        related = self._find_related(domain, subdomain)

        return {
            "domain": domain,
            "subdomain": subdomain,
            "tags": tags,
            "difficulty": difficulty,
            "applications": applications,
            "industries": industries,
            "related_formulas": related,
        }

    def _match_domain(self, formula: str, brief: str, hint: str) -> str:
        """Match formula to physics domain."""
        domain_markers = {
            "Quantum Field Theory": ["lagrangian", "feynman", "gauge", "yang-mills", "qed", "qcd", "brst", "ghost", "ward", "beta function", "renormalization", "path integral", "amplitude", "scattering matrix"],
            "Quantum Mechanics": ["schrodinger", "wavefunction", "psi", "eigenvalue", "hamiltonian", "commutator", "uncertainty", "tunneling", "de broglie", "planck"],
            "Quantum Computing": ["qubit", "quantum gate", "cnot", "hadamard", "bloch", "quantum circuit", "quantum algorithm", "quantum error"],
            "Quantum Information": ["entropy", "von neumann", "entanglement", "bell", "chsh", "ghz", "density matrix", "lindblad", "decoherence"],
            "Condensed Matter": ["superconduct", "bcs", "hubbard", "mott", "topological insulator", "quantum hall", "spin liquid", "band structure"],
            "Classical Mechanics": ["newton", "force", "mass", "acceleration", "momentum", "energy", "work", "power", "torque", "angular", "gravity", "projectile"],
            "Electromagnetism": ["maxwell", "electric field", "magnetic field", "gauss", "faraday", "ohm", "circuit", "capacitor", "inductor", "electromagnetic wave"],
            "Thermodynamics": ["entropy", "temperature", "heat", "carnot", "ideal gas", "PV=nRT", "adiabatic", "isothermal", "boltzmann"],
            "Relativity": ["einstein", "lorentz", "schwarzschild", "metric", "curvature", "ricci", "geodesic", "time dilation", "gravitational wave", "cosmology"],
            "Nuclear Physics": ["nuclear", "fission", "fusion", "radioactive", "decay", "half-life", "binding energy", "bethe", "weizsacker"],
        }

        for dom, markers in domain_markers.items():
            if any(m in formula or m in brief for m in markers):
                return dom

        # Check hint
        for dom in PHYSICS_TAXONOMY:
            if dom.lower() in hint.lower():
                return dom

        return "Physics"

    def _match_subdomain(self, formula: str, brief: str, domain: str) -> str:
        """Match to subdomain within the domain."""
        subs = PHYSICS_TAXONOMY.get(domain, ["General"])
        if len(subs) == 1:
            return subs[0]

        # Simple keyword matching
        for sub in subs:
            if sub.lower().replace("'", "").replace(" ", "") in (formula + brief).lower().replace(" ", ""):
                return sub

        return subs[0]  # Default to first

    def _extract_tags(self, formula: str, brief: str) -> list[str]:
        """Extract keyword tags."""
        combined = formula + " " + brief
        all_keywords = [
            "schrodinger", "heisenberg", "dirac", "feynman", "einstein",
            "newton", "maxwell", "faraday", "gauss", "ohm", "coulomb",
            "lagrangian", "hamiltonian", "lagrange", "hamilton",
            "wavefunction", "eigenvalue", "eigenfunction", "operator",
            "commutator", "anticommutator", "uncertainty", "tunneling",
            "superposition", "entanglement", "decoherence",
            "gauge", "symmetry", "conservation", "invariance",
            "field", "potential", "force", "energy", "momentum",
            "quantum", "classical", "relativistic", "non-relativistic",
            "linear", "nonlinear", "perturbative", "non-perturbative",
            "exact", "approximate", "numerical", "analytical",
        ]
        tags = []
        for kw in all_keywords:
            if kw in combined.lower():
                tags.append(kw)
        return tags[:10]

    def _estimate_difficulty(self, formula: str) -> str:
        """Estimate formula difficulty based on complexity markers."""
        complex_markers = [
            "mathcal", "mathfrak", "mathscr", "iiiint", "prod_",
            "otimes", "oplus", "hookrightarrow", "twoheadrightarrow",
        ]
        advanced_markers = [
            "sum_", "int_", "partial", "nabla", "frac{", "sqrt",
            "langle", "rangle", "infty",
        ]

        complex_count = sum(1 for m in complex_markers if m in formula)
        adv_count = sum(1 for m in advanced_markers if m in formula)

        if complex_count >= 5:
            return "PhD / Research"
        elif complex_count >= 2 or adv_count >= 8:
            return "Graduate"
        elif adv_count >= 4:
            return "Undergraduate"
        else:
            return "High School / Basic"

    def _find_applications(self, domain: str, subdomain: str, tags: list[str]) -> list[str]:
        """Find real-world applications."""
        application_map = {
            "Quantum Mechanics": [
                "Semiconductor device design", "Quantum chemistry simulation",
                "Medical imaging (MRI)", "Laser technology",
                "Atomic clocks", "Quantum sensors",
            ],
            "Quantum Field Theory": [
                "Particle physics predictions", "Standard Model precision tests",
                "Early universe cosmology", "Neutron star physics",
            ],
            "Quantum Computing": [
                "Drug discovery simulation", "Financial modeling",
                "Cryptography breaking", "Optimization problems",
                "Machine learning acceleration",
            ],
            "Quantum Information": [
                "Quantum key distribution", "Quantum random number generation",
                "Quantum teleportation", "Quantum error correction",
            ],
            "Condensed Matter": [
                "Superconducting magnets (MRI)", "Semiconductor chips",
                "Solar cells", "LED lighting", "Battery materials",
            ],
            "Classical Mechanics": [
                "Vehicle design", "Robotics", "Structural engineering",
                "Satellite orbits", "Sports science",
            ],
            "Electromagnetism": [
                "Wireless communication", "Electric power grid",
                "Radar systems", "Electric motors", "Generators",
            ],
            "Thermodynamics": [
                "Power plants", "Refrigeration", "Engine design",
                "Climate modeling", "Chemical processing",
            ],
            "Relativity": [
                "GPS satellite correction", "Gravitational wave detection",
                "Cosmology research", "Particle accelerator design",
            ],
            "Nuclear Physics": [
                "Nuclear power generation", "Medical isotopes",
                "Carbon dating", "Smoke detectors", "Food irradiation",
            ],
        }
        return application_map.get(domain, ["Scientific research"])[:5]

    def _find_industries(self, domain: str, subdomain: str, applications: list[str]) -> list[str]:
        """Map applications to industries."""
        industry_keywords = {
            "semiconductor": ["Electronics", "Computing Hardware"],
            "chip": ["Electronics", "Manufacturing"],
            "medical": ["Healthcare", "Medical Devices"],
            "imaging": ["Healthcare", "Medical Imaging"],
            "laser": ["Manufacturing", "Medical", "Telecommunications"],
            "cryptography": ["Cybersecurity", "Finance", "Defense"],
            "satellite": ["Aerospace", "Telecommunications"],
            "solar": ["Energy", "Renewable Energy"],
            "battery": ["Energy", "Electric Vehicles", "Electronics"],
            "nuclear": ["Energy", "Defense", "Healthcare"],
            "drug": ["Pharmaceuticals", "Healthcare"],
            "quantum": ["Computing Hardware", "Research", "Finance"],
            "wireless": ["Telecommunications", "IoT"],
            "radar": ["Defense", "Aerospace", "Automotive"],
            "motor": ["Manufacturing", "Transportation", "Robotics"],
            "engine": ["Automotive", "Aerospace", "Marine"],
            "power plant": ["Energy", "Utilities"],
            "gps": ["Aerospace", "Navigation", "Automotive"],
            "mri": ["Healthcare", "Medical Devices"],
            "superconduct": ["Energy", "Healthcare", "Research"],
        }

        industries = set()
        for app in applications:
            for kw, inds in industry_keywords.items():
                if kw in app.lower():
                    industries.update(inds)

        # Add domain-default industries
        domain_industries = INDUSTRY_MAP.get(domain, ["Research"])
        industries.update(domain_industries[:2])

        return list(industries)[:5]

    def _find_related(self, domain: str, subdomain: str) -> list[str]:
        """Suggest related formulas."""
        # Simplified cross-references
        relations = {
            ("Quantum Mechanics", "Wave Mechanics"): [
                "Planck-Einstein Relation (E = hf)",
                "De Broglie Wavelength (λ = h/p)",
                "Uncertainty Principle (ΔxΔp ≥ ħ/2)",
            ],
            ("Quantum Field Theory", "QED"): [
                "Dirac Equation",
                "Maxwell's Equations",
                "Ward-Takahashi Identity",
            ],
            ("Quantum Computing", "Qubit Theory"): [
                "Von Neumann Entropy",
                "Bell Inequality (CHSH)",
                "Lindblad Master Equation",
            ],
        }
        return relations.get((domain, subdomain), [])[:3]


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    async def main():
        classifier = FormulaClassifier()
        from formulagate.agents.graph import AgentState
        state = AgentState()
        state.discoveries = [
            FormulaDiscovery(
                agent="test", source="test",
                formula="i\\hbar\\frac{\\partial}{\\partial t}\\Psi = \\hat{H}\\Psi",
                brief="Quantum Mechanics: Time-Dependent Schrodinger Equation",
                domain="Quantum Mechanics",
            ),
            FormulaDiscovery(
                agent="test", source="test",
                formula="E = mc^2",
                brief="Relativity: Mass-Energy Equivalence",
                domain="Relativity",
            ),
        ]
        enriched = await classifier.run(state)
        for d in enriched:
            e = d._enrichment
            print(f"\n{e['domain']} > {e['subdomain']} [{e['difficulty']}]")
            print(f"  Tags: {', '.join(e['tags'][:5])}")
            print(f"  Applications: {', '.join(e['applications'][:3])}")
            print(f"  Industries: {', '.join(e['industries'][:3])}")

    asyncio.run(main())