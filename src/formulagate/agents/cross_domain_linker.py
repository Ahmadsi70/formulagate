"""Agent 7: Cross-Domain Problem Solver.

Connects physics formulas to real-world problems across any domain.
Given a problem description, finds relevant physics formulas that could solve it.

Key capabilities:
- Problem → Formula matching (semantic search across corpus)
- Cross-domain innovation (applying quantum formulas to finance, etc.)
- "What if" scenario generation
- Technology readiness assessment

Example queries:
    "How can we reduce computer chip heat at 3nm?"
    → Landauer formula + quantum thermal transport

    "Better battery materials?"
    → Density Functional Theory + Butler-Volmer equation

    "Faster drug discovery?"
    → Schrödinger equation + VQE quantum algorithm
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from formulagate.agents.graph import BaseAgent, FormulaDiscovery

logger = logging.getLogger(__name__)

# ─── Pre-built cross-domain knowledge ─────────────────────────────────────────

CROSS_DOMAIN_SOLUTIONS: list[dict] = [
    # Quantum → Computing
    {
        "problem": "Computer chips overheating",
        "problem_domain": "Computing / Hardware",
        "formula": "Quantum thermal transport (Landauer)",
        "formula_latex": "Q = \\frac{\\pi^2 k_B^2}{6\\hbar}T\\Delta T",
        "physics_domain": "Quantum Mechanics",
        "explanation": "At nanoscale, heat transport becomes quantized. Landauer formula predicts maximum heat conductance per quantum channel, enabling design of thermal-aware 3D chip architectures.",
        "feasibility": "High — already used in IBM/Intel 3nm research",
    },
    # Condensed Matter → Energy
    {
        "problem": "Energy grid instability with renewables",
        "problem_domain": "Energy / Grid",
        "formula": "Superconducting magnetic energy storage (SMES)",
        "formula_latex": "E = \\frac{1}{2}LI^2 \\text{ (stored in superconducting coil)}",
        "physics_domain": "Condensed Matter / Superconductivity",
        "explanation": "High-Tc superconductors enable lossless energy storage for grid balancing. BCS theory predicts critical temperatures for new materials.",
        "feasibility": "Medium — requires better high-Tc materials",
    },
    # Optics → Agriculture
    {
        "problem": "Crop disease detection too slow",
        "problem_domain": "Agriculture",
        "formula": "Hyperspectral imaging + Beer-Lambert law",
        "formula_latex": "I(\\lambda) = I_0(\\lambda) e^{-\\sum_i \\alpha_i(\\lambda) c_i L}",
        "physics_domain": "Optics / Spectroscopy",
        "explanation": "Each plant disease has a unique spectral signature. Drones with hyperspectral cameras scan fields, detecting disease 2 weeks before visible symptoms.",
        "feasibility": "High — John Deere and DJI already deploying",
    },
    # Fluid Dynamics → Healthcare
    {
        "problem": "Aneurysm rupture risk assessment",
        "problem_domain": "Healthcare",
        "formula": "Navier-Stokes + Wall Shear Stress",
        "formula_latex": "\\tau_w = \\mu\\left.\\frac{\\partial u}{\\partial y}\\right|_{y=0}",
        "physics_domain": "Fluid Dynamics",
        "explanation": "CFD simulation of blood flow through aneurysms predicts rupture risk based on wall shear stress, avoiding unnecessary surgery.",
        "feasibility": "High — FDA-approved software exists",
    },
    # Quantum → Finance
    {
        "problem": "Portfolio optimization too slow",
        "problem_domain": "Finance",
        "formula": "Quantum Approximate Optimization Algorithm (QAOA)",
        "formula_latex": "|\\psi(\\gamma,\\beta)\\rangle = \\prod_{p=1}^P U_B(\\beta_p)U_C(\\gamma_p)|+\\rangle^{\\otimes n}",
        "physics_domain": "Quantum Computing",
        "explanation": "QAOA maps portfolio optimization to a quantum circuit. For 1000+ assets, exponential speedup over classical solvers.",
        "feasibility": "Emerging — requires 100+ logical qubits",
    },
    # Statistical Mechanics → AI
    {
        "problem": "Neural networks are black boxes",
        "problem_domain": "Artificial Intelligence",
        "formula": "Boltzmann distribution + Energy-based models",
        "formula_latex": "P(x) = \\frac{e^{-E(x)/k_BT}}{Z}, \\quad Z = \\sum_x e^{-E(x)/k_BT}",
        "physics_domain": "Statistical Mechanics",
        "explanation": "Treating neural networks as energy-based models provides interpretable probability landscapes. Explains why deep networks generalize well.",
        "feasibility": "Medium — active research at MIT/Stanford",
    },
    # Acoustics → Construction
    {
        "problem": "Building collapse in earthquakes",
        "problem_domain": "Construction / Civil Engineering",
        "formula": "Wave equation + Resonance analysis",
        "formula_latex": "\\frac{\\partial^2 u}{\\partial t^2} = c^2\\nabla^2 u",
        "physics_domain": "Acoustics / Wave Mechanics",
        "explanation": "Buildings have natural frequencies. Tuned mass dampers (giant pendulums) absorb seismic energy through destructive interference.",
        "feasibility": "High — Taipei 101 uses 660-ton damper",
    },
    # Optics → Displays
    {
        "problem": "VR headset eye strain and bulkiness",
        "problem_domain": "Consumer Electronics",
        "formula": "Metalens (subwavelength optics)",
        "formula_latex": "\\phi(r) = -\\frac{2\\pi}{\\lambda}\\left(\\sqrt{r^2 + f^2} - f\\right)",
        "physics_domain": "Optics / Nanophotonics",
        "explanation": "Metalenses replace thick glass lenses with nanostructured surfaces 1000x thinner. Enables glasses-thin VR/AR headsets.",
        "feasibility": "High — Meta and Apple developing now",
    },
    # Nuclear → Medicine
    {
        "problem": "Cancer treatment damages healthy cells",
        "problem_domain": "Healthcare / Oncology",
        "formula": "Targeted Alpha Therapy (TAT)",
        "formula_latex": "D = \\int_0^\\infty \\dot{D}(t) \\, dt = \\frac{A_0}{\\lambda}(1-e^{-\\lambda T})",
        "physics_domain": "Nuclear Physics",
        "explanation": "Alpha particles deposit energy over 50-100μm (a few cell diameters). Attach alpha emitters to antibodies for cancer-cell-specific killing.",
        "feasibility": "High — FDA-approved Xofigo for prostate cancer",
    },
]

# ─── Agent Implementation ─────────────────────────────────────────────────────


class CrossDomainLinker(BaseAgent):
    """Links physics formulas to problems in ANY domain.

    Searches across the knowledge base and discovered formulas to find
    physics solutions for real-world problems.

    Args:
        corpus_path: Corpus directory
    """

    def __init__(
        self,
        corpus_path: Path | None = None,
    ):
        super().__init__("crosslinker", corpus_path)

    async def run(self, state) -> list[FormulaDiscovery]:
        """Find cross-domain physics solutions."""
        self.log("Linking physics formulas to industry problems...")
        discoveries: list[FormulaDiscovery] = []

        # ── 1. Pre-built cross-domain knowledge ──────────────────────────
        self.log(f"  Cross-domain KB: {len(CROSS_DOMAIN_SOLUTIONS)} solutions")

        for entry in CROSS_DOMAIN_SOLUTIONS:
            ok, detail = self.validate_with_gate(
                f"{entry['physics_domain']} → {entry['problem_domain']}: {entry['problem']}",
                entry["formula_latex"],
            )

            discovery = FormulaDiscovery(
                agent="crosslinker",
                source=f"Cross-Domain: {entry['physics_domain']} → {entry['problem_domain']}",
                brief=f"Problem: {entry['problem']} | Solution: {entry['explanation'][:200]}",
                formula=entry["formula_latex"],
                domain=f"{entry['physics_domain']} → {entry['problem_domain']}",
                confidence=0.85,
                gate_passed=ok,
                gate_detail=detail,
            )
            discovery._enrichment = {
                "problem": entry["problem"],
                "problem_domain": entry["problem_domain"],
                "physics_domain": entry["physics_domain"],
                "explanation": entry["explanation"],
                "feasibility": entry["feasibility"],
                "type": "cross_domain",
            }
            discoveries.append(discovery)
            self.log(f"  {'✅' if ok else '❌'} {entry['problem_domain']}: {entry['problem'][:50]}…")

        # ── 2. Semantic search: match discovered formulas to problems ────
        if state and hasattr(state, "discoveries"):
            matched = self._match_formulas_to_problems(state.discoveries)
            discoveries.extend(matched)
            self.log(f"  Semantic matches: {len(matched)}")

        self.log(f"Done: {len(discoveries)} cross-domain solutions")
        return discoveries

    def _match_formulas_to_problems(self, discoveries: list[FormulaDiscovery]) -> list[FormulaDiscovery]:
        """Match discovered formulas to industry problems via keyword mapping."""
        # Problem keywords → Formula domain mapping
        problem_to_formula_map = {
            "heat": ["Thermodynamics", "Quantum Mechanics"],
            "cool": ["Thermodynamics", "Condensed Matter"],
            "chip": ["Quantum Mechanics", "Condensed Matter", "Semiconductor"],
            "battery": ["Thermodynamics", "Electromagnetism", "Condensed Matter"],
            "cancer": ["Nuclear Physics", "Optics", "Quantum Mechanics"],
            "drug": ["Quantum Mechanics", "Quantum Computing"],
            "energy": ["Thermodynamics", "Nuclear Physics", "Electromagnetism"],
            "solar": ["Electromagnetism", "Condensed Matter", "Quantum Mechanics"],
            "fusion": ["Nuclear Physics", "Plasma Physics", "Electromagnetism"],
            "satellite": ["Classical Mechanics", "Relativity", "Electromagnetism"],
            "gps": ["Relativity", "Electromagnetism"],
            "encrypt": ["Quantum Computing", "Quantum Information"],
            "security": ["Quantum Information", "Quantum Computing"],
            "climate": ["Thermodynamics", "Fluid Dynamics"],
            "water": ["Fluid Dynamics", "Thermodynamics"],
            "robot": ["Classical Mechanics", "Electromagnetism"],
            "plane": ["Fluid Dynamics", "Classical Mechanics"],
            "rocket": ["Classical Mechanics", "Thermodynamics", "Nuclear Physics"],
            "car": ["Classical Mechanics", "Electromagnetism", "Thermodynamics"],
            "mri": ["Electromagnetism", "Quantum Mechanics"],
            "laser": ["Optics", "Quantum Mechanics"],
            "fiber": ["Optics", "Electromagnetism"],
            "wireless": ["Electromagnetism"],
            "radar": ["Electromagnetism", "Optics"],
            "steel": ["Condensed Matter", "Classical Mechanics"],
            "concrete": ["Classical Mechanics"],
            "bridge": ["Classical Mechanics"],
            "earthquake": ["Acoustics", "Classical Mechanics"],
        }

        results = []
        for d in discoveries:
            for keyword, domains in problem_to_formula_map.items():
                if d.domain in domains:
                    results.append(FormulaDiscovery(
                        agent="crosslinker_match",
                        source=f"Match: {d.source}",
                        brief=f"[{keyword}] Cross-domain application of: {d.brief}",
                        formula=d.formula,
                        domain=f"{d.domain} → {keyword.title()} Industry",
                        confidence=d.confidence * 0.6,
                        gate_passed=d.gate_passed,
                    ))
                    break  # One match per formula

        return results

    def search_by_problem(self, problem_text: str) -> list[dict]:
        """Search for physics formulas that could solve a given problem.

        Args:
            problem_text: Description of the problem to solve

        Returns:
            List of relevant cross-domain solutions
        """
        problem_lower = problem_text.lower()
        results = []

        for entry in CROSS_DOMAIN_SOLUTIONS:
            # Simple keyword overlap scoring
            problem_words = set(re.findall(r'[a-z]{4,}', problem_lower))
            entry_text = (entry["problem"] + " " + entry["explanation"]).lower()
            entry_words = set(re.findall(r'[a-z]{4,}', entry_text))
            overlap = len(problem_words & entry_words)

            if overlap > 0:
                results.append({
                    **entry,
                    "relevance_score": overlap,
                })

        results.sort(key=lambda r: -r["relevance_score"])
        return results[:5]


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    async def main():
        from formulagate.agents.graph import AgentState
        linker = CrossDomainLinker()
        state = AgentState()
        state.discoveries = [
            FormulaDiscovery(agent="test", source="test",
                           formula="iħ∂ψ/∂t = Ĥψ", brief="Schrödinger equation",
                           domain="Quantum Mechanics"),
        ]
        discoveries = await linker.run(state)
        for d in discoveries:
            e = getattr(d, "_enrichment", {})
            print(f"  [{e.get('problem_domain', '?')}] {e.get('problem', d.brief)[:80]}")

        # Test search
        print("\nSearching 'chip overheating'...")
        results = linker.search_by_problem("How to fix chip overheating at 3nm scale")
        for r in results:
            print(f"  Score={r['relevance_score']}: {r['problem_domain']} — {r['explanation'][:80]}")

    asyncio.run(main())