"""Agent 6: Industry Application Researcher.

For each formula, researches real-world applications across industries.
Uses a built-in knowledge base + LLM (Ollama/DeepSeek) for deep research.

Key functions:
- Map formulas to specific industry problems they solve
- Find companies/technologies that use each formula
- Identify cross-industry applications (e.g., quantum mechanics in finance)
- Generate "problem → formula → solution" chains

Industries covered:
    Computing, Healthcare, Energy, Aerospace, Defense,
    Telecommunications, Manufacturing, Transportation, Finance,
    Environmental, Materials Science, Construction, Agriculture
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

from formulagate.agents.graph import BaseAgent, FormulaDiscovery

logger = logging.getLogger(__name__)

# ─── Industry problem → formula solution knowledge base ───────────────────────

INDUSTRY_KNOWLEDGE_BASE: list[dict] = [
    # Computing
    {
        "industry": "Computing / IT",
        "problem": "Microchips overheating at 3nm scale",
        "formula": "Landauer formula for heat dissipation",
        "formula_latex": "Q = k_B T \\ln 2 \\text{ per bit erased}",
        "solution": "Use ballistic thermal transport models",
        "impact": "Enable continued Moore's Law scaling",
    },
    {
        "industry": "Computing / IT",
        "problem": "Classical encryption vulnerable to quantum attacks",
        "formula": "BB84 Quantum Key Distribution protocol",
        "formula_latex": "|\\psi\\rangle = \\frac{1}{\\sqrt{2}}(|0\\rangle + |1\\rangle)",
        "solution": "Quantum cryptography for unbreakable encryption",
        "impact": "Post-quantum security for banking and government",
    },
    # Healthcare
    {
        "industry": "Healthcare",
        "problem": "Protein folding takes years to compute classically",
        "formula": "Quantum variational eigensolver (VQE)",
        "formula_latex": "E(\\theta) = \\langle\\psi(\\theta)|H|\\psi(\\theta)\\rangle",
        "solution": "Quantum computers simulate molecular structures in hours",
        "impact": "Accelerate drug discovery by 1000x",
    },
    {
        "industry": "Healthcare",
        "problem": "Tumor detection requires better imaging contrast",
        "formula": "NMR spin relaxation (Bloch equations)",
        "formula_latex": "\\frac{d\\mathbf{M}}{dt} = \\gamma\\mathbf{M}\\times\\mathbf{B} - \\frac{M_x\\hat{i}+M_y\\hat{j}}{T_2} - \\frac{M_z-M_0}{T_1}\\hat{k}",
        "solution": "Advanced MRI pulse sequences for tissue differentiation",
        "impact": "Earlier cancer detection with 10x resolution",
    },
    {
        "industry": "Healthcare",
        "problem": "Radiation therapy damages healthy tissue",
        "formula": "Bragg peak (Bethe-Bloch formula)",
        "formula_latex": "-\\frac{dE}{dx} = \\frac{4\\pi n z^2}{m_e c^2\\beta^2}\\left(\\frac{e^2}{4\\pi\\epsilon_0}\\right)^2\\left[\\ln\\left(\\frac{2m_e c^2\\beta^2}{I(1-\\beta^2)}\\right)-\\beta^2\\right]",
        "solution": "Proton therapy deposits energy precisely at tumor depth",
        "impact": "90% reduction in collateral tissue damage",
    },
    # Energy
    {
        "industry": "Energy",
        "problem": "Fusion reactors can't sustain plasma long enough",
        "formula": "Lawson criterion for fusion ignition",
        "formula_latex": "n\\tau_E \\geq \\frac{12k_B T}{\\langle\\sigma v\\rangle E_{\\alpha}}",
        "solution": "Optimize plasma confinement (tokamak/stellarator design)",
        "impact": "Unlimited clean energy from seawater",
    },
    {
        "industry": "Energy",
        "problem": "Solar cells are inefficient at 25% conversion",
        "formula": "Shockley-Queisser limit",
        "formula_latex": "\\eta = \\frac{P_{max}}{P_{in}} \\leq 33.7\\% \\text{ (single junction)}",
        "solution": "Multi-junction cells with quantum dot layers",
        "impact": "Solar efficiency approaching 50%",
    },
    {
        "industry": "Energy",
        "problem": "Battery capacity degrades over charge cycles",
        "formula": "Butler-Volmer equation (electrochemistry)",
        "formula_latex": "j = j_0\\left[e^{\\alpha_a F\\eta/RT} - e^{-\\alpha_c F\\eta/RT}\\right]",
        "solution": "Design electrodes with optimal reaction kinetics",
        "impact": "EV batteries lasting 1M+ miles",
    },
    # Aerospace
    {
        "industry": "Aerospace",
        "problem": "Satellite clocks drift causing GPS errors",
        "formula": "General Relativity time dilation",
        "formula_latex": "\\Delta t = \\frac{\\Delta t_0}{\\sqrt{1 - 2GM/rc^2}}",
        "solution": "GPS satellites apply relativistic corrections",
        "impact": "GPS accuracy from 10km to 5m (30cm with military)",
    },
    {
        "industry": "Aerospace",
        "problem": "Spacecraft thermal management in vacuum",
        "formula": "Stefan-Boltzmann law for radiative cooling",
        "formula_latex": "P = \\epsilon\\sigma A T^4",
        "solution": "Optimize radiator panel design for heat rejection",
        "impact": "Long-duration deep space missions feasible",
    },
    # Telecommunications
    {
        "industry": "Telecommunications",
        "problem": "Fiber optic signal loss over long distances",
        "formula": "Beer-Lambert law for optical attenuation",
        "formula_latex": "I = I_0 e^{-\\alpha L}",
        "solution": "Develop ultra-low-loss optical fibers + amplifiers",
        "impact": "Transatlantic cables without repeaters",
    },
    {
        "industry": "Telecommunications",
        "problem": "5G/6G requires higher bandwidth density",
        "formula": "Shannon-Hartley theorem",
        "formula_latex": "C = B\\log_2(1 + S/N)",
        "solution": "Massive MIMO antenna arrays + beamforming",
        "impact": "100 Gbps mobile data rates",
    },
    # Finance
    {
        "industry": "Finance",
        "problem": "Stock market crashes are unpredictable",
        "formula": "Black-Scholes equation (diffusion PDE from physics)",
        "formula_latex": "\\frac{\\partial V}{\\partial t} + \\frac{1}{2}\\sigma^2 S^2\\frac{\\partial^2 V}{\\partial S^2} + rS\\frac{\\partial V}{\\partial S} - rV = 0",
        "solution": "Options pricing with physics-inspired stochastic calculus",
        "impact": "$600 trillion derivatives market enabled",
    },
    {
        "industry": "Finance",
        "problem": "Portfolio optimization is NP-hard",
        "formula": "Quantum Approximate Optimization Algorithm (QAOA)",
        "formula_latex": "|\\psi(\\gamma,\\beta)\\rangle = \\prod_{p=1}^P e^{-i\\beta_p H_M}e^{-i\\gamma_p H_C}|+\\rangle^{\\otimes n}",
        "solution": "Quantum computers find optimal portfolios exponentially faster",
        "impact": "Real-time risk-adjusted portfolio balancing",
    },
    # Environmental
    {
        "industry": "Environmental",
        "problem": "Climate models need higher resolution",
        "formula": "Navier-Stokes equations for atmospheric flow",
        "formula_latex": "\\rho\\left(\\frac{\\partial\\mathbf{v}}{\\partial t} + \\mathbf{v}\\cdot\\nabla\\mathbf{v}\\right) = -\\nabla p + \\mu\\nabla^2\\mathbf{v} + \\rho\\mathbf{g}",
        "solution": "Quantum-enhanced CFD for kilometer-scale climate grids",
        "impact": "10x more accurate hurricane and drought prediction",
    },
    {
        "industry": "Environmental",
        "problem": "CO2 capture materials are inefficient",
        "formula": "Density Functional Theory (Kohn-Sham equations)",
        "formula_latex": "\\left[-\\frac{\\hbar^2}{2m}\\nabla^2 + V_{\\mathrm{eff}}(\\mathbf{r})\\right]\\psi_i(\\mathbf{r}) = \\epsilon_i\\psi_i(\\mathbf{r})",
        "solution": "DFT simulations screen millions of MOF candidates",
        "impact": "Direct air capture at $50/ton CO2",
    },
]

# ─── Agent Implementation ─────────────────────────────────────────────────────


class IndustryResearcher(BaseAgent):
    """Researches industry applications for physics formulas.

    Uses a built-in knowledge base of 15+ industry problem→formula→solution
    chains, plus LLM for generating new connections.

    Args:
        corpus_path: Where to save results
        use_ollama: Use Ollama for LLM research
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
        super().__init__("industry", corpus_path)
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.openrouter_key = openrouter_key

    async def run(self, state) -> list[FormulaDiscovery]:
        """Research industry applications for discovered formulas."""
        self.log("Researching industry applications...")
        discoveries = state.discoveries if state and hasattr(state, "discoveries") else []
        self.log(f"  Input: {len(discoveries)} formulas to research")

        # ── 1. Built-in knowledge base ───────────────────────────────────
        kb_discoveries: list[FormulaDiscovery] = []
        self.log(f"  Knowledge base: {len(INDUSTRY_KNOWLEDGE_BASE)} industry problems")

        for entry in INDUSTRY_KNOWLEDGE_BASE:
            ok, detail = self.validate_with_gate(
                f"{entry['industry']}: {entry['problem']}",
                entry["formula_latex"],
            )

            discovery = FormulaDiscovery(
                agent="industry",
                source=f"Industry KB: {entry['industry']}",
                brief=f"Problem: {entry['problem']} → Solution: {entry['solution']}",
                formula=entry["formula_latex"],
                domain=entry["industry"],
                confidence=0.90,
                gate_passed=ok,
                gate_detail=detail,
            )
            discovery._enrichment = {
                "industry": entry["industry"],
                "problem": entry["problem"],
                "solution": entry["solution"],
                "impact": entry["impact"],
                "type": "industry_application",
            }
            kb_discoveries.append(discovery)
            self.log(f"  {'✅' if ok else '❌'} {entry['industry']}: {entry['problem'][:50]}…")

        # ── 2. Cross-reference with discovered formulas ──────────────────
        cross_discoveries = self._cross_reference(discoveries)
        self.log(f"  Cross-references: {len(cross_discoveries)} matches")

        all_discoveries = kb_discoveries + cross_discoveries
        self.log(f"Done: {len(all_discoveries)} industry applications")
        return all_discoveries

    def _cross_reference(self, discoveries: list[FormulaDiscovery]) -> list[FormulaDiscovery]:
        """Match discovered formulas to industry problems."""
        results = []

        domain_to_industry = {
            "Quantum Computing": "Computing / IT",
            "Quantum Information": "Cybersecurity",
            "Quantum Mechanics": "Semiconductor Industry",
            "Condensed Matter": "Materials Science",
            "Classical Mechanics": "Mechanical Engineering",
            "Electromagnetism": "Electrical Engineering",
            "Thermodynamics": "Energy",
            "Nuclear Physics": "Nuclear Energy",
            "Relativity": "Aerospace",
        }

        for d in discoveries:
            industry = domain_to_industry.get(d.domain, "Research")
            results.append(FormulaDiscovery(
                agent="industry_cross",
                source=f"Cross-ref: {d.source}",
                brief=f"[{industry}] Application of: {d.brief}",
                formula=d.formula,
                domain=industry,
                confidence=d.confidence * 0.7,
                gate_passed=d.gate_passed,
            ))

        return results

    def _research_with_llm(self, formula: str, domain: str) -> list[dict]:
        """Use Ollama/OpenRouter to find industry applications."""
        prompt = f"""You are a physicist turned industry consultant.
For this physics formula from {domain}:
{formula}

List 3 real-world applications across different industries.
For each, specify:
1. The industry
2. The specific problem it solves
3. How the formula provides the solution

Return as JSON list:
[{{"industry": "...", "problem": "...", "solution": "..."}}]"""

        if self.openrouter_key:
            return self._call_openrouter(prompt)
        elif self.use_ollama:
            return self._call_ollama(prompt)
        return []

    def _call_openrouter(self, prompt: str) -> list[dict]:
        try:
            payload = json.dumps({
                "model": "deepseek/deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 500,
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.openrouter_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return json.loads(data["choices"][0]["message"]["content"])
        except Exception:
            return []

    def _call_ollama(self, prompt: str) -> list[dict]:
        try:
            payload = json.dumps({
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3},
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return json.loads(data.get("response", "[]"))
        except Exception:
            return []


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    async def main():
        from formulagate.agents.graph import AgentState
        researcher = IndustryResearcher()
        state = AgentState()
        state.discoveries = [
            FormulaDiscovery(agent="test", source="test",
                           formula="E = mc^2", brief="Relativity: Mass-Energy",
                           domain="Relativity"),
        ]
        discoveries = await researcher.run(state)
        for d in discoveries:
            e = getattr(d, "_enrichment", {})
            print(f"  [{e.get('industry', d.domain)}] {e.get('problem', d.brief)[:80]}")

    asyncio.run(main())