"""Agent 4: Quantum Physics Specialist.

Specialized agent for discovering quantum physics formulas:
- Quantum Field Theory (QFT) — Lagrangians, Feynman rules, path integrals
- Quantum Mechanics (QM) — Schrödinger, Heisenberg, Dirac equations
- Quantum Electrodynamics (QED) — scattering amplitudes, Ward identities
- Quantum Chromodynamics (QCD) — asymptotic freedom, confinement
- Quantum Computing — qubit gates, entanglement, error correction
- Quantum Information — von Neumann entropy, Bell inequalities
- Condensed Matter — BCS theory, Hubbard model, topological order

Sources:
- arXiv quant-ph papers
- arXiv hep-th papers (QFT)
- arXiv cond-mat papers
- Built-in quantum formula encyclopedia

Each formula is tagged with subdomain, difficulty, and applications.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

from formulagate.agents.graph import BaseAgent, FormulaDiscovery

logger = logging.getLogger(__name__)

# ─── Quantum domains and their arXiv categories ──────────────────────────────

QUANTUM_DOMAINS = {
    "quant-ph": {
        "name": "Quantum Physics",
        "subdomains": ["Quantum Mechanics", "Quantum Information", "Quantum Computing",
                       "Quantum Optics", "Quantum Measurement", "Entanglement"],
    },
    "hep-th": {
        "name": "Quantum Field Theory",
        "subdomains": ["QFT", "QED", "QCD", "Electroweak", "Yang-Mills",
                       "Renormalization", "Path Integrals", "Supersymmetry"],
    },
    "cond-mat": {
        "name": "Condensed Matter",
        "subdomains": ["Superconductivity", "BCS Theory", "Hubbard Model",
                       "Topological Insulators", "Quantum Hall Effect",
                       "Spin Liquids", "Mott Insulators"],
    },
}

# ─── Built-in quantum formula encyclopedia ────────────────────────────────────

QUANTUM_FORMULA_ENCYCLOPEDIA: list[dict] = [
    # Quantum Mechanics
    {
        "formula": "i\\hbar\\frac{\\partial}{\\partial t}\\Psi = \\hat{H}\\Psi",
        "name": "Time-Dependent Schrödinger Equation",
        "domain": "Quantum Mechanics",
        "subdomain": "Wave Mechanics",
        "tags": ["schrodinger", "wavefunction", "hamiltonian", "time-evolution"],
        "applications": ["Quantum chemistry", "Solid-state physics", "Quantum computing"],
        "industries": ["Pharmaceuticals", "Materials Science", "Information Technology"],
    },
    {
        "formula": "\\hat{H}\\psi_n = E_n\\psi_n",
        "name": "Time-Independent Schrödinger Equation",
        "domain": "Quantum Mechanics",
        "subdomain": "Eigenvalue Problems",
        "tags": ["schrodinger", "eigenvalue", "stationary-state", "energy-level"],
        "applications": ["Atomic structure", "Molecular orbitals", "Band theory"],
        "industries": ["Chemical Engineering", "Electronics", "Energy"],
    },
    {
        "formula": "[\\hat{x}, \\hat{p}] = i\\hbar",
        "name": "Canonical Commutation Relation",
        "domain": "Quantum Mechanics",
        "subdomain": "Operator Algebra",
        "tags": ["commutation", "heisenberg", "uncertainty", "operator"],
        "applications": ["Quantum measurement theory", "Phase space quantization"],
        "industries": ["Precision Measurement", "Quantum Sensors"],
    },
    {
        "formula": "\\Delta x \\Delta p \\geq \\frac{\\hbar}{2}",
        "name": "Heisenberg Uncertainty Principle",
        "domain": "Quantum Mechanics",
        "subdomain": "Measurement Theory",
        "tags": ["heisenberg", "uncertainty", "measurement", "limit"],
        "applications": ["Quantum cryptography", "Quantum metrology", "Microscopy"],
        "industries": ["Security", "Defense", "Medical Imaging"],
    },
    # Quantum Field Theory
    {
        "formula": "\\mathcal{L}_{\\mathrm{QED}} = \\bar{\\psi}(i\\gamma^{\\mu}D_{\\mu} - m)\\psi - \\frac{1}{4}F_{\\mu\\nu}F^{\\mu\\nu}",
        "name": "QED Lagrangian Density",
        "domain": "Quantum Field Theory",
        "subdomain": "Quantum Electrodynamics",
        "tags": ["qed", "lagrangian", "dirac", "photon", "gauge-theory"],
        "applications": ["Particle physics predictions", "Precision tests of Standard Model"],
        "industries": ["Research", "Particle Accelerators"],
    },
    {
        "formula": "F_{\\mu\\nu}^a = \\partial_{\\mu}A_{\\nu}^a - \\partial_{\\nu}A_{\\mu}^a + g f^{abc}A_{\\mu}^b A_{\\nu}^c",
        "name": "Yang-Mills Field Strength Tensor",
        "domain": "Quantum Field Theory",
        "subdomain": "Yang-Mills Theory",
        "tags": ["yang-mills", "gauge", "non-abelian", "qcd"],
        "applications": ["Strong force theory", "Quark-gluon plasma"],
        "industries": ["Research", "Nuclear Physics"],
    },
    {
        "formula": "\\beta(g) = \\mu\\frac{\\partial g}{\\partial\\mu} = -b_0 g^3 - b_1 g^5 + \\mathcal{O}(g^7)",
        "name": "Beta Function (Renormalization Group)",
        "domain": "Quantum Field Theory",
        "subdomain": "Renormalization Group",
        "tags": ["beta-function", "renormalization", "running-coupling", "asymptotic-freedom"],
        "applications": ["QCD asymptotic freedom", "Critical phenomena", "Phase transitions"],
        "industries": ["Research", "Statistical Physics"],
    },
    # Quantum Computing
    {
        "formula": "|\\psi\\rangle = \\alpha|0\\rangle + \\beta|1\\rangle, \\quad |\\alpha|^2 + |\\beta|^2 = 1",
        "name": "Qubit Superposition State",
        "domain": "Quantum Computing",
        "subdomain": "Qubit Theory",
        "tags": ["qubit", "superposition", "quantum-computing", "bloch-sphere"],
        "applications": ["Quantum algorithms", "Quantum simulation", "Quantum cryptography"],
        "industries": ["Information Technology", "Finance", "Security"],
    },
    {
        "formula": "U_{\\mathrm{CNOT}} = |0\\rangle\\langle 0| \\otimes I + |1\\rangle\\langle 1| \\otimes X",
        "name": "CNOT Quantum Gate",
        "domain": "Quantum Computing",
        "subdomain": "Quantum Gates",
        "tags": ["cnot", "quantum-gate", "entanglement", "universal-gate"],
        "applications": ["Quantum error correction", "Quantum circuits", "Bell state generation"],
        "industries": ["Computing Hardware", "Cryptography"],
    },
    # Condensed Matter
    {
        "formula": "\\Delta_k = -\\sum_{k'} V_{kk'} \\frac{\\Delta_{k'}}{2E_{k'}} \\tanh\\left(\\frac{E_{k'}}{2k_B T}\\right)",
        "name": "BCS Gap Equation",
        "domain": "Condensed Matter",
        "subdomain": "Superconductivity",
        "tags": ["bcs", "superconductivity", "gap-equation", "pairing"],
        "applications": ["Superconducting magnets", "MRI machines", "Quantum computers"],
        "industries": ["Healthcare", "Energy", "Transportation"],
    },
    {
        "formula": "H = -t\\sum_{\\langle ij\\rangle,\\sigma}(c_{i\\sigma}^{\\dagger}c_{j\\sigma} + \\mathrm{h.c.}) + U\\sum_i n_{i\\uparrow}n_{i\\downarrow}",
        "name": "Hubbard Model Hamiltonian",
        "domain": "Condensed Matter",
        "subdomain": "Strongly Correlated Electrons",
        "tags": ["hubbard", "mott-insulator", "correlated-electrons", "high-tc"],
        "applications": ["High-Tc superconductors", "Magnetic materials", "Quantum simulation"],
        "industries": ["Electronics", "Energy", "Computing"],
    },
    # Quantum Information
    {
        "formula": "S = -\\mathrm{Tr}(\\rho\\ln\\rho)",
        "name": "Von Neumann Entropy",
        "domain": "Quantum Information",
        "subdomain": "Entanglement Measures",
        "tags": ["von-neumann", "entropy", "entanglement", "density-matrix"],
        "applications": ["Entanglement quantification", "Quantum thermodynamics", "Black hole physics"],
        "industries": ["Research", "Quantum Computing"],
    },
    {
        "formula": "S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')| \\leq 2",
        "name": "CHSH Bell Inequality",
        "domain": "Quantum Information",
        "subdomain": "Quantum Foundations",
        "tags": ["bell", "chsh", "nonlocality", "entanglement", "epr"],
        "applications": ["Device-independent quantum cryptography", "Quantum randomness certification"],
        "industries": ["Security", "Gaming", "Lottery"],
    },
    # Additional Quantum Formulas
    {
        "formula": "E = h\\nu = \\hbar\\omega",
        "name": "Planck-Einstein Relation",
        "domain": "Quantum Mechanics",
        "subdomain": "Quantum Foundations",
        "tags": ["planck", "photon", "energy", "frequency"],
        "applications": ["Photovoltaics", "LEDs", "Spectroscopy", "Laser technology"],
        "industries": ["Energy", "Lighting", "Medical", "Telecommunications"],
    },
    {
        "formula": "\\lambda = \\frac{h}{p} = \\frac{h}{mv}",
        "name": "De Broglie Wavelength",
        "domain": "Quantum Mechanics",
        "subdomain": "Wave-Particle Duality",
        "tags": ["de-broglie", "wavelength", "matter-wave", "electron-diffraction"],
        "applications": ["Electron microscopy", "Neutron diffraction", "Quantum transport"],
        "industries": ["Materials Science", "Semiconductor Industry"],
    },
    {
        "formula": "T \\approx e^{-2\\int_a^b \\sqrt{2m(V(x)-E)}/\\hbar \\, dx}",
        "name": "Quantum Tunneling Probability (WKB)",
        "domain": "Quantum Mechanics",
        "subdomain": "Tunneling Phenomena",
        "tags": ["tunneling", "wkb", "barrier", "scanning-tunneling"],
        "applications": ["Scanning tunneling microscopy", "Flash memory", "Nuclear fusion", "Tunnel diodes"],
        "industries": ["Electronics", "Energy", "Research Instruments"],
    },
    {
        "formula": "\\rho = \\sum_i p_i |\\psi_i\\rangle\\langle\\psi_i|",
        "name": "Density Matrix",
        "domain": "Quantum Mechanics",
        "subdomain": "Quantum Statistics",
        "tags": ["density-matrix", "mixed-state", "decoherence", "open-quantum-systems"],
        "applications": ["Quantum decoherence theory", "Quantum thermodynamics"],
        "industries": ["Quantum Computing", "Research"],
    },
    {
        "formula": "\\frac{d\\rho}{dt} = -\\frac{i}{\\hbar}[H,\\rho] + \\sum_k \\gamma_k\\mathcal{D}[L_k]\\rho",
        "name": "Lindblad Master Equation",
        "domain": "Quantum Information",
        "subdomain": "Open Quantum Systems",
        "tags": ["lindblad", "master-equation", "dissipation", "quantum-noise"],
        "applications": ["Quantum error correction", "Quantum control", "Laser cooling"],
        "industries": ["Quantum Computing", "Precision Measurement"],
    },
]

# ─── Agent Implementation ─────────────────────────────────────────────────────


class QuantumSpecialist(BaseAgent):
    """Specialized agent for quantum physics formula discovery.

    Searches arXiv quantum physics categories AND a built-in encyclopedia
    of quantum formulas with pre-tagged applications and industries.

    Args:
        corpus_path: Where to save discovered formulas
        max_results: Max papers per arXiv query
        use_ollama: Use Ollama for enrichment
        ollama_model: Ollama model name
        openrouter_key: OpenRouter API key for high-quality enrichment
    """

    def __init__(
        self,
        corpus_path: Path | None = None,
        max_results: int = 3,
        use_ollama: bool = False,
        ollama_model: str = "qwen2.5:1.5b",
        openrouter_key: str | None = None,
    ):
        super().__init__("quantum", corpus_path)
        self.max_results = max_results
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.openrouter_key = openrouter_key

    async def run(self, state) -> list[FormulaDiscovery]:
        """Main entry point."""
        self.log("Scanning quantum physics sources...")
        discoveries: list[FormulaDiscovery] = []

        # ── 1. Built-in encyclopedia (guaranteed high-quality) ────────────
        self.log(f"  Encyclopedia: {len(QUANTUM_FORMULA_ENCYCLOPEDIA)} quantum formulas")
        for entry in QUANTUM_FORMULA_ENCYCLOPEDIA:
            brief = f"{entry['domain']} > {entry['subdomain']}: {entry['name']}"
            ok, detail = self.validate_with_gate(brief, entry["formula"])

            discovery = FormulaDiscovery(
                agent="quantum",
                source=f"Quantum Encyclopedia: {entry['name']}",
                brief=brief,
                formula=entry["formula"],
                domain=entry["domain"],
                confidence=0.95,
                gate_passed=ok,
                gate_detail=detail,
            )
            # Attach enrichment metadata
            discovery._enrichment = {
                "tags": entry.get("tags", []),
                "subdomain": entry.get("subdomain", ""),
                "applications": entry.get("applications", []),
                "industries": entry.get("industries", []),
            }
            discoveries.append(discovery)
            self.log(f"  {'✅' if ok else '❌'} {entry['name']}")

        # ── 2. arXiv quantum papers ──────────────────────────────────────
        for cat, info in QUANTUM_DOMAINS.items():
            papers = self._fetch_arxiv_papers(cat, max_results=self.max_results)
            self.log(f"  arXiv {cat}: {len(papers)} papers")

            for paper in papers:
                formulas = self._extract_from_paper(paper)
                for f_data in formulas:
                    formula = f_data.get("formula", "")
                    name = f_data.get("name", "Quantum Formula")
                    domain = info["name"]

                    if not formula or len(formula) < 5:
                        continue

                    brief = f"{domain}: {name}"
                    ok, detail = self.validate_with_gate(brief, formula)

                    discovery = FormulaDiscovery(
                        agent="quantum",
                        source=paper.get("title", name),
                        source_url=paper.get("url", ""),
                        brief=brief,
                        formula=formula,
                        domain=domain,
                        confidence=f_data.get("confidence", 0.5),
                        gate_passed=ok,
                        gate_detail=detail,
                    )
                    discoveries.append(discovery)
                    self.log(f"    {'✅' if ok else '❌'} {formula[:50]}…")

        self.log(f"Done: {len(discoveries)} quantum formulas")
        return discoveries

    def _fetch_arxiv_papers(self, category: str, max_results: int = 3) -> list[dict]:
        """Query arXiv API."""
        query = f"cat:{category}"
        params = {
            "search_query": query,
            "start": "0",
            "max_results": str(max_results),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Formulagate/0.3"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read().decode("utf-8")

            root = ET.fromstring(data)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            papers = []
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                link = entry.find("atom:id", ns)
                if summary is not None and summary.text and len(summary.text) > 50:
                    papers.append({
                        "title": title.text.strip() if title is not None and title.text else "",
                        "summary": summary.text.strip(),
                        "url": link.text.strip() if link is not None and link.text else "",
                        "category": category,
                    })
            return papers
        except Exception as exc:
            logger.warning("arXiv %s: %s", category, exc)
            return []

    def _extract_from_paper(self, paper: dict) -> list[dict]:
        """Extract formulas from paper abstract."""
        text = f"{paper.get('title', '')}\n{paper.get('summary', '')}"

        # Try regex for LaTeX formulas first (fast)
        results = []
        latex_patterns = [
            r'\$\$(.+?)\$\$',
            r'\$(.+?)\$',
            r'\\\[(.+?)\\\]',
            r'\\begin\{equation\}(.+?)\\end\{equation\}',
        ]
        seen = set()
        for pat in latex_patterns:
            for match in re.findall(pat, text, re.DOTALL):
                formula = match.strip()
                if len(formula) > 10 and formula not in seen and any(
                    kw in formula.lower() for kw in
                    ["hbar", "psi", "hat", "mathcal", "sum", "int", "partial",
                     "delta", "gamma", "sigma", "lambda", "rho", "phi", "omega"]
                ):
                    seen.add(formula)
                    results.append({
                        "formula": formula[:500],
                        "name": "Extracted quantum formula",
                        "confidence": 0.5,
                    })
        return results[:3]


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    async def main():
        qs = QuantumSpecialist(max_results=1)
        discoveries = await qs.run(None)
        for d in discoveries:
            enrich = getattr(d, "_enrichment", {})
            apps = enrich.get("applications", [])
            print(f"  {d.formula[:60]}…  [{', '.join(apps[:2]) if apps else d.domain}]")

    asyncio.run(main())