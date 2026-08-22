"""Formulagate discovery agents.

Discovery Layer (citable sources only):
  1. ArxivScanner       — Extract formulas from arXiv papers
  2. QuantumSpecialist  — Quantum-physics-focused arXiv discovery

Enrichment Layer:
  3. FormulaClassifier  — Tag formulas with domain, subdomain, industries
  4. IndustryResearcher — Map formulas to real-world industry problems

Application Layer:
  5. CrossDomainLinker  — Find physics solutions for an industry problem

All formulas pass through the Formulagate Semantic Gate for validation.
Agents that produced simulated observations (synthetic trajectories, caption
placeholders) were removed: a formula corpus is only as trustworthy as the
provenance of its weakest record.

Usage:
  from formulagate.agents import AgentSupervisor
  supervisor = AgentSupervisor()
  results = await supervisor.run()
"""

from formulagate.agents.arxiv_scanner import ArxivScanner
from formulagate.agents.quantum_specialist import QuantumSpecialist
from formulagate.agents.formula_classifier import FormulaClassifier
from formulagate.agents.industry_researcher import IndustryResearcher
from formulagate.agents.cross_domain_linker import CrossDomainLinker
from formulagate.agents.graph import AgentSupervisor, AgentState, FormulaDiscovery, BaseAgent

__all__ = [
    "ArxivScanner",
    "QuantumSpecialist",
    "FormulaClassifier",
    "IndustryResearcher",
    "CrossDomainLinker",
    "AgentSupervisor",
    "AgentState",
    "FormulaDiscovery",
    "BaseAgent",
]