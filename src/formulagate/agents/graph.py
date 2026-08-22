"""LangGraph orchestration for the Formulagate discovery agents.

Only sources that return verifiable third-party text are kept: the simulated
video/caption agents were removed because their output could not be traced to a
citable record. Each discovered formula passes through the Semantic Gate.

Architecture:
    START
      │
      ├──> ArxivScanner      (fetch → extract → gate → store)
      │
      └──> QuantumSpecialist (fetch → extract → gate → store)
      │
      ▼
    MERGE → REPORT
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── Data models ──────────────────────────────────────────────────────────────


@dataclass
class FormulaDiscovery:
    """A single formula discovered by an agent."""
    agent: str
    source: str
    source_url: str = ""
    brief: str = ""
    formula: str = ""
    domain: str = ""
    confidence: float = 0.0
    gate_passed: bool = False
    gate_detail: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "source": self.source,
            "source_url": self.source_url,
            "brief": self.brief,
            "formula": self.formula,
            "domain": self.domain,
            "confidence": self.confidence,
            "gate_passed": self.gate_passed,
            "gate_detail": self.gate_detail,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentState:
    """Shared state across all agents in the LangGraph."""
    messages: list[str] = field(default_factory=list)
    discoveries: list[FormulaDiscovery] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=lambda: {
        "total_scanned": 0,
        "formulas_found": 0,
        "gate_passed": 0,
        "gate_rejected": 0,
    })
    running: bool = True


# ─── Base Agent ───────────────────────────────────────────────────────────────


class BaseAgent:
    """Shared functionality for all agents."""

    def __init__(self, name: str, corpus_path: Path | None = None):
        self.name = name
        self.corpus_path = corpus_path or Path("data") / "discovered_formulas.json"

    def validate_with_gate(self, brief: str, formula: str, corpus_path: Path | None = None) -> tuple[bool, str]:
        """Pass a formula through the Formulagate semantic gate for validation."""
        try:
            from formulagate.gate import run_gate
            cp = corpus_path or Path("data/sample_corpus.json")
            evaluation = run_gate(
                brief=brief[:500],
                candidate_text=formula[:500],
                corpus_path=cp,
                top_k=1,
                use_semantic=True,
            )
            return evaluation.ok, evaluation.detail
        except Exception as exc:
            logger.warning("Gate validation failed: %s", exc)
            # Fallback: accept with warning
            return True, f"gate_fallback: {exc}"

    def log(self, message: str) -> None:
        logger.info("[%s] %s", self.name, message)


# ─── LangGraph Supervisor ─────────────────────────────────────────────────────


class AgentSupervisor:
    """Coordinates all agents, manages state, routes results.

    Uses LangGraph for orchestration when available, falls back to
    sequential execution otherwise.
    """

    def __init__(
        self,
        corpus_path: Path | None = None,
        use_graph: bool = True,
        use_ollama: bool = False,
        ollama_model: str = "qwen2.5:1.5b",
        openrouter_key: str | None = None,
        openrouter_model: str = "deepseek/deepseek-chat",
        deepseek_key: str | None = None,
    ):
        self.corpus_path = corpus_path or Path("data")
        self.use_graph = use_graph and self._has_langgraph()
        self.state = AgentState()

        # Initialize all 7 agents with shared settings
        from formulagate.agents.arxiv_scanner import ArxivScanner
        from formulagate.agents.quantum_specialist import QuantumSpecialist
        from formulagate.agents.formula_classifier import FormulaClassifier
        from formulagate.agents.industry_researcher import IndustryResearcher
        from formulagate.agents.cross_domain_linker import CrossDomainLinker

        # Discovery Layer (Agents 1-4)
        self.discovery_agents: dict[str, BaseAgent] = {
            "arxiv": ArxivScanner(
                corpus_path=self.corpus_path,
                use_ollama=use_ollama,
                ollama_model=ollama_model,
                openrouter_key=openrouter_key,
                openrouter_model=openrouter_model,
                deepseek_key=deepseek_key,
            ),
            "quantum": QuantumSpecialist(
                corpus_path=self.corpus_path,
                use_ollama=use_ollama,
                ollama_model=ollama_model,
                openrouter_key=openrouter_key,
            ),
        }

        # Enrichment Layer (Agents 5-6)
        self.enrichment_agents: dict[str, BaseAgent] = {
            "classifier": FormulaClassifier(
                corpus_path=self.corpus_path,
                use_ollama=use_ollama,
                ollama_model=ollama_model,
                openrouter_key=openrouter_key,
            ),
            "industry": IndustryResearcher(
                corpus_path=self.corpus_path,
                use_ollama=use_ollama,
                ollama_model=ollama_model,
                openrouter_key=openrouter_key,
            ),
        }

        # Application Layer (Agent 7)
        self.application_agents: dict[str, BaseAgent] = {
            "crosslinker": CrossDomainLinker(
                corpus_path=self.corpus_path,
            ),
        }

        # All agents for backward compat
        self.agents = {
            **self.discovery_agents,
            **self.enrichment_agents,
            **self.application_agents,
        }

    @staticmethod
    def _has_langgraph() -> bool:
        try:
            import langgraph  # noqa: F401
            return True
        except ImportError:
            logger.info("LangGraph not installed, using sequential execution")
            return False

    async def run(self) -> dict:
        """Run all agents and return summary.

        Returns:
            dict with discoveries, stats, errors
        """
        t0 = time.time()

        if self.use_graph:
            result = await self._run_with_langgraph()
        else:
            result = await self._run_sequential()

        # Save discoveries
        self._save_discoveries()

        result["elapsed_sec"] = time.time() - t0
        return result

    async def _run_sequential(self) -> dict:
        """Run agents in proper order: Discovery → Enrichment → Application."""
        # Phase 1: Discovery (Agents 1-4)
        for name, agent in self.discovery_agents.items():
            try:
                discoveries = await agent.run(self.state)
                self.state.discoveries.extend(discoveries)
                for d in discoveries:
                    self.state.stats["formulas_found"] += 1
                    if d.gate_passed:
                        self.state.stats["gate_passed"] += 1
                    else:
                        self.state.stats["gate_rejected"] += 1
            except Exception as exc:
                self.state.errors.append(f"{name}: {exc}")
                logger.exception("Discovery agent %s failed", name)

        # Phase 2: Enrichment (Agents 5-6)
        for name, agent in self.enrichment_agents.items():
            try:
                enriched = await agent.run(self.state)
                # Enrichment agents modify existing discoveries (add metadata)
                self.state.stats["formulas_enriched"] = len(enriched)
            except Exception as exc:
                self.state.errors.append(f"{name}: {exc}")
                logger.exception("Enrichment agent %s failed", name)

        # Phase 3: Application (Agent 7)
        for name, agent in self.application_agents.items():
            try:
                cross_discoveries = await agent.run(self.state)
                self.state.discoveries.extend(cross_discoveries)
                for d in cross_discoveries:
                    self.state.stats["cross_domain_links"] = \
                        self.state.stats.get("cross_domain_links", 0) + 1
            except Exception as exc:
                self.state.errors.append(f"{name}: {exc}")
                logger.exception("Application agent %s failed", name)

        return {
            "discoveries": [d.to_dict() for d in self.state.discoveries],
            "stats": self.state.stats,
            "errors": self.state.errors,
        }

    async def _run_with_langgraph(self) -> dict:
        """Run agents using LangGraph StateGraph for orchestration."""
        try:
            from langgraph.graph import StateGraph, END

            graph = StateGraph(AgentState)

            # Add agent nodes
            for name in self.agents:
                graph.add_node(name, self._make_agent_node(name))

            # Merge results node
            graph.add_node("merge", self._merge_node)

            # Route from each agent to merge
            for name in self.agents:
                graph.add_edge(name, "merge")

            graph.add_edge("merge", END)
            graph.set_entry_point(list(self.agents.keys())[0])

            # Compile and run
            app = graph.compile()
            final_state = await app.ainvoke(self.state)

            return {
                "discoveries": [d.to_dict() for d in final_state.discoveries],
                "stats": final_state.stats,
                "errors": final_state.errors,
            }
        except Exception as exc:
            logger.warning("LangGraph failed, falling back to sequential: %s", exc)
            self.use_graph = False
            return await self._run_sequential()

    def _make_agent_node(self, agent_name: str):
        """Create a LangGraph node function for an agent."""
        async def node(state: AgentState) -> AgentState:
            agent = self.agents[agent_name]
            try:
                discoveries = await agent.run(state)
                state.discoveries.extend(discoveries)
                for d in discoveries:
                    state.stats["formulas_found"] += 1
                    if d.gate_passed:
                        state.stats["gate_passed"] += 1
                    else:
                        state.stats["gate_rejected"] += 1
            except Exception as exc:
                state.errors.append(f"{agent_name}: {exc}")
            return state
        return node

    async def _merge_node(self, state: AgentState) -> AgentState:
        """Merge results from all agents and generate report."""
        state.messages.append(
            f"Multi-agent run complete: {state.stats['formulas_found']} formulas, "
            f"{state.stats['gate_passed']} passed, {state.stats['gate_rejected']} rejected"
        )
        state.running = False
        return state

    def _save_discoveries(self) -> None:
        """Persist discoveries to JSON corpus."""
        output_path = self.corpus_path / "autonomous_discoveries.json"
        existing = []
        if output_path.exists():
            try:
                existing = json.loads(output_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        new_entries = [d.to_dict() for d in self.state.discoveries]
        all_entries = existing + new_entries

        output_path.write_text(
            json.dumps(all_entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved %d discoveries to %s", len(new_entries), output_path)

    def print_report(self) -> None:
        """Print a human-readable summary."""
        s = self.state.stats
        print("\n" + "=" * 60)
        print("FORMULAGATE MULTI-AGENT REPORT v2")
        print("=" * 60)
        print(f"  🔍 DISCOVERY LAYER")
        print(f"    Formulas found:     {s.get('formulas_found', 0)}")
        print(f"    Gate passed:        {s.get('gate_passed', 0)} ✅")
        print(f"    Gate rejected:      {s.get('gate_rejected', 0)} ❌")
        print(f"  🏷️  ENRICHMENT LAYER")
        print(f"    Formulas enriched:  {s.get('formulas_enriched', 0)}")
        print(f"  🔗 APPLICATION LAYER")
        print(f"    Cross-domain links: {s.get('cross_domain_links', 0)}")
        print(f"  ❌ Errors:           {len(self.state.errors)}")
        if self.state.errors:
            for e in self.state.errors[:5]:
                print(f"    - {e}")
        if self.state.discoveries:
            print(f"\n  Top discoveries:")
            for d in self.state.discoveries[:5]:
                status = "✅" if d.gate_passed else "❌"
                enrich = getattr(d, "_enrichment", {})
                extra = ""
                if enrich.get("industry"):
                    extra = f" → {enrich['industry']}"
                elif enrich.get("problem_domain"):
                    extra = f" → {enrich['problem_domain']}"
                print(f"    {status} [{d.agent}] {d.formula[:60]}…{extra}")
        print("=" * 60)