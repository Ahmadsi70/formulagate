"""Agent 1: arXiv Formula Scanner.

Scans arXiv papers for physics formulas using the arXiv API and Ollama/LLM
for formula extraction. Each formula passes through Formulagate's gate.

Workflow (LangGraph-compatible):
    1. fetch_arxiv  — Query arXiv API for physics papers
    2. extract_text — Download paper abstract/text
    3. extract_formulas — Use Ollama/LLM to find LaTeX formulas
    4. validate_gate — Pass through Formulagate Semantic Gate
    5. store_result — Add to corpus

Supports:
    - Ollama (local, free) — qwen2.5, llama3
    - DeepSeek API (cloud, high quality) — when API key available
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

# ─── arXiv API ────────────────────────────────────────────────────────────────

ARXIV_API = "http://export.arxiv.org/api/query"

PHYSICS_CATEGORIES = [
    "hep-th",    # High Energy Physics - Theory
    "hep-ph",    # High Energy Physics - Phenomenology
    "gr-qc",     # General Relativity and Quantum Cosmology
    "quant-ph",  # Quantum Physics
    "cond-mat",  # Condensed Matter
    "astro-ph",  # Astrophysics
    "physics",   # General Physics
    "nucl-th",   # Nuclear Theory
    "hep-ex",    # High Energy Physics - Experiment
    "hep-lat",   # High Energy Physics - Lattice
]

# ─── Ollama / LLM prompts ─────────────────────────────────────────────────────

FORMULA_EXTRACTION_PROMPT = """You are a PhD physicist extracting formulas from a research paper.
Extract ALL LaTeX formulas from this text. For each formula:
1. Write the LaTeX inside $$...$$ 
2. Give a ONE-LINE description of what it represents
3. State the physics domain (e.g., Quantum Field Theory, General Relativity, etc.)

Return as JSON list:
[{"formula": "E = mc^2", "description": "Mass-energy equivalence", "domain": "Relativity"}]

Text to analyze:
{text}

Return ONLY valid JSON. No extra text."""


FORMULA_VALIDATION_PROMPT = """Validate this physics formula:

Formula: {formula}
Description: {description}
Domain: {domain}

Check:
1. Dimensional analysis — do units match?
2. Physical consistency — does it violate known conservation laws?
3. Mathematical correctness — proper LaTeX, well-formed expression?

Return JSON:
{{"valid": true/false, "confidence": 0.0-1.0, "issues": ["..."], "corrected_formula": "..."}}"""

# ─── Agent Implementation ─────────────────────────────────────────────────────


class ArxivScanner(BaseAgent):
    """Scans arXiv for physics formulas.

    Args:
        corpus_path: Where to save discovered formulas
        max_results: Max papers per query
        use_ollama: Use local Ollama (True) or cloud API (False)
        ollama_model: Ollama model name for extraction
        openrouter_key: OpenRouter API key (uses DeepSeek via OpenRouter)
        openrouter_model: Model on OpenRouter (default: deepseek/deepseek-chat)
        deepseek_key: Direct DeepSeek API key (legacy, prefer openrouter_key)
    """

    def __init__(
        self,
        corpus_path: Path | None = None,
        max_results: int = 5,
        use_ollama: bool = True,
        ollama_model: str = "qwen2.5:1.5b",
        openrouter_key: str | None = None,
        openrouter_model: str = "deepseek/deepseek-chat",
        deepseek_key: str | None = None,
    ):
        super().__init__("arxiv", corpus_path)
        self.max_results = max_results
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.openrouter_key = openrouter_key
        self.openrouter_model = openrouter_model
        self.deepseek_key = deepseek_key  # Legacy direct API

    async def run(self, state) -> list[FormulaDiscovery]:
        """Main entry point for LangGraph."""
        self.log(f"Scanning arXiv (max={self.max_results} papers)...")
        discoveries = []

        for category in PHYSICS_CATEGORIES[:3]:  # Top 3 categories for speed
            papers = self._fetch_arxiv_papers(category, max_results=self.max_results)
            self.log(f"  {category}: {len(papers)} papers found")

            for paper in papers:
                try:
                    formulas = self._extract_formulas(paper)
                    for f_data in formulas:
                        formula = f_data.get("formula", "")
                        desc = f_data.get("description", "")
                        domain = f_data.get("domain", "Physics")

                        if not formula or len(formula) < 5:
                            continue

                        # Validate through Formulagate Gate
                        brief = f"{domain}: {desc}"
                        ok, detail = self.validate_with_gate(brief, formula)

                        discovery = FormulaDiscovery(
                            agent="arxiv",
                            source=paper.get("title", "Unknown"),
                            source_url=paper.get("url", ""),
                            brief=brief,
                            formula=formula,
                            domain=domain,
                            confidence=f_data.get("confidence", 0.5),
                            gate_passed=ok,
                            gate_detail=detail,
                        )
                        discoveries.append(discovery)

                        status = "✅" if ok else "❌"
                        self.log(f"  {status} {formula[:60]}…")

                except Exception as exc:
                    logger.warning("arXiv paper extraction failed: %s", exc)
                    continue

        self.log(f"Done: {len(discoveries)} formulas discovered")
        return discoveries

    def _fetch_arxiv_papers(self, category: str, max_results: int = 5) -> list[dict]:
        """Query arXiv API for recent papers in a category."""
        query = f"cat:{category}"
        params = {
            "search_query": query,
            "start": "0",
            "max_results": str(max_results),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Formulagate/0.3"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read().decode("utf-8")

            root = ET.fromstring(data)
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }

            papers = []
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                link = entry.find("atom:id", ns)

                title_text = title.text.strip() if title is not None and title.text else ""
                summary_text = summary.text.strip() if summary is not None and summary.text else ""
                link_text = link.text.strip() if link is not None and link.text else ""

                # Skip if no useful content
                if len(summary_text) < 50:
                    continue

                papers.append({
                    "title": title_text,
                    "summary": summary_text,
                    "url": link_text,
                    "category": category,
                })

            return papers

        except Exception as exc:
            # No synthetic fallback: a failed fetch must yield zero papers, never
            # invented ones, or the corpus silently absorbs fabricated physics.
            logger.warning("arXiv API failed for %s: %s", category, exc)
            return []

    def _extract_formulas(self, paper: dict) -> list[dict]:
        """Extract LaTeX formulas from a paper using available LLM."""
        text = f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('summary', '')}"

        # Priority: OpenRouter → Ollama → Direct DeepSeek → Regex
        if self.openrouter_key:
            return self._extract_with_openrouter(text)
        elif self.use_ollama:
            return self._extract_with_ollama(text)
        elif self.deepseek_key:
            return self._extract_with_deepseek(text)
        else:
            return self._extract_regex(text)

    def _extract_with_ollama(self, text: str) -> list[dict]:
        """Use local Ollama for formula extraction."""
        prompt = FORMULA_EXTRACTION_PROMPT.format(text=text[:3000])

        try:
            payload = json.dumps({
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 1000},
            }).encode("utf-8")

            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            response = str(data.get("response", ""))

            # Try to parse JSON from response
            return self._parse_formula_json(response)

        except Exception as exc:
            logger.warning("Ollama extraction failed: %s", exc)
            return self._extract_regex(text)

    def _extract_with_openrouter(self, text: str) -> list[dict]:
        """Use OpenRouter (DeepSeek) for formula extraction."""
        prompt = FORMULA_EXTRACTION_PROMPT.format(text=text[:3000])

        try:
            payload = json.dumps({
                "model": self.openrouter_model,
                "messages": [
                    {"role": "system", "content": "You are a PhD physicist. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 1000,
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "HTTP-Referer": "https://github.com/formulagate",  # Optional: for rankings
                    "X-Title": "Formulagate Physics Discovery",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            response = data["choices"][0]["message"]["content"]
            logger.info("OpenRouter response: %s", response[:200])
            return self._parse_formula_json(response)

        except Exception as exc:
            logger.warning("OpenRouter extraction failed: %s", exc)
            return self._extract_regex(text)

    def _extract_with_deepseek(self, text: str) -> list[dict]:
        """Use DeepSeek API for high-quality formula extraction."""
        prompt = FORMULA_EXTRACTION_PROMPT.format(text=text[:3000])

        try:
            payload = json.dumps({
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1000,
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.deepseek.com/v1/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.deepseek_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            response = data["choices"][0]["message"]["content"]
            return self._parse_formula_json(response)

        except Exception as exc:
            logger.warning("DeepSeek extraction failed: %s", exc)
            return self._extract_regex(text)

    def _extract_regex(self, text: str) -> list[dict]:
        """Fallback: regex-based LaTeX formula extraction."""
        formulas = []

        # Match $$...$$ and $...$ patterns
        patterns = [
            r'\$\$(.+?)\$\$',
            r'\$(.+?)\$',
            r'\\\[(.+?)\\\]',
            r'\\\((.+?)\\\)',
            r'\\begin\{equation\}(.+?)\\end\{equation\}',
            r'\\begin\{align\}(.+?)\\end\{align\}',
        ]

        seen = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                formula = match.strip()
                if len(formula) > 5 and formula not in seen:
                    seen.add(formula)
                    formulas.append({
                        "formula": formula[:500],
                        "description": "Extracted from paper abstract",
                        "domain": "Physics",
                        "confidence": 0.3,
                    })

        return formulas[:5]

    def _parse_formula_json(self, response: str) -> list[dict]:
        """Parse JSON formula list from LLM response."""
        # Try direct JSON parse
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "formulas" in data:
                return data["formulas"]
        except json.JSONDecodeError:
            pass

        # Try to extract JSON array from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: regex formulas from response text
        return self._extract_regex(response)


# ─── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    async def main():
        scanner = ArxivScanner(max_results=2, use_ollama=False)
        discoveries = await scanner.run(None)
        for d in discoveries:
            print(f"  [{d.domain}] {d.formula[:80]}")

    asyncio.run(main())