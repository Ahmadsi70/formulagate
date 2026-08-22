"""Build a leak-free physics benchmark from live arXiv metadata.

Why this replaces the old golden sets:
  The previous benchmarks copied a record's own text into the query
  (``brief = row['english']``, ``draft = row['math_formula']``, gold = that same
  row), so retrieval was answered before it started. Here every field comes from
  a different human-written surface of a real paper:

    corpus record : abstract + the LaTeX spans it contains
    brief         : the paper TITLE (written by the authors, not the abstract)
    positive draft: a math-bearing sentence from that paper's abstract
    negative draft: a math-bearing sentence from a paper in a DIFFERENT arXiv
                    primary category

  Negatives are paired deterministically across categories and the retriever
  always sees the whole corpus — no per-case corpus slicing that pre-decides the
  answer. Nothing in this file is synthesised: if arXiv returns nothing, the
  benchmark is empty.

Usage:
    python scripts/fetch_arxiv_benchmark.py --per-category 60
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "real"

ARXIV_API = "http://export.arxiv.org/api/query"
USER_AGENT = "Formulagate/0.5 (benchmark builder; contact: repository owner)"
REQUEST_PAUSE_SEC = 3.0  # arXiv asks for >=3s between calls.

CATEGORIES = (
    "hep-th",
    "gr-qc",
    "quant-ph",
    "cond-mat.str-el",
    "math-ph",
    "nucl-th",
    "physics.optics",
    "astro-ph.CO",
)

_NS = {"atom": "http://www.w3.org/2005/Atom"}
_MATH_SPAN = re.compile(r"\$[^$]{4,}\$|\\\[[^\]]{4,}\\\]|\\\([^)]{4,}\\\)")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _fetch_category(category: str, want: int) -> list[dict[str, Any]]:
    """Pull the most recent submissions for one arXiv category."""

    params = urllib.parse.urlencode(
        {
            "search_query": f"cat:{category}",
            "start": 0,
            "max_results": want,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    request = urllib.request.Request(f"{ARXIV_API}?{params}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read().decode("utf-8")

    papers: list[dict[str, Any]] = []
    for entry in ET.fromstring(payload).findall("atom:entry", _NS):
        title_el = entry.find("atom:title", _NS)
        summary_el = entry.find("atom:summary", _NS)
        id_el = entry.find("atom:id", _NS)
        if title_el is None or summary_el is None or id_el is None:
            continue
        title = " ".join((title_el.text or "").split())
        abstract = " ".join((summary_el.text or "").split())
        arxiv_id = (id_el.text or "").rsplit("/", 1)[-1]
        if not title or len(abstract) < 200 or not arxiv_id:
            continue
        papers.append(
            {
                "id": f"arxiv:{arxiv_id}",
                "title": title,
                "abstract": abstract,
                "category": category,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
            }
        )
    return papers


def _math_spans(text: str) -> list[str]:
    return [m.group(0) for m in _MATH_SPAN.finditer(text)]


def _math_sentence(abstract: str) -> str | None:
    """First sentence that actually carries mathematics."""

    for sentence in _SENTENCE.split(abstract):
        if _MATH_SPAN.search(sentence) and len(sentence) >= 40:
            return sentence.strip()
    return None


def build(per_category: int) -> tuple[list[dict], list[dict], dict]:
    papers: list[dict[str, Any]] = []
    per_cat_kept: dict[str, int] = {}

    for i, category in enumerate(CATEGORIES):
        if i:
            time.sleep(REQUEST_PAUSE_SEC)
        # Over-fetch: only abstracts containing real LaTeX survive the filter.
        fetched = _fetch_category(category, want=per_category * 3)
        kept = [p for p in fetched if _math_spans(p["abstract"]) and _math_sentence(p["abstract"])]
        kept = kept[:per_category]
        per_cat_kept[category] = len(kept)
        papers.extend(kept)
        print(f"  {category}: fetched {len(fetched)} -> kept {len(kept)}", flush=True)

    corpus = [
        {
            "id": p["id"],
            "english": p["abstract"],
            "math_formula": " ".join(_math_spans(p["abstract"]))[:600],
            "scientific_domain": p["category"],
            "title": p["title"],
            "source": "arxiv",
            "url": p["url"],
        }
        for p in papers
    ]

    cases: list[dict[str, Any]] = []
    for p in papers:
        cases.append(
            {
                "brief": p["title"],
                "draft": _math_sentence(p["abstract"]),
                "expect_generate": True,
                "expected_top_id": p["id"],
                "case_kind": "grounded",
            }
        )

    # Deterministic cross-category pairing: paper i borrows the draft of the
    # paper half a list away, skipping same-category neighbours.
    n = len(papers)
    for i, p in enumerate(papers):
        for offset in range(n // 2, n):
            other = papers[(i + offset) % n]
            if other["category"] != p["category"]:
                cases.append(
                    {
                        "brief": p["title"],
                        "draft": _math_sentence(other["abstract"]),
                        "expect_generate": False,
                        "distractor_id": other["id"],
                        "case_kind": "cross_category_distractor",
                    }
                )
                break

    provenance = {
        "built_on": date.today().isoformat(),
        "source": "arXiv Atom API (live)",
        "categories": per_cat_kept,
        "papers": len(papers),
        "cases": len(cases),
        "grounded_cases": sum(1 for c in cases if c["expect_generate"]),
        "distractor_cases": sum(1 for c in cases if not c["expect_generate"]),
        "leak_controls": [
            "query is the paper title; corpus text is the abstract",
            "drafts are verbatim author sentences, never generated",
            "negatives come from a different arXiv primary category",
            "no per-case corpus filtering — retriever always sees every record",
        ],
    }
    return corpus, cases, provenance


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=60)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(CATEGORIES)} arXiv categories ...", flush=True)
    corpus, cases, provenance = build(args.per_category)
    if not corpus:
        print("arXiv returned no usable papers — nothing written.")
        return 1

    (OUT_DIR / "arxiv_corpus.json").write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "arxiv_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
