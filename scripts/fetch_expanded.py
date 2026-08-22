"""Fetch new arXiv papers for expanded benchmark.

Downloads papers from specific arXiv categories to fill audience gaps:
  - q-bio.QM → Pharmaceutical (quantitative biology / pharmacokinetics)
  - physics.flu-dyn → Oil & Gas / Aerospace
  - physics.class-ph → Civil Engineering
  - physics.ed-ph → STEM Education

Usage: python scripts/fetch_expanded.py --category q-bio.QM --limit 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULLTEXT = ROOT / "data" / "real" / "fulltext"
USER_AGENT = "Formulagate/0.7 (benchmark; contact: repo owner)"
REQUEST_PAUSE = 3.0
ARXIV_API = "https://export.arxiv.org/api/query"

CATEGORIES = {
    "q-bio.QM": "Pharmaceutical R&D",
    "q-bio.TO": "Pharmaceutical R&D",
    "physics.flu-dyn": "Oil & Gas / Aerospace",
    "physics.class-ph": "Civil Engineering",
    "physics.ed-ph": "STEM Education",
    "physics.med-ph": "Medical Physics",
}


def search_arxiv(category: str, max_results: int = 15) -> list[dict]:
    """Search arXiv API for papers in a category."""
    params = urllib.parse.urlencode({
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    time.sleep(REQUEST_PAUSE)

    with urllib.request.urlopen(req) as resp:
        data = resp.read().decode("utf-8")

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(data)
    papers = []
    for entry in root.findall("atom:entry", ns):
        paper_id = entry.find("atom:id", ns).text.strip()
        paper_id = paper_id.split("/")[-1]  # extract ID from URL
        title = entry.find("atom:title", ns).text.strip()
        # Skip papers already downloaded
        if (FULLTEXT / paper_id).exists():
            print(f"  SKIP {paper_id} (already exists)")
            continue
        papers.append({"id": paper_id, "title": title[:80]})
    return papers


def download_paper(paper_id: str) -> bool:
    """Download LaTeX source for one paper."""
    dest = FULLTEXT / paper_id
    if dest.exists():
        return True

    url = f"https://arxiv.org/e-print/{paper_id}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    time.sleep(REQUEST_PAUSE)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = resp.read()
    except Exception as e:
        print(f"  FAIL {paper_id}: {e}")
        return False

    dest.mkdir(parents=True, exist_ok=True)
    try:
        tar = tarfile.open(fileobj=__import__("io").BytesIO(payload), mode="r:*")
        written = 0
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            if name.lower().endswith((".tex", ".ltx")):
                f = tar.extractfile(member)
                if f:
                    (dest / name).write_bytes(f.read())
                    written += 1
        if written == 0:
            # Single .tex file upload
            (dest / "paper.tex").write_bytes(payload)
            written = 1
    except tarfile.ReadError:
        (dest / "paper.tex").write_bytes(payload)

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="q-bio.QM", help="arXiv category")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--all", action="store_true", help="Download all categories")
    args = ap.parse_args()

    categories = CATEGORIES if args.all else {args.category: CATEGORIES.get(args.category, args.category)}

    manifest_path = FULLTEXT / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for cat, audience in categories.items():
        print(f"\n{'='*60}")
        print(f"  {cat} → {audience}")
        print(f"{'='*60}")

        papers = search_arxiv(cat, args.limit)
        downloaded = 0
        for p in papers:
            pid = p["id"]
            status = "ok" if download_paper(pid) else "download_failed"
            if status == "ok":
                downloaded += 1
            manifest[pid] = {"category": cat, "audience": audience, "status": status, "title": p["title"]}
            print(f"  [{downloaded}/{len(papers)}] {pid}: {status}")

        print(f"\n  Downloaded: {downloaded}/{len(papers)}")

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nManifest updated: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
