"""Phase 15: Download pharmaceutical-related papers from arXiv + medrXiv.

Downloads q-bio papers (biomolecules, pharmacokinetics, toxicology) plus
medical physics papers.  Adds them to the existing fulltext corpus.

Usage: python scripts/fetch_pharma.py
"""

from __future__ import annotations

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
USER_AGENT = "Formulagate/0.7 (benchmark; repo owner)"
REQUEST_PAUSE = 5.0
ARXIV_TIMEOUT = 120
ARXIV_API = "https://export.arxiv.org/api/query"

CATEGORIES = [
    ("q-bio.QM", 15, "Pharmaceutical R&D"),
    ("q-bio.TO", 15, "Pharmaceutical R&D"),
    ("q-bio.BM", 10, "Pharmaceutical R&D"),
    ("physics.med-ph", 10, "Medical Physics"),
    ("q-bio.CB", 5, "Pharmaceutical R&D"),
]


def search_arxiv(category, max_results):
    params = urllib.parse.urlencode({
        "search_query": f"cat:{category}",
        "start": 0, "max_results": max_results,
        "sortBy": "submittedDate", "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    time.sleep(REQUEST_PAUSE)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read().decode("utf-8")
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(data)
    papers = []
    for entry in root.findall("atom:entry", ns):
        pid = entry.find("atom:id", ns).text.strip().split("/")[-1]
        title = entry.find("atom:title", ns).text.strip()
        if (FULLTEXT / pid).exists():
            continue
        papers.append({"id": pid, "title": title[:80]})
    return papers


def download_paper(paper_id):
    dest = FULLTEXT / paper_id
    if dest.exists():
        return True
    url = f"https://arxiv.org/e-print/{paper_id}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    time.sleep(REQUEST_PAUSE)
    try:
        with urllib.request.urlopen(req, timeout=ARXIV_TIMEOUT) as resp:
            payload = resp.read()
    except Exception as e:
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
            (dest / "paper.tex").write_bytes(payload)
    except tarfile.ReadError:
        (dest / "paper.tex").write_bytes(payload)
    return True


def main():
    manifest_path = FULLTEXT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    total_downloaded = 0
    for cat, limit, audience in CATEGORIES:
        print(f"\n{'='*60}")
        print(f"  {cat} → {audience} (limit={limit})")
        print(f"{'='*60}")
        papers = search_arxiv(cat, limit)
        for p in papers:
            pid = p["id"]
            ok = download_paper(pid)
            if ok:
                total_downloaded += 1
                manifest[pid] = {"category": cat, "audience": audience, "status": "ok", "title": p["title"]}
            else:
                manifest[pid] = {"category": cat, "audience": audience, "status": "download_failed", "title": p["title"]}
            print(f"  [{total_downloaded}] {pid}: {'ok' if ok else 'FAIL'}")

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTotal downloaded: {total_downloaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
