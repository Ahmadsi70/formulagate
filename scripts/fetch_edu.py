"""Phase 16: Download STEM education papers from arXiv.

Downloads physics.ed-ph, physics.pop-ph papers plus classic physics
textbooks that have equations suitable for student error detection.
"""

import json, os, sys, tarfile, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULLTEXT = ROOT / "data" / "real" / "fulltext"
USER_AGENT = "Formulagate/0.7 (benchmark; repo owner)"
REQUEST_PAUSE = 5.0
TIMEOUT = 120
ARXIV_API = "https://export.arxiv.org/api/query"

CATEGORIES = [
    ("physics.ed-ph", 20, "STEM Education"),
    ("physics.pop-ph", 10, "STEM Education"),
    ("physics.class-ph", 5, "STEM Education"),
    ("astro-ph.IM", 5, "STEM Education"),
]


def search(cat, n):
    params = urllib.parse.urlencode({
        "search_query": f"cat:{cat}", "start": 0, "max_results": n,
        "sortBy": "submittedDate", "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    time.sleep(REQUEST_PAUSE)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        data = resp.read().decode("utf-8")
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(data)
    papers = []
    for e in root.findall("atom:entry", ns):
        pid = e.find("atom:id", ns).text.strip().split("/")[-1]
        title = e.find("atom:title", ns).text.strip()
        if (FULLTEXT / pid).exists():
            continue
        papers.append({"id": pid, "title": title[:80]})
    return papers


def download(pid):
    dest = FULLTEXT / pid
    if dest.exists():
        return True
    url = f"https://arxiv.org/e-print/{pid}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    time.sleep(REQUEST_PAUSE)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            payload = resp.read()
    except Exception:
        return False
    dest.mkdir(parents=True, exist_ok=True)
    try:
        tar = tarfile.open(fileobj=__import__("io").BytesIO(payload), mode="r:*")
        w = 0
        for m in tar.getmembers():
            if not m.isfile():
                continue
            n = os.path.basename(m.name)
            if n.lower().endswith((".tex", ".ltx")):
                f = tar.extractfile(m)
                if f:
                    (dest / n).write_bytes(f.read())
                    w += 1
        if w == 0:
            (dest / "paper.tex").write_bytes(payload)
    except tarfile.ReadError:
        (dest / "paper.tex").write_bytes(payload)
    return True


def main():
    mp = FULLTEXT / "manifest.json"
    manifest = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    total = 0
    for cat, limit, audience in CATEGORIES:
        print(f"\n{'='*50}")
        print(f"  {cat} -> {audience}")
        print(f"{'='*50}")
        papers = search(cat, limit)
        for p in papers:
            ok = download(p["id"])
            if ok:
                total += 1
                manifest[p["id"]] = {"category": cat, "audience": audience, "status": "ok", "title": p["title"]}
            else:
                manifest[p["id"]] = {"category": cat, "audience": audience, "status": "download_failed", "title": p["title"]}
            print(f"  [{total}] {p['id']}: {'ok' if ok else 'FAIL'}")
    mp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTotal: {total}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
