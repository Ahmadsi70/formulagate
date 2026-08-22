"""Download real full-text LaTeX sources for the arXiv benchmark corpus.

Why this exists:
  The existing benchmark reads abstracts, which quote equation *fragments*
  (``E_0``, ``p = 1``), not complete equations. The dimensional veto needs
  complete equations to compare two terms. The honest source of those is the
  paper's own LaTeX source, which arXiv serves at the ``e-print`` endpoint.

  This script is resumable: sources already on disk are skipped, so an
  interrupted run continues from where it stopped. Nothing here is synthesised
  — if arXiv serves no source, the paper is simply marked ``no_source``.

Usage:
    python scripts/fetch_arxiv_sources.py --limit 60 --resume
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FULLTEXT = ROOT / "data" / "real" / "fulltext"
USER_AGENT = "Formulagate/0.7 (benchmark builder; contact: repository owner)"
REQUEST_PAUSE_SEC = 3.0
_TEX_SUFFIXES = {".tex", ".ltx", ".bibtex"}


def _is_tex(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(".tex") or lower.endswith(".ltx")


def _extract_sources(payload: bytes, dest: Path) -> list[str]:
    """Write every LaTeX source file from an e-print payload to ``dest``.

    Returns the list of written file names. Handles single .tex uploads,
    gzipped .tex, and tar/gzip/tar.gz archives.
    """
    written: list[str] = []
    dest.mkdir(parents=True, exist_ok=True)

    try:
        tar = tarfile.open(fileobj=__import__("io").BytesIO(payload), mode="r:*")
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if not _is_tex(member.name):
                continue
            base = os.path.basename(member.name)
            if not base:
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            content = f.read()
            (dest / base).write_bytes(content)
            written.append(base)
        return written
    except tarfile.TarError:
        pass

    # Single-file submission: plain .tex or gzipped .tex.
    for candidate in (payload,):
        try:
            import gzip

            try:
                decompressed = gzip.decompress(payload)
                if len(decompressed) >= len(payload):
                    candidate = decompressed
            except Exception:
                candidate = payload
        except Exception:
            candidate = payload

        text = candidate.decode("utf-8", errors="replace")
        # arXiv single-file .tex sources often start with a comment or
        # a documentclass line.
        if "documentclass" in text[:2000] or text.startswith("%"):
            (dest / "main.tex").write_bytes(candidate)
            written.append("main.tex")
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=ROOT / "data" / "real" / "arxiv_corpus.json")
    ap.add_argument("--limit", type=int, help="Cap how many papers to attempt")
    ap.add_argument(
        "--every", type=int, default=1,
        help="Process every Nth paper (spread the sample across categories)",
    )
    ap.add_argument("--resume", action="store_true", help="Skip papers already on disk")
    ap.add_argument("--pause", type=float, default=REQUEST_PAUSE_SEC)
    args = ap.parse_args()

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    manifest_path = FULLTEXT / "manifest.json"

    manifest: dict[str, Any] = {}
    if args.resume and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    arxiv_ids = [r["id"].replace("arxiv:", "") for r in corpus]
    selected = arxiv_ids[:: args.every][: args.limit]

    ok = skipped = no_source = failed = 0
    for i, arxiv_id in enumerate(selected):
        entry = manifest.get(arxiv_id)
        if args.resume and entry and entry.get("status") in ("ok", "no_source", "pdf_only"):
            skipped += 1
            continue
        if i:
            time.sleep(args.pause)
        dest = FULLTEXT / arxiv_id
        try:
            req = urllib.request.Request(
                f"https://export.arxiv.org/e-print/{arxiv_id}",
                headers={"User-Agent": USER_AGENT},
            )
            payload = urllib.request.urlopen(req, timeout=120).read()
            written = _extract_sources(payload, dest)
            if written:
                manifest[arxiv_id] = {"status": "ok", "files": written}
                ok += 1
            else:
                # A tar that only contained PDFs, or a binary blob: no LaTeX.
                manifest[arxiv_id] = {"status": "pdf_only", "files": []}
                no_source += 1
        except Exception as exc:
            manifest[arxiv_id] = {"status": "failed", "error": repr(exc)[:200]}
            failed += 1
        if (i + 1) % 25 == 0 or i == len(selected) - 1:
            print(
                f"  {i + 1}/{len(selected)} ok={ok} no_source={no_source} "
                f"failed={failed} skipped={skipped}",
                flush=True,
            )
        FULLTEXT.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"attempted": len(selected), "ok": ok, "no_source": no_source,
                      "failed": failed, "skipped": skipped}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
