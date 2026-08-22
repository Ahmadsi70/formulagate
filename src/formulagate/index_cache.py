"""FAISS index cache — save/load pre-built FAISS indices to disk.

Why: DenseHybridRetriever rebuilds the FAISS index from scratch on every run.
For 5000+ record corpora, embedding takes 60-120 seconds. Caching eliminates
this overhead for repeated benchmarks.

Format:
- {path}.faiss       — FAISS index (faiss.write_index)
- {path}.ids.json    — corpus id list
- {path}.hash.txt    — corpus content hash (MD5 of JSON) for invalidation
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _corpus_hash(corpus: list[dict[str, Any]]) -> str:
    """Compute deterministic hash of corpus content for cache invalidation."""
    serialized = json.dumps(
        [{"id": r.get("id", ""), "mf": str(r.get("math_formula", ""))[:100]}
         for r in corpus],
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.md5(serialized.encode()).hexdigest()[:12]


def save_index(
    retriever: Any,
    path: Path,
    corpus: list[dict[str, Any]],
) -> None:
    """Save FAISS index + metadata to disk.

    Args:
        retriever: DenseHybridRetriever instance with built index
        path: Base path (will create path.faiss, path.ids.json, path.hash.txt)
        corpus: Original corpus for hash verification
    """
    import faiss

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss_path = path.with_suffix(".faiss")
    faiss.write_index(retriever._index, str(faiss_path))
    logger.info("Saved FAISS index: %s", faiss_path)

    # Save id map
    ids_path = path.with_suffix(".ids.json")
    ids_path.write_text(json.dumps(retriever._ids, ensure_ascii=True))
    logger.info("Saved id map: %s", ids_path)

    # Save corpus hash
    hash_path = path.with_suffix(".hash.txt")
    hash_path.write_text(_corpus_hash(corpus))
    logger.info("Saved corpus hash: %s", hash_path)


def load_index(
    path: Path,
    corpus: list[dict[str, Any]],
) -> tuple[Any, list[str]] | None:
    """Load FAISS index from disk if still valid.

    Returns (faiss_index, id_list) or None if cache is missing/stale.

    Args:
        path: Base path (reads path.faiss, path.ids.json, path.hash.txt)
        corpus: Current corpus to verify hash match

    Returns:
        Tuple of (faiss_index, id_list) or None
    """
    import faiss

    path = Path(path)
    faiss_path = path.with_suffix(".faiss")
    ids_path = path.with_suffix(".ids.json")
    hash_path = path.with_suffix(".hash.txt")

    if not faiss_path.exists() or not ids_path.exists():
        logger.info("FAISS cache miss: files don't exist")
        return None

    # Verify hash
    if hash_path.exists():
        stored_hash = hash_path.read_text().strip()
        current_hash = _corpus_hash(corpus)
        if stored_hash != current_hash:
            logger.info(
                "FAISS cache stale: hash %s != %s", stored_hash, current_hash
            )
            return None
    else:
        logger.info("FAISS cache: no hash file, assuming valid")

    # Load FAISS index
    try:
        index = faiss.read_index(str(faiss_path))
    except Exception as exc:
        logger.warning("FAISS cache read failed: %s", exc)
        return None

    # Load id map
    try:
        ids = json.loads(ids_path.read_text())
    except Exception as exc:
        logger.warning("FAISS id map read failed: %s", exc)
        return None

    logger.info("FAISS cache hit: %s (%d vectors)", faiss_path, index.ntotal)
    return index, ids


def clear_cache(path: Path) -> None:
    """Remove cached FAISS files."""
    for suffix in (".faiss", ".ids.json", ".hash.txt"):
        p = path.with_suffix(suffix)
        if p.exists():
            p.unlink()
            logger.info("Removed: %s", p)