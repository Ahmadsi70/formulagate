"""ML-based domain classifier using SentenceTransformer embeddings.

Why: Lexical markers fail on undergrad physics vocabulary (mass, charge, pipe,
planet, temperature) that doesn't appear in the 322 PhD-level PHYSICS_BRIEF_MARKERS.
Embedding similarity captures semantic relatedness beyond exact token match.

Design:
- Four prototype texts describe each domain at a high level
- Brief is embedded and compared to prototypes via cosine similarity
- Falls back to lexical markers when similarity is ambiguous
- Singleton pattern: model loaded once, reused across calls
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

Domain = Literal["mathematics", "physics", "chemistry", "general"]

# ─── Prototype descriptions for each domain ──────────────────────────────────

_MATH_PROTOTYPE = (
    "mathematics number theory algebra geometry calculus topology analysis "
    "theorem proof lemma conjecture equation function operator integral derivative "
    "modular arithmetic prime divisor congruence group ring field vector space "
    "manifold homology cohomology category functor sheaf scheme moduli "
    "eigenvalue eigenvector matrix linear polynomial"
)

_PHYSICS_PROTOTYPE = (
    "physics mechanics thermodynamics electromagnetism quantum relativity "
    "force energy mass velocity acceleration momentum work power gravity "
    "electric magnetic charge current voltage resistance field wave particle "
    "temperature heat entropy pressure volume fluid gas liquid solid "
    "optics light lens mirror reflection refraction diffraction interference "
    "nuclear atomic electron proton neutron photon decay radiation isotope "
    "circuit capacitor inductor diode transistor semiconductor "
    "orbit satellite planet star galaxy universe cosmology "
    "spring friction torque angular collision impulse oscillation "
    "wavelength frequency amplitude photon laser "
    "superconductor plasma accelerator detector experiment measurement"
)

_CHEMISTRY_PROTOTYPE = (
    "chemistry molecule atom element compound reaction catalyst enzyme "
    "polymer bond orbital electron configuration periodic table acid base "
    "pH oxidation reduction equilibrium kinetics thermodynamics "
    "organic inorganic analytical physical biochemistry "
    "solution concentration solvent solute precipitate "
    "spectroscopy chromatography titration synthesis"
)

_GENERAL_PROTOTYPE = (
    "general knowledge history literature art music philosophy economics "
    "politics sociology psychology anthropology linguistics education "
    "law medicine engineering computer science business management "
    "cooking recipe food drink travel geography culture religion "
)

# ─── Model singleton ──────────────────────────────────────────────────────────

_MODEL = None
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_model():
    """Lazy-load SentenceTransformer model (singleton)."""
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL = SentenceTransformer(_MODEL_NAME)
            logger.info("SemanticDomainClassifier: loaded %s", _MODEL_NAME)
        except ImportError:
            logger.warning("sentence-transformers not available, falling back to lexical")
            return None
        except Exception as exc:
            logger.warning("Failed to load SentenceTransformer: %s", exc)
            return None
    return _MODEL


# ─── Cached prototype embeddings ─────────────────────────────────────────────

_PROTOTYPE_EMBEDDINGS: dict[str, np.ndarray] | None = None


def _get_prototype_embeddings() -> dict[str, np.ndarray]:
    """Encode domain prototypes once and cache."""
    global _PROTOTYPE_EMBEDDINGS
    if _PROTOTYPE_EMBEDDINGS is not None:
        return _PROTOTYPE_EMBEDDINGS

    model = _get_model()
    if model is None:
        _PROTOTYPE_EMBEDDINGS = {}  # empty = model unavailable
        return _PROTOTYPE_EMBEDDINGS

    prototypes = {
        "mathematics": _MATH_PROTOTYPE,
        "physics": _PHYSICS_PROTOTYPE,
        "chemistry": _CHEMISTRY_PROTOTYPE,
        "general": _GENERAL_PROTOTYPE,
    }
    texts = list(prototypes.values())
    keys = list(prototypes.keys())
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    _PROTOTYPE_EMBEDDINGS = {
        keys[i]: np.asarray(vecs[i], dtype=np.float32) for i in range(len(keys))
    }
    return _PROTOTYPE_EMBEDDINGS


# ─── Main classifier ──────────────────────────────────────────────────────────

def classify_domain_ml(brief: str) -> Domain | None:
    """Classify brief domain via embedding similarity to prototypes.

    Returns None when the model is unavailable or similarity is too low/ambiguous.
    The caller should fall back to lexical classification.

    Similarity thresholds (cosine, L2-normalized Inner Product):
    - >= 0.45: strong match → use this domain
    - >= 0.30: weak match → use only if unambiguous
    - < 0.30: too ambiguous → fall back to lexical
    """
    proto_embs = _get_prototype_embeddings()
    if not proto_embs:
        return None  # Model unavailable

    model = _get_model()
    if model is None:
        return None

    brief_emb = model.encode(
        [brief[:512]], normalize_embeddings=True, show_progress_bar=False
    )
    brief_vec = np.asarray(brief_emb[0], dtype=np.float32)

    scores: dict[str, float] = {}
    for domain, proto_vec in proto_embs.items():
        # Cosine similarity = dot product (vectors are L2-normalized)
        sim = float(np.dot(brief_vec, proto_vec))
        scores[domain] = sim

    best_domain = max(scores, key=lambda d: scores[d])
    best_score = scores[best_domain]

    # Find second-best for ambiguity check
    others = {d: s for d, s in scores.items() if d != best_domain}
    second_best = max(others.values()) if others else 0.0
    margin = best_score - second_best

    # Strong, clear match
    if best_score >= 0.45 and margin >= 0.08:
        return best_domain  # type: ignore[return-value]

    # Weak but unambiguous match
    if best_score >= 0.30 and margin >= 0.15:
        return best_domain  # type: ignore[return-value]

    # Too ambiguous — fall back to lexical
    return None


# ─── Batch classification (for benchmarks) ────────────────────────────────────

def classify_batch_ml(briefs: list[str]) -> list[Domain | None]:
    """Classify a batch of briefs, returning None for ambiguous cases."""
    proto_embs = _get_prototype_embeddings()
    if not proto_embs:
        return [None] * len(briefs)

    model = _get_model()
    if model is None:
        return [None] * len(briefs)

    truncated = [b[:512] for b in briefs]
    brief_embs = model.encode(
        truncated, normalize_embeddings=True, show_progress_bar=False
    )
    brief_vecs = np.asarray(brief_embs, dtype=np.float32)

    proto_vecs = np.stack([proto_embs[d] for d in ["mathematics", "physics", "chemistry", "general"]])
    # proto_vecs shape: (4, dim)
    # brief_vecs shape: (n, dim)
    # sims shape: (n, 4)
    sims = np.dot(brief_vecs, proto_vecs.T)  # (n, 4)

    domain_names = ["mathematics", "physics", "chemistry", "general"]
    results: list[Domain | None] = []

    for i in range(len(briefs)):
        row = sims[i]
        best_idx = int(np.argmax(row))
        best_score = float(row[best_idx])

        # Second best
        mask = np.ones(4, dtype=bool)
        mask[best_idx] = False
        second_best = float(row[mask].max()) if mask.any() else 0.0
        margin = best_score - second_best

        if best_score >= 0.45 and margin >= 0.08:
            results.append(domain_names[best_idx])  # type: ignore[arg-type]
        elif best_score >= 0.30 and margin >= 0.15:
            results.append(domain_names[best_idx])  # type: ignore[arg-type]
        else:
            results.append(None)

    return results


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke test
    tests = [
        "A ball rolls down a hill starting from rest",
        "Find the electric field at point P due to two charges",
        "Calculate the pH of a 0.1M HCl solution",
        "Prove that the sum of two even numbers is even",
        "quantum field theory renormalization group beta function",
        "general relativity einstein field equations curvature",
        "What is the capital of France?",
    ]
    for t in tests:
        result = classify_domain_ml(t)
        print(f"  {t[:60]:<60s} -> {result}")