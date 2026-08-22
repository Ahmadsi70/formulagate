"""Formulagate: a physics-aware gate that abstains instead of hallucinating.

Integrators should import :mod:`formulagate.sdk` — a small, stable surface
(``Formulagate``, ``verify``, ``check``). Everything exported here is the
internal machinery those two calls are built from, and it may move between
minor versions.
"""

from formulagate.atomic import (
    AtomicClaim,
    AtomicReport,
    AtomicVerdict,
    AtomicVerifier,
    decompose_rule,
)
from formulagate.audit import build_rag_audit
from formulagate.bench import RerankBenchmark, benchmark_rerankers
from formulagate.calibration import (
    CalibrationError,
    CalibrationParams,
    CalibrationReport,
    MultiCalibration,
    calibrate_confidence,
    evaluate_calibration,
    fit_calibration,
    fit_multi_calibration,
    get_calibration,
    get_multi_calibration,
    load_calibration,
    load_multi_calibration,
    save_calibration,
    save_multi_calibration,
    set_calibration,
    set_multi_calibration,
)
from formulagate.conformal import (
    ConformalCalibration,
    ConformalError,
    conformal_threshold,
    fit_conformal,
)
from formulagate.dense import DenseHybridRetriever, HashEmbedder, SentenceTransformerEmbedder
from formulagate.dimensions import Dimension, DimensionVerdict, check_dimensions
from formulagate.equivalence import EquivalenceVerdict, check_equivalence
from formulagate.formula_extract import (
    Formula,
    canonicalize,
    extract_formulas,
    extract_latex_spans,
    structural_match,
)
from formulagate.gate import (
    DiscoveryResult,
    GateEvaluation,
    GateSignals,
    discover_linkages,
    evaluate_gate,
    evaluate_gate_signals,
)
from formulagate.grounding import (
    GroundingScore,
    check_grounding,
    semantic_grounding,
    token_level_grounding,
)
from formulagate.hybrid import HybridRrfRetriever, rrf_fuse
from formulagate.metrics import (
    AbstainReport,
    evaluate_abstain_cases,
    evaluate_golden_cases,
    load_golden_cases,
)
from formulagate.physics_constraints import (
    ConstraintVerdict,
    ConstraintsReport,
    check_physics_constraints,
)
from formulagate.physics_signals import PhysicsSignals, physics_signals
from formulagate.rag import LexicalStubRetriever, RagDecision, run_scientific_rag
from formulagate.rerank import CrossEncoderReranker, FormulagateReranker, RerankingRetriever
from formulagate.scoring import SCORE_WEIGHT_REL, score_record
from formulagate.semantic_entropy import (
    SemanticEntropy,
    compute_semantic_entropy,
    jaccard_cluster,
    nli_cluster,
)

__all__ = [
    "SCORE_WEIGHT_REL",
    "AbstainReport",
    "AtomicClaim",
    "AtomicReport",
    "AtomicVerdict",
    "AtomicVerifier",
    "CalibrationError",
    "CalibrationParams",
    "CalibrationReport",
    "ConformalCalibration",
    "ConformalError",
    "CrossEncoderReranker",
    "DenseHybridRetriever",
    "Dimension",
    "DimensionVerdict",
    "DiscoveryResult",
    "EquivalenceVerdict",
    "Formula",
    "FormulagateReranker",
    "GateEvaluation",
    "GateSignals",
    "GroundingScore",
    "HashEmbedder",
    "HybridRrfRetriever",
    "LexicalStubRetriever",
    "MultiCalibration",
    "PhysicsSignals",
    "RagDecision",
    "RerankBenchmark",
    "RerankingRetriever",
    "SemanticEntropy",
    "SentenceTransformerEmbedder",
    "benchmark_rerankers",
    "build_rag_audit",
    "calibrate_confidence",
    "canonicalize",
    "check_dimensions",
    "check_equivalence",
    "check_grounding",
    "conformal_threshold",
    "decompose_rule",
    "discover_linkages",
    "evaluate_abstain_cases",
    "evaluate_calibration",
    "evaluate_gate",
    "evaluate_gate_signals",
    "evaluate_golden_cases",
    "extract_formulas",
    "extract_latex_spans",
    "fit_calibration",
    "fit_conformal",
    "fit_multi_calibration",
    "get_calibration",
    "get_multi_calibration",
    "load_calibration",
    "load_golden_cases",
    "load_multi_calibration",
    "physics_signals",
    "rrf_fuse",
    "run_scientific_rag",
    "save_calibration",
    "save_multi_calibration",
    "score_record",
    "set_calibration",
    "set_multi_calibration",
    "structural_match",
]

__version__ = "1.0.0"
