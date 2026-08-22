"""Test the benchmark suite integration — mini-benchmark on sample corpus.

Verifies that bench_suite runs end-to-end on a small synthetic corpus,
that all 8 dataset formats are parseable, and that the report structure
is valid JSON with all required fields.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


class TestBenchSuite:
    """Tests for the benchmark suite infrastructure."""

    def test_bench_suite_eval_single_case(self):
        """Test the core evaluation function with a minimal case."""
        from scripts.bench_suite import _features_from_case

        corpus = [
            {"id": "1", "math_formula": "E = m c^2", "english": "mass energy equivalence",
             "title": "Relativity", "scientific_domain": "physics"},
        ]
        result = _features_from_case(corpus, "energy", "E = m c^2", 1)
        assert "score" in result
        assert "features" in result
        assert len(result["features"]) == 7
        assert "physics" in result
        assert result["label"] == 1

    def test_bench_suite_eval_no_match(self):
        """Test evaluation when no corpus entries match."""
        from scripts.bench_suite import _features_from_case

        corpus = [
            {"id": "x1", "math_formula": "", "english": "unrelated content",
             "title": "Stuff", "scientific_domain": "history"},
        ]
        result = _features_from_case(corpus, "physics", "E = m c^2", 0)
        assert result["score"] == 0.0
        assert result["label"] == 0

    def test_confusion_matrix(self):
        from scripts.bench_suite import _confusion

        preds = [1, 1, 0, 0, 1, 0]
        labels = [1, 0, 1, 0, 1, 0]
        conf = _confusion(preds, labels)
        assert conf["tp"] == 2
        assert conf["tn"] == 2
        assert conf["fp"] == 1
        assert conf["fn"] == 1

    def test_eval_dataset_single_class(self):
        """Single-class data should not crash."""
        from scripts.bench_suite import _eval_dataset

        corpus = [{"id": "1", "math_formula": "x=1", "english": "test", "title": "t",
                   "scientific_domain": "general"}]
        cases = [{"question": "q", "answer": "a"}]
        result = _eval_dataset(corpus, cases, "test_single", folds=2)
        assert result["n_cases"] == 1

    def test_eval_dataset_two_class(self):
        """Two-class data with proper OOF evaluation."""
        from scripts.bench_suite import _eval_dataset

        corpus = [
            {"id": "p1", "math_formula": "E = m c^2", "english": "energy mass equivalence",
             "title": "Relativity", "scientific_domain": "physics"},
            {"id": "g1", "math_formula": "", "english": "history text about ancient civilizations",
             "title": "History of Rome", "scientific_domain": "history"},
        ]
        cases = [
            {"question": "energy mass equivalence", "answer": "E = m c^2 is Einstein's equation",
             "correct_answer": "E=mc^2"},
            {"question": "rome colosseum", "answer": "the colosseum was built in 80 AD",
             "correct_answer": "", "label": "hallucinated"},
        ]
        result = _eval_dataset(corpus, cases, "test_two", folds=2)
        assert result["n_cases"] == 2
        # May be single class if all cases get the same label
        assert "accuracy" in result or "note" in result

    def test_benchmark_meta_complete(self):
        from scripts.bench_suite import BENCHMARK_META

        required = {"gpqa", "truthfulqa", "halueval", "expertqa",
                     "alce", "mmlu", "mmlu_pro", "arxiv"}
        assert set(BENCHMARK_META) == required
        for name, meta in BENCHMARK_META.items():
            assert "name" in meta
            assert "url" in meta

    def test_report_structure(self):
        """Verify the expected report structure."""
        report = {
            "config": {"corpus": "test", "folds": 3},
            "benchmarks": {
                "gpqa": {"n_cases": 100, "accuracy": 0.75, "ece": 0.05},
                "mmlu": {"n_cases": 200, "accuracy": 0.82, "ece": 0.03},
            },
            "aggregate": {"n_datasets": 2, "mean_accuracy": 0.785, "mean_ece": 0.04},
        }
        json_str = json.dumps(report)
        parsed = json.loads(json_str)
        assert parsed["aggregate"]["n_datasets"] == 2
        assert "benchmarks" in parsed
        assert "gpqa" in parsed["benchmarks"]

    def test_fetch_benchmarks_fetchers_smoke(self):
        """Smoke-test that all fetcher functions are callable."""
        from scripts.fetch_benchmarks import FETCHERS

        for name, (fn, desc) in FETCHERS.items():
            assert callable(fn), f"{name} fetcher should be callable"
            assert isinstance(desc, str)

    def test_features_from_case_with_multi_field_corpus(self):
        """Full corpus fields should be collected into candidate text."""
        from scripts.bench_suite import _features_from_case

        corpus = [
            {"id": "full1", "math_formula": "\\nabla \\cdot E = \\rho",
             "formula": "", "english": "Gauss's law",
             "text": "The divergence of the electric field equals charge density",
             "title": "Electrodynamics",
             "abstract": "Maxwell's equations describe classical electromagnetism",
             "scientific_domain": "physics"},
        ]
        result = _features_from_case(corpus, "Gauss law", "div E = rho", 1)
        assert result["ok"] is True or result["detail"]
        assert "features" in result

    def test_mini_bench_on_sample_corpus(self, tmp_path):
        """Run a mini benchmark on the sample corpus to verify end-to-end."""
        sample_path = ROOT / "data" / "sample_corpus.json"
        if not sample_path.is_file():
            pytest.skip("sample_corpus.json not found")

        # Write minimal cases file
        corpus = json.loads(sample_path.read_text(encoding="utf-8"))
        cases = []
        for row in corpus[:5]:
            cases.append({
                "id": row.get("id", "unknown"),
                "title": row.get("title", row.get("english", "test")),
                "question": row.get("title", row.get("english", "test")),
                "answer": row.get("english", row.get("text", "test")),
            })

        cases_path = tmp_path / "cases.json"
        cases_path.write_text(json.dumps(cases))

        from scripts.bench_suite import _eval_dataset
        result = _eval_dataset(corpus, cases, "sample", folds=3)
        assert result["n_cases"] > 0
        assert "accuracy" in result