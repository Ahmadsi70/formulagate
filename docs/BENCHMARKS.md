# Benchmark Registry — Formulagate v1.0.1

The authoritative, freshly reproduced benchmark numbers live in the README's
[Benchmarks — real data, reproducible](https://github.com/Ahmadsi70/formulagate#benchmarks--real-data-reproducible)
section. Every claim there was re-measured from scratch (no caches) on a clean
host; the raw per-run reports are committed under `data/real/bench_*_server*.json`.

Older per-server and per-phase numbers (Server 1/2 FPR tables, "320+ papers",
"0.0% FPR") were superseded by the honest fresh-run measurement — 142 papers,
7/706 = 0.99% false rejects on real equations — and are no longer maintained
here.

## Reproducing

```bash
python scripts/fetch_arxiv_benchmark.py --per-category 60   # corpus
python scripts/bench_real.py --out data/calibration.json --multi-out data/calibration_fused.json
python scripts/fetch_benchmarks.py --all                    # MMLU, GPQA, TruthfulQA, HaluEval…
python scripts/bench_suite.py --all
python scripts/bench_fulltext.py                            # needs scripts/fetch_arxiv_sources.py first
```

## Key Definitions

- **FPR_smt**: false-positive rate — SMT rejects a CORRECT equation
- **anchored_fp_rate**: FPR on equations with certain-symbol proof
- **decidable_smt**: SMT can prove consistent OR inconsistent
- **grounding_coverage**: equations with ≥1 known symbol dimension