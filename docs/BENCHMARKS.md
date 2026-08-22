# Benchmark Registry — Formulagate v1.0.0

All benchmarks run on 320+ real arXiv papers across 7 scientific domains.
Every benchmark is **honest** — corruptions are real, anchors are real
prose, no synthetic data.

## Quick Reference (Combined)

```json
{
  "total_papers": 320,
  "total_equations": 10000,
  "servers_tested": 2,
  "fpr_smt": "~0.3%",
  "anchored_fp_rate": "0.026",
  "tests": 343
}
```

## Server 1 Results (200 papers, 8 audiences)

| audience | papers | equations | FPR | anchored |
|---|---|---|---|---|
| National Labs (CERN/NASA) | 86 | 3,467 | 0.0% | 37 |
| Oil & Gas / Aerospace | 30 | 1,161 | 0.0% | 37 |
| Pharmaceutical R&D | 34 | 529 | 0.0% | 8 |
| Civil Engineering | 7 | 466 | 0.0% | 23 |
| Academic Journals | 11 | 296 | 0.0% | 7 |
| R&D Semiconductor | 16 | 244 | 0.0% | 6 |
| Medical Physics | 5 | 98 | 0.0% | 1 |
| STEM Education | 11 | 85 | 0.0% | 0 |
| **Total** | **200** | **6,346** | **0.0%** | **119** |

## Server 2 Results (121 papers, 7 audiences — incl. Chemistry & Biology)

| audience | papers | equations | FPR | anchored |
|---|---|---|---|---|
| Biology | 39 | 1,092 | 1.5% | 46 |
| Chemistry | 22 | 676 | 0.0% | 27 |
| Civil Engineering | 12 | 671 | 0.0% | 27 |
| Oil & Gas / Aerospace | 14 | 634 | 0.0% | 30 |
| Medical Physics | 8 | 363 | 0.0% | 16 |
| Pharmaceutical R&D | 21 | 286 | 1.9% | 10 |
| STEM Education | 5 | 38 | 0.0% | 0 |
| **Total** | **121** | **3,760** | **~0.7%** | **156** |

## Per-Phase History

| phase | file | key metric | value |
|---|---|---|---|
| 1 | `bench_fulltext_final.json` | FPR_smt | 0.0% (was 6.8%) |
| 2 | `bench_phase2e.json` | anchored_decidable | 6 (was 3) |
| 3 | `bench_phase3b.json` | anchored_in_scope | 75 (was 60) |
| 4 | `bench_phase4.json` | gold-set precision | 100% |
| 6 | `bench_phase6b.json` | anchored_decidable | 75 (+1150%) |
| 7 | `bench_phase7b.json` | anchored_in_scope | 79 (+5%) |
| 9 | `bench_phase9.json` | decidable_smt | 608 (+5.4%) |
| 11 | `bench_phase11.json` | grounding | 96.8% |
| 13-14 | `bench_audience_v2.json` | 8 audiences | all FPR 0.0% |
| 15 | `bench_phase15.json` | pharmaceutical | 34 papers |
| 16 | `bench_phase16.json` | STEM education | 11 papers |
| — | `bench_chembio.json` | chemistry + biology | 61 papers, FPR 0.7% |

## Data Files

| file | location | size | description |
|---|---|---|---|
| `bench_*.json` | `data/real/` | 2-3 KB | Per-phase summary reports |
| `bench_*_cases.json` | `data/real/` | 2-3 MB | Per-equation details |
| `fulltext/` | `data/real/fulltext/` | ~500 MB | Raw LaTeX sources |
| `manifest.json` | `data/real/fulltext/` | ~50 KB | Paper metadata |

## Reproducing

```bash
# Full benchmark on all papers
python scripts/bench_fulltext.py --report data/real/bench_new.json

# Per-audience benchmark
python scripts/bench_per_audience.py --report data/real/bench_audience.json

# Profession test suite (8 industries)
python scripts/test_professions.py

# Download new papers
python scripts/fetch_expanded.py --all
python scripts/fetch_chembio.py
```

## Key Definitions

- **FPR_smt**: false-positive rate — SMT rejects a CORRECT equation
- **anchored_fp_rate**: FPR on equations with certain-symbol proof
- **decidable_smt**: SMT can prove consistent OR inconsistent
- **grounding_coverage**: equations with ≥1 known symbol dimension
