"""Quick pharma benchmark run — use existing corpus data."""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from formulagate.sdk import Formulagate, Source
from formulagate.physics_signals import physics_signals
from formulagate.dimensions import check_dimensions
from formulagate.formula_extract import canonicalize

PHARMA_CORPUS = [
    {"id": "pk_clearance", "math_formula": "CL = Dose/AUC",
     "english": "Total clearance is volume of plasma cleared per unit time clearance rate.",
     "scientific_domain": "pharmacology", "title": "Clinical Pharmacokinetics", "text": "pharmacokinetics clearance AUC"},
    {"id": "pk_halflife", "math_formula": "t_half = 0.693 Vd/CL",
     "english": "Elimination half-life equals log2 times volume divided by clearance.",
     "scientific_domain": "pharmacology", "title": "PK Principles", "text": "half life pharmacokinetics"},
    {"id": "pk_vd", "math_formula": "Vd = Dose/C0",
     "english": "Volume of distribution equals dose divided by initial concentration.",
     "scientific_domain": "pharmacology", "title": "PK Parameters", "text": "volume distribution drug"},
    {"id": "enzyme_mm", "math_formula": "v = Vmax S/(Km+S)",
     "english": "Michaelis-Menten equation enzyme kinetics substrate rate.",
     "scientific_domain": "pharmacology", "title": "Enzyme Kinetics", "text": "michaelis menten enzyme drug"},
    {"id": "dissolution", "math_formula": "dM/dt = D A(Cs-C)/h",
     "english": "Noyes-Whitney dissolution rate of solid drug in solution.",
     "scientific_domain": "pharmacology", "title": "Drug Dissolution", "text": "dissolution noyes whitney drug"},
    {"id": "pbpk_hepatic", "math_formula": "CLh = Qh fu CLint/(Qh+fu CLint)",
     "english": "Well-stirred hepatic clearance model with blood flow and intrinsic clearance.",
     "scientific_domain": "pharmacology", "title": "PBPK Modeling", "text": "hepatic clearance PBPK liver drug"},
    {"id": "dose_loading", "math_formula": "LD = Vd Cp/F",
     "english": "Loading dose equals volume times target concentration divided by bioavailability.",
     "scientific_domain": "pharmacology", "title": "Dose Individualization", "text": "loading dose clinical pharmacology"},
    {"id": "bioavail", "math_formula": "F = AUC_oral Dose_IV/(AUC_IV Dose_oral)",
     "english": "Absolute oral bioavailability compared to intravenous administration.",
     "scientific_domain": "pharmacology", "title": "Bioavailability", "text": "bioavailability oral IV drug"},
    {"id": "mrt", "math_formula": "MRT = AUMC/AUC",
     "english": "Mean residence time equals area under moment curve divided by AUC.",
     "scientific_domain": "pharmacology", "title": "PK Parameters", "text": "mean residence time drug"},
    {"id": "therapeutic", "math_formula": "TI = TD50/ED50",
     "english": "Therapeutic index ratio of toxic dose to effective dose.",
     "scientific_domain": "pharmacology", "title": "Safety", "text": "therapeutic index toxic effective dose"},
]

GROUND_TRUTH = [
    # (name, latex, consistent, multi_letter)
    ("Clearance", r"\mathrm{CL} = \frac{\mathrm{Dose}}{\mathrm{AUC}}", True, True),
    ("Volume of Distribution", r"\mathrm{Vd} = \frac{\mathrm{Dose}}{\mathrm{C0}}", True, True),
    ("Mean Residence Time", r"\mathrm{MRT} = \frac{\mathrm{AUMC}}{\mathrm{AUC}}", True, True),
    ("Absolute Bioavailability", r"\mathrm{F} = \frac{\mathrm{AUC}_{oral} \cdot \mathrm{Dose}_{IV}}{\mathrm{AUC}_{IV} \cdot \mathrm{Dose}_{oral}}", True, True),
    ("Loading Dose", r"\mathrm{LD} = \frac{\mathrm{Vd} \cdot \mathrm{Cp}}{\mathrm{F}}", True, True),
    ("Therapeutic Index", r"\mathrm{TI} = \frac{\mathrm{TD50}}{\mathrm{ED50}}", True, True),
    ("Cheng-Prusoff IC50", r"\mathrm{IC50} = \mathrm{Ki} \cdot (1 + \frac{[S]}{\mathrm{Km}})", True, True),
    ("Hepatic Clearance", r"\mathrm{CLh} = \frac{\mathrm{Qh} \cdot f_u \cdot \mathrm{CLint}}{\mathrm{Qh} + f_u \cdot \mathrm{CLint}}", True, True),
    # Hallucinations (deliberate errors)
    ("HALLUC: CL = Dose*AUC", r"\mathrm{CL} = \mathrm{Dose} \cdot \mathrm{AUC}", False, True),
    ("HALLUC: Vd = C0/Dose", r"\mathrm{Vd} = \frac{\mathrm{C0}}{\mathrm{Dose}}", False, True),
    ("HALLUC: LD inverted", r"\mathrm{LD} = \frac{\mathrm{Cp}}{\mathrm{Vd} \cdot \mathrm{F}}", False, True),
]


def main():
    sources = [Source(id=c["id"], text=c.get("english", ""), formula=c.get("math_formula", ""),
                       domain=c.get("scientific_domain", "")) for c in PHARMA_CORPUS]
    gate = Formulagate(sources=sources, use_physics=True, top_k=3)

    print("=" * 65)
    print("  FORMULAGATE PHARMA BENCHMARK — Dimensional Analysis")
    print("  (No LLM key — all deterministic algebra)")
    print("=" * 65)

    dim_pass = 0
    hall_caught = 0
    valid_count = sum(1 for _, _, c, _ in GROUND_TRUTH if c)
    hall_count = sum(1 for _, _, c, _ in GROUND_TRUTH if not c)

    for name, latex, consistent, multi_letter in GROUND_TRUTH:
        canon = canonicalize(latex)
        dim = check_dimensions(canon, None, domain="pharmacology")
        dim_ok = dim.status == "consistent"

        gate_result = gate.check(brief=name, draft=latex)

        if consistent:
            if dim_ok:
                dim_pass += 1
                status = "PASS"
            else:
                status = "UNKNOWN"
        else:
            if not dim_ok:
                hall_caught += 1
                status = "CAUGHT"
            else:
                status = "MISSED"

        symbol = {"PASS": "OK", "UNKNOWN": "??", "CAUGHT": "!!", "MISSED": "XX"}.get(status, "  ")
        print(f"  [{symbol}] dim={dim.status:12s} gate={gate_result.action:8s} conf={str(gate_result.confidence)[:6]}  {name}")

    print(f"\n  {'='*50}")
    print(f"  DIMENSIONAL CONSISTENCY:   {dim_pass}/{valid_count} passed  ({dim_pass/max(valid_count,1):.0%})")
    print(f"  HALLUCINATION DETECTION:   {hall_caught}/{hall_count} caught   ({hall_caught/max(hall_count,1):.0%})")
    print(f"  {'='*50}")

    return {
        "dimensional_pass_rate": dim_pass / max(valid_count, 1),
        "hallucination_catch_rate": hall_caught / max(hall_count, 1),
        "valid_count": valid_count,
        "hall_count": hall_count,
        "dim_pass": dim_pass,
        "hall_caught": hall_caught,
    }


if __name__ == "__main__":
    result = main()
    # Save report
    report_path = ROOT / "data" / "real" / "pharma_bench_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved -> {report_path}")