"""Real-world profession tests for Formulagate.

Tests the dimensional veto against realistic equations from 8 industries.
Each test: setup sources → verify correct equation (should pass) → verify
corrupted equation (should be caught by anchored veto).

Run: python scripts/test_professions.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from formulagate.sdk import Formulagate, Source


@dataclass
class ProfessionTest:
    name: str
    audience: str
    sources: list[Source]
    correct_cases: list[tuple[str, str]]  # (brief, draft) — should pass
    corrupt_cases: list[tuple[str, str]]  # (brief, draft) — should be caught


def run_tests():
    all_tests = []

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Pharmaceutical R&D — Drug Dosage Calculations
    # ═══════════════════════════════════════════════════════════════════════
    all_tests.append(ProfessionTest(
        name="Pharmaceutical R&D",
        audience="Pfizer, Merck, Novartis — drug dosage and pharmacokinetics",
        sources=[
            Source(id="pk-1", text="Drug concentration C(t) = Dose * e^{-kt} / V where V is the volume of distribution",
                   formula=r"C = \frac{D}{V} e^{-k t}"),
            Source(id="pk-2", text="Half-life t_1/2 = ln(2) / k where k is the elimination rate constant",
                   formula=r"t_{1/2} = \frac{\ln 2}{k}"),
        ],
        correct_cases=[
            ("Check drug concentration after 4 hours", r"$C(4) = \frac{500}{40} e^{-0.15 \times 4}$"),
            ("Verify half-life formula", r"$t_{1/2} = \frac{\ln 2}{k} \approx \frac{0.693}{0.15}$"),
        ],
        corrupt_cases=[
            ("Dose error: time in denominator", r"$C = \frac{D}{V t}$"),  # dim mismatch
            ("Half-life dimension error: k * ln2", r"$t_{1/2} = k \ln 2$"),  # T vs T^-1
        ],
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Aerospace — Rocket Equation & Orbital Mechanics
    # ═══════════════════════════════════════════════════════════════════════
    all_tests.append(ProfessionTest(
        name="Aerospace & Defense",
        audience="SpaceX, Lockheed, Boeing — rocket propulsion and orbital dynamics",
        sources=[
            Source(id="rocket-1", text="Tsiolkovsky rocket equation: Delta-v = v_e * ln(m0/mf)",
                   formula=r"\Delta v = v_e \ln\frac{m_0}{m_f}"),
            Source(id="orbit-1", text="Orbital velocity: v = sqrt(G M / r)",
                   formula=r"v = \sqrt{\frac{G M}{r}}"),
            Source(id="orbit-2", text="Kepler's third law: T^2 = 4 pi^2 a^3 / (G M)",
                   formula=r"T^2 = \frac{4\pi^2 a^3}{G M}"),
        ],
        correct_cases=[
            ("Delta-v for Falcon 9", r"$\Delta v = 3000 \ln 5$"),
            ("LEO orbital velocity", r"$v = \sqrt{\frac{G M_{\oplus}}{R_{\oplus}}}$"),
        ],
        corrupt_cases=[
            ("Rocket: wrong exponent", r"$\Delta v = v_e \ln\frac{m_0}{m_f^2}$"),
            ("Orbit: mass in numerator", r"$v = \sqrt{\frac{G M^2}{r}}$"),
        ],
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # 3. National Labs — Particle Physics & Cosmology
    # ═══════════════════════════════════════════════════════════════════════
    all_tests.append(ProfessionTest(
        name="National Laboratories",
        audience="CERN, NASA, LANL — particle physics, cosmology, GR",
        sources=[
            Source(id="em-1", text="Coulomb force: F = q1 q2 / (4 pi epsilon_0 r^2)",
                   formula=r"F = \frac{q_1 q_2}{4\pi\epsilon_0 r^2}"),
            Source(id="gr-1", text="Einstein field equations: G_munu = 8 pi G T_munu",
                   formula=r"R_{\mu\nu} - \frac{1}{2}R g_{\mu\nu} = 8\pi G T_{\mu\nu}"),
            Source(id="cosmo-1", text="Friedmann equation: H^2 = 8 pi G rho / 3",
                   formula=r"H^2 = \frac{8\pi G}{3}\rho"),
        ],
        correct_cases=[
            ("Coulomb between two protons", r"$F = \frac{e^2}{4\pi\epsilon_0 r^2}$"),
            ("Einstein-Hilbert action", r"$S = \frac{1}{16\pi G} \int R \sqrt{-g} d^4x$"),
        ],
        corrupt_cases=[
            ("Coulomb with r^3", r"$F = \frac{q_1 q_2}{4\pi\epsilon_0 r^3}$"),
            ("Friedmann with H instead of H^2", r"$H = \frac{8\pi G}{3}\rho$"),
        ],
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Academic Journals — Peer Review
    # ═══════════════════════════════════════════════════════════════════════
    all_tests.append(ProfessionTest(
        name="Academic Journals",
        audience="Nature, APS, IEEE — automated submission screening",
        sources=[
            Source(id="ref-1", text="Planck energy: E_p = sqrt(hbar c^5 / G)",
                   formula=r"E_p = \sqrt{\frac{\hbar c^5}{G}}"),
            Source(id="ref-2", text="Bohr radius: a_0 = 4 pi epsilon_0 hbar^2 / (m_e e^2)",
                   formula=r"a_0 = \frac{4\pi\epsilon_0 \hbar^2}{m_e e^2}"),
        ],
        correct_cases=[
            ("Check Planck energy dimension", r"$E_p = \sqrt{\frac{\hbar c^5}{G}}$"),
            ("Verify Bohr radius", r"$a_0 = \frac{4\pi\epsilon_0 \hbar^2}{m_e e^2}$"),
        ],
        corrupt_cases=[
            ("Planck energy: missing sqrt", r"$E_p = \frac{\hbar c^5}{G}$"),
            ("Bohr: wrong hbar exponent", r"$a_0 = \frac{4\pi\epsilon_0 \hbar}{m_e e^2}$"),
        ],
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Enterprise RAG — Document Verification
    # ═══════════════════════════════════════════════════════════════════════
    all_tests.append(ProfessionTest(
        name="Enterprise RAG Systems",
        audience="LangChain, LlamaIndex users — document retrieval with formula check",
        sources=[
            Source(id="doc-1", text="Ohm's law: voltage = current * resistance",
                   formula=r"V = I R"),
            Source(id="doc-2", text="Power: P = I V = I^2 R",
                   formula=r"P = I V = I^2 R"),
            Source(id="doc-3", text="Kinetic energy: K = 1/2 m v^2",
                   formula=r"K = \frac{1}{2} m v^2"),
        ],
        correct_cases=[
            ("What is Ohm's law?", r"$V = I R$ is the voltage-current relation."),
            ("Power in a circuit", r"$P = I V = I^2 R$"),
        ],
        corrupt_cases=[
            ("Wrong Ohm's law", r"$V = I / R$ is the voltage-current relation."),
            ("Wrong kinetic energy", r"$K = m v^3$"),
        ],
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # 6. STEM Education — Student Error Detection
    # ═══════════════════════════════════════════════════════════════════════
    all_tests.append(ProfessionTest(
        name="STEM Education",
        audience="Khan Academy, Coursera, edX — automated homework checking",
        sources=[
            Source(id="phy-1", text="Newton's second law: F = m a",
                   formula=r"F = m a"),
            Source(id="phy-2", text="Gravitational potential energy: U = m g h",
                   formula=r"U = m g h"),
            Source(id="phy-3", text="Momentum: p = m v",
                   formula=r"p = m v"),
        ],
        correct_cases=[
            ("Student calculates force", r"$F = 10 \cdot 2 = 20 \text{ N}$"),
            ("Potential energy of a book", r"$U = 2 \cdot 9.8 \cdot 5 = 98 \text{ J}$"),
        ],
        corrupt_cases=[
            ("Force = mass + acceleration (wrong)", r"$F = m + a$"),
            ("Momentum = m / v (wrong)", r"$p = \frac{m}{v}$"),
            ("Energy = m g / h (wrong)", r"$U = \frac{m g}{h}$"),
            ("Kinetic energy = m v (missing v)", r"$K = m v$"),
        ],
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # 7. Civil Engineering — Structural Mechanics
    # ═══════════════════════════════════════════════════════════════════════
    all_tests.append(ProfessionTest(
        name="Civil Engineering",
        audience="Arup, AECOM — structural analysis, beam theory",
        sources=[
            Source(id="struct-1", text="Stress = Force / Area: sigma = F / A",
                   formula=r"\sigma = \frac{F}{A}"),
            Source(id="struct-2", text="Strain = delta_L / L",
                   formula=r"\epsilon = \frac{\Delta L}{L}"),
            Source(id="struct-3", text="Hooke's law: sigma = E * epsilon",
                   formula=r"\sigma = E \epsilon"),
            Source(id="struct-4", text="Moment = Force * distance: M = F d",
                   formula=r"M = F d"),
        ],
        correct_cases=[
            ("Stress in a steel beam", r"$\sigma = \frac{100000}{0.01} = 10 \text{ MPa}$"),
            ("Check Hooke's law", r"$\sigma = 200e9 \cdot 0.002 = 400 \text{ MPa}$"),
        ],
        corrupt_cases=[
            ("Stress = F * A (wrong)", r"$\sigma = F A$"),
            ("Moment = F / d (wrong)", r"$M = \frac{F}{d}$"),
        ],
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # 8. Oil & Gas — Fluid Flow & Reservoir Engineering
    # ═══════════════════════════════════════════════════════════════════════
    all_tests.append(ProfessionTest(
        name="Oil & Gas",
        audience="Shell, Schlumberger — reservoir engineering, fluid dynamics",
        sources=[
            Source(id="flow-1", text="Darcy's law: Q = k A Delta_P / (mu L)",
                   formula=r"Q = \frac{k A \Delta P}{\mu L}"),
            Source(id="flow-2", text="Reynolds number: Re = rho v D / mu",
                   formula=r"Re = \frac{\rho v D}{\mu}"),
            Source(id="flow-3", text="Continuity: A1 v1 = A2 v2",
                   formula=r"A_1 v_1 = A_2 v_2"),
            Source(id="flow-4", text="Bernoulli: P + 1/2 rho v^2 + rho g h = const",
                   formula=r"P + \frac{1}{2}\rho v^2 + \rho g h = \text{const}"),
        ],
        correct_cases=[
            ("Flow through porous rock", r"$Q = \frac{k A \Delta P}{\mu L}$"),
            ("Check Reynolds number", r"$Re = \frac{1000 \cdot 2 \cdot 0.1}{0.001} = 200000$"),
        ],
        corrupt_cases=[
            ("Darcy with mu in numerator", r"$Q = \frac{k A \mu \Delta P}{L}$"),
            ("Reynolds missing rho", r"$Re = \frac{v D}{\mu}$"),
        ],
    ))

    # ═══ Run all tests ════════════════════════════════════════════════════
    results = {}
    for test in all_tests:
        print(f"\n{'='*70}")
        print(f"  {test.name}")
        print(f"  {test.audience}")
        print(f"{'='*70}")

        gate = Formulagate(sources=test.sources, use_physics=True)
        correct_pass = 0
        correct_total = 0
        corrupt_caught = 0
        corrupt_total = 0

        # Test correct equations
        for brief, draft in test.correct_cases:
            result = gate.check(brief=brief, draft=draft)
            correct_total += 1
            if result.action == "generate":
                correct_pass += 1
                print(f"  ✓ CORRECT PASS: {brief[:50]}")
            else:
                print(f"  ✗ FALSE REJECT: {brief[:50]} → {result.detail[:60]}")

        # Test corrupted equations
        for brief, draft in test.corrupt_cases:
            result = gate.check(brief=brief, draft=draft)
            corrupt_total += 1
            if result.action == "abstain" and "dimension" in result.detail.lower():
                corrupt_caught += 1
                print(f"  ✓ CAUGHT: {brief[:50]}")
            elif result.action == "abstain":
                print(f"  ~ ABSTAINED (other: {result.detail[:40]})")
            else:
                print(f"  ✗ MISSED: {brief[:50]} → generate")

        results[test.name] = {
            "correct_rate": f"{correct_pass}/{correct_total}",
            "detection_rate": f"{corrupt_caught}/{corrupt_total}",
            "false_rejects": correct_total - correct_pass,
        }
        print(f"  Correct: {correct_pass}/{correct_total}  |  Caught: {corrupt_caught}/{corrupt_total}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  FINAL ASSESSMENT")
    print(f"{'='*70}")
    for name, r in results.items():
        c = r["correct_rate"]
        d = r["detection_rate"]
        fr = r["false_rejects"]
        status = "✅ READY" if fr == 0 and d != "0/0" else "⚠️ REVIEW"
        print(f"  {status} | {name:30s} | correct={c:8s} | caught={d:8s}")

    return json.dumps(results, indent=2)


if __name__ == "__main__":
    print(run_tests())
