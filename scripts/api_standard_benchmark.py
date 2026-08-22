#!/usr/bin/env python3
"""Standard HTTP benchmark for the Formulagate API.

Measures what any production caller cares about:
  BENCH 1 — determinism     : identical requests must return identical bodies
  BENCH 2 — latency         : p50/p95/p99 for /verify and /check
  BENCH 3 — throughput      : requests/second, single-thread and concurrent
  BENCH 4 — correctness     : wrong formulas always rejected, right ones accepted
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

API = "http://localhost:8907"
SOURCES = [
    {"id": "rel-1", "text": "energy mass equivalence", "formula": "$E = m c^2$"},
    {"id": "newton", "text": "second law of motion", "formula": "$F = m a$"},
    {"id": "sb", "text": "Stefan Boltzmann law", "formula": "$j = \\sigma T^4$"},
]


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] * 1000  # ms


def main() -> int:
    print("=" * 72)
    print("FORMULAGATE API — STANDARD HTTP BENCHMARK")
    print("=" * 72)

    health = requests.get(f"{API}/health", timeout=10).json()
    print(f"\nServer: v{health['version']}")

    # ── BENCH 1: determinism ────────────────────────────────────────────
    print("\n[BENCH 1] DETERMINISM — 20 identical /verify requests per formula")
    formulas = ["E = m c^2", "F = m a", "E = m c^3"]
    all_det = True
    for f in formulas:
        bodies = {
            json.dumps(requests.post(f"{API}/verify", json={"formula": f}, timeout=10).json(), sort_keys=True)
            for _ in range(20)
        }
        ok = len(bodies) == 1
        all_det &= ok
        print(f"  {f:10s} -> {'deterministic (20/20)' if ok else f'NON-DETERMINISTIC ({len(bodies)} variants)'}")
    print(f"  RESULT: {'PASS' if all_det else 'FAIL'}")

    # ── BENCH 2: latency ────────────────────────────────────────────────
    print("\n[BENCH 2] LATENCY — 300 requests per endpoint (sequential)")
    for name, payload in [
        ("verify", {"formula": "E = m c^2"}),
        ("verify+equiv", {"formula": "E = m c^2", "against": "m c^2 = E"}),
        ("check", {"brief": "What relates mass and energy?", "draft": "Einstein showed E = m c^2.", "sources": SOURCES}),
        ("check-bad", {"brief": "What relates mass and energy?", "draft": "Einstein showed E = m c^3.", "sources": SOURCES}),
    ]:
        lat = []
        for _ in range(300):
            t0 = time.perf_counter()
            requests.post(f"{API}/{payload and name.split('-')[0]}", json=payload, timeout=10)
            lat.append(time.perf_counter() - t0)
        print(
            f"  {name:12s} p50={pct(lat, 0.5):7.2f}ms p95={pct(lat, 0.95):7.2f}ms "
            f"p99={pct(lat, 0.99):7.2f}ms max={max(lat) * 1000:7.2f}ms"
        )

    # ── BENCH 3: throughput ─────────────────────────────────────────────
    print("\n[BENCH 3] THROUGHPUT — 10s sustained, concurrent workers")
    for workers in (1, 8, 32):
        stop = time.perf_counter() + 10
        count = [0]

        def hit(_: int) -> None:
            while time.perf_counter() < stop:
                requests.post(f"{API}/verify", json={"formula": "F = m a"}, timeout=10)
                count[0] += 1

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(hit, range(workers)))
        print(f"  workers={workers:2d}  {count[0] / 10:8.1f} req/s")

    # ── BENCH 4: correctness under the full formula set ─────────────────
    print("\n[BENCH 4] CORRECTNESS — ground-truth formula battery")
    battery = [
        ("E = m c^2", True), ("F = m a", True), ("E_k = m v^2 / 2", True),
        ("E = m c^3", False), ("F = m a^2", False), ("E = m c", False),
    ]
    good = bad = 0
    for f, expect_ok in battery:
        r = requests.post(f"{API}/verify", json={"formula": f}, timeout=10).json()
        got = bool(r.get("ok"))
        if got == expect_ok:
            good += 1
            print(f"  {f:16s} ok={got} expected={expect_ok}  ✓")
        else:
            bad += 1
            print(f"  {f:16s} ok={got} expected={expect_ok}  ✗ MISMATCH ({r.get('dimensions')})")
    print(f"  RESULT: {good}/{good + bad} correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
