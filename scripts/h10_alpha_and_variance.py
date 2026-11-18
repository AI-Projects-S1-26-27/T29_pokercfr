#!/usr/bin/env python3
"""Two studies: Kuhn's alpha manifold, and MCCFR seed variance.


  alpha manifold   Kuhn's Nash equilibria form a one-parameter family indexed by
                   the first player's bluffing frequency alpha = P(bet | J), with
                   alpha in [0, 1/3] and P(bet | K) = 3*alpha at equilibrium.
                   Every variant converges to SOME equilibrium, so exploitability
                   cannot tell them apart -- but they need not pick the same one.

  MCCFR variance   The textbook claim is that external sampling has lower
                   variance than outcome sampling. Measured across 5 seeds.

Writes results/verify/alpha_manifold.json and results/verify/mccfr_variance_small.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from statistics import fmean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pokercfr.eval.exploitability import exploitability  # noqa: E402
from pokercfr.games.kuhn_family import game  # noqa: E402
from pokercfr.solvers.cfr import CFRSolver, ablation_cube  # noqa: E402

CHECKPOINTS = (100, 1_000, 10_000, 50_000, 100_000, 200_000)
SEEDS = (0, 1, 2, 3, 4)


def alpha_trace(label: str) -> tuple[str, list[dict]]:
    g = game("kuhn")
    solver = CFRSolver(g, ablation_cube()[label])
    trace = []
    for target in CHECKPOINTS:
        solver.train(target - solver.iterations)
        pol = solver.average_strategy()
        trace.append({
            "iterations": target,
            "alpha": pol["J"]["b"],              # bluff with the Jack
            "bet_K": pol["K"]["b"],              # should be 3*alpha at equilibrium
            "call_Q": pol["Qpb"]["b"],           # should be alpha + 1/3
            "exploitability": exploitability(g, pol),
        })
    return label, trace


def mccfr_cell(args) -> tuple[str, int, float]:
    scheme, seed = args
    from pokercfr.solvers.mccfr import MCCFRSolver
    g = game("kuhn")
    pol = MCCFRSolver(g, scheme=scheme, seed=seed).train(200_000).average_strategy()
    return scheme, seed, exploitability(g, pol)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()
    jobs = args.jobs or (mp.cpu_count() or 4)

    print("=== alpha manifold: where on [0, 1/3] does each variant land? ===", flush=True)
    with mp.Pool(jobs) as pool:
        traces = dict(pool.map(alpha_trace, list(ablation_cube())))
    print(f"  {'variant':<20}{'alpha':>9}{'3*alpha':>9}{'bet_K':>9}{'call_Q':>9}{'expl':>11}")
    for label, tr in sorted(traces.items(), key=lambda kv: kv[1][-1]["alpha"]):
        last = tr[-1]
        print(f"  {label:<20}{last['alpha']:>9.4f}{3*last['alpha']:>9.4f}"
              f"{last['bet_K']:>9.4f}{last['call_Q']:>9.4f}{last['exploitability']:>11.2e}")
    alphas = [tr[-1]["alpha"] for tr in traces.values()]
    print(f"\n  alpha spread across the 8 variants: {min(alphas):.4f} to {max(alphas):.4f}")
    print(f"  all inside Kuhn's [0, 1/3] family: {all(-1e-6 <= a <= 1/3 + 1e-6 for a in alphas)}")

    print("\n=== MCCFR seed variance (5 seeds, 200k iterations) ===", flush=True)
    cells = [(s, d) for s in ("outcome", "external", "chance") for d in SEEDS]
    with mp.Pool(jobs) as pool:
        rows = pool.map(mccfr_cell, cells)
    by_scheme: dict[str, list[float]] = {}
    for scheme, _seed, expl in rows:
        by_scheme.setdefault(scheme, []).append(expl)
    print(f"  {'scheme':<12}{'mean expl':>12}{'stdev':>12}{'CV':>8}")
    for scheme, vals in by_scheme.items():
        cv = stdev(vals) / fmean(vals) if fmean(vals) else 0.0
        print(f"  {scheme:<12}{fmean(vals):>12.3e}{stdev(vals):>12.3e}{cv:>8.2%}")
    ext, out = stdev(by_scheme["external"]), stdev(by_scheme["outcome"])
    print(f"\n  external lower variance than outcome: {ext < out}  "
          f"({ext:.2e} vs {out:.2e})")

    (ROOT / "results/verify/alpha_manifold.json").write_text(json.dumps(
        {"scale": "reduced", "iterations": CHECKPOINTS[-1],
         "registered_iterations": 50_000_000, "traces": traces}, indent=2))
    (ROOT / "results/verify/mccfr_variance_small.json").write_text(json.dumps(
        {"scale": "reduced", "iterations": 200_000, "registered_iterations": 10_000_000,
         "seeds": list(SEEDS),
         "by_scheme": by_scheme,
         "external_lower_variance_than_outcome": ext < out}, indent=2))
    print("\nwrote results/verify/alpha_manifold.json, "
          "results/verify/mccfr_variance_small.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
