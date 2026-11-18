#!/usr/bin/env python3
"""Which equilibrium does each CFR variant pick on Kuhn Poker?

Kuhn's Nash equilibria are not a point but a one-parameter family indexed by the
first player's bluffing frequency alpha = P(bet | J), with alpha in [0, 1/3]. At
equilibrium P(bet | K) = 3*alpha and P(call | Q facing a bet) = alpha + 1/3
(Kuhn 1950). Every variant converges to SOME equilibrium, so exploitability
cannot distinguish them -- but they need not pick the same one, and nobody
reports which one they pick.

This traces alpha over training for all eight corners of the ablation cube and
checks the two closed-form identities that must hold at equilibrium.

Writes results/verify/alpha_manifold.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pokercfr.eval.exploitability import exploitability  # noqa: E402
from pokercfr.games.kuhn_family import game  # noqa: E402
from pokercfr.solvers.cfr import CFRSolver, ablation_cube  # noqa: E402

CHECKPOINTS = (100, 1_000, 10_000, 100_000, 500_000)


def trace(label: str) -> tuple[str, list[dict]]:
    g = game("kuhn")
    solver = CFRSolver(g, ablation_cube()[label])
    out = []
    for target in CHECKPOINTS:
        solver.train(target - solver.iterations)
        pol = solver.average_strategy()
        alpha = pol["J"]["b"]
        out.append({
            "iterations": target,
            "alpha": alpha,
            "bet_K": pol["K"]["b"],
            "bet_K_predicted": 3 * alpha,
            "call_Q": pol["Qpb"]["b"],
            "call_Q_predicted": alpha + 1 / 3,
            "exploitability": exploitability(g, pol),
        })
    return label, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()

    with mp.Pool(args.jobs or mp.cpu_count() or 4) as pool:
        traces = dict(pool.map(trace, list(ablation_cube())))

    print(f"  {'variant':<20}{'alpha':>8}{'bet K':>8}{'pred':>8}"
          f"{'call Q':>9}{'pred':>8}{'expl':>11}")
    for label, tr in sorted(traces.items(), key=lambda kv: kv[1][-1]["alpha"]):
        r = tr[-1]
        print(f"  {label:<20}{r['alpha']:>8.4f}{r['bet_K']:>8.4f}"
              f"{r['bet_K_predicted']:>8.4f}{r['call_Q']:>9.4f}"
              f"{r['call_Q_predicted']:>8.4f}{r['exploitability']:>11.2e}")

    alphas = [tr[-1]["alpha"] for tr in traces.values()]
    in_family = all(-1e-6 <= a <= 1 / 3 + 1e-6 for a in alphas)
    worst_k = max(abs(tr[-1]["bet_K"] - tr[-1]["bet_K_predicted"]) for tr in traces.values())
    worst_q = max(abs(tr[-1]["call_Q"] - tr[-1]["call_Q_predicted"]) for tr in traces.values())
    print(f"\n  alpha spread across the eight variants: "
          f"{min(alphas):.4f} to {max(alphas):.4f}")
    print(f"  every variant inside Kuhn's [0, 1/3] family: {in_family}")
    print(f"  worst deviation from P(bet|K) = 3*alpha      : {worst_k:.2e}")
    print(f"  worst deviation from P(call|Q) = alpha + 1/3 : {worst_q:.2e}")

    (ROOT / "results/verify/alpha_manifold.json").write_text(json.dumps({
        "iterations": CHECKPOINTS[-1], "checkpoints": list(CHECKPOINTS),
        "traces": traces, "alpha_min": min(alphas), "alpha_max": max(alphas),
        "all_inside_family": in_family,
        "worst_bet_K_deviation": worst_k, "worst_call_Q_deviation": worst_q,
    }, indent=2))
    print("\nwrote results/verify/alpha_manifold.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
