#!/usr/bin/env python3
"""Does DCFR's parameter family unify CFR and CFR+ on our implementation?

Brown & Sandholm present DCFR as a family whose limits recover CFR and CFR+. We
have both reference algorithms as corners of the ablation cube, so the claim is
directly checkable: train each reference and each DCFR setting on the same game
and compare whole exploitability curves, not just endpoints.

Writes results/verify/dcfr_limits.json and prints a verdict.

    python scripts/h1_dcfr_limits.py                 # default 2000 iterations
    python scripts/h1_dcfr_limits.py --iterations 500
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pokercfr.eval.best_response import exploitability  # noqa: E402
from pokercfr.games.kuhn_family import game  # noqa: E402
from pokercfr.solvers.cfr import (  # noqa: E402
    DCFR_AS_CFR,
    DCFR_AS_CFR_PLUS,
    CFRConfig,
    CFRSolver,
    ablation_cube,
)

OUT = ROOT / "results" / "verify" / "dcfr_limits.json"

RUNGS = ("kuhn", "4rank-2suit")
CHECKPOINTS = (10, 20, 50, 100, 200, 500, 1000, 2000)
GRID = [(a, b, g) for a in (1.0, 1.5, 2.0, 3.0) for b in (0.0, 0.5, 1.0) for g in (1.0, 2.0, 3.0)]

# Does pairing the discount schedule with alternating updates close the gap to
# CFR+? The simultaneous grid never does, and alternating updates is the one
# CFR+ component the discount family cannot express.
ALT_SETTINGS = ((1.5, 0.0, 2.0), (2.0, 0.0, 1.0), (3.0, 0.0, 1.0))


def curve(rung: str, config: CFRConfig, iterations: int) -> dict[int, float]:
    g = game(rung)
    solver = CFRSolver(g, config)
    out = {}
    for target in CHECKPOINTS:
        if target > iterations:
            break
        solver.train(target - solver.iterations)
        out[target] = exploitability(g, solver.average_strategy())
    return out


def curve_distance(a: dict[int, float], b: dict[int, float]) -> float:
    """Mean absolute log-ratio over shared checkpoints.

    Compares shape across the whole run: two curves can cross at one checkpoint
    while differing everywhere else, so an endpoint comparison would mislead.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        return math.inf
    terms = []
    for t in shared:
        x, y = a[t], b[t]
        if x <= 0 or y <= 0:
            terms.append(0.0 if x == y else math.inf)
        else:
            terms.append(abs(math.log(x / y)))
    return sum(terms) / len(terms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=2000)
    args = ap.parse_args()

    cube = ablation_cube()
    record: dict = {"iterations": args.iterations, "rungs": {}}
    started = time.perf_counter()

    for rung in RUNGS:
        print(f"\n=== {rung} ===", flush=True)
        refs = {}
        for name in ("cfr", "cfr+", "rm+/linear/sim"):
            refs[name] = curve(rung, cube[name], args.iterations)
            print(f"  reference {name:<16} final={refs[name][max(refs[name])]:.10f}", flush=True)

        limits = {}
        for label, config, expected in (
            ("dcfr(a=inf,b=inf,g=0)", DCFR_AS_CFR, "cfr"),
            ("dcfr(a=inf,b=-inf,g=0)", DCFR_AS_CFR_PLUS, "rm+/linear/sim"),
        ):
            c = curve(rung, config, args.iterations)
            d = curve_distance(c, refs[expected])
            limits[label] = {"curve": c, "expected": expected, "distance": d,
                             "reproduces": d < 1e-9}
            verdict = "MATCH" if d < 1e-9 else f"DIFFERS (dist={d:.4g})"
            print(f"  limit {label:<24} vs {expected:<16} {verdict}", flush=True)

        grid = {}
        for alpha, beta, gamma in GRID:
            label = f"dcfr-a{alpha:g}-b{beta:g}-g{gamma:g}"
            c = curve(rung, CFRConfig(averaging="linear", dcfr=(alpha, beta, gamma)),
                      args.iterations)
            grid[label] = {
                "curve": c,
                "final": c[max(c)],
                "distance_to_cfr": curve_distance(c, refs["cfr"]),
                "distance_to_cfr_plus": curve_distance(c, refs["cfr+"]),
            }
        print(f"  grid: {len(grid)} settings", flush=True)

        alt = {}
        for alpha, beta, gamma in ALT_SETTINGS:
            label = f"dcfr-alt-a{alpha:g}-b{beta:g}-g{gamma:g}"
            c = curve(rung, CFRConfig(averaging="linear", alternating=True,
                                      dcfr=(alpha, beta, gamma)), args.iterations)
            alt[label] = {"curve": c, "final": c[max(c)],
                          "distance_to_cfr_plus": curve_distance(c, refs["cfr+"])}
            print(f"  alternating {label:<26} final={alt[label]['final']:.10f}", flush=True)

        best = min(grid, key=lambda k: grid[k]["final"])
        closest_cfr = min(grid, key=lambda k: grid[k]["distance_to_cfr"])
        closest_plus = min(grid, key=lambda k: grid[k]["distance_to_cfr_plus"])
        print(f"  best final exploitability : {best} ({grid[best]['final']:.10f})")
        print(f"  closest to cfr            : {closest_cfr}")
        print(f"  closest to cfr+           : {closest_plus}")

        best_alt = min(alt, key=lambda k: alt[k]["final"])
        cfr_plus_final = refs["cfr+"][max(refs["cfr+"])]
        print(f"  best simultaneous DCFR    : {grid[best]['final']:.10f} "
              f"({grid[best]['final'] / cfr_plus_final:.1f}x cfr+)")
        print(f"  best alternating DCFR     : {alt[best_alt]['final']:.10f} "
              f"({alt[best_alt]['final'] / cfr_plus_final:.1f}x cfr+)")

        record["rungs"][rung] = {
            "references": {k: v for k, v in refs.items()},
            "limits": limits,
            "grid": grid,
            "alternating": alt,
            "best_alternating": best_alt,
            "best_final": best,
            "closest_to_cfr": closest_cfr,
            "closest_to_cfr_plus": closest_plus,
        }

    reproduces = all(lim["reproduces"]
                     for r in record["rungs"].values() for lim in r["limits"].values())
    record["unification_reproduces"] = reproduces
    record["seconds"] = time.perf_counter() - started

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2))

    print("\n" + "=" * 62)
    print(f"VERDICT: DCFR's stated limits {'DO' if reproduces else 'DO NOT'} reproduce "
          f"their reference algorithms")
    print("Note: DCFR cannot express alternating updates, so its CFR+ limit is the")
    print("rm+/linear/sim corner rather than cfr+ itself.")
    print(f"wrote {OUT.relative_to(ROOT)}  ({record['seconds']:.1f}s)")
    return 0 if reproduces else 1


if __name__ == "__main__":
    raise SystemExit(main())
