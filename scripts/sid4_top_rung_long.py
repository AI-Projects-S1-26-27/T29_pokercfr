"""EXPLORATORY (not pre-registered): the 52-card rung run 100x longer.

OWNER: Siddhant. Produces results/verify/top_rung_long/<variant>.json and a
summary results/verify/top_rung_long.json (+ .log via tee).

The pre-registered `top-rung` matrix stops at 1,000 iterations because that was
all the SCALAR traversal could afford (~95-190 s/iteration). With the vectorized
traversal 1,000 iterations take ~10 s, which is too short to say anything about
asymptotic rates, so this script runs every corner of the ablation cube for
100,000 iterations on `13rank-4suit-2dealt-2p` and measures each average
strategy against the EXACT equilibrium value from the LP oracle
(results/verify/lp_top_rung.json, produced by `sid1_lp_values.py --top-rung`).

It deliberately does NOT write into results/ and is NOT part of the registry:
it was decided after the pre-registration freeze, so it is reported as
exploratory and must not be pooled with registered cells. Partial files are
rewritten after every checkpoint, so an interrupted run keeps its curve.

    python scripts/sid4_top_rung_long.py --jobs 2     # ~1 hour on 2 cores
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOP = "13rank-4suit-2dealt-2p"
ITERS = 100_000
OUT_DIR = ROOT / "results" / "verify" / "top_rung_long"
SUMMARY = ROOT / "results" / "verify" / "top_rung_long.json"
LP_FILE = ROOT / "results" / "verify" / "lp_top_rung.json"


def run(label: str) -> dict:
    sys.path.insert(0, str(ROOT))
    from pokercfr.experiments.registry import checkpoints
    from pokercfr.games.kuhn_family import game
    from pokercfr.solvers.cfr import ablation_cube
    from pokercfr.solvers.vectorized import VectorizedCFRSolver, nash_conv

    lp_value = json.loads(LP_FILE.read_text())["lp"][0]["value"] if LP_FILE.exists() else None
    g = game(TOP)
    solver = VectorizedCFRSolver(g, ablation_cube()[label])
    curve, train = [], 0.0
    path = OUT_DIR / f"{label.replace('/', '-')}.json"
    for target in checkpoints(ITERS):
        started = time.perf_counter()
        while solver.iterations < target:
            solver.step()
        train += time.perf_counter() - started
        started = time.perf_counter()
        result = nash_conv(g, solver.average_strategy())
        point = {"iterations": solver.iterations, "train_seconds": train,
                 "eval_seconds": time.perf_counter() - started,
                 "exploitability": result.exploitability, "gains": result.gains,
                 "value_seat0": result.policy_values[0],
                 "value_gap_to_lp": (None if lp_value is None
                                     else result.policy_values[0] - lp_value)}
        curve.append(point)
        record = {"exploratory": True, "pre_registered": False, "game": TOP, "variant": label,
                  "traversal": "vectorized", "iterations_target": ITERS, "lp_value": lp_value,
                  "curve": curve, "complete": solver.iterations >= ITERS}
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2))
    return record


def main() -> int:
    from pokercfr.solvers.cfr import ablation_cube

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    if not LP_FILE.exists():
        print(f"note: {LP_FILE.relative_to(ROOT)} missing; value gaps will be null")
    with mp.get_context("spawn").Pool(args.jobs) as pool:
        records = pool.map(run, list(ablation_cube()))
    print(f"{'variant':<18}{'T':>9}{'train s':>10}{'exploitability':>16}{'value - LP':>14}")
    for r in sorted(records, key=lambda r: r["curve"][-1]["exploitability"]):
        last = r["curve"][-1]
        gap = last["value_gap_to_lp"]
        print(f"{r['variant']:<18}{last['iterations']:>9,}{last['train_seconds']:>10.0f}"
              f"{last['exploitability']:>16.3e}{'-' if gap is None else f'{gap:+.2e}':>14}")
    SUMMARY.write_text(json.dumps({"exploratory": True, "pre_registered": False,
                                   "runs": records}, indent=2))
    print(f"\nwrote {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
