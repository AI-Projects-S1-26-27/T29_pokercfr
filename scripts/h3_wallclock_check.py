#!/usr/bin/env python3
"""Is the equal-wall-clock budget mode a fair comparison, and does it matter?

H1 claims CFR+ beats vanilla CFR "at equal wall-clock budgets". Three checks:

  fairness   every variant given the same budget actually trains for the same
             number of seconds, otherwise the comparison is void
  isolation  evaluation time is excluded from the training clock, so measuring
             more often does not cost a variant iterations
  verdict    whether ranking the variants by equal-iterations and by
             equal-wall-clock gives the same answer. Alternating corners do one
             traversal per player, so they cost ~N x more per iteration -- if
             the two rankings differ, iteration-matched comparisons have been
             flattering them and H1 needed this mode to be tested honestly.

Writes results/verify/wallclock_check.json.

    python scripts/h3_wallclock_check.py
    python scripts/h3_wallclock_check.py --budget 20 --rung 4rank-2suit-2dealt
    python scripts/h3_wallclock_check.py --budget 2 --out /tmp/smoke.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pokercfr.core.config import RunSpec  # noqa: E402
from pokercfr.eval.best_response import exploitability  # noqa: E402
from pokercfr.experiments import runner  # noqa: E402
from pokercfr.games.kuhn_family import game  # noqa: E402
from pokercfr.solvers.cfr import CFRSolver, ablation_cube  # noqa: E402

DEFAULT_OUT = ROOT / "results" / "verify" / "wallclock_check.json"
FAIRNESS_TOLERANCE = 0.02
ISOLATION_TOLERANCE = 0.10


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def timed(rung: str, variant: str, budget: float, fractions=(1.0,)) -> dict:
    spec = RunSpec(game=rung, variant=variant, seed=0,
                   extra={"time_budget": budget, "time_checkpoints": tuple(fractions)})
    record = runner.run_one(spec, resume=False, progress=lambda *_: None)
    last = record["curve"][-1]
    return {"iterations": last["iterations"], "train_seconds": last["train_seconds"],
            "exploitability": last["exploitability"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=10.0, help="seconds per variant")
    ap.add_argument("--rung", default="4rank-2suit")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="where to write the JSON; point a smoke run somewhere "
                         "else so it cannot clobber a real one")
    args = ap.parse_args()

    # run_one writes a result file per cell. These are throwaway measurements,
    # so redirect the runner at a temp tree rather than littering results/ and
    # leaving strays behind if the script dies partway.
    scratch = Path(tempfile.mkdtemp())
    real_root, real_ckpt = runner.ROOT, runner.CKPT_DIR
    runner.ROOT = scratch
    runner.CKPT_DIR = scratch / "results" / ".ckpt"
    try:
        return _run(args)
    finally:
        runner.ROOT, runner.CKPT_DIR = real_root, real_ckpt
        shutil.rmtree(scratch, ignore_errors=True)


def _run(args) -> int:

    cube = ablation_cube()
    record: dict = {"rung": args.rung, "budget_seconds": args.budget}
    started = time.perf_counter()

    print(f"=== fairness: {len(cube)} variants, {args.budget}s each on {args.rung} ===")
    by_time = {}
    for label in cube:
        by_time[label] = timed(args.rung, label, args.budget)
        r = by_time[label]
        print(f"  {label:<20} {r['train_seconds']:>6.2f}s  T={r['iterations']:>9,}  "
              f"expl={r['exploitability']:.8f}", flush=True)

    seconds = [r["train_seconds"] for r in by_time.values()]
    spread = max(seconds) / min(seconds) - 1.0
    fair = spread <= FAIRNESS_TOLERANCE
    print(f"  spread in train_seconds: {spread:.2%}  "
          f"({'FAIR' if fair else 'UNFAIR'}, tolerance {FAIRNESS_TOLERANCE:.0%})")

    print("\n=== isolation: sparse vs dense evaluation at the same budget ===")
    sparse = timed(args.rung, "cfr+", args.budget, fractions=(1.0,))
    dense = timed(args.rung, "cfr+", args.budget,
                  fractions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    drift = abs(dense["iterations"] - sparse["iterations"]) / sparse["iterations"]
    isolated = drift <= ISOLATION_TOLERANCE
    print(f"  1 evaluation : T={sparse['iterations']:,}")
    print(f"  10 evaluations: T={dense['iterations']:,}")
    print(f"  iteration drift: {drift:.2%}  "
          f"({'ISOLATED' if isolated else 'EVAL TIME LEAKING'}, tolerance {ISOLATION_TOLERANCE:.0%})")

    print("\n=== verdict: does equal-iterations rank the same as equal-wall-clock? ===")
    matched_iters = min(r["iterations"] for r in by_time.values())
    print(f"  matching every variant at T={matched_iters:,} iterations")
    by_iters = {}
    for label, config in cube.items():
        g = game(args.rung)
        solver = CFRSolver(g, config).train(matched_iters)
        by_iters[label] = exploitability(g, solver.average_strategy())

    rank_time = sorted(by_time, key=lambda k: by_time[k]["exploitability"])
    rank_iters = sorted(by_iters, key=lambda k: by_iters[k])
    print(f"\n  {'variant':<20}{'rank @ equal time':>19}{'rank @ equal iters':>20}{'moved':>8}")
    for label in rank_time:
        a, b = rank_time.index(label) + 1, rank_iters.index(label) + 1
        print(f"  {label:<20}{a:>19}{b:>20}{a - b:>+8}")
    same = rank_time == rank_iters
    note = ("iteration-matched comparisons give the same H1 answer" if same
            else "iteration-matched comparisons give a DIFFERENT H1 answer")
    print(f"\n  rankings {'AGREE' if same else 'DIFFER'} — {note}")

    record.update({
        "by_wallclock": by_time,
        "by_iterations": {"matched_iterations": matched_iters, "exploitability": by_iters},
        "fairness": {"spread": spread, "tolerance": FAIRNESS_TOLERANCE, "pass": fair},
        "isolation": {"sparse": sparse, "dense": dense, "drift": drift,
                      "tolerance": ISOLATION_TOLERANCE, "pass": isolated},
        "ranking_by_wallclock": rank_time,
        "ranking_by_iterations": rank_iters,
        "rankings_agree": same,
        "seconds": time.perf_counter() - started,
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2))

    print("\n" + "=" * 62)
    print(f"fairness   {'PASS' if fair else 'FAIL'}")
    print(f"isolation  {'PASS' if isolated else 'FAIL'}")
    print(f"wrote {_display(args.out)}  ({record['seconds']:.1f}s)")
    return 0 if (fair and isolated) else 1


if __name__ == "__main__":
    raise SystemExit(main())
