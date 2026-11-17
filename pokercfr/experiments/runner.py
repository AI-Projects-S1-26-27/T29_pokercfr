"""Run the results matrix, sharded across machines.

    # on laptop k of 8 (k = 0..7)
    python -m pokercfr.experiments.runner --shard 0 --of 8

Each machine takes every spec where `index % of == shard`, runs it across all
local cores, and writes one JSON per run into results/. Because a run's filename
is its RunSpec hash, two machines cannot collide and a rerun overwrites rather
than duplicating. Commit results/ when a shard finishes.

Nothing is distributed: CFR's regret tables want shared memory, so the unit of
parallelism is a whole run, never a piece of one. With 8 laptops x ~8 cores that
is still ~64 runs in flight.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import time
from pathlib import Path

from pokercfr.core.config import RunSpec

ROOT = Path(__file__).resolve().parents[2]


def build_solver(spec: RunSpec):
    """Map a spec's `variant` string onto a concrete solver."""
    from pokercfr.games import kuhn_family
    from pokercfr.solvers.cfr import CFRSolver, ablation_cube

    game = kuhn_family.game(spec.game)
    cube = ablation_cube()
    if spec.variant in cube:
        return game, CFRSolver(game, cube[spec.variant])
    if spec.variant.startswith("mccfr-"):
        from pokercfr.solvers.mccfr import MCCFRSolver
        return game, MCCFRSolver(game, scheme=spec.variant.split("-", 1)[1], seed=spec.seed)
    raise KeyError(f"unknown variant {spec.variant!r}")


def run_one(spec: RunSpec) -> dict:
    """Train, measuring exploitability at each checkpoint. Returns a result record."""
    from pokercfr.eval.best_response import nash_conv

    game, solver = build_solver(spec)
    checkpoints = spec.checkpoints or (spec.iterations,)
    curve, started = [], time.perf_counter()
    for target in checkpoints:
        if target > spec.iterations:
            break
        while solver.iterations < target:
            solver.step()
        train_seconds = time.perf_counter() - started
        # Evaluation time is excluded from train_seconds on purpose: a wall-clock
        # comparison must not be polluted by how often we chose to measure.
        eval_started = time.perf_counter()
        result = nash_conv(game, solver.average_strategy())
        curve.append({
            "iterations": solver.iterations,
            "train_seconds": train_seconds,
            "eval_seconds": time.perf_counter() - eval_started,
            "nashconv_total": result.total,
            "nashconv_mean": result.mean,
            "nashconv_max": result.max,
            "exploitability": result.exploitability if game.num_players == 2 else None,
            "gains": result.gains,
        })
        started = time.perf_counter() - train_seconds  # resume the training clock
    return {
        "spec": {**spec.__dict__, "spec_hash": spec.spec_hash()},
        "game": {"name": game.name, "players": game.num_players,
                 "info_sets": game.num_info_sets()},
        "curve": curve,
        "env": {"python": platform.python_version(), "machine": platform.machine(),
                "processor": platform.processor(), "node": platform.node()},
    }


def _worker(spec: RunSpec) -> str:
    path = ROOT / spec.result_path
    try:
        record = run_one(spec)
    except NotImplementedError as exc:
        return f"SKIP  {spec.game:<22} {spec.variant:<18} (not implemented: {exc})"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    last = record["curve"][-1] if record["curve"] else {}
    return (f"DONE  {spec.game:<22} {spec.variant:<18} "
            f"T={last.get('iterations','-'):<7} expl={last.get('exploitability')} "
            f"-> {spec.result_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, default=0, help="this machine's index")
    ap.add_argument("--of", type=int, default=1, help="total number of machines")
    ap.add_argument("--jobs", type=int, default=0, help="processes (0 = all cores)")
    ap.add_argument("--matrix", default="full", choices=("full", "ablation"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from pokercfr.experiments import registry
    specs = registry.full_matrix() if args.matrix == "full" else registry.ablation_matrix()
    mine = [s for i, s in enumerate(specs) if i % args.of == args.shard]
    print(f"shard {args.shard}/{args.of}: {len(mine)} of {len(specs)} runs")
    if args.dry_run:
        for s in mine:
            print(f"  {s.game:<22} {s.variant:<18} seed={s.seed} -> {s.result_path}")
        return
    jobs = args.jobs or os.cpu_count() or 1
    with mp.Pool(jobs) as pool:
        for line in pool.imap_unordered(_worker, mine):
            print(line, flush=True)


if __name__ == "__main__":
    main()
