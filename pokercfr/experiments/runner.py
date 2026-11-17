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
import pickle
import platform
import time
from pathlib import Path

from pokercfr.core.config import RunSpec

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "results" / ".ckpt"


def build_game(name: str):
    """Resolve a rung name to a Game. Leduc rungs look like `leduc-3p-cap2`."""
    if name.startswith("leduc-"):
        from pokercfr.games import leduc
        _, players, cap = name.split("-")
        return leduc.game(players=int(players.rstrip("p")), raise_cap=int(cap[3:]))
    from pokercfr.games import kuhn_family
    return kuhn_family.game(name)


def build_solver(spec: RunSpec):
    """Map a spec's `variant` string onto a concrete solver.

    Variant names are the stable identity of an algorithm in `results/`, so they
    are parsed here and nowhere else. Adding an algorithm means adding a branch
    here and a name in the registry -- never a new runner.
    """
    from pokercfr.solvers.cfr import CFRConfig, CFRSolver, ablation_cube

    game = build_game(spec.game)
    cube = ablation_cube()
    if spec.variant in cube:
        return game, CFRSolver(game, cube[spec.variant])
    if spec.variant.startswith("dcfr-"):
        # dcfr-a1.5-b0-g2 -> DCFR with (alpha, beta, gamma) = (1.5, 0, 2)
        parts = dict(p[0] and (p[0], float(p[1:])) for p in spec.variant.split("-")[1:])
        return game, CFRSolver(game, CFRConfig(
            averaging="linear", dcfr=(parts["a"], parts["b"], parts["g"])))
    if spec.variant.startswith("mccfr-"):
        from pokercfr.solvers.mccfr import MCCFRSolver
        return game, MCCFRSolver(game, scheme=spec.variant.split("-", 1)[1], seed=spec.seed)
    if spec.variant == "lp-oracle":
        from pokercfr.solvers.lp_oracle import LPOracleSolver
        return game, LPOracleSolver(game)
    raise KeyError(f"unknown variant {spec.variant!r}")


def _ckpt_path(spec: RunSpec) -> Path:
    return CKPT_DIR / f"{spec.spec_hash()}.pkl"


def _save_checkpoint(spec: RunSpec, solver, curve: list[dict], elapsed: float) -> None:
    """Persist solver state so a killed run resumes instead of restarting.

    The upper rungs take 10+ hours per run on a laptop. Without this, a closed
    lid, a sleep or an OOM at hour 9 throws away hour 9, and with a deadline that
    is unrecoverable. Written atomically (temp file + replace) so a kill DURING
    the write cannot leave a corrupt checkpoint that poisons the resume.
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _ckpt_path(spec).with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump({"solver": solver, "curve": curve, "elapsed": elapsed}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(_ckpt_path(spec))


def _load_checkpoint(spec: RunSpec):
    path = _ckpt_path(spec)
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None  # corrupt or stale pickle: start over rather than crash the shard


def run_one(spec: RunSpec, resume: bool = True, progress=print) -> dict:
    """Train, measuring exploitability at each checkpoint. Returns a result record.

    Partial results are written after EVERY checkpoint, not only at the end, so a
    run that is still going (or that died) has already contributed a usable
    convergence curve. That matters when the analysis has to start before the
    longest runs finish.
    """
    from pokercfr.eval.best_response import nash_conv

    game, solver = build_solver(spec)
    checkpoints = spec.checkpoints or (spec.iterations,)
    curve, elapsed = [], 0.0

    if resume:
        state = _load_checkpoint(spec)
        if state is not None:
            solver, curve, elapsed = state["solver"], state["curve"], state["elapsed"]
            progress(f"RESUME {spec.game}/{spec.variant} s{spec.seed} "
                     f"from iteration {solver.iterations:,}")

    started = time.perf_counter() - elapsed
    for target in checkpoints:
        if target > spec.iterations:
            break
        if curve and target <= curve[-1]["iterations"]:
            continue  # already done before the interruption
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
        _save_checkpoint(spec, solver, curve, train_seconds)
        _write_result(spec, _record(spec, game, curve, complete=False))
        if spec.max_seconds is not None and train_seconds >= spec.max_seconds:
            progress(f"BUDGET {spec.game}/{spec.variant} s{spec.seed} stopped at "
                     f"{solver.iterations:,} iters ({train_seconds/3600:.1f} h)")
            break
    _ckpt_path(spec).unlink(missing_ok=True)  # finished: drop the resume state
    return _record(spec, game, curve, complete=True)


def _record(spec: RunSpec, game, curve: list[dict], complete: bool) -> dict:
    return {
        "spec": {**spec.__dict__, "spec_hash": spec.spec_hash()},
        "game": {"name": game.name, "players": game.num_players,
                 "info_sets": game.num_info_sets()},
        "curve": curve,
        # `complete` lets the analysis include an in-flight run's partial curve
        # while marking clearly that it has not reached its budget.
        "complete": complete,
        "env": {"python": platform.python_version(), "machine": platform.machine(),
                "processor": platform.processor(), "node": platform.node()},
    }


def _write_result(spec: RunSpec, record: dict) -> None:
    path = ROOT / spec.result_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(path)


def _worker(spec: RunSpec) -> str:
    try:
        record = run_one(spec)
    except NotImplementedError as exc:
        return f"SKIP  {spec.game:<22} {spec.variant:<18} (not implemented: {exc})"
    except Exception as exc:  # one bad cell must not kill a 15-hour shard
        return f"ERROR {spec.game:<22} {spec.variant:<18} {type(exc).__name__}: {exc}"
    _write_result(spec, record)
    last = record["curve"][-1] if record["curve"] else {}
    hours = last.get("train_seconds", 0) / 3600
    return (f"DONE  {spec.game:<22} {spec.variant:<18} "
            f"T={last.get('iterations','-'):<9} {hours:>6.2f}h "
            f"expl={last.get('exploitability')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, default=0, help="this machine's index")
    ap.add_argument("--of", type=int, default=1, help="total number of machines")
    ap.add_argument("--jobs", type=int, default=0, help="processes (0 = all cores)")
    ap.add_argument("--matrix", default="full",
                    help="which matrix to run; see registry.MATRICES")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--estimate", action="store_true",
                    help="measure s/iter on each rung and print projected CPU-hours")
    args = ap.parse_args()

    from pokercfr.experiments import registry
    if args.matrix not in registry.MATRICES:
        raise SystemExit(f"unknown matrix {args.matrix!r}; choose from: "
                         f"{', '.join(registry.MATRICES)}")
    specs = registry.MATRICES[args.matrix]()
    mine = [s for i, s in enumerate(specs) if i % args.of == args.shard]
    print(f"shard {args.shard}/{args.of}: {len(mine)} of {len(specs)} runs "
          f"from matrix {args.matrix!r}")
    if args.dry_run:
        for s in mine:
            done = "[done]" if (ROOT / s.result_path).exists() else ""
            print(f"  {s.game:<24} {s.variant:<20} seed={s.seed} {done}")
        return
    if args.estimate:
        registry.print_estimate(mine)
        return
    jobs = args.jobs or os.cpu_count() or 1
    with mp.Pool(jobs) as pool:
        for line in pool.imap_unordered(_worker, mine):
            print(line, flush=True)


if __name__ == "__main__":
    main()
