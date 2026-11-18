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
from dataclasses import replace
from pathlib import Path

from pokercfr.core.config import RunSpec

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "results" / ".ckpt"


def build_game(name: str):
    """Resolve a rung name to a Game.

    Leduc rungs are `leduc-<N>p-cap<M>` on the standard 6-card deck, or
    `leduc-r<R>s<S>-<N>p-cap<M>` to pin a different deck -- which H3 needs,
    because the deck has to be held fixed across player counts AND be large
    enough for the largest one. This mirrors leduc._config_name exactly, so a
    game's own name round-trips back to the same game.
    """
    if name.startswith("leduc-"):
        from pokercfr.games import leduc
        fields = name.split("-")[1:]
        deck = {}
        if fields[0].startswith("r") and "s" in fields[0]:
            ranks, suits = fields[0][1:].split("s")
            deck = {"ranks": int(ranks), "suits": int(suits)}
            fields = fields[1:]
        players, cap = fields
        return leduc.game(players=int(players.rstrip("p")),
                          raise_cap=int(cap.removeprefix("cap")), **deck)
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
        # dcfr-a1.5-b0-g2, or dcfr-alt-a1.5-b0-g2 for alternating updates
        fields = spec.variant.split("-")[1:]
        alternating = fields and fields[0] == "alt"
        if alternating:
            fields = fields[1:]
        parts = {f[0]: float(f[1:]) for f in fields}
        return game, CFRSolver(game, CFRConfig(
            averaging="linear", alternating=alternating,
            dcfr=(parts["a"], parts["b"], parts["g"])))
    if spec.variant.startswith("mccfr-"):
        from pokercfr.solvers.mccfr import MCCFRSolver
        return game, MCCFRSolver(game, scheme=spec.variant.split("-", 1)[1], seed=spec.seed)
    if spec.variant == "lp-oracle":
        from pokercfr.solvers.lp_oracle import LPOracleSolver
        return game, LPOracleSolver(game)
    raise KeyError(f"unknown variant {spec.variant!r}")


TIME_CHECKPOINTS = (0.05, 0.1, 0.2, 0.5, 1.0)
MAX_BATCH = 50_000


def time_budget_of(spec: RunSpec) -> float | None:
    """Seconds of training this spec is allotted, or None for an iteration run.

    Carried in `extra` rather than as a RunSpec field so that adding it does not
    change the hash of every spec already on disk.
    """
    budget = spec.extra.get("time_budget")
    return float(budget) if budget else None


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


def _run_timed(spec: RunSpec, game, solver, curve: list[dict], elapsed: float,
               progress) -> tuple[list[dict], float]:
    """Train against a wall-clock budget, measuring at fractions of it.

    H1 is stated in wall-clock terms, so iterations are an output here, not an
    input. Evaluation is timed separately and never added to `elapsed`: exact
    best response costs more than several training iterations on the bigger
    rungs, so counting it would penalise whichever variant we happened to
    measure more often.
    """
    from pokercfr.eval.exploitability import curve_point, evaluate

    budget = time_budget_of(spec)
    fractions = tuple(spec.extra.get("time_checkpoints", TIME_CHECKPOINTS))
    per_iter: float | None = None

    for fraction in sorted(fractions):
        target = fraction * budget
        if curve and elapsed >= target:
            continue
        while elapsed < target:
            remaining = target - elapsed
            if per_iter is None:
                batch = 1
            else:
                batch = min(MAX_BATCH, max(1, int(0.9 * remaining / per_iter)))
            started = time.perf_counter()
            solver.train(batch)
            taken = time.perf_counter() - started
            elapsed += taken
            per_iter = taken / batch

        eval_started = time.perf_counter()
        result = evaluate(game, solver.average_strategy())
        curve.append(curve_point(result, iterations=solver.iterations,
                                 train_seconds=elapsed,
                                 eval_seconds=time.perf_counter() - eval_started,
                                 time_fraction=fraction))
        _save_checkpoint(spec, solver, curve, elapsed)
        _write_result(spec, _record(spec, game, curve, complete=False))
        progress(f"  {spec.variant:<22} {fraction:>5.0%} of budget  "
                 f"T={solver.iterations:<10,} {elapsed:>7.2f}s")

    return curve, elapsed


def run_one(spec: RunSpec, resume: bool = True, progress=print) -> dict:
    """Train, measuring exploitability at each checkpoint. Returns a result record.

    Partial results are written after EVERY checkpoint, not only at the end, so a
    run that is still going (or that died) has already contributed a usable
    convergence curve. That matters when the analysis has to start before the
    longest runs finish.
    """
    from pokercfr.eval.exploitability import curve_point, evaluate

    game, solver = build_solver(spec)
    checkpoints = spec.checkpoints or (spec.iterations,)
    curve, elapsed = [], 0.0

    if resume:
        state = _load_checkpoint(spec)
        if state is not None:
            solver, curve, elapsed = state["solver"], state["curve"], state["elapsed"]
            progress(f"RESUME {spec.game}/{spec.variant} s{spec.seed} "
                     f"from iteration {solver.iterations:,}")

    if time_budget_of(spec) is not None:
        curve, elapsed = _run_timed(spec, game, solver, curve, elapsed, progress)
        _ckpt_path(spec).unlink(missing_ok=True)
        return _record(spec, game, curve, complete=True)

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
        result = evaluate(game, solver.average_strategy())
        curve.append(curve_point(result, iterations=solver.iterations,
                                 train_seconds=train_seconds,
                                 eval_seconds=time.perf_counter() - eval_started))
        started = time.perf_counter() - train_seconds  # resume the training clock
        _save_checkpoint(spec, solver, curve, train_seconds)
        _write_result(spec, _record(spec, game, curve, complete=False))
        expl = curve[-1]["exploitability"]
        progress(f"    {spec.game}/{spec.variant} s{spec.seed}  "
                 f"T={solver.iterations:,}/{spec.iterations:,}  "
                 f"{train_seconds / 60:.1f}m  "
                 + (f"expl={expl:.3e}" if expl is not None
                    else f"nashconv={curve[-1]['nashconv_mean']:.3e}"))
        if spec.max_seconds is not None and train_seconds >= spec.max_seconds:
            progress(f"BUDGET {spec.game}/{spec.variant} s{spec.seed} stopped at "
                     f"{solver.iterations:,} iters ({train_seconds/3600:.1f} h)")
            break
    _ckpt_path(spec).unlink(missing_ok=True)  # finished: drop the resume state
    return _record(spec, game, curve, complete=True)


def _record(spec: RunSpec, game, curve: list[dict], complete: bool) -> dict:
    from pokercfr.eval.exploitability import RECORD_SCHEMA_VERSION

    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "spec": {**spec.__dict__, "spec_hash": spec.spec_hash()},
        "game": {"name": game.name, "players": game.num_players,
                 "info_sets": game.num_info_sets()},
        "curve": curve,
        "time_budget": time_budget_of(spec),
        "iterations_reached": curve[-1]["iterations"] if curve else 0,
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


def estimate_seconds(spec: RunSpec, cache: dict) -> float:
    """Rough cost of one cell, for scheduling only -- never for reporting.

    A timed cell costs its budget. An iteration cell is measured from a single
    warm iteration, which is noisy in absolute terms but good enough to rank
    cells that differ by two orders of magnitude.
    """
    budget = time_budget_of(spec)
    if budget:
        return budget
    key = (spec.game, spec.variant)
    if key not in cache:
        try:
            _, solver = build_solver(spec)
            solver.step()                      # warm up the tables
            started = time.perf_counter()
            solver.step()
            cache[key] = time.perf_counter() - started
        except Exception:
            cache[key] = 0.0
    cap = (spec.max_seconds or float("inf"))
    return min(spec.iterations * cache[key], cap)


def schedule_longest_first(specs: list[RunSpec]) -> list[RunSpec]:
    """Longest-processing-time-first ordering.

    A pool hands the next queued cell to whichever core frees up, so the cheap
    cells should be left till last: they fill the gaps while the long ones are
    still running. Feeding cells in matrix order instead leaves cores idle at
    the tail waiting on one straggler -- measured at 14.0h versus 10.6h for the
    same 33-cell shard, against a perfect-balance bound of 10.5h.

    This cannot speed up an individual run: CFR is sequential, iteration t+1
    needs t's regrets. It only stops cores sitting idle.
    """
    cache: dict = {}
    return sorted(specs, key=lambda s: -estimate_seconds(s, cache))


def _already_complete(spec: RunSpec) -> bool:
    """True if this cell has a finished result on disk.

    Lets a killed or timed-out worker be relaunched without redoing work -- which
    matters for 12-hour Kaggle sessions and for anyone closing a laptop.
    """
    path = ROOT / spec.result_path
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text()).get("complete"))
    except Exception:
        return False


PROGRESS_EVERY_SECONDS = 60.0


def _worker(spec: RunSpec) -> str:
    # Checkpoints are log-spaced, so the first dozen land within milliseconds on
    # a cheap rung. Throttle to roughly one line a minute per cell: enough to
    # see a run is alive over seven hours, without burying the DONE lines.
    last = [0.0]

    def progress(line: str) -> None:
        now = time.perf_counter()
        if now - last[0] >= PROGRESS_EVERY_SECONDS or last[0] == 0.0:
            last[0] = now
            print(line, flush=True)

    try:
        record = run_one(spec, progress=progress)
    except NotImplementedError as exc:
        return f"SKIP  {spec.game:<22} {spec.variant:<18} (not implemented: {exc})"
    except Exception as exc:  # one bad cell must not kill a 7-hour shard
        return f"ERROR {spec.game:<22} {spec.variant:<18} {type(exc).__name__}: {exc}"
    _write_result(spec, record)
    last = record["curve"][-1] if record["curve"] else {}
    hours = last.get("train_seconds", 0) / 3600
    expl = last.get("exploitability")
    metric = f"expl={expl:.4e}" if isinstance(expl, float) else \
        f"nashconv={last.get('nashconv_mean', 0):.4e}"
    return (f"DONE  {spec.game:<22} {spec.variant:<18} "
            f"T={last.get('iterations', '-'):<9} {hours:>5.2f}h {metric}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", default="0",
                    help="this worker's shard index, or a comma-separated list "
                         "(e.g. 0,1,2,3) so a fast machine can take several")
    ap.add_argument("--of", type=int, default=1, help="total number of shards")
    ap.add_argument("--jobs", type=int, default=0, help="processes (0 = all cores)")
    ap.add_argument("--matrix", default="full",
                    help="which matrix to run; see registry.MATRICES")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--estimate", action="store_true",
                    help="measure s/iter on each rung and print projected CPU-hours")
    ap.add_argument("--only", choices=("timed", "iteration"), default=None,
                    help="restrict to wall-clock-budgeted or iteration-budgeted cells. "
                         "A timed comparison is only valid between runs on the SAME "
                         "hardware, so those cells must not be spread across machines; "
                         "iteration-budgeted cells are hardware-independent and can go "
                         "anywhere")
    ap.add_argument("--budget", default=None,
                    help="comma-separated time budgets in seconds to keep "
                         "(e.g. 600 for H1, or 3600,21600 for the H3 Leduc cells)")
    ap.add_argument("--games", default=None,
                    help="comma-separated rung-name prefixes to keep")
    ap.add_argument("--order", choices=("longest-first", "matrix"), default="longest-first",
                    help="longest-first keeps cores busy to the end; matrix "
                         "preserves registry order")
    ap.add_argument("--force", action="store_true",
                    help="re-run cells whose result is already complete")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="train each cell for this many wall-clock seconds instead of "
                         "a fixed iteration count (H1 is a wall-clock hypothesis)")
    args = ap.parse_args()

    from pokercfr.experiments import registry
    if args.matrix not in registry.MATRICES:
        raise SystemExit(f"unknown matrix {args.matrix!r}; choose from: "
                         f"{', '.join(registry.MATRICES)}")
    specs = registry.MATRICES[args.matrix]()
    if args.time_budget is not None:
        specs = [replace(s, extra={**s.extra, "time_budget": args.time_budget})
                 for s in specs]
    try:
        wanted = {int(k) for k in str(args.shard).split(",") if k != ""}
    except ValueError as exc:
        raise SystemExit(
            f"--shard must be an int or a comma-separated list, got {args.shard!r}") from exc
    if any(k >= args.of for k in wanted):
        raise SystemExit(f"--shard index must be < --of ({args.of})")
    if args.only == "timed":
        specs = [s for s in specs if time_budget_of(s)]
    elif args.only == "iteration":
        specs = [s for s in specs if not time_budget_of(s)]
    if args.budget:
        keep = {float(b) for b in args.budget.split(",")}
        specs = [s for s in specs if time_budget_of(s) in keep]
    if args.games:
        prefixes = tuple(args.games.split(","))
        specs = [s for s in specs if s.game.startswith(prefixes)]
    if not specs:
        print("no cells match those filters")
        return

    mine = [s for i, s in enumerate(specs) if i % args.of in wanted]
    print(f"shard {sorted(wanted)} of {args.of}: {len(mine)} of {len(specs)} runs "
          f"from matrix {args.matrix!r}")

    if not args.force:
        before = len(mine)
        mine = [s for s in mine if not _already_complete(s)]
        if before != len(mine):
            print(f"skipping {before - len(mine)} already-complete runs "
                  f"(--force to redo them)")
    if args.dry_run:
        for s in mine:
            done = "[done]" if (ROOT / s.result_path).exists() else ""
            print(f"  {s.game:<24} {s.variant:<20} seed={s.seed} {done}")
        return
    if args.estimate:
        registry.print_estimate(mine)
        return
    if not mine:
        print("nothing to do: every cell in this shard is already complete")
        return

    if args.order == "longest-first" and len(mine) > 1:
        print("estimating cell costs to order longest-first...", flush=True)
        mine = schedule_longest_first(mine)

    jobs = args.jobs or os.cpu_count() or 1
    total = len(mine)
    started = time.perf_counter()
    done = 0
    print(f"running {total} cells on {jobs} cores\n", flush=True)
    with mp.Pool(jobs) as pool:
        for line in pool.imap_unordered(_worker, mine):
            done += 1
            elapsed = time.perf_counter() - started
            # Rough: remaining cells assumed to average what finished ones did.
            # Costs vary a lot per cell, so treat it as an order of magnitude.
            eta = (elapsed / done) * (total - done)
            print(f"[{done}/{total}] {line}   "
                  f"| elapsed {elapsed / 3600:.2f}h, eta ~{eta / 3600:.2f}h",
                  flush=True)
    print(f"\nshard complete: {total} cells in "
          f"{(time.perf_counter() - started) / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
