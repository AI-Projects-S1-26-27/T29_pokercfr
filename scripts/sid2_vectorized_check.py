"""Scalar vs vectorized CFR: equivalence AND speed, measured on the same runs.

OWNER: Siddhant. Produces results/verify/vectorized_check.json (+ .log via tee).

Every measurement runs in a FRESH process (multiprocessing "spawn"), so the peak
RSS reported for one solver is not polluted by another's tables, and both
solvers see the same cold start. For each (rung, config):

  scalar      build, 1 warm-up iteration (it allocates the tables, so the
              registry's estimator excludes it too), then k timed iterations
  vectorized  build, 1 warm-up, then m timed iterations

Tables of BOTH are snapshotted at iteration EQUIV_AT (30; on the 52-card rung,
the last scalar iteration, 3) and compared there. They are deliberately not
compared at the end of the timing loop: over thousands of iterations CFR's
dynamics amplify summation-order rounding geometrically on some rungs, for two
scalar runs as much as for scalar vs vectorized -- scripts/sid3 measures that
drift with a scalar-vs-scalar control.

Equivalence compares regret_sum, strategy_sum, average and current strategy,
info-set keys and exploitability. Speedup = scalar s/iter / vectorized s/iter.
On the 52-card rung it also times one exact evaluation through eval/ (the
runner's evaluator) against solvers/vectorized.nash_conv on the same policy.

    python scripts/sid2_vectorized_check.py           # ~30 min, dominated by the 52-card scalar runs
    python scripts/sid2_vectorized_check.py --quick   # skip the 52-card rung

Exits 1 if any equivalence check fails.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "verify" / "vectorized_check.json"
TOP = "13rank-4suit-2dealt-2p"
RUNGS = ("kuhn", "4rank", "4rank-2suit", "4rank-2suit-2dealt", "kuhn-3p", "4rank-2suit-3p",
         "4rank-2suit-2dealt-3p", "4rank-2suit-4p", TOP)
CONFIGS = ("cfr", "cfr+")          # simultaneous and alternating, the two traversal shapes
EQUIV_AT = 30
REL_TOL = 1e-11                    # same bar as tests/test_vectorized.py; see its docstring
SCALAR_SECONDS = 30.0              # timing budget per scalar measurement (min 2 iterations)
VECTOR_SECONDS = 10.0


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KiB


def _copy(table: dict) -> dict:
    return {key: dict(row) for key, row in table.items()}


def _child_train(rung: str, config: str, impl: str, timed: int | None, snapshot_at: int | None,
                 budget: float, conn) -> None:
    sys.path.insert(0, str(ROOT))
    from pokercfr.games.kuhn_family import game
    from pokercfr.solvers.cfr import CFRSolver, ablation_cube
    from pokercfr.solvers.vectorized import VectorizedCFRSolver

    baseline = _rss_mb()
    g = game(rung)
    cls = CFRSolver if impl == "scalar" else VectorizedCFRSolver
    t0 = time.perf_counter()
    solver = cls(g, ablation_cube()[config])
    build = time.perf_counter() - t0
    t0 = time.perf_counter()
    solver.step()
    warmup = time.perf_counter() - t0
    times, snapshot = [], None
    while True:
        if snapshot_at is not None and solver.iterations == snapshot_at:
            # Copies: CFRSolver.regret_sum is its live table and keeps changing.
            snapshot = {"regret_sum": _copy(solver.regret_sum),
                        "strategy_sum": _copy(solver.strategy_sum),
                        "average": solver.average_strategy(), "current": solver.current_strategy()}
        done_timing = (len(times) >= timed) if timed else (sum(times) >= budget and len(times) >= 2)
        if done_timing and (snapshot_at is None or solver.iterations >= snapshot_at):
            break
        t0 = time.perf_counter()
        solver.step()
        times.append(time.perf_counter() - t0)
    tables = snapshot or {"regret_sum": _copy(solver.regret_sum),
                          "strategy_sum": _copy(solver.strategy_sum),
                          "average": solver.average_strategy(),
                          "current": solver.current_strategy()}
    conn.send({"build_seconds": build, "warmup_seconds": warmup, "iteration_seconds": times,
               "iterations_total": solver.iterations, "baseline_rss_mb": baseline,
               "peak_rss_mb": _rss_mb(), "tables": tables,
               "tables_at": snapshot_at or solver.iterations})
    conn.close()


def _child_eval(rung: str, which: str, policy: dict, conn) -> None:
    sys.path.insert(0, str(ROOT))
    from pokercfr.games.kuhn_family import game

    baseline = _rss_mb()
    g = game(rung)
    if which == "eval":
        from pokercfr.eval.exploitability import evaluate as fn
    else:
        from pokercfr.solvers.vectorized import nash_conv as fn
    t0 = time.perf_counter()
    result = fn(g, policy)
    conn.send({"seconds": time.perf_counter() - t0, "gains": result.gains,
               "policy_values": result.policy_values, "baseline_rss_mb": baseline,
               "peak_rss_mb": _rss_mb()})
    conn.close()


def run_child(target, *args) -> dict:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=target, args=(*args, child))
    proc.start()
    out = parent.recv()
    proc.join()
    return out


def summarize_times(times: list[float]) -> dict:
    """Per-iteration timings reduced to summary statistics, so the JSON stays
    small enough to commit (the raw lists run to tens of thousands of entries)."""
    ordered = sorted(times)
    return {"count": len(times), "mean": sum(times) / len(times),
            "median": ordered[len(ordered) // 2], "min": ordered[0], "max": ordered[-1]}


def gap(a: dict, b: dict) -> dict:
    keys_equal = set(a) == set(b) and all(set(a[k]) == set(b[k]) for k in a)
    if not keys_equal:
        return {"keys_equal": False}
    abs_gap = max((abs(a[k][x] - b[k][x]) for k in a for x in a[k]), default=0.0)
    scale = max((abs(a[k][x]) for k in a for x in a[k]), default=0.0)
    return {"keys_equal": True, "max_abs_diff": abs_gap, "max_abs_entry": scale,
            "rel_diff": abs_gap / max(scale, 1.0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help=f"skip {TOP}")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    rungs = [r for r in RUNGS if not (args.quick and r == TOP)]

    rows, failures = [], []
    print(f"{'rung':<24}{'config':<7}{'scalar s/it':>13}{'vector s/it':>13}{'speedup':>10}"
          f"{'scalar MB':>11}{'vector MB':>11}{'max rel diff':>14}")
    for rung in rungs:
        for config in CONFIGS:
            timed = 2 if rung == TOP else None
            at = None if rung == TOP else EQUIV_AT
            sc = run_child(_child_train, rung, config, "scalar", timed, at, SCALAR_SECONDS)
            vec = run_child(_child_train, rung, config, "vectorized", None, sc["tables_at"],
                            VECTOR_SECONDS)
            diffs = {name: gap(sc["tables"][name], vec["tables"][name]) for name in sc["tables"]}
            ok = all(d["keys_equal"] and d["rel_diff"] <= REL_TOL for d in diffs.values())
            if not ok:
                failures.append(f"{rung}/{config}: {diffs}")
            s_it = sum(sc["iteration_seconds"]) / len(sc["iteration_seconds"])
            v_it = sum(vec["iteration_seconds"]) / len(vec["iteration_seconds"])
            row = {"rung": rung, "config": config, "compared_at_iteration": sc["tables_at"],
                   "equivalent": ok, "diffs": diffs, "speedup": s_it / v_it}
            for name, m in (("scalar", sc), ("vectorized", vec)):
                row[name] = {k: v for k, v in m.items() if k != "tables"}
                row[name]["iteration_seconds"] = summarize_times(m["iteration_seconds"])
                row[name]["seconds_per_iteration"] = s_it if name == "scalar" else v_it
            rows.append(row)
            worst = max(d.get("rel_diff", float("inf")) for d in diffs.values())
            print(f"{rung:<24}{config:<7}{s_it:>13.3e}{v_it:>13.3e}{s_it / v_it:>10.1f}"
                  f"{sc['peak_rss_mb']:>11.0f}{vec['peak_rss_mb']:>11.0f}{worst:>14.2e}"
                  f"  {'OK' if ok else 'MISMATCH'}", flush=True)
            if rung == TOP and config == "cfr+":
                policy = sc["tables"]["average"]

    evals = None
    if TOP in rungs:
        print(f"\nexact evaluation of the scalar CFR+ average strategy on {TOP}")
        evals = {w: run_child(_child_eval, TOP, w, policy) for w in ("eval", "vectorized")}
        worst = max(abs(a - b) for a, b in zip(evals["eval"]["gains"],
                                                evals["vectorized"]["gains"], strict=True))
        evals["max_abs_gain_diff"] = worst
        evals["speedup"] = evals["eval"]["seconds"] / evals["vectorized"]["seconds"]
        if worst > 1e-10:
            failures.append(f"{TOP} evaluators disagree by {worst:.3e}")
        for w in ("eval", "vectorized"):
            e = evals[w]
            print(f"  {w:<11}{e['seconds']:>10.2f}s  peak {e['peak_rss_mb']:>7.0f} MB  "
                  f"gains {e['gains']}")
        print(f"  max |gain diff| {worst:.2e}, speedup {evals['speedup']:.0f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "rel_tol": REL_TOL, "rows": rows, "top_rung_evaluation": evals, "failures": failures,
        "env": {"python": platform.python_version(), "machine": platform.machine(),
                "processor": platform.processor(), "node": platform.node(),
                "cpus": mp.cpu_count()}}, indent=2))
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    print("ALL EQUIVALENT" if not failures else "FAILURES:\n  " + "\n  ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
