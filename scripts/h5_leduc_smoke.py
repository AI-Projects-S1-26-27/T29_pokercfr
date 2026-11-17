#!/usr/bin/env python3
"""Does Leduc run through the solver harness unchanged, and how far does it go?

Leduc is the only game with a chance node in the MIDDLE of the tree, which is
why the frozen contract has a generic is_chance_node() rather than a root-only
deal. The claim is that the solver therefore needs no Leduc-specific code at
all. "Should work unchanged" is a design claim, so this checks it.

Second job: the feasibility frontier. H3 wants player counts 2-6 with the deck
held fixed, but information sets grow explosively with seats, and a deck large
enough for six players is larger still. This measures where exact evaluation
stops being possible on this hardware, which the proposal promised as a result
in its own right.

Writes results/verify/leduc_smoke.json.

    python scripts/h5_leduc_smoke.py
    python scripts/h5_leduc_smoke.py --iterations 50 --out /tmp/smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pokercfr.eval.exploitability import evaluate  # noqa: E402
from pokercfr.experiments.runner import build_game  # noqa: E402
from pokercfr.solvers.cfr import CFRSolver, ablation_cube  # noqa: E402

DEFAULT_OUT = ROOT / "results" / "verify" / "leduc_smoke.json"

# Candidate rungs for H3. The deck must be held fixed across player counts and
# be large enough for the largest, so the bigger counts need an explicit deck.
FRONTIER = ["leduc-2p-cap2", "leduc-3p-cap2", "leduc-4p-cap2", "leduc-5p-cap2",
            "leduc-6p-cap2", "leduc-r4s2-2p-cap2", "leduc-r4s2-3p-cap2",
            "leduc-r4s2-4p-cap2", "leduc-r4s2-5p-cap2", "leduc-r4s3-6p-cap2"]
MEASURE_UNDER_INFO_SETS = 200_000


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def walk_checks(game, max_states: int = 60_000) -> dict:
    """Zero-sum at terminals, and chance probabilities summing to 1 everywhere."""
    seen = chance_nodes = terminals = 0
    worst_zero_sum = 0.0
    worst_chance = 0.0
    stack = [game.new_initial_state()]
    while stack and seen < max_states:
        state = stack.pop()
        seen += 1
        if state.is_terminal():
            terminals += 1
            worst_zero_sum = max(worst_zero_sum, abs(sum(state.utilities())))
            continue
        if state.is_chance_node():
            chance_nodes += 1
            outcomes = state.chance_outcomes()
            worst_chance = max(worst_chance, abs(sum(p for _, p in outcomes) - 1.0))
            stack.extend(child for child, _ in outcomes)
            continue
        stack.extend(state.apply_action(a) for a in state.legal_actions())
    return {"states_walked": seen, "chance_nodes": chance_nodes, "terminals": terminals,
            "exhaustive": seen < max_states,
            "worst_zero_sum_error": worst_zero_sum,
            "worst_chance_prob_error": worst_chance}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    record: dict = {"iterations": args.iterations}
    failures: list[str] = []
    started = time.perf_counter()

    print("=== tree checks: zero-sum payoffs and chance probabilities ===")
    record["walks"] = {}
    for rung in ("leduc-2p-cap2", "leduc-2p-cap1", "leduc-3p-cap2"):
        checks = walk_checks(build_game(rung))
        record["walks"][rung] = checks
        ok = checks["worst_zero_sum_error"] < 1e-9 and checks["worst_chance_prob_error"] < 1e-9
        if not ok:
            failures.append(f"{rung}: tree checks failed")
        print(f"  {rung:<20} states={checks['states_walked']:>7,} "
              f"chance_nodes={checks['chance_nodes']:>6,} "
              f"zero_sum_err={checks['worst_zero_sum_error']:.2e} "
              f"chance_err={checks['worst_chance_prob_error']:.2e} "
              f"{'OK' if ok else 'FAIL'}", flush=True)

    print("\n=== every ablation corner runs on Leduc with no solver changes ===")
    rung = "leduc-2p-cap2"
    game = build_game(rung)
    record["corners"] = {}
    for label, config in ablation_cube().items():
        solver = CFRSolver(game, config)
        t0 = time.perf_counter()
        solver.train(10)
        early = evaluate(game, solver.average_strategy())
        solver.train(args.iterations - 10)
        late = evaluate(game, solver.average_strategy())
        per_iter = (time.perf_counter() - t0) / args.iterations
        discovered = len(solver.average_strategy())
        decreased = late.exploitability < early.exploitability
        complete = discovered == game.num_info_sets()
        record["corners"][label] = {
            "exploitability_at_10": early.exploitability,
            "exploitability_at_end": late.exploitability,
            "decreased": decreased,
            "info_sets_discovered": discovered,
            "info_sets_expected": game.num_info_sets(),
            "all_info_sets_visited": complete,
            "seconds_per_iteration": per_iter,
        }
        if not decreased:
            failures.append(f"{label}: exploitability did not decrease on {rung}")
        if not complete:
            failures.append(f"{label}: visited {discovered} of {game.num_info_sets()} info sets")
        print(f"  {label:<20} {early.exploitability:.6f} -> {late.exploitability:.6f} "
              f"info_sets={discovered}/{game.num_info_sets()} {per_iter:>7.4f}s/it "
              f"{'OK' if decreased and complete else 'FAIL'}", flush=True)

    print("\n=== multiplayer path: seat-wise NashConv on 3-player Leduc ===")
    g3 = build_game("leduc-3p-cap2")
    solver = CFRSolver(g3, ablation_cube()["cfr+"])
    solver.train(max(10, args.iterations // 10))
    result = evaluate(g3, solver.average_strategy())
    record["multiplayer"] = {
        "rung": "leduc-3p-cap2", "iterations": solver.iterations,
        "nashconv_total": result.total, "nashconv_mean": result.mean,
        "nashconv_max": result.max, "max_seat": result.max_seat, "gains": result.gains,
    }
    nonneg = all(g >= -1e-12 for g in result.gains)
    if not nonneg:
        failures.append("3-player gains went negative")
    print(f"  gains={[round(g, 6) for g in result.gains]} mean={result.mean:.6f} "
          f"max_seat={result.max_seat} {'OK' if nonneg else 'FAIL'}")

    print("\n=== feasibility frontier for H3 ===")
    print(f"  {'rung':<22}{'deck':>6}{'info sets':>13}{'s/iter':>10}  status")
    record["frontier"] = {}
    for rung in FRONTIER:
        try:
            g = build_game(rung)
        except Exception as exc:
            record["frontier"][rung] = {"buildable": False, "error": str(exc)}
            print(f"  {rung:<22}{'':>6}{'':>13}{'':>10}  CANNOT BUILD: {str(exc)[:44]}")
            continue
        size = g.num_info_sets()
        entry = {"buildable": True, "deck_size": g.config.deck_size, "info_sets": size}
        if size <= MEASURE_UNDER_INFO_SETS:
            solver = CFRSolver(g, ablation_cube()["cfr+"])
            t0 = time.perf_counter()
            solver.train(2)
            entry["seconds_per_iteration"] = (time.perf_counter() - t0) / 2
            entry["status"] = "measured"
            print(f"  {rung:<22}{g.config.deck_size:>6}{size:>13,}"
                  f"{entry['seconds_per_iteration']:>10.4f}  measured", flush=True)
        else:
            entry["status"] = "too large to attempt"
            print(f"  {rung:<22}{g.config.deck_size:>6}{size:>13,}{'-':>10}  "
                  f"too large (> {MEASURE_UNDER_INFO_SETS:,} info sets)")
        record["frontier"][rung] = entry

    record["failures"] = failures
    record["seconds"] = time.perf_counter() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2))

    print("\n" + "=" * 68)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("all checks passed: Leduc runs through the harness with no solver changes")
    print(f"wrote {_display(args.out)}  ({record['seconds']:.1f}s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
