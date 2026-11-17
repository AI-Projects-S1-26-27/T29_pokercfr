#!/usr/bin/env python3
"""Fixed-deck multiplayer check: the evidence the report's H3 section cites.

For players 2..6 on the SAME 6-card deck and the same raise cap, train CFR+ under
an equal wall-clock budget and record seat-wise NashConv at log-spaced
checkpoints. Reports, per player count:

  - whether per-seat (mean) NashConv settles, and whether it settles above zero
  - which seat has the largest deviation incentive (position effects)

and, across player counts, whether the level rises with N. Rising is judged on
the MEAN, never the total: the total grows with N simply because there are more
seats that could deviate (see pokercfr/eval/exploitability.py). It is judged at
the largest iteration count every player count reached, so a slow N is never
compared at fewer iterations than a fast one. The budget-end values are recorded
too, labelled as such.

The deck is asserted identical across every run and written to the output, so
the report can state the control instead of asserting it. A player count the deck
cannot serve is recorded as infeasible, never resized.

Writes results/verify/fixed_deck_h3.json.

    python scripts/s7_fixed_deck_check.py | tee results/verify/fixed_deck_h3.log
    python scripts/s7_fixed_deck_check.py --budget 60 --players 2 3   # quick look
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pokercfr.eval.exploitability import curve_point, evaluate  # noqa: E402
from pokercfr.experiments.registry import epsilon_for  # noqa: E402
from pokercfr.games import leduc  # noqa: E402
from pokercfr.solvers.cfr import CFR_PLUS, CFRSolver  # noqa: E402

OUT = ROOT / "results" / "verify" / "fixed_deck_h3.json"
SETTLED = 0.10  # mean NashConv moved < 10% over the last checkpoint doubling


def run(g: leduc.LeducGame, budget: float, max_iters: int) -> list[dict]:
    """CFR+ until `budget` training seconds, evaluating at iterations 1, 2, 4, ...
    Evaluation time is excluded from the budget and recorded separately."""
    solver, curve, train, target = CFRSolver(g, CFR_PLUS), [], 0.0, 1
    while solver.iterations < max_iters:
        start = time.perf_counter()
        solver.step()
        train += time.perf_counter() - start
        if solver.iterations == target or train >= budget:
            start = time.perf_counter()
            result = evaluate(g, solver.average_strategy())
            curve.append(curve_point(result, iterations=solver.iterations, train_seconds=train,
                                     eval_seconds=time.perf_counter() - start))
            print(f"    T={solver.iterations:>6} train={train:7.1f}s mean={result.mean:.6f} "
                  f"max={result.max:.6f} (seat {result.max_seat})", flush=True)
            target *= 2
        if train >= budget:
            break
    return curve


def summarize(curve: list[dict], epsilon: float) -> dict:
    last = curve[-1]
    prev = next((p for p in reversed(curve) if p["iterations"] <= last["iterations"] // 2), None)
    change = abs(last["nashconv_mean"] - prev["nashconv_mean"]) / prev["nashconv_mean"] \
        if prev and prev["nashconv_mean"] > 0 else None
    return {
        "iterations_reached": last["iterations"],
        "final_mean": last["nashconv_mean"],
        "final_total": last["nashconv_total"],
        "final_max": last["nashconv_max"],
        "max_seat": last["nashconv_max_seat"],
        "gains": last["gains"],
        "last_doubling_rel_change": change,
        "settled": change is not None and change < SETTLED,
        "epsilon": epsilon,
        "above_epsilon": last["nashconv_mean"] > epsilon,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--players", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--cap", type=int, default=1,
                        help="raise cap held fixed across N (default 1: the only cap that reaches 4p)")
    parser.add_argument("--budget", type=float, default=600.0,
                        help="training seconds per player count (default 600)")
    parser.add_argument("--max-iters", type=int, default=100_000)
    args = parser.parse_args()

    runs, decks = [], set()
    for n in args.players:
        try:
            g = leduc.game(players=n, raise_cap=args.cap)
        except ValueError as err:
            runs.append({"players": n, "feasible": False, "reason": str(err)})
            print(f"{n}p: infeasible on the fixed deck -- {err}")
            continue
        decks.add(g.config.deck_size)
        print(f"{g.name}: deck={g.config.deck_size} info_sets={g.num_info_sets():,}", flush=True)
        curve = run(g, args.budget, args.max_iters)
        runs.append({"players": n, "feasible": True, "game": g.name, "deck_size": g.config.deck_size,
                     "info_sets": g.num_info_sets(), "curve": curve,
                     "summary": summarize(curve, epsilon_for(g.name))})

    assert len(decks) == 1, f"deck size varied across player counts: {decks}"
    feasible = [r for r in runs if r["feasible"]]

    # Compare across N at the largest checkpoint every player count reached.
    common = max(set.intersection(*({p["iterations"] for p in r["curve"]} for r in feasible)))
    at_common = {r["players"]: next(p for p in r["curve"] if p["iterations"] == common)["nashconv_mean"]
                 for r in feasible}
    ns = sorted(at_common)
    rises = all(at_common[a] < at_common[b] for a, b in zip(ns, ns[1:]))

    print(f"\n{'N':>2} {'iters':>7} {'mean':>10} {'total':>10} {'max seat':>8} {'settled':>7} "
          f"{'> eps':>5}  per-seat gains")
    for r in feasible:
        s = r["summary"]
        print(f"{r['players']:>2} {s['iterations_reached']:>7} {s['final_mean']:>10.6f} "
              f"{s['final_total']:>10.6f} {s['max_seat']:>8} {str(s['settled']):>7} "
              f"{str(s['above_epsilon']):>5}  {[round(x, 4) for x in s['gains']]}")
    print(f"\nmean NashConv at common T={common}: "
          + ", ".join(f"{n}p={at_common[n]:.6f}" for n in ns))
    print(f"rises monotonically with N at common T: {rises}")
    print(f"deck held fixed at {decks.pop() if decks else '?'} cards; cap {args.cap}; "
          f"{args.budget:g}s CFR+ training per N")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "solver": "cfr+", "raise_cap": args.cap, "budget_seconds": args.budget,
        "deck_size": feasible[0]["deck_size"], "deck_held_fixed": True,
        "infeasible_players": [r["players"] for r in runs if not r["feasible"]],
        "common_iterations": common, "mean_nashconv_at_common_iterations": at_common,
        "rises_with_players_at_common_iterations": rises,
        "settled_threshold": SETTLED,
        "env": {"machine": platform.platform(), "python": platform.python_version()},
        "runs": runs,
    }, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
