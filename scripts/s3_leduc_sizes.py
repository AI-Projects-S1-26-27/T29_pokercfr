#!/usr/bin/env python3
"""Leduc size census: the Leduc half of the game-size ladder.

For players 2..6 and raise caps 1..3 on the FIXED standard deck, reports info
sets, terminal betting histories, deck size, and measured seconds per CFR+
iteration. Cells too slow to time are PROJECTED from an exact node count and a
measured node rate, and labelled as such. The timings are what tell the team whether the large-N runs in the
leduc-sweep matrix are feasible on a laptop at all.

Two things are asserted, not just printed:
  - the deck size is identical down every player column (the H3 control)
  - a player count the deck cannot serve is rejected, never silently resized

Writes results/verify/leduc_sizes.json.

    python scripts/s3_leduc_sizes.py
    python scripts/s3_leduc_sizes.py --time-budget 30   # seconds of timing per cell
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

from pokercfr.games import leduc  # noqa: E402
from pokercfr.solvers.cfr import CFR_PLUS, CFRSolver  # noqa: E402

OUT = ROOT / "results" / "verify" / "leduc_sizes.json"
PLAYERS = (2, 3, 4, 5, 6)
CAPS = (1, 2, 3)


def betting_tree(g: leduc.LeducGame) -> tuple[int, int]:
    """(decision histories, terminal histories) for one fixed deal -- the betting
    tree is identical for every deal, so this is the per-deal tree size."""
    counts = {"decision": 0, "terminal": 0}

    def walk(state) -> None:
        if state.is_terminal():
            counts["terminal"] += 1
        elif state.is_chance_node():
            walk(state.chance_outcomes()[0][0])
        else:
            counts["decision"] += 1
            for a in state.legal_actions():
                walk(state.apply_action(a))

    walk(g.new_initial_state().chance_outcomes()[0][0])
    return counts["decision"], counts["terminal"]


def nodes_per_iteration(g: leduc.LeducGame) -> int:
    """Exact node visits in one CFR+ iteration: one full-tree traversal per player
    (CFR+ uses alternating updates). Checked against the solver's own counter on
    every cell we actually time, so the projections below rest on a verified count."""
    def subtree(state) -> int:
        if state.is_terminal():
            return 1
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            return 1 + len(outcomes) * subtree(outcomes[0][0])  # same betting tree per card
        return 1 + sum(subtree(state.apply_action(a)) for a in state.legal_actions())

    root = g.new_initial_state()
    per_traversal = 1 + len(root.chance_outcomes()) * subtree(root.chance_outcomes()[0][0])
    return g.num_players * per_traversal


def measure(g: leduc.LeducGame, budget: float) -> tuple[float, int]:
    """Mean wall-clock seconds per CFR+ iteration. Times at least one iteration,
    then keeps going until `budget` seconds are spent."""
    solver = CFRSolver(g, CFR_PLUS)
    start = time.perf_counter()
    solver.step()
    assert solver._nodes_touched == nodes_per_iteration(g), \
        f"{g.name}: node count {nodes_per_iteration(g)} != solver's {solver._nodes_touched}"
    while time.perf_counter() - start < budget:
        solver.step()
    return (time.perf_counter() - start) / solver.iterations, solver.iterations


def _fmt_seconds(sec: float) -> str:
    for unit, size in (("d", 86400), ("h", 3600), ("min", 60)):
        if sec >= size:
            return f"{sec / size:.1f}{unit}"
    return f"{sec:.3f}s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--time-budget", type=float, default=10.0,
                        help="seconds of CFR+ timing per cell (default 10)")
    args = parser.parse_args()

    rows, rate = [], None  # rate: (seconds per node, cell it was measured on)
    print(f"{'players':>7} {'cap':>3} {'deck':>4} {'info sets':>10} {'deals':>7} "
          f"{'terminals':>9} {'per iter':>9}  timing")
    for players in PLAYERS:
        for cap in CAPS:
            row = {"players": players, "raise_cap": cap}
            try:
                g = leduc.game(players=players, raise_cap=cap)
            except ValueError as err:
                # Expected for N where N private cards + 1 public exceed the deck.
                row.update(feasible=False, deck_size=leduc.STANDARD.deck_size, reason=str(err))
                rows.append(row)
                print(f"{players:>7} {cap:>3} {row['deck_size']:>4} {'-':>10} {'-':>7} {'-':>9} "
                      f"{'-':>9}  infeasible: deck too small")
                continue
            decisions, terminals = betting_tree(g)
            deals = len(g.new_initial_state().chance_outcomes())
            nodes = nodes_per_iteration(g)
            row.update(feasible=True, name=g.name, deck_size=g.config.deck_size,
                       info_sets=g.num_info_sets(), deals=deals,
                       decision_histories_per_deal=decisions, terminal_histories_per_deal=terminals,
                       nodes_per_iteration=nodes)
            # Measure when one iteration fits the budget; otherwise project from the
            # node rate of the largest-player cell measured so far, and say so.
            if rate is None or nodes * rate[0] <= args.time_budget:
                sec, iters = measure(g, args.time_budget)
                rate = (sec / nodes, g.name)
                row.update(seconds_per_cfr_plus_iteration=sec, timing="measured", iterations_timed=iters)
                note = f"measured over {iters} iter"
            else:
                sec = nodes * rate[0]
                row.update(seconds_per_cfr_plus_iteration=sec, timing="projected",
                           projected_from=rate[1], seconds_per_node=rate[0])
                note = f"PROJECTED from {rate[1]} node rate"
            rows.append(row)
            print(f"{players:>7} {cap:>3} {g.config.deck_size:>4} {g.num_info_sets():>10,} "
                  f"{deals:>7,} {terminals:>9,} {_fmt_seconds(sec):>9}  {note}")

    # The H3 control: the deck never changes with player count.
    decks = {r["deck_size"] for r in rows}
    assert decks == {leduc.STANDARD.deck_size}, f"deck size varied across player counts: {decks}"
    infeasible = sorted({r["players"] for r in rows if not r["feasible"]})
    assert infeasible == [p for p in PLAYERS if p + 1 > leduc.STANDARD.deck_size], \
        f"unexpected feasibility pattern: {infeasible}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "deck_size": leduc.STANDARD.deck_size,
        "deck_held_fixed": True,
        "infeasible_player_counts": infeasible,
        "timing": {"solver": "cfr+", "time_budget_seconds": args.time_budget,
                   "machine": platform.platform(), "python": platform.python_version()},
        "rows": rows,
    }, indent=2) + "\n")
    print(f"\ndeck size constant at {leduc.STANDARD.deck_size} across all player counts")
    if infeasible:
        print(f"infeasible on the fixed deck: N = {infeasible} (N private + 1 public > deck)")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
