#!/usr/bin/env python3
"""Cross-check our Leduc engine against OpenSpiel's `leduc_poker`.

Our exploitability machinery already agrees with OpenSpiel to 1e-12 on Kuhn.
This extends that to the game H1 actually runs on. Our own tests cannot catch a
misreading of the rules baked into both the engine and the tests; an independent
implementation can. Four checks, each sharper than the last:

  info sets   same count -> same betting abstraction and raise cap
  uniform     exploitability of the uniform policy -> pot accounting and showdown
  evaluator   one of OUR trained policies, translated into OpenSpiel and scored by
              THEIR evaluator, must match our evaluator's number
  cfr+        our CFR+ and OpenSpiel's CFRPlusSolver at matched iterations

The CFR+ trajectory cannot be held to 1e-9 past ~100 iterations, by either
implementation: regret-matching+ clamps regrets at zero, so a last-bit rounding
difference can flip an action in or out of the strategy and then compounds. We
measure that noise floor directly -- a second copy of OUR solver that visits
chance outcomes in reverse order (identical maths, different float summation)
-- and require OpenSpiel to agree with us within NOISE_FACTOR x how well we agree
with ourselves. The rule checks (counts, uniform, evaluator) stay exact.

How to read a disagreement: if info-set counts differ, suspect the betting rules
or the cap. If counts match but exploitability does not, suspect pot accounting
or the showdown comparison. Either is a real bug, not a convention difference.

Writes results/verify/leduc_crosscheck.json. Exits 1 on any disagreement, so
nothing downstream (H1, the sweep) runs on an engine that failed this gate.

    python scripts/s5_openspiel_leduc_crosscheck.py
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pokercfr.eval.best_response import exploitability  # noqa: E402
from pokercfr.games import leduc  # noqa: E402
from pokercfr.games.leduc import LeducState  # noqa: E402
from pokercfr.solvers.cfr import CFR_PLUS, CFRSolver  # noqa: E402

OUT = ROOT / "results" / "verify" / "leduc_crosscheck.json"
TOLERANCE = 1e-9
NOISE_FACTOR = 10
CHECKPOINTS = (1, 10, 50, 100, 200)
OS_ACTION = {0: "f", 1: "c", 2: "r"}  # OpenSpiel's Fold / Call / Raise ids


def os_info_set_count(osg) -> int:
    seen = set()

    def walk(st) -> None:
        if st.is_terminal():
            return
        if st.is_chance_node():
            for a, _ in st.chance_outcomes():
                walk(st.child(a))
            return
        seen.add(st.information_state_string())
        for a in st.legal_actions():
            walk(st.child(a))

    walk(osg.new_initial_state())
    return len(seen)


def key_map(osg, ours) -> dict[str, str]:
    """Walk both trees in lockstep: OpenSpiel info-state string -> our info-set key.
    OpenSpiel deals one card per chance node; we deal every private card at once,
    so the first `players` OpenSpiel chance actions select our deal outcome."""
    deals = {s.private: s for s, _ in ours.new_initial_state().chance_outcomes()}
    mapping: dict[str, str] = {}

    def walk(os_state, our_state, dealt: tuple[int, ...]) -> None:
        if os_state.is_terminal():
            assert our_state is not None and our_state.is_terminal()
            return
        if os_state.is_chance_node():
            for a, _ in os_state.chance_outcomes():
                if our_state is None:  # still dealing private cards
                    cards = dealt + (a,)
                    nxt = deals[cards] if len(cards) == ours.num_players else None
                    walk(os_state.child(a), nxt, cards)
                else:  # the public card
                    child = next(s for s, _ in our_state.chance_outcomes() if s.public == a)
                    walk(os_state.child(a), child, dealt)
            return
        player = os_state.current_player()
        assert player == our_state.current_player(), "acting seat disagrees"
        assert [OS_ACTION[a] for a in os_state.legal_actions()] == our_state.legal_actions(), \
            f"legal actions disagree at {our_state.history!r}"
        mapping[os_state.information_state_string()] = our_state.information_set_key(player)
        for a in os_state.legal_actions():
            walk(os_state.child(a), our_state.apply_action(OS_ACTION[a]), dealt)

    walk(osg.new_initial_state(), None, ())
    return mapping


def to_openspiel(osg, policy, mapping):
    """Our Policy dict -> an OpenSpiel TabularPolicy, via the lockstep key map."""
    from open_spiel.python import policy as os_policy

    tab = os_policy.TabularPolicy(osg)
    for os_key, row in tab.state_lookup.items():
        probs = policy[mapping[os_key]]
        tab.action_probability_array[row, :] = 0.0
        for action_id, label in OS_ACTION.items():
            if label in probs:
                tab.action_probability_array[row, action_id] = probs[label]
    return tab


@contextmanager
def reversed_chance_order():
    """Same game, chance outcomes visited in reverse: changes only float summation order."""
    original = LeducState.chance_outcomes
    LeducState.chance_outcomes = lambda self: list(reversed(original(self)))
    try:
        yield
    finally:
        LeducState.chance_outcomes = original


def compare(name: str, ours: float, theirs: float, checks: list[dict],
            noise_floor: float | None = None) -> None:
    ours, theirs = float(ours), float(theirs)
    diff = abs(ours - theirs)
    rel = diff / max(abs(theirs), 1e-300)
    ok = diff <= TOLERANCE or (noise_floor is not None and diff <= NOISE_FACTOR * noise_floor)
    row = {"check": name, "ours": ours, "openspiel": theirs, "abs_diff": diff, "rel_diff": rel,
           "agree": bool(ok)}
    if noise_floor is not None:
        row["self_noise_floor"] = noise_floor
    checks.append(row)
    floor = f" noise_floor={noise_floor:.2e}" if noise_floor is not None else ""
    print(f"{'ok  ' if ok else 'FAIL'} {name:38} ours={ours:.15f} openspiel={theirs:.15f} "
          f"rel_diff={rel:.2e}{floor}")


def main() -> int:
    try:
        import pyspiel
        from open_spiel.python import policy as os_policy
        from open_spiel.python.algorithms import cfr as os_cfr
        from open_spiel.python.algorithms import exploitability as os_exp
    except ImportError:
        print("SKIP: open_spiel not installed (pip install -e '.[verify]')")
        return 0

    osg, ours = pyspiel.load_game("leduc_poker"), leduc.game()
    checks: list[dict] = []

    # 1. Information-set counts.
    os_count, our_count = os_info_set_count(osg), ours.num_info_sets()
    checks.append({"check": "info_set_count", "ours": our_count, "openspiel": os_count,
                   "agree": os_count == our_count})
    print(f"{'ok  ' if os_count == our_count else 'FAIL'} {'info set count':34} "
          f"ours={our_count} openspiel={os_count}")

    # 2. Uniform-random exploitability: needs no policy translation at all.
    compare("uniform random exploitability", exploitability(ours, {}),
            os_exp.exploitability(osg, os_policy.UniformRandomPolicy(osg)), checks)

    # 3 + 4. Matched-iteration CFR+, scored by each side's own evaluator, plus our
    # policy translated into OpenSpiel and scored by THEIR evaluator.
    mapping = key_map(osg, ours)
    assert len(mapping) == os_count, "lockstep walk missed info sets"
    theirs_solver, our_solver = os_cfr.CFRPlusSolver(osg), CFRSolver(ours, CFR_PLUS)
    reordered = CFRSolver(ours, CFR_PLUS)
    for t in CHECKPOINTS:
        while theirs_solver._iteration < t:
            theirs_solver.evaluate_and_update_policy()
        our_solver.train(t - our_solver.iterations)
        with reversed_chance_order():
            reordered.train(t - reordered.iterations)
        our_policy = our_solver.average_strategy()
        our_value = exploitability(ours, our_policy)
        noise = abs(our_value - exploitability(ours, reordered.average_strategy()))
        compare(f"cfr+ T={t}", our_value,
                os_exp.exploitability(osg, theirs_solver.average_policy()), checks, noise_floor=noise)
        compare(f"our cfr+ policy, their evaluator T={t}", our_value,
                os_exp.exploitability(osg, to_openspiel(osg, our_policy, mapping)), checks)

    agree = all(c["agree"] for c in checks)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"game": ours.name, "openspiel_game": "leduc_poker",
                               "tolerance": TOLERANCE, "noise_factor": NOISE_FACTOR,
                               "all_agree": agree,
                               "checks": checks}, indent=2) + "\n")
    print(f"\n{'PASS' if agree else 'FAIL'}: {sum(c['agree'] for c in checks)}/{len(checks)} "
          f"checks agree with OpenSpiel (exact checks within {TOLERANCE:g}; CFR+ trajectories "
          f"within {NOISE_FACTOR}x our own float-order noise floor)")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0 if agree else 1


if __name__ == "__main__":
    sys.exit(main())
