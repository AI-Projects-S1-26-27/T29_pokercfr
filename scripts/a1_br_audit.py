#!/usr/bin/env python3
"""Audit of the exact best-response instrument (Aarush, step 1).

Every number in the report is measured by `pokercfr/eval/best_response.py`, so if
it is wrong, everything is wrong in the same direction. This script re-verifies
the properties that must hold, records each check with its margin, and exits 1 if
any fails. Checks:

  nonneg      every per-seat gain >= 0 across every Kuhn-family rung (incl. 3p/4p)
              at several iteration counts -- a best response cannot lose to the
              policy it responds to
  uniform     uniform-random Kuhn exploitability == 11/24 exactly
  realised    playing the computed BR policy against the original policy really
              yields the claimed BR value, for every seat on every rung
  closed form CFR+ on Kuhn converges to the game value -1/18
  openspiel   exploitability and NashConv agree with OpenSpiel's, deltas recorded

    python scripts/a1_br_audit.py [--out results/verify/br_audit.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pokercfr.eval.best_response import best_response, exploitability, nash_conv  # noqa: E402
from pokercfr.games.kuhn_family import game  # noqa: E402
from pokercfr.solvers.cfr import (  # noqa: E402
    CFR_PLUS,
    VANILLA_CFR,
    CFRSolver,
    policy_expected_values,
)

RUNGS = ("kuhn", "4rank", "4rank-2suit", "kuhn-3p", "4rank-2suit-3p")
ITERATIONS = (1, 10, 100)
NONNEG_TOL = 1e-12
REALISE_TOL = 1e-9
CLOSED_FORM_TOL = 1e-4
OPENSPIEL_TOL = 1e-12

CHECKS: list[dict] = []


def record(name: str, passed: bool, margin: float | None, tolerance: float | None,
           detail: str = "") -> None:
    """margin = how far inside the pass condition we are (positive = pass)."""
    CHECKS.append({"check": name, "passed": bool(passed), "margin": margin,
                   "tolerance": tolerance, "detail": detail})
    flag = "PASS" if passed else "FAIL"
    m = f"margin {margin:.3e}" if margin is not None else ""
    print(f"  [{flag}] {name:<58} {m} {detail}")


def check_nonneg() -> None:
    print("nonneg: every per-seat gain >= 0")
    for rung in RUNGS:
        g = game(rung)
        for solver_name, cfg in (("cfr", VANILLA_CFR), ("cfr+", CFR_PLUS)):
            solver = CFRSolver(g, cfg)
            done = 0
            for t in ITERATIONS:
                solver.train(t - done)
                done = t
                gains = nash_conv(g, solver.average_strategy()).gains
                lowest = min(gains)
                record(f"nonneg {rung} {solver_name} T={t}", lowest >= -NONNEG_TOL,
                       lowest + NONNEG_TOL, NONNEG_TOL, f"min gain {lowest:.3e}")


def check_uniform() -> None:
    print("uniform: Kuhn uniform-random exploitability == 11/24")
    value = exploitability(game("kuhn"), {})
    err = abs(value - 11 / 24)
    record("uniform kuhn exploitability = 11/24", err < 1e-12, 1e-12 - err, 1e-12,
           f"{value!r}")


def check_realised() -> None:
    print("realised: BR policy played against the policy realises the BR value")
    for rung in RUNGS:
        g = game(rung)
        policy = CFRSolver(g, CFR_PLUS).train(30).average_strategy()
        worst = 0.0
        for seat in range(g.num_players):
            claimed, br = best_response(g, policy, seat)
            # Seat `seat` plays br; everyone else plays `policy`. The two
            # policies only share keys by player prefix inside the key string,
            # so overlay br last: it only holds the responder's info sets.
            realised = policy_expected_values(g, {**_filled(g, policy), **br})[seat]
            worst = max(worst, abs(claimed - realised))
        record(f"realised {rung} (all seats)", worst < REALISE_TOL,
               REALISE_TOL - worst, REALISE_TOL, f"max |claimed-realised| {worst:.3e}")


def _filled(g, policy):
    from pokercfr.eval.best_response import _full_policy
    return _full_policy(g, policy)


def check_closed_form() -> None:
    print("closed form: CFR+ on Kuhn -> -1/18")
    g = game("kuhn")
    policy = CFRSolver(g, CFR_PLUS).train(20_000).average_strategy()
    v = policy_expected_values(g, policy)[0]
    err = abs(v + 1 / 18)
    record("CFR+ kuhn game value = -1/18", err < CLOSED_FORM_TOL, CLOSED_FORM_TOL - err,
           CLOSED_FORM_TOL, f"value {v:.9f}, err {err:.3e}")
    expl = exploitability(g, policy)
    record("CFR+ kuhn exploitability < 1e-4 at T=20000", expl < 1e-4, 1e-4 - expl, 1e-4,
           f"expl {expl:.3e}")


def check_openspiel() -> None:
    print("openspiel: exploitability / NashConv vs OpenSpiel (Kuhn)")
    try:
        import pyspiel
        from open_spiel.python import policy as os_policy
        from open_spiel.python.algorithms import cfr as os_cfr
        from open_spiel.python.algorithms import exploitability as os_exp
    except ImportError:
        record("openspiel available", False, None, None,
               "pip install -e '.[verify]' -- cannot audit against OpenSpiel")
        return
    osg = pyspiel.load_game("kuhn_poker")
    ours = game("kuhn")

    uniform = os_policy.UniformRandomPolicy(osg)
    d = abs(os_exp.exploitability(osg, uniform) - exploitability(ours, {}))
    record("openspiel uniform exploitability delta", d < OPENSPIEL_TOL, OPENSPIEL_TOL - d,
           OPENSPIEL_TOL, f"delta {d:.3e}")
    d = abs(os_exp.nash_conv(osg, uniform) - nash_conv(ours, {}).total)
    record("openspiel uniform NashConv delta", d < OPENSPIEL_TOL, OPENSPIEL_TOL - d,
           OPENSPIEL_TOL, f"delta {d:.3e}")

    theirs, mine = os_cfr.CFRPlusSolver(osg), CFRSolver(ours, CFR_PLUS)
    for target in (10, 100, 1000):
        while theirs._iteration < target:
            theirs.evaluate_and_update_policy()
        mine.train(target - mine.iterations)
        a = os_exp.exploitability(osg, theirs.average_policy())
        b = exploitability(ours, mine.average_strategy())
        n1 = os_exp.nash_conv(osg, theirs.average_policy())
        n2 = nash_conv(ours, mine.average_strategy()).total
        record(f"openspiel CFR+ T={target} exploitability delta",
               abs(a - b) < OPENSPIEL_TOL, OPENSPIEL_TOL - abs(a - b), OPENSPIEL_TOL,
               f"delta {abs(a - b):.3e}")
        record(f"openspiel CFR+ T={target} NashConv delta",
               abs(n1 - n2) < OPENSPIEL_TOL, OPENSPIEL_TOL - abs(n1 - n2), OPENSPIEL_TOL,
               f"delta {abs(n1 - n2):.3e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "results" / "verify" / "br_audit.json"))
    args = ap.parse_args()

    check_nonneg()
    check_uniform()
    check_realised()
    check_closed_form()
    check_openspiel()

    failed = [c for c in CHECKS if not c["passed"]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"passed": not failed, "n_checks": len(CHECKS),
                               "n_failed": len(failed), "checks": CHECKS}, indent=2))
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed -> {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
