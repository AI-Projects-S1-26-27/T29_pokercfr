"""Exact equilibrium values of every feasible two-player rung, by sequence-form LP.

OWNER: Siddhant. Produces results/verify/lp_values.json (+ .log via tee).

For each rung: build the sequence form, solve both LPs with HiGHS, and check the
answer three ways that share no code with the LP:
  1. the LP policy's exploitability under eval/exploitability (our exact BR),
  2. on Kuhn-family rungs, the same under solvers/vectorized.nash_conv,
  3. the value the policy actually earns, by plain tree walking.

Then the module-docstring acceptance check: the LP value must agree with a
200,000-iteration CFR+ run to < 1e-4 on every two-player ladder rung up to
`4rank-2suit-2dealt`. CFR+ is run with the vectorized traversal (proven
table-equivalent to the scalar one, see sid2) and, with --scalar, with the scalar
solver too, so the agreement does not rest on the new code.

    python scripts/sid1_lp_values.py              # ~2 minutes
    python scripts/sid1_lp_values.py --scalar     # + scalar CFR+ on 3 rungs, ~30 min
    python scripts/sid1_lp_values.py --top-rung   # ONLY the 52-card rung (~2.2 GB, ~2 min),
                                                  # into results/verify/lp_top_rung.json

Exits 1 if any check fails.
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

from pokercfr.eval.exploitability import evaluate  # noqa: E402
from pokercfr.games import kuhn_family, leduc  # noqa: E402
from pokercfr.solvers.cfr import CFR_PLUS, CFRSolver, policy_expected_values  # noqa: E402
from pokercfr.solvers.lp_oracle import solve  # noqa: E402
from pokercfr.solvers.vectorized import VectorizedCFRSolver, nash_conv  # noqa: E402

OUT = ROOT / "results" / "verify" / "lp_values.json"
OUT_TOP = ROOT / "results" / "verify" / "lp_top_rung.json"
KUHN_RUNGS = ("kuhn", "4rank", "4rank-2dealt", "4rank-2suit", "4rank-2suit-2dealt")
CFR_PLUS_RUNGS = ("kuhn", "4rank", "4rank-2suit", "4rank-2suit-2dealt")  # docstring criterion
CFR_PLUS_ITERS = 200_000
# Scalar CFR+ at 200k iterations costs ~0.05 s/iter on 4rank-2suit-2dealt (~2.6 h),
# so the scalar cross-check covers the three smaller rungs; the vectorized run
# covers all four.
SCALAR_RUNGS = ("kuhn", "4rank", "4rank-2suit")
TOP = "13rank-4suit-2dealt-2p"
EXPL_TOL = 1e-9
AGREE_TOL = 1e-4


def lp_row(g, kuhn_family_game: bool) -> dict:
    sol = solve(g)
    started = time.perf_counter()
    expl = evaluate(g, sol.policy).exploitability if g.name != TOP else None
    eval_seconds = time.perf_counter() - started
    row = {"game": g.name, "info_sets": g.num_info_sets(), **sol.stats(),
           "exploitability_eval": expl, "eval_seconds": eval_seconds,
           "policy_value": None if g.name == TOP else policy_expected_values(g, sol.policy)[0]}
    if kuhn_family_game:
        row["exploitability_vectorized"] = nash_conv(g, sol.policy).exploitability
    return row


def cfr_plus_value(g, solver_cls) -> tuple[float, float, float]:
    started = time.perf_counter()
    avg = solver_cls(g, CFR_PLUS).train(CFR_PLUS_ITERS).average_strategy()
    seconds = time.perf_counter() - started
    return policy_expected_values(g, avg)[0], nash_conv(g, avg).exploitability, seconds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scalar", action="store_true", help="also run scalar CFR+ (slow)")
    ap.add_argument("--top-rung", action="store_true", help=f"solve only {TOP}, by LP")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    args.out = args.out or (OUT_TOP if args.top_rung else OUT)

    rows, failures = [], []
    if args.top_rung:
        games = [(kuhn_family.game(TOP), True)]
    else:
        games = [(kuhn_family.game(r), True) for r in KUHN_RUNGS]
        games += [(leduc.game(raise_cap=c), False) for c in (1, 2, 3)]
    print(f"{'game':<26}{'value (seat 0)':>22}{'seqs':>8}{'nnz':>10}"
          f"{'expl (eval)':>13}{'expl (vec)':>12}{'solve s':>9}")
    for g, is_kuhn in games:
        row = lp_row(g, is_kuhn)
        rows.append(row)
        for field in ("exploitability_eval", "exploitability_vectorized"):
            if row.get(field) is not None and row[field] > EXPL_TOL:
                failures.append(f"{g.name}: {field} = {row[field]:.3e}")
        if row["policy_value"] is not None and abs(row["policy_value"] - row["value"]) > EXPL_TOL:
            failures.append(f"{g.name}: policy earns {row['policy_value']!r}, LP says {row['value']!r}")
        fmt = lambda v: "-" if v is None else f"{v:.2e}"  # noqa: E731
        print(f"{g.name:<26}{row['value']:>22.15f}{row['sequences'][0]:>8}{row['payoff_nnz']:>10}"
              f"{fmt(row['exploitability_eval']):>13}{fmt(row.get('exploitability_vectorized')):>12}"
              f"{row['solve_seconds']:>9.2f}", flush=True)

    agreement = []
    if args.top_rung:
        return _finish(args, rows, agreement, failures)
    kuhn_value = rows[0]["value"]
    if abs(kuhn_value + 1 / 18) > 1e-9:
        failures.append(f"kuhn value {kuhn_value!r} != -1/18")
    print(f"\nKuhn: LP value + 1/18 = {kuhn_value + 1 / 18:.3e}")

    solvers = [("vectorized", VectorizedCFRSolver), ("scalar", CFRSolver)]
    print(f"\nLP vs {CFR_PLUS_ITERS:,}-iteration CFR+ (criterion |diff| < {AGREE_TOL:g})")
    for rung in CFR_PLUS_RUNGS:
        g = kuhn_family.game(rung)
        lp = next(r["value"] for r in rows if r["game"] == rung)
        for name, cls in solvers:
            if name == "scalar" and not (args.scalar and rung in SCALAR_RUNGS):
                continue
            value, expl, seconds = cfr_plus_value(g, cls)
            diff = abs(value - lp)
            agreement.append({"game": rung, "solver": name, "iterations": CFR_PLUS_ITERS,
                              "cfr_plus_value": value, "lp_value": lp, "abs_diff": diff,
                              "cfr_plus_exploitability": expl, "train_seconds": seconds})
            ok = diff < AGREE_TOL
            if not ok:
                failures.append(f"{rung}/{name}: |CFR+ - LP| = {diff:.3e}")
            print(f"  {rung:<22}{name:<12}CFR+ {value:.12f}  LP {lp:.12f}  |diff| {diff:.2e}  "
                  f"expl {expl:.2e}  {seconds:7.1f}s  {'PASS' if ok else 'FAIL'}", flush=True)

    return _finish(args, rows, agreement, failures)


def _finish(args, rows, agreement, failures) -> int:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "lp": rows, "cfr_plus_agreement": agreement, "failures": failures,
        "env": {"python": platform.python_version(), "machine": platform.machine(),
                "node": platform.node()}}, indent=2))
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    print("ALL CHECKS PASS" if not failures else "FAILURES:\n  " + "\n  ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
