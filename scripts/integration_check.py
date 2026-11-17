#!/usr/bin/env python3
"""The cross-module gate. Must pass before the freeze.

`main` is unprotected and there are no pull requests, so nothing else looks at
all eight modules together. This is the review: it checks the seams between
owners, which is where the expensive failures have actually come from.

Every bug this project has hit so far was silent -- tests passed, runs
completed, and the damage showed up as missing or wrong data rather than an
error. So each check here asserts something that would otherwise fail quietly:

  imports     every module imports, and an unfinished one raises
              NotImplementedError rather than TypeError from a drifted signature
  games       every rung name in every matrix resolves through build_game
  solvers     every variant name in every matrix resolves through build_solver
  paths       no result path nests into a directory (slash-bearing variant
              labels once hid 20 of cube-deep's runs from a results/*.json glob)
  hashes      every committed result's spec hash matches its filename, which
              catches a record edited by hand
  schema      every committed result validates against the frozen schema
  complete    the `complete` flag agrees with whether the curve reached budget
  ground      Kuhn still reproduces -1/18, and the cube corners still match
              OpenSpiel to 1e-12

Exits non-zero on any failure. Run it before the freeze, and again after the
last result lands.

    python scripts/integration_check.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pokercfr.core.config import RunSpec  # noqa: E402

MODULES = [
    "pokercfr.core.protocols", "pokercfr.core.config",
    "pokercfr.games.kuhn_family", "pokercfr.games.leduc", "pokercfr.games.kuhn_reference",
    "pokercfr.solvers.cfr", "pokercfr.solvers.mccfr", "pokercfr.solvers.lp_oracle",
    "pokercfr.eval.best_response", "pokercfr.eval.exploitability", "pokercfr.eval.tournament",
    "pokercfr.agents.baselines", "pokercfr.analysis.stats", "pokercfr.analysis.plots",
    "pokercfr.experiments.registry", "pokercfr.experiments.runner",
]


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
        if not ok:
            self.failures.append(f"{name}: {detail}")
        return ok

    def note(self, text: str) -> None:
        print(f"  note  {text}")
        self.notes.append(text)


def check_imports(r: Report) -> None:
    import importlib

    print("\n=== modules import ===")
    for name in MODULES:
        try:
            importlib.import_module(name)
            r.check(name, True)
        except Exception as exc:
            r.check(name, False, f"{type(exc).__name__}: {exc}")


def check_matrices(r: Report) -> None:
    from pokercfr.experiments import registry
    from pokercfr.experiments.runner import build_game, build_solver

    print("\n=== every matrix cell resolves ===")
    specs = registry.full_matrix()
    r.check("matrix is non-empty", bool(specs), f"{len(specs)} cells")

    games, variants = {}, {}
    unbuilt_solvers: dict[str, str] = {}
    for spec in specs:
        if spec.game not in games:
            try:
                build_game(spec.game)
                games[spec.game] = None
            except Exception as exc:
                games[spec.game] = f"{type(exc).__name__}: {exc}"
        key = (spec.game, spec.variant)
        if spec.variant not in variants:
            try:
                build_solver(spec)
                variants[spec.variant] = None
            except NotImplementedError:
                variants[spec.variant] = None
                unbuilt_solvers[spec.variant] = "not implemented yet"
            except Exception as exc:
                variants[spec.variant] = f"{type(exc).__name__}: {exc}"
        del key

    bad_games = {k: v for k, v in games.items() if v}
    r.check("all rung names resolve", not bad_games,
            "; ".join(f"{k} -> {v}" for k, v in list(bad_games.items())[:3])
            or f"{len(games)} distinct rungs")
    bad_variants = {k: v for k, v in variants.items() if v}
    r.check("all variant names resolve", not bad_variants,
            "; ".join(f"{k} -> {v}" for k, v in list(bad_variants.items())[:3])
            or f"{len(variants)} distinct variants")
    if unbuilt_solvers:
        r.note(f"{len(unbuilt_solvers)} variant(s) not implemented yet: "
               f"{', '.join(sorted(unbuilt_solvers)[:6])}")

    print("\n=== result paths stay flat ===")
    nested = [s.variant for s in specs if "/" in s.result_path[len("results/"):]]
    r.check("no result path nests into a directory", not nested,
            f"{len(set(nested))} offending variants" if nested else f"{len(specs)} cells")
    paths = [s.result_path for s in specs]
    r.check("result paths are unique", len(paths) == len(set(paths)),
            f"{len(paths) - len(set(paths))} collisions")


def check_committed_results(r: Report, results_dir: Path) -> None:
    from pokercfr.eval.exploitability import validate_record

    print("\n=== committed results ===")
    files = sorted(results_dir.glob("*.json"))
    if not files:
        r.note("no committed run records yet")
        return
    bad_hash, bad_schema, bad_complete = [], [], []
    for path in files:
        record = json.loads(path.read_text())
        spec_blob = record.get("spec", {})
        try:
            spec = RunSpec(
                game=spec_blob["game"], variant=spec_blob["variant"],
                seed=spec_blob["seed"], iterations=spec_blob["iterations"],
                max_seconds=spec_blob.get("max_seconds"),
                checkpoints=tuple(spec_blob.get("checkpoints", ())),
                extra=spec_blob.get("extra", {}))
            if spec.spec_hash() != spec_blob.get("spec_hash") or \
                    Path(spec.result_path).name != path.name:
                bad_hash.append(path.name)
        except Exception as exc:
            bad_hash.append(f"{path.name} ({exc})")
        errors = validate_record(record)
        if errors:
            bad_schema.append(f"{path.name}: {errors[0]}")
        curve = record.get("curve") or []
        if curve and record.get("complete"):
            budget = record.get("time_budget")
            reached = (curve[-1]["train_seconds"] >= 0.9 * budget if budget
                       else curve[-1]["iterations"] >= spec_blob.get("iterations", 0))
            if not reached:
                bad_complete.append(path.name)
    r.check("spec hash matches filename", not bad_hash,
            "; ".join(bad_hash[:3]) or f"{len(files)} files")
    r.check("records match the frozen schema", not bad_schema,
            "; ".join(bad_schema[:3]) or f"{len(files)} files")
    r.check("complete flag agrees with the curve", not bad_complete,
            "; ".join(bad_complete[:3]) or f"{len(files)} files")


def check_ground_truth(r: Report) -> None:
    from pokercfr.eval.exploitability import exploitability
    from pokercfr.games.kuhn_family import game
    from pokercfr.solvers.cfr import CFR_PLUS, CFRConfig, CFRSolver, policy_expected_values

    print("\n=== ground truth still holds ===")
    g = game("kuhn")
    policy = CFRSolver(g, CFR_PLUS).train(20_000).average_strategy()
    value = policy_expected_values(g, policy)[0]
    r.check("Kuhn value reproduces -1/18", abs(value + 1 / 18) < 1e-5,
            f"got {value:.8f}, error {abs(value + 1/18):.2e}")
    r.check("uniform random Kuhn exploitability is 11/24",
            abs(exploitability(g, {}) - 11 / 24) < 1e-12)

    try:
        import pyspiel
        from open_spiel.python.algorithms import cfr as os_cfr
        from open_spiel.python.algorithms import exploitability as os_exp
    except ImportError:
        r.note("open_spiel not installed; skipped the external cross-check")
        return

    osg = pyspiel.load_game("kuhn_poker")
    for label, their_solver, ours in (
        ("OpenSpiel CFRSolver == rm/uniform/alt", os_cfr.CFRSolver(osg),
         CFRSolver(g, CFRConfig(alternating=True))),
        ("OpenSpiel CFRPlusSolver == cfr+", os_cfr.CFRPlusSolver(osg),
         CFRSolver(g, CFR_PLUS)),
    ):
        while their_solver._iteration < 1000:
            their_solver.evaluate_and_update_policy()
        ours.train(1000)
        theirs = os_exp.exploitability(osg, their_solver.average_policy())
        mine = exploitability(g, ours.average_strategy())
        r.check(label, abs(theirs - mine) < 1e-12, f"|diff| = {abs(theirs - mine):.2e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--skip-ground-truth", action="store_true",
                    help="skip the 20k-iteration Kuhn checks (a few seconds)")
    args = ap.parse_args()

    r = Report()
    check_imports(r)
    check_matrices(r)
    check_committed_results(r, args.results)
    if not args.skip_ground_truth:
        check_ground_truth(r)

    print("\n" + "=" * 68)
    if r.failures:
        print(f"{len(r.failures)} FAILURE(S) — do not freeze until these are resolved:")
        for f in r.failures:
            print(f"  - {f}")
        return 1
    print("integration check PASSED"
          + (f" ({len(r.notes)} note(s))" if r.notes else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
