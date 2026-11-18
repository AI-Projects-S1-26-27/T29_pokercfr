"""How far apart do two CORRECT CFR runs drift when only summation order differs?

OWNER: Siddhant. Produces results/verify/float_sensitivity.json (+ .log via tee).

Background: the vectorized traversal matches the scalar one to ~1e-15 relative
for the first ~1,000 iterations, then on some rungs the tables separate
geometrically. A logic bug would show up at iteration 1, not iteration 1,500, so
the hypothesis is that CFR's dynamics amplify rounding differences. This script
tests that hypothesis with a CONTROL that involves no new code at all:

  scalar            solvers/cfr.CFRSolver, as is
  scalar-reversed   the SAME solver on the SAME game, but with the root deal
                    enumerated in reverse order -- mathematically identical, it
                    only changes the order deals are summed in
  vectorized        solvers/vectorized.VectorizedCFRSolver

If scalar vs scalar-reversed drifts like scalar vs vectorized, the drift is a
property of CFR on that game under floating point, not of the vectorization. We
also report the exploitability of each run's average strategy at log-spaced
checkpoints: that is the quantity the study reports, so the question that matters
is whether it agrees, not whether every current-iterate regret does.

Part 2 -- the regret-matching TIE. On `4rank-2suit-4p` with alternating updates
the vectorized tables separate at iteration 2, not after thousands. Cause:
regret matching is discontinuous at zero. At info sets where two actions are
exactly tied (true regret 0), one summation order yields 0.0 (-> uniform) and
another +-1e-18 (-> a pure strategy). The control is the scalar solver with the
deal order SHUFFLED: if shuffling alone flips those info sets too, the split is a
property of CFR in floating point, not of the vectorization. We then follow
NashConv (mean per seat, the H3 metric) for all runs.

    python scripts/sid3_float_sensitivity.py      # ~15 minutes
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pokercfr.games.kuhn_family import ChanceRoot, game  # noqa: E402
from pokercfr.solvers.cfr import CFRSolver, ablation_cube  # noqa: E402
from pokercfr.solvers.vectorized import VectorizedCFRSolver, nash_conv  # noqa: E402

OUT = ROOT / "results" / "verify" / "float_sensitivity.json"
CASES = (("kuhn", "cfr"), ("4rank", "cfr"), ("4rank", "cfr+"), ("4rank-2suit", "cfr"),
         ("4rank-2suit", "cfr+"))
ITERS = 10_000
CHECKPOINTS = (10, 30, 100, 300, 1_000, 2_000, 3_000, 5_000, 10_000)


TIE_CASES = (("4rank-2suit-4p", "rm/uniform/alt"), ("4rank-2suit-4p", "cfr+"))
TIE_ITERS = 60
TIE_CHECKPOINTS = (1, 2, 5, 10, 20, 30, 40, 60)
TIE_KEY = "Qhbpp"   # an exactly-tied info set (call vs fold) after iteration 1
SHUFFLE_SEEDS = (0, 1, 2, 3, 4, 5)


class _ShuffledRoot(ChanceRoot):
    seed = 0

    def chance_outcomes(self):
        out = super().chance_outcomes()
        random.Random(_ShuffledRoot.seed).shuffle(out)
        return out


class _ShuffledGame:
    def __init__(self, inner, seed):
        self._inner, self._seed = inner, seed

    @property
    def num_players(self):
        return self._inner.num_players

    def new_initial_state(self):
        _ShuffledRoot.seed = self._seed
        return _ShuffledRoot(self._inner.config)


class _ReversedRoot(ChanceRoot):
    def chance_outcomes(self):
        return list(reversed(super().chance_outcomes()))


class _ReversedGame:
    """The same game with the root deal enumerated in reverse order."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def num_players(self):
        return self._inner.num_players

    def new_initial_state(self):
        return _ReversedRoot(self._inner.config)


def rel_gap(a: dict, b: dict) -> float:
    scale = max(max(abs(v) for v in row.values()) for row in a.values())
    return max(abs(a[k][x] - b[k][x]) for k in a for x in a[k]) / max(scale, 1.0)


def main() -> int:
    rows = []
    for rung, label in CASES:
        g, cfg = game(rung), ablation_cube()[label]
        runs = {"scalar": CFRSolver(g, cfg), "scalar-reversed": CFRSolver(_ReversedGame(g), cfg),
                "vectorized": VectorizedCFRSolver(g, cfg)}
        print(f"\n{rung} / {label}")
        print(f"{'T':>7}{'rel regret gap':>34}{'exploitability of average':>48}")
        print(f"{'':>7}{'scal vs rev':>17}{'scal vs vec':>17}"
              f"{'scalar':>16}{'reversed':>16}{'vectorized':>16}")
        curve = []
        for t in range(1, ITERS + 1):
            for solver in runs.values():
                solver.step()
            if t in CHECKPOINTS:
                ref = runs["scalar"].regret_sum
                rev, vec = runs["scalar-reversed"].regret_sum, runs["vectorized"].regret_sum
                point = {"iterations": t,
                         "rel_regret_gap_scalar_vs_reversed": rel_gap(ref, rev),
                         "rel_regret_gap_scalar_vs_vectorized": rel_gap(ref, vec),
                         "exploitability": {name: nash_conv(g, s.average_strategy()).exploitability
                                            for name, s in runs.items()}}
                curve.append(point)
                e = point["exploitability"]
                print(f"{t:>7}{point['rel_regret_gap_scalar_vs_reversed']:>17.2e}"
                      f"{point['rel_regret_gap_scalar_vs_vectorized']:>17.2e}"
                      f"{e['scalar']:>16.4e}{e['scalar-reversed']:>16.4e}{e['vectorized']:>16.4e}",
                      flush=True)
        rows.append({"rung": rung, "config": label, "curve": curve})
    ties = tie_control()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"cases": rows, "tie_control": ties}, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


def tie_control() -> dict:
    from pokercfr.solvers.cfr import regret_matching

    g = game(TIE_CASES[0][0])
    cfg = ablation_cube()[TIE_CASES[0][1]]
    print(f"\n=== tie control: regret at {TIE_KEY!r} after ONE iteration, {g.name} / "
          f"{TIE_CASES[0][1]} ===")
    one_iter = {}
    for seed in SHUFFLE_SEEDS:
        # One solver per seed, stepped before the next is built: the shuffle seed
        # is read when the root is created.
        s = CFRSolver(_ShuffledGame(g, seed), cfg)
        s.step()
        r = dict(s.regret_sum[TIE_KEY])
        one_iter[f"scalar-shuffled-{seed}"] = {"regret": r, "strategy": regret_matching(r, ["p", "b"])}
    for name, s in (("scalar", CFRSolver(g, cfg)), ("vectorized", VectorizedCFRSolver(g, cfg))):
        s.step()
        r = dict(s.regret_sum[TIE_KEY])
        one_iter[name] = {"regret": r, "strategy": regret_matching(r, ["p", "b"])}
    for name, row in one_iter.items():
        print(f"  {name:<20} regret {row['regret']}  ->  strategy {row['strategy']}")

    cases = []
    for rung, label in TIE_CASES:
        g, cfg = game(rung), ablation_cube()[label]
        # shuffle seed 1 happens to round the tie to exactly 0.0, like the vectorized run
        runs = {"scalar": CFRSolver(g, cfg),
                "scalar-shuffled-1": CFRSolver(_ShuffledGame(g, 1), cfg),
                "vectorized": VectorizedCFRSolver(g, cfg)}
        print(f"\n{rung} / {label}: rel regret gap vs scalar, and mean NashConv per seat")
        print(f"{'T':>4}{'shuffled-1':>13}{'vectorized':>13}{'shuf-1 vs vec':>15}"
              f"{'scalar':>13}{'shuffled-1':>13}{'vectorized':>13}")
        curve = []
        for t in range(1, TIE_ITERS + 1):
            for s in runs.values():
                s.step()
            if t in TIE_CHECKPOINTS:
                ref = runs["scalar"].regret_sum
                p = {"iterations": t,
                     "rel_gap_shuffled1": rel_gap(ref, runs["scalar-shuffled-1"].regret_sum),
                     "rel_gap_vectorized": rel_gap(ref, runs["vectorized"].regret_sum),
                     "rel_gap_shuffled1_vs_vectorized": rel_gap(
                         runs["scalar-shuffled-1"].regret_sum, runs["vectorized"].regret_sum),
                     "nashconv_mean": {n: nash_conv(g, s.average_strategy()).mean
                                       for n, s in runs.items()}}
                curve.append(p)
                m = p["nashconv_mean"]
                print(f"{t:>4}{p['rel_gap_shuffled1']:>13.2e}{p['rel_gap_vectorized']:>13.2e}"
                      f"{p['rel_gap_shuffled1_vs_vectorized']:>15.2e}{m['scalar']:>13.4e}"
                      f"{m['scalar-shuffled-1']:>13.4e}{m['vectorized']:>13.4e}", flush=True)
        cases.append({"rung": rung, "config": label, "curve": curve})
    return {"tie_key": TIE_KEY, "after_one_iteration": one_iter, "cases": cases}


if __name__ == "__main__":
    raise SystemExit(main())
