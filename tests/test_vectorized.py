"""The vectorized traversal must compute the SAME thing as solvers/cfr.py.

A fast traversal that computes the wrong thing is worse than a slow one, so this
file compares the vectorized solver against the scalar one table-for-table, on
every corner of the ablation cube plus DCFR, on two- and three-player rungs.

Tolerance, and why it is not zero
---------------------------------
Both solvers do the same arithmetic in a different ORDER: the scalar solver
accumulates counterfactual values deal by deal, the vectorized one as matrix-
vector products. Floating-point addition is not associative, so the two agree to
within a few ulps of the magnitudes involved -- measured at <= ~1e-14 RELATIVE to
the largest entry of each table after 30 iterations (scripts/sid2 reports the
exact figures). We assert 1e-11 relative: three orders of magnitude of headroom
over the observed rounding, and still far below any real algorithmic difference
(a wrong reach, a missing chance factor or a mis-ordered commit shows up at
1e-2 or larger -- see test_a_real_algorithmic_change_is_detected).
"""
import pickle
import unittest

from pokercfr.eval.exploitability import evaluate
from pokercfr.games import leduc
from pokercfr.games.kuhn_family import game
from pokercfr.solvers.cfr import DCFR, CFRConfig, CFRSolver, ablation_cube
from pokercfr.solvers.vectorized import VectorizedCFRSolver, nash_conv

REL_TOL = 1e-11
ITERS = 30
CONFIGS = {**ablation_cube(), "dcfr": DCFR,
           "dcfr-alt": CFRConfig(averaging="linear", alternating=True, dcfr=(1.5, 0.0, 2.0)),
           "quadratic": CFRConfig(averaging="quadratic")}
RUNGS = ("kuhn", "4rank", "4rank-2dealt", "4rank-2suit", "4rank-2suit-2dealt", "kuhn-3p",
         "4rank-2suit-3p")


def table_gap(a: dict, b: dict) -> tuple[float, float]:
    """(max abs difference, max abs entry), after asserting identical keys/actions."""
    if set(a) != set(b):
        raise AssertionError(f"info-set keys differ: {sorted(set(a) ^ set(b))[:5]}")
    gap = scale = 0.0
    for key in a:
        if set(a[key]) != set(b[key]):
            raise AssertionError(f"actions differ at {key!r}")
        for act in a[key]:
            gap = max(gap, abs(a[key][act] - b[key][act]))
            scale = max(scale, abs(a[key][act]))
    return gap, scale


class TestScalarEquivalence(unittest.TestCase):
    def assertTablesMatch(self, a, b, what):
        gap, scale = table_gap(a, b)
        self.assertLessEqual(gap, REL_TOL * max(scale, 1.0), f"{what}: gap {gap:.3e}")

    def test_every_config_on_every_rung(self):
        for rung in RUNGS:
            g = game(rung)
            for label, cfg in CONFIGS.items():
                with self.subTest(rung=rung, config=label):
                    s = CFRSolver(g, cfg).train(ITERS)
                    v = VectorizedCFRSolver(g, cfg).train(ITERS)
                    self.assertEqual(s.iterations, v.iterations)
                    self.assertTablesMatch(s.regret_sum, v.regret_sum, "regret_sum")
                    self.assertTablesMatch(s.strategy_sum, v.strategy_sum, "strategy_sum")
                    self.assertTablesMatch(s.average_strategy(), v.average_strategy(), "average")
                    self.assertTablesMatch(s.current_strategy(), v.current_strategy(), "current")

    def test_exploitability_matches(self):
        for rung in ("kuhn", "4rank-2suit-2dealt"):
            g = game(rung)
            for label in ("cfr", "cfr+"):
                with self.subTest(rung=rung, config=label):
                    s = CFRSolver(g, CONFIGS[label]).train(100).average_strategy()
                    v = VectorizedCFRSolver(g, CONFIGS[label]).train(100).average_strategy()
                    self.assertAlmostEqual(evaluate(g, s).exploitability,
                                           evaluate(g, v).exploitability, delta=1e-12)

    def test_info_set_count_matches_the_game(self):
        for rung in RUNGS:
            g = game(rung)
            self.assertEqual(len(VectorizedCFRSolver(g).train(1).regret_sum), g.num_info_sets())

    def test_converges_to_kuhn_value(self):
        from pokercfr.solvers.cfr import policy_expected_values
        g = game("kuhn")
        avg = VectorizedCFRSolver(g, CONFIGS["cfr+"]).train(2000).average_strategy()
        self.assertAlmostEqual(policy_expected_values(g, avg)[0], -1 / 18, delta=1e-4)

    def test_a_real_algorithmic_change_is_detected(self):
        """Guard against a tolerance so loose it would pass anything: a genuinely
        different update rule (simultaneous vs alternating) must FAIL the check."""
        g = game("4rank")
        s = CFRSolver(g, CONFIGS["cfr"]).train(ITERS)
        v = VectorizedCFRSolver(g, CONFIGS["rm/uniform/alt"]).train(ITERS)
        gap, scale = table_gap(s.regret_sum, v.regret_sum)
        self.assertGreater(gap, 1e-3 * scale)

    def test_before_any_step(self):
        v = VectorizedCFRSolver(game("kuhn"))
        s = CFRSolver(game("kuhn"))
        self.assertEqual(v.regret_sum, s.regret_sum)
        self.assertEqual(v.iterations, 0)


class TestFourPlayerRegretTie(unittest.TestCase):
    """The one configuration where scalar and vectorized tables SPLIT:
    4rank-2suit-4p with alternating updates. This pins down why, so the
    exception is tested rather than only explained in a script.

    At info set Qhbpp the two actions are exactly tied (true regret 0). Regret
    matching is discontinuous at 0: a summation order that rounds the regret to
    0.0 plays uniform, one that rounds it to +1e-18 plays pure. The default
    scalar run rounds to +1e-18; the vectorized one to 0.0. Claims checked:
      1. the SCALAR solver alone splits the same way when only its deal order
         is shuffled -- so the split is not caused by vectorization;
      2. a scalar run whose shuffle also rounds the tie to 0.0 matches the
         vectorized run to the usual tolerance.
    Not claimed: that the default scalar and vectorized runs are equal here.
    """

    ITERS = 5

    @classmethod
    def setUpClass(cls):
        import random

        from pokercfr.games.kuhn_family import ChanceRoot

        class ShuffledRoot(ChanceRoot):
            def chance_outcomes(self):
                out = super().chance_outcomes()
                random.Random(1).shuffle(out)  # seed 1 rounds the Qhbpp tie to 0.0
                return out

        class ShuffledGame:
            def __init__(self, inner):
                self.inner = inner
                self.num_players = inner.num_players

            def new_initial_state(self):
                return ShuffledRoot(self.inner.config)

        g = game("4rank-2suit-4p")
        cfg = CONFIGS["rm/uniform/alt"]
        cls.scalar = CFRSolver(g, cfg).train(cls.ITERS)
        cls.shuffled = CFRSolver(ShuffledGame(g), cfg).train(cls.ITERS)
        cls.vector = VectorizedCFRSolver(g, cfg).train(cls.ITERS)

    def test_scalar_splits_from_itself_under_reordering(self):
        gap, scale = table_gap(self.scalar.regret_sum, self.shuffled.regret_sum)
        self.assertGreater(gap, 1e-3 * scale)

    def test_vectorized_matches_scalar_with_the_same_tie_rounding(self):
        gap, scale = table_gap(self.shuffled.regret_sum, self.vector.regret_sum)
        self.assertLessEqual(gap, REL_TOL * max(scale, 1.0))
        gap, scale = table_gap(self.shuffled.strategy_sum, self.vector.strategy_sum)
        self.assertLessEqual(gap, REL_TOL * max(scale, 1.0))


class TestFastEvaluator(unittest.TestCase):
    """vectorized.nash_conv must agree with eval/best_response.py exactly."""

    def test_matches_exact_best_response(self):
        for rung in RUNGS:
            g = game(rung)
            for policy in ({}, CFRSolver(g, CONFIGS["cfr+"]).train(20).average_strategy(),
                           CFRSolver(g, CONFIGS["cfr"]).train(3).current_strategy()):
                with self.subTest(rung=rung):
                    ref, fast = evaluate(g, policy), nash_conv(g, policy)
                    for a, b in zip(ref.gains, fast.gains, strict=True):
                        self.assertAlmostEqual(a, b, delta=1e-12)
                    for a, b in zip(ref.policy_values, fast.policy_values, strict=True):
                        self.assertAlmostEqual(a, b, delta=1e-12)

    def test_pure_policy_with_unreachable_info_sets(self):
        # A deterministic policy makes opponent info sets unreachable -- the case
        # that exercises the probability-zero branches of the best response.
        g = game("4rank-2suit")
        pure = {k: {a: float(a == "p") for a in row}
                for k, row in CFRSolver(g).train(1).average_strategy().items()}
        ref, fast = evaluate(g, pure), nash_conv(g, pure)
        for a, b in zip(ref.gains, fast.gains, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-12)


class TestScopeAndPlumbing(unittest.TestCase):
    def test_rejects_leduc(self):
        with self.assertRaisesRegex(TypeError, "Kuhn-family"):
            VectorizedCFRSolver(leduc.game())

    def test_refuses_rungs_whose_tensors_do_not_fit(self):
        with self.assertRaisesRegex(MemoryError, "GiB"):
            VectorizedCFRSolver(game("6rank-4suit-2dealt-4p"))

    def test_pickles_for_runner_checkpoints(self):
        v = VectorizedCFRSolver(game("4rank-2suit"), CONFIGS["cfr+"]).train(5)
        blob = pickle.dumps(v)
        self.assertNotIn(b"_space", blob)  # the derived tensors are not checkpointed
        w = pickle.loads(blob)
        self.assertEqual(w.regret_sum, v.regret_sum)
        w.step()
        v.step()
        self.assertEqual(w.regret_sum, v.regret_sum)

    def test_deterministic(self):
        a = VectorizedCFRSolver(game("4rank-2suit"), CONFIGS["cfr+"]).train(10)
        b = VectorizedCFRSolver(game("4rank-2suit"), CONFIGS["cfr+"]).train(10)
        self.assertEqual(a.regret_sum, b.regret_sum)
        self.assertEqual(a.strategy_sum, b.strategy_sum)


class TestRunnerTraversalFlag(unittest.TestCase):
    """`runner --traversal vectorized` must give the same curve as the scalar
    traversal, in the frozen schema, without changing any spec hash."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        from pokercfr.experiments import runner
        self.runner = runner
        self._tmp = Path(tempfile.mkdtemp())
        self._saved = runner.ROOT, runner.CKPT_DIR, runner.TRAVERSAL
        runner.ROOT, runner.CKPT_DIR = self._tmp, self._tmp / "results" / ".ckpt"

    def tearDown(self):
        import shutil
        r = self.runner
        r.ROOT, r.CKPT_DIR = self._saved[0], self._saved[1]
        r._set_traversal(self._saved[2])
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _curve(self, traversal, spec):
        self.runner._set_traversal(traversal)
        return self.runner.run_one(spec, resume=False, progress=lambda *_: None)

    def test_same_curve_same_schema(self):
        from pokercfr.core.config import RunSpec
        from pokercfr.eval.exploitability import validate_record

        spec = RunSpec(game="4rank-2suit", variant="cfr+", iterations=50, checkpoints=(10, 50))
        scalar, vector = self._curve("scalar", spec), self._curve("vectorized", spec)
        self.assertEqual(validate_record(vector), [])
        self.assertEqual((scalar["traversal"], vector["traversal"]), ("scalar", "vectorized"))
        self.assertEqual(scalar["spec"]["spec_hash"], vector["spec"]["spec_hash"])
        for a, b in zip(scalar["curve"], vector["curve"], strict=True):
            self.assertEqual(a["iterations"], b["iterations"])
            for x, y in zip(a["gains"], b["gains"], strict=True):
                self.assertAlmostEqual(x, y, delta=1e-12)

    def test_vectorized_flag_on_leduc_is_an_error_not_a_silent_fallback(self):
        from pokercfr.core.config import RunSpec

        self.runner._set_traversal("vectorized")
        with self.assertRaises(TypeError):
            self.runner.build_solver(RunSpec(game="leduc-2p-cap2", variant="cfr"))

    def test_unknown_traversal_rejected(self):
        with self.assertRaises(ValueError):
            self.runner._set_traversal("gpu")


if __name__ == "__main__":
    unittest.main()
