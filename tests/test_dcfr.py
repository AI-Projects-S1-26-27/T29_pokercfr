"""Discounted CFR: the paper's limits, and numerical safety across the grid."""
import math
import unittest

from pokercfr.core.config import RunSpec
from pokercfr.eval.best_response import exploitability
from pokercfr.experiments.runner import build_solver
from pokercfr.games.kuhn_family import game
from pokercfr.solvers.cfr import (
    DCFR,
    DCFR_AS_CFR,
    DCFR_AS_CFR_PLUS,
    VANILLA_CFR,
    CFRConfig,
    CFRSolver,
    ablation_cube,
)

GRID = [(a, b, g) for a in (1.0, 1.5, 2.0, 3.0) for b in (0.0, 0.5, 1.0) for g in (1.0, 2.0, 3.0)]


class TestDCFRLimits(unittest.TestCase):
    def test_no_discount_reproduces_vanilla_cfr(self):
        g = game("kuhn")
        vanilla = CFRSolver(g, VANILLA_CFR).train(300).average_strategy()
        limit = CFRSolver(g, DCFR_AS_CFR).train(300).average_strategy()
        self.assertAlmostEqual(exploitability(g, vanilla), exploitability(g, limit), places=12)

    def test_no_discount_limit_matches_on_a_larger_rung(self):
        g = game("4rank-2suit")
        vanilla = CFRSolver(g, VANILLA_CFR).train(100).average_strategy()
        limit = CFRSolver(g, DCFR_AS_CFR).train(100).average_strategy()
        self.assertAlmostEqual(exploitability(g, vanilla), exploitability(g, limit), places=12)

    def test_negative_infinite_beta_clamps_negative_regret(self):
        # DCFR discounts at the start of an iteration, so the clamp lands one
        # iteration after the regret is accrued -- unlike RM+, which clamps on
        # the update itself.
        solver = CFRSolver(game("kuhn"), DCFR_AS_CFR_PLUS).train(200)
        solver.iterations += 1
        solver._apply_dcfr_discount()
        worst = min(r for regrets in solver.regret_sum.values() for r in regrets.values())
        self.assertGreaterEqual(worst, 0.0)

    def test_negative_infinite_beta_is_exactly_regret_matching_plus(self):
        # Deferring the clamp by one iteration changes nothing: RM+ clamps just
        # after adding a delta, DCFR just before adding the next, so cumulative
        # regret is non-negative whenever a delta lands. Part of the paper's
        # unification claim, verified rather than cited.
        g = game("kuhn")
        dcfr = CFRSolver(g, DCFR_AS_CFR_PLUS).train(200)
        rm_plus = CFRSolver(g, CFRConfig(regret_matching_plus=True, averaging="linear")).train(200)
        self.assertAlmostEqual(exploitability(g, dcfr.average_strategy()),
                               exploitability(g, rm_plus.average_strategy()), places=12)

    def test_dcfr_cannot_express_alternating_updates(self):
        """Its CFR+ limit is the rm+/linear/sim corner, not cfr+ itself."""
        g = game("kuhn")
        limit = CFRSolver(g, DCFR_AS_CFR_PLUS).train(300).average_strategy()
        corner = CFRSolver(g, ablation_cube()["rm+/linear/sim"]).train(300).average_strategy()
        cfr_plus = CFRSolver(g, ablation_cube()["cfr+"]).train(300).average_strategy()
        self.assertLess(abs(exploitability(g, limit) - exploitability(g, corner)),
                        abs(exploitability(g, limit) - exploitability(g, cfr_plus)))


class TestDiscountIsApplied(unittest.TestCase):
    def test_gamma_shrinks_the_strategy_sum(self):
        g = game("kuhn")
        undiscounted = CFRSolver(g, CFRConfig(averaging="linear", dcfr=(1.5, 0.0, 0.0))).train(200)
        discounted = CFRSolver(g, CFRConfig(averaging="linear", dcfr=(1.5, 0.0, 3.0))).train(200)
        def total(solver):
            return sum(v for sums in solver.strategy_sum.values() for v in sums.values())

        self.assertLess(total(discounted), total(undiscounted))

    def test_first_iteration_is_not_discounted(self):
        g = game("kuhn")
        self.assertAlmostEqual(
            exploitability(g, CFRSolver(g, DCFR).train(1).average_strategy()), 11 / 24, places=12)


class TestNumericalSafety(unittest.TestCase):
    def test_no_nan_or_inf_across_the_grid(self):
        g = game("4rank-2suit")
        for alpha, beta, gamma in GRID:
            solver = CFRSolver(g, CFRConfig(averaging="linear", dcfr=(alpha, beta, gamma)))
            solver.train(200)
            values = [v for t in (solver.regret_sum, solver.strategy_sum)
                      for d in t.values() for v in d.values()]
            self.assertTrue(all(math.isfinite(v) for v in values), f"({alpha},{beta},{gamma})")

    def test_large_exponent_does_not_overflow(self):
        solver = CFRSolver(game("kuhn"), CFRConfig(dcfr=(200.0, 200.0, 0.0)))
        solver.iterations = 5000
        solver._apply_dcfr_discount()
        solver.train(5)
        self.assertTrue(all(math.isfinite(v) for d in solver.regret_sum.values()
                            for v in d.values()))


class TestRegistryNaming(unittest.TestCase):
    def test_dcfr_variant_names_resolve(self):
        for alpha, beta, gamma in GRID[:6]:
            name = f"dcfr-a{alpha:g}-b{beta:g}-g{gamma:g}"
            _, solver = build_solver(RunSpec(game="kuhn", variant=name))
            self.assertEqual(solver.config.dcfr, (alpha, beta, gamma), name)


if __name__ == "__main__":
    unittest.main()
