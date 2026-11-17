"""Ground-truth validation. These tests are the project's correctness argument.

Every headline number is checked against something external: Kuhn's closed form
(Kuhn 1950), OpenSpiel's independent implementation, and -- once the LP oracle
lands -- an exact sequence-form solution. The OpenSpiel tests skip cleanly if
pyspiel is not installed, so CI stays green on a bare checkout, but they are the
reason we can claim the measurement instrument is right rather than hope it is.
"""
import unittest

from pokercfr.eval.best_response import best_response, exploitability, nash_conv
from pokercfr.games.kuhn_family import game
from pokercfr.solvers.cfr import (
    CFR_PLUS,
    VANILLA_CFR,
    CFRConfig,
    CFRSolver,
    ablation_cube,
    policy_expected_values,
)

try:
    import pyspiel  # noqa: F401
    from open_spiel.python import policy as os_policy
    from open_spiel.python.algorithms import cfr as os_cfr
    from open_spiel.python.algorithms import exploitability as os_exp
    HAVE_OPENSPIEL = True
except ImportError:
    HAVE_OPENSPIEL = False

needs_openspiel = unittest.skipUnless(HAVE_OPENSPIEL, "open_spiel not installed (pip install -e '.[verify]')")


class TestBestResponseCorrectness(unittest.TestCase):
    def test_deviation_gain_is_never_negative(self):
        """A best response can never do worse than the policy it responds to.
        If this fails, the two-pass info-set grouping is broken."""
        for rung in ("kuhn", "4rank", "4rank-2suit", "kuhn-3p"):
            g = game(rung)
            result = nash_conv(g, CFRSolver(g, CFR_PLUS).train(50).average_strategy())
            for seat, gain in enumerate(result.gains):
                self.assertGreaterEqual(gain, -1e-12, f"{rung} seat {seat}")

    def test_uniform_random_kuhn_exploitability_is_11_over_24(self):
        self.assertAlmostEqual(exploitability(game("kuhn"), {}), 11 / 24, places=12)

    def test_best_response_to_itself_is_zero_sum_consistent(self):
        g = game("kuhn")
        policy = CFRSolver(g, CFR_PLUS).train(500).average_strategy()
        value, br = best_response(g, policy, responder=0)
        self.assertAlmostEqual(value, policy_expected_values(g, {**policy, **br})[0], places=9)

    def test_br_value_is_realised_for_every_seat_including_multiplayer(self):
        """The BR value a solver claims must be achievable by actually playing the
        BR policy against the original one -- for every seat, not just seat 0."""
        from pokercfr.eval.best_response import _full_policy
        for rung in ("kuhn", "4rank", "kuhn-3p"):
            g = game(rung)
            policy = CFRSolver(g, CFR_PLUS).train(20).average_strategy()
            full = _full_policy(g, policy)
            for seat in range(g.num_players):
                claimed, br = best_response(g, policy, seat)
                realised = policy_expected_values(g, {**full, **br})[seat]
                self.assertAlmostEqual(claimed, realised, places=9, msg=f"{rung} seat {seat}")

    def test_best_response_beats_or_ties_the_policy_per_seat_at_many_budgets(self):
        for rung in ("kuhn-3p", "4rank-2suit-3p"):
            g = game(rung)
            solver = CFRSolver(g, VANILLA_CFR)
            for t in (1, 5, 25):
                solver.train(t - solver.iterations)
                for gain in nash_conv(g, solver.average_strategy()).gains:
                    self.assertGreaterEqual(gain, -1e-12, f"{rung} T={t}")

    def test_exploitability_rejects_three_players(self):
        g = game("kuhn-3p")
        result = nash_conv(g, CFRSolver(g, VANILLA_CFR).train(5).average_strategy())
        with self.assertRaises(ValueError):
            _ = result.exploitability


class TestKuhnClosedForm(unittest.TestCase):
    def test_game_value_is_minus_one_eighteenth(self):
        g = game("kuhn")
        policy = CFRSolver(g, CFR_PLUS).train(20_000).average_strategy()
        self.assertAlmostEqual(policy_expected_values(g, policy)[0], -1 / 18, places=5)

    def test_cfr_plus_drives_exploitability_below_1e4(self):
        g = game("kuhn")
        self.assertLess(exploitability(g, CFRSolver(g, CFR_PLUS).train(1000).average_strategy()), 1e-4)


class TestAblationCube(unittest.TestCase):
    def test_cube_has_eight_distinct_corners(self):
        cube = ablation_cube()
        self.assertEqual(len(cube), 8)
        self.assertEqual(len({(c.regret_matching_plus, c.averaging, c.alternating)
                              for c in cube.values()}), 8)

    def test_corners_are_vanilla_and_cfr_plus(self):
        cube = ablation_cube()
        self.assertEqual(cube["cfr"], VANILLA_CFR)
        self.assertEqual(cube["cfr+"], CFR_PLUS)

    def test_first_iteration_average_is_uniform_for_every_corner(self):
        """After one iteration all regrets are zero, so the average strategy must
        be uniform -- i.e. exactly as exploitable as random. This is the test that
        catches regrets leaking into the strategy mid-iteration."""
        g = game("kuhn")
        for label, cfg in ablation_cube().items():
            policy = CFRSolver(g, cfg).train(1).average_strategy()
            self.assertAlmostEqual(exploitability(g, policy), 11 / 24, places=12, msg=label)

    def test_every_corner_converges(self):
        g = game("kuhn")
        for label, cfg in ablation_cube().items():
            e = exploitability(g, CFRSolver(g, cfg).train(2000).average_strategy())
            self.assertLess(e, 0.02, f"{label} failed to converge: {e}")

    def test_alternating_updates_dominate_on_kuhn(self):
        """Our headline R1 finding, locked in as a regression test: on Kuhn every
        alternating corner beats every simultaneous corner at matched iterations."""
        g = game("kuhn")
        scores = {label: exploitability(g, CFRSolver(g, cfg).train(1000).average_strategy())
                  for label, cfg in ablation_cube().items()}
        alt = [v for label, v in scores.items() if ablation_cube()[label].alternating]
        sim = [v for label, v in scores.items() if not ablation_cube()[label].alternating]
        self.assertLess(max(alt), min(sim), f"alternating did not dominate: {scores}")


@needs_openspiel
class TestAgainstOpenSpiel(unittest.TestCase):
    """Our exploitability machinery vs an independent implementation."""

    def setUp(self):
        self.osg = pyspiel.load_game("kuhn_poker")
        self.ours = game("kuhn")

    def test_uniform_random_exploitability_matches(self):
        theirs = os_exp.exploitability(self.osg, os_policy.UniformRandomPolicy(self.osg))
        self.assertAlmostEqual(theirs, exploitability(self.ours, {}), places=12)

    def test_uniform_random_nashconv_matches(self):
        theirs = os_exp.nash_conv(self.osg, os_policy.UniformRandomPolicy(self.osg))
        self.assertAlmostEqual(theirs, nash_conv(self.ours, {}).total, places=12)

    def test_openspiel_cfr_is_our_alternating_corner(self):
        """OpenSpiel's CFRSolver documents that it uses alternating updates, so it
        is NOT the algorithm as published -- it is our (rm, uniform, alt) corner.
        Matching it to 1e-12 proves both that our traversal is right and that the
        ablation flags mean what we say they mean."""
        theirs, ours = os_cfr.CFRSolver(self.osg), CFRSolver(self.ours, CFRConfig(alternating=True))
        for target in (10, 100, 1000):
            while theirs._iteration < target:
                theirs.evaluate_and_update_policy()
            ours.train(target - ours.iterations)
            self.assertAlmostEqual(os_exp.exploitability(self.osg, theirs.average_policy()),
                                   exploitability(self.ours, ours.average_strategy()),
                                   places=12, msg=f"T={target}")

    def test_openspiel_cfr_plus_matches_our_cfr_plus(self):
        theirs, ours = os_cfr.CFRPlusSolver(self.osg), CFRSolver(self.ours, CFR_PLUS)
        for target in (10, 100, 1000):
            while theirs._iteration < target:
                theirs.evaluate_and_update_policy()
            ours.train(target - ours.iterations)
            self.assertAlmostEqual(os_exp.exploitability(self.osg, theirs.average_policy()),
                                   exploitability(self.ours, ours.average_strategy()),
                                   places=12, msg=f"T={target}")


if __name__ == "__main__":
    unittest.main()
