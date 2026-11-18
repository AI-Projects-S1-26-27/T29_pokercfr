"""Acceptance tests for the sequence-form LP oracle (solvers/lp_oracle.py).

The oracle is only useful as an INDEPENDENT check if it is itself checked against
things it shares no code with: Kuhn's closed form, our exact best response,
OpenSpiel's exploitability, and small hand-built games whose value is known.
"""
import shutil
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np

from pokercfr.core.config import RunSpec
from pokercfr.eval.exploitability import evaluate, validate_record
from pokercfr.experiments import runner
from pokercfr.games import leduc
from pokercfr.games.kuhn_family import game
from pokercfr.solvers import lp_oracle
from pokercfr.solvers.cfr import policy_expected_values
from pokercfr.solvers.lp_oracle import (
    LPOracleError,
    LPOracleSolver,
    build_sequence_form,
    solve,
    solve_exact,
)

try:
    import pyspiel
    from open_spiel.python import policy as os_policy
    from open_spiel.python.algorithms import exploitability as os_exp
    HAVE_OPENSPIEL = True
except ImportError:
    HAVE_OPENSPIEL = False

needs_openspiel = unittest.skipUnless(HAVE_OPENSPIEL, "open_spiel not installed")

TWO_PLAYER_RUNGS = ("kuhn", "4rank", "4rank-2dealt", "4rank-2suit", "4rank-2suit-2dealt")


# ---- a tiny explicit-tree game, for structures the poker engines never produce ----

@dataclass(frozen=True)
class _Node:
    """kind: 'T' terminal (payoff), 'C' chance (children with probs), 'D' decision."""
    kind: str
    payoff: tuple = ()
    player: int = 0
    key: str = ""
    children: tuple = ()      # ((label_or_prob, _Node), ...)

    def is_terminal(self):
        return self.kind == "T"

    def is_chance_node(self):
        return self.kind == "C"

    def chance_outcomes(self):
        return [(child, p) for p, child in self.children]

    def current_player(self):
        return self.player

    def legal_actions(self):
        return [a for a, _ in self.children]

    def apply_action(self, action):
        return dict(self.children)[action]

    def utilities(self):
        return list(self.payoff)

    def information_set_key(self, player):
        return self.key


class _TreeGame:
    def __init__(self, root, players=2):
        self._root, self._players = root, players

    name = "toy"

    @property
    def num_players(self):
        return self._players

    def new_initial_state(self):
        return self._root

    def num_info_sets(self):
        return 0


def T(u0):
    return _Node("T", payoff=(u0, -u0))


def D(player, key, **children):
    return _Node("D", player=player, key=key, children=tuple(children.items()))


def matching_pennies():
    """Seat 1 picks without seeing seat 0's choice: value 0, both mix 50/50."""
    return _TreeGame(D(0, "root", h=D(1, "guess", h=T(1), t=T(-1)),
                       t=D(1, "guess", h=T(-1), t=T(1))))


# ---- acceptance criteria from the module docstring -------------------------------

class TestKuhnGroundTruth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = game("kuhn")
        cls.sol = solve(cls.game)

    def test_value_is_minus_one_eighteenth(self):
        self.assertAlmostEqual(self.sol.value, -1 / 18, delta=1e-9)
        self.assertAlmostEqual(self.sol.value_check, -1 / 18, delta=1e-9)

    def test_policy_actually_achieves_the_value(self):
        # The value the LP reports must equal what its policy really earns.
        self.assertAlmostEqual(policy_expected_values(self.game, self.sol.policy)[0],
                               -1 / 18, delta=1e-9)

    def test_policy_is_unexploitable_by_our_best_response(self):
        self.assertLess(evaluate(self.game, self.sol.policy).exploitability, 1e-9)

    def test_lies_on_kuhns_alpha_family(self):
        # Kuhn (1950): seat 0 bets J w.p. alpha in [0, 1/3], K w.p. 3 alpha, and
        # calls with Q facing a bet w.p. alpha + 1/3.
        b = lambda k: self.sol.policy[k]["b"]  # noqa: E731
        alpha = b("J")
        self.assertLessEqual(alpha, 1 / 3 + 1e-9)
        self.assertAlmostEqual(b("K"), 3 * alpha, delta=1e-9)
        self.assertAlmostEqual(b("Qpb"), alpha + 1 / 3, delta=1e-9)

    def test_sequence_form_sizes(self):
        # 6 info sets per seat (3 cards x 2 histories), 2 actions each, + empty.
        self.assertEqual(self.sol.form.num_infosets, (6, 6))
        self.assertEqual(self.sol.form.num_sequences, (13, 13))

    def test_solve_exact_matches_the_stub_signature(self):
        value, policy = solve_exact(self.game)
        self.assertAlmostEqual(value, -1 / 18, delta=1e-9)
        self.assertEqual(set(policy), set(self.sol.policy))


class TestEveryTwoPlayerRung(unittest.TestCase):
    def test_exploitability_is_zero_and_value_is_earned(self):
        for rung in TWO_PLAYER_RUNGS:
            with self.subTest(rung=rung):
                g = game(rung)
                sol = solve(g)
                self.assertAlmostEqual(sol.value, sol.value_check, delta=1e-9)
                self.assertLess(evaluate(g, sol.policy).exploitability, 1e-9)
                self.assertAlmostEqual(policy_expected_values(g, sol.policy)[0], sol.value,
                                       delta=1e-9)

    def test_leduc(self):
        for cap in (1, 2):
            with self.subTest(cap=cap):
                g = leduc.game(raise_cap=cap)
                sol = solve(g)
                self.assertLess(evaluate(g, sol.policy).exploitability, 1e-9)
                self.assertEqual(sum(sol.form.num_infosets), g.num_info_sets())

    def test_leduc_value(self):
        # Standard Leduc's seat-0 value, -0.0856..., matches the literature value
        # and OpenSpiel (see TestOpenSpiel below for the exploitability check).
        self.assertAlmostEqual(solve(leduc.game()).value, -0.0856064240780, delta=1e-9)


@needs_openspiel
class TestOpenSpiel(unittest.TestCase):
    """A third, fully external judge of the LP's policy."""

    def test_kuhn(self):
        osg = pyspiel.load_game("kuhn_poker")
        sol = solve(game("kuhn"))
        tab = os_policy.TabularPolicy(osg)
        names = "JQK"
        for os_key, row in tab.state_lookup.items():
            card, history = int(os_key[0]), os_key[1:]
            probs = sol.policy[names[card] + history]
            tab.action_probability_array[row] = [probs["p"], probs["b"]]
        self.assertLess(os_exp.exploitability(osg, tab), 1e-9)

    def test_leduc(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from s5_openspiel_leduc_crosscheck import key_map, to_openspiel

        osg, g = pyspiel.load_game("leduc_poker"), leduc.game()
        sol = solve(g)
        self.assertLess(os_exp.exploitability(osg, to_openspiel(osg, sol.policy, key_map(osg, g))),
                        1e-9)


class TestScope(unittest.TestCase):
    def test_rejects_more_than_two_players(self):
        for g in (game("kuhn-3p"), leduc.game(players=3)):
            with self.subTest(game=g.name):
                with self.assertRaisesRegex(ValueError, "TWO-player zero-sum"):
                    solve(g)

    def test_solver_wrapper_rejects_too(self):
        with self.assertRaises(ValueError):
            LPOracleSolver(game("kuhn-3p")).step()


# ---- numerical robustness -------------------------------------------------------

class TestRobustness(unittest.TestCase):
    def test_matching_pennies(self):
        sol = solve(matching_pennies())
        self.assertAlmostEqual(sol.value, 0.0, delta=1e-12)
        for key in ("root", "guess"):
            for p in sol.policy[key].values():
                self.assertAlmostEqual(p, 0.5, delta=1e-9)

    def test_dominated_action_gives_a_pure_strategy(self):
        g = _TreeGame(D(0, "r", good=T(3), bad=T(-1)))
        sol = solve(g)
        self.assertAlmostEqual(sol.value, 3.0, delta=1e-12)
        self.assertEqual(sol.policy["r"], {"good": 1.0, "bad": 0.0})

    def test_single_action_and_degenerate_game(self):
        g = _TreeGame(D(0, "r", only=D(1, "s", only=T(2))))
        sol = solve(g)
        self.assertAlmostEqual(sol.value, 2.0, delta=1e-12)
        self.assertEqual(sol.policy, {"r": {"only": 1.0}, "s": {"only": 1.0}})

    def test_unreachable_info_set_gets_uniform(self):
        # Seat 0 never plays "bad", so seat 0's OWN later info set "later" has a
        # zero-weight parent sequence: unreachable under its plan -> uniform.
        later = D(0, "later", x=T(-5), y=T(-6), z=T(-7))
        g = _TreeGame(D(0, "r", good=T(3), bad=D(1, "opp", go=later, stop=T(-1))))
        sol = solve(g)
        self.assertIn("later", sol.unreachable)
        self.assertEqual(sol.policy["later"], {a: 1 / 3 for a in "xyz"})
        # Seat 1's "opp" is unreachable only because of the OPPONENT. The sequence
        # form tracks own-sequence weight, so it is not flagged; it must still be
        # a valid distribution.
        self.assertNotIn("opp", sol.unreachable)
        self.assertAlmostEqual(sum(sol.policy["opp"].values()), 1.0, delta=1e-12)
        self.assertAlmostEqual(sol.value, 3.0, delta=1e-12)
        # (No eval.best_response check here: it orders info sets by
        # `state.history`, which is not part of the frozen protocol, so it cannot
        # run on this toy tree. Exploitability is checked on every poker rung.)

    def test_zero_probability_chance_branch_is_skipped(self):
        root = _Node("C", children=((1.0, D(0, "a", x=T(1), y=T(0))),
                                    (0.0, D(0, "b", x=T(-9), y=T(9)))))
        sol = solve(_TreeGame(root))
        self.assertAlmostEqual(sol.value, 1.0, delta=1e-12)
        self.assertNotIn("b", sol.policy)  # never reached by chance

    def test_rows_are_normalised_and_flows_hold(self):
        for g in (game("4rank-2suit-2dealt"), leduc.game()):
            sol = solve(g)
            self.assertLess(sol.max_flow_residual, 1e-8)
            for key, row in sol.policy.items():
                self.assertAlmostEqual(sum(row.values()), 1.0, delta=1e-12, msg=key)
                self.assertTrue(all(p >= 0.0 for p in row.values()), key)

    def test_payoff_matrix_is_sparse_and_reproduces_expected_value(self):
        # x^T A y must equal the expected payoff computed by plain tree walking.
        g = game("4rank-2suit")
        sol = solve(g)
        x, y = sol.plans
        self.assertAlmostEqual(float(x @ (sol.form.payoff @ y)),
                               policy_expected_values(g, sol.policy)[0], delta=1e-9)
        n0, n1 = sol.form.num_sequences
        self.assertLess(sol.form.payoff.nnz, n0 * n1)

    def test_rejects_non_zero_sum(self):
        g = _TreeGame(D(0, "r", a=_Node("T", payoff=(1, 1)), b=T(0)))
        with self.assertRaisesRegex(ValueError, "not zero-sum"):
            build_sequence_form(g)

    def test_rejects_bad_chance_probabilities(self):
        root = _Node("C", children=((0.7, T(1)), (0.7, T(0))))
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            build_sequence_form(_TreeGame(root))

    def test_rejects_imperfect_recall(self):
        # Seat 0 forgets its own first action: same key after "l" and after "r".
        g = _TreeGame(D(0, "first", l=D(0, "forgot", a=T(1), b=T(0)),
                        r=D(0, "forgot", a=T(0), b=T(1))))
        with self.assertRaisesRegex(ValueError, "perfect recall"):
            build_sequence_form(g)

    def test_rejects_a_key_shared_by_both_seats(self):
        g = _TreeGame(D(0, "same", a=D(1, "same", a=T(1), b=T(0)), b=T(0)))
        with self.assertRaisesRegex(ValueError, "both seats"):
            build_sequence_form(g)

    def test_solver_failure_is_reported_not_swallowed(self):
        failed = mock.Mock(status=4, x=None, message="numerical difficulties")
        with mock.patch.object(lp_oracle, "linprog", return_value=failed):
            with self.assertRaisesRegex(LPOracleError, "HiGHS failed"):
                solve(game("kuhn"))

    def test_inconsistent_halves_are_refused(self):
        real = lp_oracle._solve_side
        calls = []

        def skewed(*args, **kwargs):
            value, plan = real(*args, **kwargs)
            calls.append(1)
            return (value + 1e-3 if len(calls) == 2 else value), plan

        with mock.patch.object(lp_oracle, "_solve_side", side_effect=skewed):
            with self.assertRaisesRegex(LPOracleError, "disagree"):
                solve(game("kuhn"))

    def test_plans_are_nonnegative(self):
        sol = solve(leduc.game())
        for plan in sol.plans:
            self.assertGreaterEqual(float(np.min(plan)), -1e-9)


class TestRunnerIntegration(unittest.TestCase):
    """The registry's `lp-oracle` cells go through the ordinary runner."""

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._root, self._ckpt = runner.ROOT, runner.CKPT_DIR
        runner.ROOT = self._tmp
        runner.CKPT_DIR = self._tmp / "results" / ".ckpt"

    def tearDown(self):
        runner.ROOT, runner.CKPT_DIR = self._root, self._ckpt
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_lp_oracle_cell_produces_a_valid_zero_exploitability_record(self):
        spec = RunSpec(game="kuhn", variant="lp-oracle", iterations=1, checkpoints=(1,))
        record = runner.run_one(spec, progress=lambda *_: None)
        self.assertEqual(validate_record(record), [])
        self.assertLess(record["curve"][-1]["exploitability"], 1e-9)


if __name__ == "__main__":
    unittest.main()
