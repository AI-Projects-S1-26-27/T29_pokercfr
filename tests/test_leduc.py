"""Acceptance tests for the Leduc engine, written before it.

Leduc has enough fiddly rules (two bet sizes, a per-round raise cap, pairing with
the public card, split pots) that a wrong engine still converges to *something*
and produces a plausible exploitability curve. These tests are how we find out.
"""
import unittest

from pokercfr.core.protocols import Game, GameState
from pokercfr.eval.best_response import exploitability
from pokercfr.games import leduc
from pokercfr.games.leduc import LeducConfig, LeducGame
from pokercfr.solvers.cfr import CFR_PLUS, CFRSolver

try:
    import pyspiel
    HAVE_OPENSPIEL = True
except ImportError:
    HAVE_OPENSPIEL = False


def walk(state, path=()):
    """Yield (state, action path) for every reachable state, chance outcomes included."""
    yield state, path
    if state.is_terminal():
        return
    if state.is_chance_node():
        for child, _ in state.chance_outcomes():
            yield from walk(child, path)
        return
    for a in state.legal_actions():
        yield from walk(state.apply_action(a), path + (a,))


def after_deal(g, private, public=None):
    """The decision state right after a specific deal (and public card, if given)."""
    for state, _ in g.new_initial_state().chance_outcomes():
        if state.private == private:
            return state, public
    raise AssertionError(f"deal {private} not found")


def play(state, actions, public=None):
    """Apply betting actions; when the public-card chance node appears, take `public`."""
    for a in actions:
        if state.is_chance_node():
            state = next(s for s, _ in state.chance_outcomes() if s.public == public)
        state = state.apply_action(a)
    if state.is_chance_node() and public is not None:
        state = next(s for s, _ in state.chance_outcomes() if s.public == public)
    return state


class TestProtocolConformance(unittest.TestCase):
    def setUp(self):
        self.g = leduc.game()

    def test_satisfies_protocols(self):
        self.assertIsInstance(self.g, Game)
        self.assertIsInstance(self.g.new_initial_state(), GameState)

    def test_exactly_one_node_type_everywhere(self):
        for state, _ in walk(self.g.new_initial_state()):
            decision = not state.is_terminal() and not state.is_chance_node()
            self.assertEqual(state.is_terminal() + state.is_chance_node() + decision, 1)

    def test_chance_probabilities_sum_to_one_at_both_chance_nodes(self):
        root = self.g.new_initial_state()
        self.assertTrue(root.is_chance_node())
        self.assertAlmostEqual(sum(p for _, p in root.chance_outcomes()), 1.0, places=12)
        public_nodes = 0
        for state, _ in walk(root):
            if state.is_chance_node() and state is not root:
                public_nodes += 1
                outcomes = state.chance_outcomes()
                self.assertAlmostEqual(sum(p for _, p in outcomes), 1.0, places=12)
                # The public card comes from the deck minus the private cards.
                self.assertEqual(len(outcomes), self.g.config.deck_size - self.g.num_players)
                self.assertTrue(all(s.public not in s.private for s, _ in outcomes))
        self.assertGreater(public_nodes, 0)

    def test_wrong_node_type_calls_raise(self):
        root = self.g.new_initial_state()
        with self.assertRaises(ValueError):
            root.current_player()
        with self.assertRaises(ValueError):
            root.utilities()
        decision = root.chance_outcomes()[0][0]
        with self.assertRaises(ValueError):
            decision.utilities()

    def test_history_length_is_monotone_depth(self):
        """best_response orders info sets by len(state.history): every action must lengthen it."""
        for state, _ in walk(self.g.new_initial_state()):
            if state.is_terminal() or state.is_chance_node():
                continue
            for a in state.legal_actions():
                child = state.apply_action(a)
                self.assertGreater(len(child.history), len(state.history))


class TestGameRules(unittest.TestCase):
    def setUp(self):
        self.g = leduc.game()

    def test_utilities_sum_to_zero_at_every_terminal(self):
        terminals = 0
        for state, _ in walk(self.g.new_initial_state()):
            if state.is_terminal():
                terminals += 1
                self.assertAlmostEqual(sum(state.utilities()), 0.0, places=12)
        self.assertGreater(terminals, 0)

    def test_pair_with_public_card_beats_unpaired_hand(self):
        # Cards are rank * suits + suit: 0,1 = J; 2,3 = Q; 4,5 = K.
        s, _ = after_deal(self.g, (0, 4))  # p0 holds J, p1 holds K
        end = play(s, "cccc", public=1)  # public J pairs p0
        self.assertEqual(end.utilities(), [1.0, -1.0])

    def test_higher_private_rank_wins_without_pair(self):
        s, _ = after_deal(self.g, (0, 4))
        end = play(s, "cccc", public=2)  # public Q pairs nobody
        self.assertEqual(end.utilities(), [-1.0, 1.0])

    def test_equal_private_ranks_split_the_pot(self):
        s, _ = after_deal(self.g, (0, 1))  # both hold a J
        end = play(s, "rccc", public=4)
        self.assertEqual(end.utilities(), [0.0, 0.0])

    def test_bet_sizes_are_two_then_four(self):
        s, _ = after_deal(self.g, (4, 0))  # p0 K beats p1 J
        # Round 1: p0 raises 2, p1 calls. Round 2: p0 raises 4, p1 calls.
        end = play(s, "rcrc", public=2)
        self.assertEqual(end.utilities(), [7.0, -7.0])  # ante 1 + 2 + 4

    def test_fold_ends_hand_and_pays_non_folder(self):
        s, _ = after_deal(self.g, (0, 4))
        end = play(s, "rf")  # p0 raises with J, p1 folds K
        self.assertTrue(end.is_terminal())
        self.assertEqual(end.utilities(), [1.0, -1.0])

    def test_fold_only_legal_facing_a_bet(self):
        s, _ = after_deal(self.g, (0, 4))
        self.assertEqual(s.legal_actions(), ["c", "r"])
        self.assertEqual(s.apply_action("r").legal_actions(), ["f", "c", "r"])

    def test_raise_cap_is_per_round(self):
        for cap in (1, 2, 3):
            g = leduc.game(raise_cap=cap)
            for state, _ in walk(g.new_initial_state()):
                if not state.is_terminal() and not state.is_chance_node():
                    for rnd in state.history.split("/"):
                        self.assertLessEqual(rnd.count("r"), cap, f"cap={cap} history={state.history}")
            # And the cap is reached in round 2 even after it was reached in round 1.
            s, _ = after_deal(g, (0, 4))
            end = play(s, "r" * cap + "c" + "r" * cap, public=2)
            self.assertNotIn("r", end.legal_actions())


class TestSizesAndScaling(unittest.TestCase):
    def test_info_set_count_matches_full_walk(self):
        for players in (2, 3):
            for cap in (1, 2):
                g = leduc.game(players=players, raise_cap=cap)
                keys = {(s.current_player(), s.information_set_key(s.current_player()))
                        for s, _ in walk(g.new_initial_state())
                        if not s.is_terminal() and not s.is_chance_node()}
                self.assertEqual(g.num_info_sets(), len(keys), f"{players}p cap{cap}")

    @unittest.skipUnless(HAVE_OPENSPIEL, "open_spiel not installed")
    def test_two_player_info_set_count_matches_openspiel(self):
        osg = pyspiel.load_game("leduc_poker")
        seen = set()

        def os_walk(st):
            if st.is_terminal():
                return
            if st.is_chance_node():
                for a, _ in st.chance_outcomes():
                    os_walk(st.child(a))
                return
            seen.add(st.information_state_string())
            for a in st.legal_actions():
                os_walk(st.child(a))

        os_walk(osg.new_initial_state())
        self.assertEqual(leduc.game().num_info_sets(), len(seen))

    def test_raise_cap_grows_info_sets(self):
        counts = [leduc.game(raise_cap=c).num_info_sets() for c in (1, 2, 3)]
        self.assertLess(counts[0], counts[1])
        self.assertLess(counts[1], counts[2])

    def test_deck_does_not_change_with_player_count(self):
        """The novelty claim, as a test: OpenSpiel grows the deck as 2(N+1); we do not."""
        decks = {leduc.game(players=n).config.deck_size for n in range(2, 6)}
        self.assertEqual(decks, {6})

    def test_too_many_players_for_the_deck_raises(self):
        with self.assertRaisesRegex(ValueError, "deck"):
            leduc.game(players=6)  # 6 private cards + 1 public > 6-card deck

    def test_name_matches_runner_contract(self):
        self.assertEqual(leduc.game(players=3, raise_cap=1).name, "leduc-3p-cap1")
        self.assertEqual(LeducGame(LeducConfig()).name, "leduc-2p-cap2")


class TestConvergence(unittest.TestCase):
    def test_cfr_plus_drives_exploitability_below_one_hundredth(self):
        g = leduc.game()
        policy = CFRSolver(g, CFR_PLUS).train(200).average_strategy()
        self.assertLess(exploitability(g, policy), 0.01)


if __name__ == "__main__":
    unittest.main()
