import unittest

from pokercfr.games.kuhn_family import GameConfig, PokerState, _hand_strength


class TestLadderBoundaries(unittest.TestCase):
    def test_three_card_strength_orders_trips_two_pair_pair_high(self):
        # Two suits per rank; card index // 2 is the rank.
        trips = _hand_strength((0, 1, 2), suits=3)
        pair = _hand_strength((0, 1, 4), suits=2)
        high = _hand_strength((0, 2, 4), suits=2)
        self.assertGreater(trips, pair)
        self.assertGreater(pair, high)
        self.assertGreater(_hand_strength((0, 1, 2), 3), _hand_strength((0, 3, 6), 3))

    def test_four_card_strength_orders_quads_full_house_two_pair(self):
        quads = _hand_strength((0, 1, 2, 3), suits=4)
        full_house = _hand_strength((0, 1, 2, 4), suits=4)
        two_pair = _hand_strength((0, 1, 4, 5), suits=4)
        pair = _hand_strength((0, 1, 4, 8), suits=4)
        self.assertGreater(quads, full_house)
        self.assertGreater(full_house, two_pair)
        self.assertGreater(two_pair, pair)

    def test_rejects_impossible_configs_and_malformed_states(self):
        for kwargs in ({"players": 1}, {"ranks": 0}, {"suits": 5},
                       {"ranks": 2, "players": 3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GameConfig(**kwargs)
        cfg = GameConfig(ranks=4, suits=2, cards_dealt=2, players=2)
        with self.assertRaises(ValueError):
            cfg.card_name(8)
        with self.assertRaises(ValueError):
            PokerState(cfg, ((0, 1), (1, 2)))
        with self.assertRaises(ValueError):
            PokerState(cfg, ((0,), (1,)))

    def test_late_bet_allows_every_seat_to_respond(self):
        cfg = GameConfig(ranks=4, players=3)
        state = PokerState(cfg, ((0,), (1,), (2,)), "ppbpp")
        self.assertTrue(state.is_terminal())
        with self.assertRaises(ValueError):
            PokerState(cfg, ((0,), (1,), (2,)), "ppbppp")


if __name__ == "__main__":
    unittest.main()
