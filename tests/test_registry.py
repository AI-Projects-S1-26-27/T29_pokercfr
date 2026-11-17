"""The pre-registered design must be internally consistent before it is frozen."""
import unittest

from pokercfr.experiments import registry as R


class TestRegistry(unittest.TestCase):
    def test_no_two_distinct_cells_share_a_hash_or_filename(self):
        seen_hash, seen_path = {}, {}
        for build in R.PER_PERSON.values():
            for spec in build():
                h = spec.spec_hash()
                self.assertEqual(seen_hash.setdefault(h, spec), spec)
                self.assertEqual(seen_path.setdefault(spec.result_path, h), h)

    def test_full_matrix_is_the_union_of_the_per_person_matrices(self):
        union = {s.spec_hash() for b in R.PER_PERSON.values() for s in b()}
        self.assertEqual({s.spec_hash() for s in R.full_matrix()}, union)

    def test_h1_cells_are_wallclock_budgeted_across_caps_and_seeds(self):
        cells = [s for s in R.leduc_sweep() if s.game.startswith("leduc-2p-cap")
                 and s.extra.get("time_budget") == R.H1_TIME_BUDGET]
        self.assertEqual({s.game for s in cells}, {f"leduc-2p-cap{c}" for c in R.H1_CAPS})
        self.assertEqual({s.seed for s in cells}, set(R.H1_SEEDS))
        self.assertEqual(len(cells), len(R.H1_CAPS) * 8 * len(R.H1_SEEDS))

    def test_every_leduc_rung_name_resolves(self):
        from pokercfr.experiments.runner import build_game
        for s in R.leduc_sweep():
            self.assertTrue(build_game(s.game).num_players >= 2)

    def test_multiplayer_epsilon_is_distinct_from_two_player(self):
        self.assertEqual(R.epsilon_for("kuhn"), R.EPSILON["kuhn"])
        self.assertEqual(R.epsilon_for("leduc-2p-cap2"), R.EPSILON["leduc-2p"])
        self.assertEqual(R.epsilon_for("leduc-3p-cap1"), R.EPSILON["multiplayer"])
        self.assertEqual(R.epsilon_for("4rank-2suit-4p"), R.EPSILON["multiplayer"])


if __name__ == "__main__":
    unittest.main()
