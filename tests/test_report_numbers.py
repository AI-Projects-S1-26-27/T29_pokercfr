"""The pre-registered decision rules, as code. These must match the frozen document."""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location("a7", ROOT / "scripts" / "a7_report_numbers.py")
a7 = importlib.util.module_from_spec(s)
sys.modules["a7"] = a7
s.loader.exec_module(a7)


class TestH1Rule(unittest.TestCase):
    def test_supported_needs_four_wins_and_a_20_percent_ratio(self):
        self.assertEqual(a7.h1_verdict([(1.0, 0.5)] * 5)[0], "SUPPORTED")
        self.assertEqual(a7.h1_verdict([(1.0, 0.5)] * 4 + [(1.0, 2.0)])[0], "SUPPORTED")

    def test_three_wins_fails(self):
        self.assertEqual(a7.h1_verdict([(1.0, 0.5)] * 3 + [(1.0, 2.0)] * 2)[0], "FAILS")

    def test_a_win_smaller_than_20_percent_fails(self):
        self.assertEqual(a7.h1_verdict([(1.0, 0.9)] * 5)[0], "FAILS")

    def test_missing_pairs_are_incomplete_not_a_verdict(self):
        self.assertEqual(a7.h1_verdict([(1.0, 0.5)] * 4)[0], "INCOMPLETE")


class TestH3Rule(unittest.TestCase):
    def test_monotone_rise_above_epsilon_is_supported(self):
        self.assertEqual(a7.h3_verdict({2: 0.05, 3: 0.08, 4: 0.12})[0], "SUPPORTED")

    def test_step_under_ten_percent_fails(self):
        self.assertEqual(a7.h3_verdict({2: 0.10, 3: 0.105})[0], "FAILS")

    def test_non_monotone_fails(self):
        self.assertEqual(a7.h3_verdict({2: 0.1, 3: 0.2, 4: 0.15})[0], "FAILS")

    def test_converging_below_epsilon_fails(self):
        self.assertEqual(a7.h3_verdict({2: 0.001, 3: 0.01, 4: 0.03})[0], "FAILS")

    def test_single_player_count_is_incomplete(self):
        self.assertEqual(a7.h3_verdict({2: 0.1})[0], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
