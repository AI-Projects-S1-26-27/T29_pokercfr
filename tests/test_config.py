import os
import subprocess
import sys
import unittest
from dataclasses import replace

from pokercfr.core.config import RunSpec
from pokercfr.experiments import registry


class TestResultPathIsFlat(unittest.TestCase):
    def test_no_result_path_nests_into_directories(self):
        """Five of the eight cube labels contain slashes. Unsanitised they nest
        into directories and vanish from a results/*.json glob, which would drop
        those corners from the analysis without any error."""
        for name, build in registry.PER_PERSON.items():
            for spec in build():
                tail = spec.result_path[len("results/"):]
                self.assertNotIn("/", tail, f"{name}: {spec.variant}")

    def test_result_paths_are_unique_per_spec(self):
        specs = registry.full_matrix()
        paths = [s.result_path for s in specs]
        self.assertEqual(len(paths), len(set(paths)))

    def test_sanitising_the_path_does_not_change_the_hash(self):
        spec = RunSpec(game="kuhn", variant="rm+/uniform/alt", seed=0)
        self.assertNotIn("/", spec.result_path[len("results/"):])
        self.assertIn(spec.spec_hash(), spec.result_path)


class TestRunSpecIdentity(unittest.TestCase):
    def test_spec_hash_is_pinned(self):
        """Changing this deliberately changes filenames of committed results."""
        self.assertEqual(RunSpec("kuhn", "cfr").spec_hash(), "f622906c2dd7045a")

    def test_hash_is_stable_across_processes(self):
        spec = RunSpec("kuhn", "cfr", extra={"z": [1, 2], "a": {"x": True}})
        code = ("from pokercfr.core.config import RunSpec; "
                "print(RunSpec('kuhn','cfr',extra={'z':[1,2],'a':{'x':True}}).spec_hash())")
        env = {**os.environ, "PYTHONHASHSEED": "random"}
        hashes = [subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip()
                  for _ in range(2)]
        self.assertEqual(hashes, [spec.spec_hash()] * 2)

    def test_dict_insertion_order_does_not_change_hash(self):
        left = RunSpec("kuhn", "cfr", extra={"a": 1, "b": {"x": 2, "y": 3}})
        right = RunSpec("kuhn", "cfr", extra={"b": {"y": 3, "x": 2}, "a": 1})
        self.assertEqual(left.spec_hash(), right.spec_hash())

    def test_every_field_changes_hash(self):
        base = RunSpec("kuhn", "cfr")
        changed = [
            replace(base, game="4rank"), replace(base, variant="cfr+"),
            replace(base, seed=1), replace(base, iterations=2),
            replace(base, max_seconds=1.0), replace(base, checkpoints=(1,)),
            replace(base, extra={"x": 1}),
        ]
        self.assertEqual(len({base.spec_hash(), *(s.spec_hash() for s in changed)}), 8)

    def test_integral_numbers_share_one_identity(self):
        integer = RunSpec("leduc", "cfr", extra={"time_budget": 3600})
        floating = RunSpec("leduc", "cfr", extra={"time_budget": 3600.0})
        self.assertEqual(integer.spec_hash(), floating.spec_hash())

    def test_invalid_specs_fail_at_construction(self):
        invalid = (
            {"seed": "0"},
            {"iterations": 10, "checkpoints": (5, 1)},
            {"iterations": 10, "checkpoints": (5, 11)},
            {"extra": {"x": float("nan")}},
        )
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                RunSpec("kuhn", "cfr", **fields)

    def test_full_traversal_cfr_ignores_seed(self):
        from pokercfr.experiments.runner import build_solver

        _, a = build_solver(RunSpec("kuhn", "cfr", seed=0))
        _, b = build_solver(RunSpec("kuhn", "cfr", seed=9182))
        a.train(100)
        b.train(100)
        self.assertEqual(a.regret_sum, b.regret_sum)
        self.assertEqual(a.strategy_sum, b.strategy_sum)


if __name__ == "__main__":
    unittest.main()
