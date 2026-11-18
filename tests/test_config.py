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

    def test_int_and_float_budgets_hash_differently(self):
        """Documents a deliberate limitation, not an aspiration.

        Normalising integral floats to ints before hashing would be more robust
        -- time_budget 3600 and 3600.0 would then be one cell rather than two --
        but it changes every hash, and the hash is the result filename. By the
        time this was proposed ~15 hours of runs had landed, including all of H1
        and H3, so applying it would have orphaned 260 committed results. Pass a
        budget consistently (the registry always uses floats) and this cannot
        bite. Revisit only if the results tree is ever reset.
        """
        integer = RunSpec("leduc", "cfr", extra={"time_budget": 3600})
        floating = RunSpec("leduc", "cfr", extra={"time_budget": 3600.0})
        self.assertNotEqual(integer.spec_hash(), floating.spec_hash())

    def test_committed_results_still_rehash_to_their_filenames(self):
        """The guard that would have caught the orphaning before it happened."""
        import glob
        import json
        from pathlib import Path

        files = glob.glob("results/*.json")
        if not files:
            self.skipTest("no committed results")
        for path in files:
            blob = json.loads(Path(path).read_text())["spec"]
            spec = RunSpec(game=blob["game"], variant=blob["variant"], seed=blob["seed"],
                           iterations=blob["iterations"], max_seconds=blob.get("max_seconds"),
                           checkpoints=tuple(blob.get("checkpoints", ())),
                           extra=blob.get("extra", {}))
            self.assertEqual(spec.spec_hash(), blob["spec_hash"], path)
            self.assertEqual(Path(spec.result_path).name, Path(path).name, path)

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
