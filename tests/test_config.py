import unittest

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


if __name__ == "__main__":
    unittest.main()
