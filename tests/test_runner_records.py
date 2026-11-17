"""The runner must emit records matching the frozen schema in eval/exploitability.

The runner writes the files and Aarush's module defines their shape, so this is
the seam between two owners. A drift here is silent: runs complete, files land,
and the aggregation quietly skips or misreads them.
"""
import shutil
import tempfile
import unittest
from pathlib import Path

from pokercfr.core.config import RunSpec
from pokercfr.eval.exploitability import RECORD_SCHEMA_VERSION, validate_record
from pokercfr.experiments import runner


class TestRecordsMatchTheFrozenSchema(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._root, self._ckpt = runner.ROOT, runner.CKPT_DIR
        runner.ROOT = self._tmp
        runner.CKPT_DIR = self._tmp / "results" / ".ckpt"

    def tearDown(self):
        runner.ROOT, runner.CKPT_DIR = self._root, self._ckpt
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _run(self, spec):
        return runner.run_one(spec, progress=lambda *_: None)

    def test_two_player_iteration_run_validates(self):
        spec = RunSpec(game="4rank", variant="cfr+", seed=0, iterations=20,
                       checkpoints=(10, 20))
        self.assertEqual(validate_record(self._run(spec)), [])

    def test_three_player_run_validates_and_omits_exploitability(self):
        spec = RunSpec(game="kuhn-3p", variant="cfr", seed=0, iterations=5,
                       checkpoints=(5,))
        record = self._run(spec)
        self.assertEqual(validate_record(record), [])
        self.assertIsNone(record["curve"][-1]["exploitability"])
        self.assertEqual(len(record["curve"][-1]["gains"]), 3)

    def test_timed_run_validates_and_carries_time_fraction(self):
        spec = RunSpec(game="4rank", variant="cfr+", seed=0,
                       extra={"time_budget": 1.0, "time_checkpoints": (0.5, 1.0)})
        record = self._run(spec)
        self.assertEqual(validate_record(record), [])
        self.assertEqual([p["time_fraction"] for p in record["curve"]], [0.5, 1.0])
        self.assertIsNotNone(record["time_budget"])

    def test_partial_record_written_mid_run_also_validates(self):
        spec = RunSpec(game="4rank", variant="cfr+", seed=0, iterations=20,
                       checkpoints=(10, 20))
        self._run(spec)
        import json
        written = json.loads((self._tmp / spec.result_path).read_text())
        self.assertEqual(validate_record(written), [])

    def test_schema_version_is_stamped(self):
        spec = RunSpec(game="4rank", variant="cfr", seed=0, iterations=5,
                       checkpoints=(5,))
        self.assertEqual(self._run(spec)["schema_version"], RECORD_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
