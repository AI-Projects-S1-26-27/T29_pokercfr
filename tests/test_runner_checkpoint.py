"""Checkpoint and resume. These runs are 10+ hours, so this is load-bearing.

The failure mode these guard against is silent: an unpicklable solver table, or
a resume that restarts from scratch, both look like "the run is still going" and
only reveal themselves as a lost night.
"""
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

from pokercfr.core.config import RunSpec
from pokercfr.experiments import runner
from pokercfr.games.kuhn_family import game
from pokercfr.solvers.cfr import CFR_PLUS, CFRSolver


class TestSolverIsPicklable(unittest.TestCase):
    def test_solver_round_trips_through_pickle(self):
        """`defaultdict(lambda: ...)` is the natural spelling of the regret tables
        and is NOT picklable, which would break every checkpoint. Guard the
        module-level factory that fixes it."""
        solver = CFRSolver(game("kuhn"), CFR_PLUS).train(5)
        revived = pickle.loads(pickle.dumps(solver))
        self.assertEqual(revived.iterations, solver.iterations)
        self.assertEqual(revived.average_strategy(), solver.average_strategy())

    def test_revived_solver_keeps_training(self):
        solver = CFRSolver(game("kuhn"), CFR_PLUS).train(5)
        revived = pickle.loads(pickle.dumps(solver))
        revived.train(5)
        solver.train(5)
        self.assertEqual(revived.average_strategy(), solver.average_strategy())


class TestResume(unittest.TestCase):
    """Redirects the runner at a temp directory: these tests call run_one, which
    writes result files, and must not touch the committed results/ tree."""

    spec = RunSpec(game="4rank", variant="cfr+", seed=0, iterations=50,
                   checkpoints=(10, 20, 50))

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._root, self._ckpt = runner.ROOT, runner.CKPT_DIR
        runner.ROOT = self._tmp
        runner.CKPT_DIR = self._tmp / "results" / ".ckpt"

    def tearDown(self):
        runner.ROOT, runner.CKPT_DIR = self._root, self._ckpt
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_resume_reproduces_the_uninterrupted_run_exactly(self):
        baseline = runner.run_one(self.spec)

        # Simulate a process killed at iteration 20.
        _, solver = runner.build_solver(self.spec)
        solver.train(20)
        kept = [c for c in baseline["curve"] if c["iterations"] <= 20]
        runner._save_checkpoint(self.spec, solver, kept, 1.0)

        resumed = runner.run_one(self.spec)
        self.assertEqual([c["iterations"] for c in resumed["curve"]],
                         [c["iterations"] for c in baseline["curve"]])
        self.assertAlmostEqual(resumed["curve"][-1]["exploitability"],
                               baseline["curve"][-1]["exploitability"], places=15)

    def test_checkpoint_is_cleaned_up_once_the_run_finishes(self):
        runner.run_one(self.spec)
        self.assertFalse(runner._ckpt_path(self.spec).exists())

    def test_completed_run_is_marked_complete(self):
        self.assertTrue(runner.run_one(self.spec)["complete"])

    def test_run_does_not_write_outside_its_root(self):
        runner.run_one(self.spec)
        self.assertTrue((self._tmp / "results").exists())


if __name__ == "__main__":
    unittest.main()
