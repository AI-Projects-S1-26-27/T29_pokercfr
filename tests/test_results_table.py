"""Results aggregation: hash integrity, partial marking, coverage."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec_ = importlib.util.spec_from_file_location("a6", ROOT / "scripts" / "a6_results_table.py")
a6 = importlib.util.module_from_spec(spec_)
sys.modules["a6"] = a6
spec_.loader.exec_module(a6)

from pokercfr.core.config import RunSpec  # noqa: E402
from pokercfr.eval.exploitability import RECORD_SCHEMA_VERSION  # noqa: E402


def write(dirpath: Path, spec: RunSpec, complete=True, tamper=False) -> Path:
    gains = [0.1, 0.1]
    rec = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "spec": {**spec.__dict__, "checkpoints": list(spec.checkpoints),
                 "spec_hash": spec.spec_hash()},
        "game": {"name": spec.game, "players": 2, "info_sets": 12},
        "curve": [{"iterations": 10, "train_seconds": 1.0, "eval_seconds": 0.1,
                   "nashconv_total": 0.2, "nashconv_mean": 0.1, "nashconv_max": 0.1,
                   "nashconv_max_seat": 0, "exploitability": 0.1, "gains": gains}],
        "time_budget": None, "iterations_reached": 10, "complete": complete, "env": {},
    }
    path = dirpath / Path(spec.result_path).name
    if tamper:
        rec["spec"]["seed"] = 99          # edited by hand; hash no longer matches
    path.write_text(json.dumps(rec))
    return path


class TestResultsTable(unittest.TestCase):
    def test_valid_results_load_and_partial_is_marked(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write(d, RunSpec("kuhn", "cfr", iterations=10, checkpoints=(10,)))
            write(d, RunSpec("kuhn", "cfr+", iterations=10), complete=False)
            good, errors = a6.load_results(d)
            self.assertEqual(errors, [])
            rows = [r for p, rec, s in good for r in a6.rows_of(p, rec, s, lambda g: 1e-4)]
            self.assertEqual({r["status"] for r in rows}, {"complete", "PARTIAL"})

    def test_hand_edited_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write(d, RunSpec("kuhn", "cfr", iterations=10), tamper=True)
            good, errors = a6.load_results(d)
            self.assertEqual(good, [])
            self.assertTrue(any("edited by hand" in e for e in errors))

    def test_renamed_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            p = write(d, RunSpec("kuhn", "cfr", iterations=10))
            p.rename(d / "kuhn__cfr__s0__deadbeefdeadbeef.json")
            self.assertTrue(a6.load_results(d)[1])

    def test_coverage_counts_complete_partial_missing_and_unregistered(self):
        prereg = {"cells": [
            {"spec_hash": "a", "matrices": ["m"], "game": "g", "variant": "v", "seed": 0},
            {"spec_hash": "b", "matrices": ["m"], "game": "g", "variant": "v", "seed": 1},
            {"spec_hash": "c", "matrices": ["m"], "game": "g", "variant": "v", "seed": 2}]}
        cov = a6.coverage(prereg, {"a": True, "b": False, "zzz": True})
        self.assertEqual(dict(cov["totals"]), {"complete": 1, "partial": 1, "missing": 1})
        self.assertEqual(cov["unregistered"], ["zzz"])


if __name__ == "__main__":
    unittest.main()
