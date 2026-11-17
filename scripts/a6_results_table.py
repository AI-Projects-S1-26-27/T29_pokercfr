#!/usr/bin/env python3
"""Aggregate every result file into one tidy table + a coverage summary (Aarush, step 6).

Walks results/*.json, validates each record against the frozen schema, verifies
that its spec hash matches both its own spec and its filename, and writes one row
per measured checkpoint. Incomplete runs are INCLUDED but marked, so analysis can
start before the longest runs finish without anyone mistaking a partial curve for
a finished one.

The coverage summary compares what landed against the frozen pre-registration
(results/verify/preregistration.json): which cells are complete, which are
partial, which are missing, and which result files belong to no registered cell.

A hash mismatch means a result file was edited by hand or renamed. That is fatal
(exit 2): nothing downstream should run on a result it cannot trace.

    python scripts/a6_results_table.py [--results-dir results] [--out CSV]
                                       [--prereg JSON] [--list-missing]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pokercfr.core.config import RunSpec  # noqa: E402
from pokercfr.eval.exploitability import validate_record  # noqa: E402

COLUMNS = ["game", "variant", "seed", "spec_hash", "budget_type", "budget", "players",
           "info_sets", "point", "is_last", "iterations", "train_seconds", "eval_seconds",
           "time_fraction", "nashconv_total", "nashconv_mean", "nashconv_max",
           "nashconv_max_seat", "exploitability", "epsilon", "complete", "status", "file"]


class IntegrityError(Exception):
    pass


def spec_of(record: dict) -> RunSpec:
    s = {k: v for k, v in record["spec"].items() if k != "spec_hash"}
    s["checkpoints"] = tuple(s.get("checkpoints", ()))
    return RunSpec(**s)


def check_integrity(path: Path, record: dict) -> RunSpec:
    """The recorded hash must equal the recomputed one AND the filename's."""
    spec = spec_of(record)
    recomputed = spec.spec_hash()
    claimed = record["spec"].get("spec_hash")
    in_name = path.stem.rsplit("__", 1)[-1]
    if not (recomputed == claimed == in_name):
        raise IntegrityError(f"{path.name}: recomputed {recomputed}, record says {claimed}, "
                             f"filename says {in_name} -- edited by hand or renamed")
    return spec


def load_results(results_dir: Path) -> tuple[list[tuple[Path, dict, RunSpec]], list[str]]:
    good, errors = [], []
    for path in sorted(results_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}: unreadable JSON ({exc})")
            continue
        problems = validate_record(record)
        if problems:
            errors.append(f"{path.name}: schema: {'; '.join(problems[:3])}")
            continue
        try:
            spec = check_integrity(path, record)
        except IntegrityError as exc:
            errors.append(str(exc))
            continue
        good.append((path, record, spec))
    return good, errors


def rows_of(path: Path, record: dict, spec: RunSpec, epsilon_for) -> list[dict]:
    budget = spec.extra.get("time_budget")
    complete = bool(record["complete"])
    curve = record["curve"]
    rows = []
    for i, p in enumerate(curve):
        rows.append({
            "game": spec.game, "variant": spec.variant, "seed": spec.seed,
            "spec_hash": spec.spec_hash(),
            "budget_type": "wall_clock_seconds" if budget else "iterations",
            "budget": budget if budget else spec.iterations,
            "players": record["game"]["players"], "info_sets": record["game"]["info_sets"],
            "point": i, "is_last": i == len(curve) - 1,
            "iterations": p["iterations"], "train_seconds": p["train_seconds"],
            "eval_seconds": p["eval_seconds"], "time_fraction": p.get("time_fraction", ""),
            "nashconv_total": p["nashconv_total"], "nashconv_mean": p["nashconv_mean"],
            "nashconv_max": p["nashconv_max"], "nashconv_max_seat": p["nashconv_max_seat"],
            "exploitability": "" if p["exploitability"] is None else p["exploitability"],
            "epsilon": epsilon_for(spec.game), "complete": complete,
            "status": "complete" if complete else "PARTIAL", "file": path.name,
        })
    return rows


def coverage(prereg: dict, landed: dict[str, bool]) -> dict:
    """landed maps spec_hash -> complete flag, for every result file present."""
    per_matrix: dict[str, Counter] = defaultdict(Counter)
    missing: list[dict] = []
    for cell in prereg["cells"]:
        h = cell["spec_hash"]
        state = "missing" if h not in landed else ("complete" if landed[h] else "partial")
        if state == "missing":
            missing.append(cell)
        for m in cell["matrices"]:
            per_matrix[m][state] += 1
    registered = {c["spec_hash"] for c in prereg["cells"]}
    return {"per_matrix": per_matrix, "missing": missing,
            "unregistered": sorted(set(landed) - registered),
            "totals": Counter("missing" if c["spec_hash"] not in landed else
                              "complete" if landed[c["spec_hash"]] else "partial"
                              for c in prereg["cells"])}


def print_coverage(cov: dict, n_cells: int, list_missing: bool) -> None:
    print(f"\ncoverage against the pre-registration ({n_cells} cells)")
    print(f"  {'matrix':<20}{'complete':>9}{'partial':>9}{'missing':>9}")
    for name in sorted(cov["per_matrix"]):
        c = cov["per_matrix"][name]
        print(f"  {name:<20}{c['complete']:>9}{c['partial']:>9}{c['missing']:>9}")
    t = cov["totals"]
    print(f"  {'ALL DISTINCT CELLS':<20}{t['complete']:>9}{t['partial']:>9}{t['missing']:>9}")
    if cov["unregistered"]:
        print(f"  {len(cov['unregistered'])} result file(s) match no registered cell: "
              f"{', '.join(cov['unregistered'][:5])}{' ...' if len(cov['unregistered']) > 5 else ''}")
    if list_missing:
        print("\nmissing cells:")
        for c in cov["missing"]:
            print(f"  {c['game']:<26}{c['variant']:<20}seed={c['seed']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "results" / "verify" / "results_table.csv"))
    ap.add_argument("--prereg", default=str(ROOT / "results" / "verify" / "preregistration.json"))
    ap.add_argument("--list-missing", action="store_true")
    args = ap.parse_args()

    from pokercfr.experiments.registry import epsilon_for

    good, errors = load_results(Path(args.results_dir))
    integrity = [e for e in errors if "edited by hand" in e]
    rows = [r for path, rec, spec in good for r in rows_of(path, rec, spec, epsilon_for)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    partial = sum(1 for _, rec, _ in good if not rec["complete"])
    print(f"{len(good)} result files ({len(good) - partial} complete, {partial} partial), "
          f"{len(rows)} rows -> {out}")

    prereg_path = Path(args.prereg)
    if prereg_path.exists():
        prereg = json.loads(prereg_path.read_text())
        landed = {spec.spec_hash(): bool(rec["complete"]) for _, rec, spec in good}
        print_coverage(coverage(prereg, landed), len(prereg["cells"]), args.list_missing)
    else:
        print(f"no pre-registration at {prereg_path}; coverage skipped")

    for e in errors:
        print(f"ERROR {e}", file=sys.stderr)
    if integrity:
        print(f"\n{len(integrity)} result file(s) FAILED the spec-hash check", file=sys.stderr)
        return 2
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
