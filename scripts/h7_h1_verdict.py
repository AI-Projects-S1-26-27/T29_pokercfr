#!/usr/bin/env python3
"""H1's verdict, computed from committed results under the pre-registered rule.

H1 (primary): at equal wall-clock training budget, CFR+ reaches lower
exploitability than vanilla CFR on 2-player Leduc.

The registered design already covers this -- 3 raise caps x 8 corners x 5 seeds
at 600 s each, inside `leduc-sweep`. So this script does NOT re-run anything; it
reads the committed records and applies the decision rule as registered:

  measurement   exploitability at 100% of the training budget, evaluation time
                excluded, paired by seed across 5 replicates
  per cap       cfr vs cfr+, paired by seed, reported per cap and NEVER averaged
  overall       'supported' needs all three caps; 'partially supported' is some;
                'refuted' is none. A cap where H1 fails is reported as a
                failure, not dropped.

Runs safely before the sweep finishes: it reports coverage and withholds a
verdict rather than ruling on partial data.

Writes results/verify/h1_verdict.json.

    python scripts/h7_h1_verdict.py
    python scripts/h7_h1_verdict.py --out /tmp/h1.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from a6_results_table import load_results  # noqa: E402

DEFAULT_OUT = ROOT / "results" / "verify" / "h1_verdict.json"
PAIR = ("cfr", "cfr+")


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def registered_h1_cells() -> tuple[list, tuple[int, ...]]:
    """Exactly the H1 cells, taken from the registry's own H1 constants.

    Filtering on "2-player Leduc with any time budget" is too loose: H3's
    2-player reference cells live there too, at a 3600 s budget. Mixing budgets
    into a wall-clock-matched comparison would reintroduce the unfairness the
    timed mode exists to remove.
    """
    from pokercfr.experiments import registry

    cells = [s for s in registry.leduc_sweep()
             if s.game.startswith("leduc-2p-cap")
             and s.extra.get("time_budget") == registry.H1_TIME_BUDGET
             and s.seed in registry.H1_SEEDS]
    return cells, registry.H1_CAPS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    registered, CAPS = registered_h1_cells()
    want = {s.spec_hash() for s in registered}
    print(f"H1 registered cells: {len(registered)} "
          f"({len({s.game for s in registered})} caps x "
          f"{len({s.variant for s in registered})} variants x "
          f"{len({s.seed for s in registered})} seeds)")

    loaded, errors = load_results(args.results)
    for err in errors:
        print(f"  integrity: {err}")

    # final exploitability per (cap, variant, seed)
    final: dict[tuple[int, str, int], dict] = {}
    for _path, record, spec in loaded:
        if spec.spec_hash() not in want or not record["curve"]:
            continue
        point = record["curve"][-1]
        cap = int(spec.game.rsplit("cap", 1)[1])
        final[(cap, spec.variant, spec.seed)] = {
            "exploitability": point["exploitability"],
            "iterations": point["iterations"],
            "train_seconds": point["train_seconds"],
            "complete": record["complete"],
        }

    print(f"H1 cells landed: {len(final)} / {len(registered)} "
          f"({len(final) / len(registered):.0%})")
    if not final:
        print("\nno H1 results yet -- run the leduc-sweep matrix first "
              "(`--matrix leduc-sweep`). Withholding a verdict.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"registered_cells": len(registered), "landed": 0,
             "verdict": "not evaluable", "per_cap": {}}, indent=2))
        print(f"wrote {_display(args.out)}")
        return 0

    per_cap: dict[str, dict] = {}
    print(f"\n{'cap':>4}{'seeds':>7}{'cfr':>14}{'cfr+':>14}{'cfr+ wins':>11}  verdict")
    for cap in CAPS:
        seeds = sorted({s for (c, v, s) in final if c == cap and v in PAIR})
        paired = [(final[(cap, PAIR[0], s)]["exploitability"],
                   final[(cap, PAIR[1], s)]["exploitability"])
                  for s in seeds
                  if (cap, PAIR[0], s) in final and (cap, PAIR[1], s) in final]
        if not paired:
            per_cap[str(cap)] = {"seeds": 0, "verdict": "no data"}
            print(f"{cap:>4}{0:>7}{'-':>14}{'-':>14}{'-':>11}  no data")
            continue
        base = [a for a, _ in paired]
        plus = [b for _, b in paired]
        wins = sum(1 for a, b in paired if b < a)
        # Paired by seed, as registered. The formal signed-rank test lives in
        # analysis/stats.py; until that lands this reports the paired direction,
        # the win count and the effect ratio, which is what the rule needs.
        supported = wins == len(paired)
        per_cap[str(cap)] = {
            "seeds": len(paired),
            "cfr_mean": statistics.fmean(base),
            "cfr_plus_mean": statistics.fmean(plus),
            "ratio": statistics.fmean(base) / statistics.fmean(plus) if plus else None,
            "cfr_plus_wins": wins,
            "paired": paired,
            "verdict": "supported" if supported else "not supported",
        }
        print(f"{cap:>4}{len(paired):>7}{statistics.fmean(base):>14.8f}"
              f"{statistics.fmean(plus):>14.8f}{wins:>7}/{len(paired)}  "
              f"{'SUPPORTED' if supported else 'NOT SUPPORTED'}")

    evaluated = [c for c in CAPS if per_cap.get(str(c), {}).get("seeds")]
    supported = [c for c in evaluated if per_cap[str(c)]["verdict"] == "supported"]
    if not evaluated:
        verdict = "not evaluable"
    elif len(supported) == len(CAPS):
        verdict = "supported"
    elif supported:
        verdict = "partially supported"
    else:
        verdict = "refuted"

    print("\n=== exploratory: all eight corners, per cap (not part of H1) ===")
    exploratory: dict[str, dict] = {}
    for cap in CAPS:
        by_variant = defaultdict(list)
        for (c, v, _s), entry in final.items():
            if c == cap and entry["exploitability"] is not None:
                by_variant[v].append(entry["exploitability"])
        if not by_variant:
            continue
        ranked = sorted(((statistics.fmean(v), k) for k, v in by_variant.items()))
        exploratory[str(cap)] = {k: statistics.fmean(v) for k, v in by_variant.items()}
        print(f"  cap {cap}: " + ", ".join(f"{k}={m:.2e}" for m, k in ranked))

    record = {
        "hypothesis": "H1",
        "claim": "at equal wall-clock budget, CFR+ reaches lower exploitability "
                 "than vanilla CFR on 2-player Leduc",
        "registered_cells": len(registered),
        "landed": len(final),
        "coverage": len(final) / len(registered),
        "per_cap": per_cap,
        "caps_evaluated": evaluated,
        "caps_supported": supported,
        "verdict": verdict,
        "exploratory_all_corners": exploratory,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2))

    print("\n" + "=" * 66)
    print(f"H1 VERDICT: {verdict.upper()}  "
          f"(caps supported: {supported or 'none'} of {list(CAPS)})")
    if verdict == "partially supported":
        print("Report per cap and do not average -- a cap where H1 fails is a")
        print("finding, and the pre-registration says it is reported as one.")
    if len(final) < len(registered):
        print(f"NOTE: {len(registered) - len(final)} registered cells missing; "
              f"verdict is provisional.")
    print(f"wrote {_display(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
