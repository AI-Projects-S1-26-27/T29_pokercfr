#!/usr/bin/env python3
"""Freeze the experimental design (Aarush, step 4).

Turns `pokercfr/experiments/registry.py` into a committed, timestamped record,
BEFORE the final sweeps run. Fixing epsilon targets, budgets, seeds, raise caps
and decision rules up front is what separates a genuine prediction from "CFR+ was
better at the settings we chose after seeing the results".

Writes:
  results/verify/preregistration.json   every cell: game, variant, seed, budget,
                                        epsilon, spec hash; plus the decision rules
  docs/report/preregistration.md        the human-readable version

Fails (exit 1) if two distinct cells collide on spec hash or on result filename,
since the hash IS the result filename and a collision silently overwrites a run.

    python scripts/a4_preregister.py [--out results/verify/preregistration.json]
                                     [--md docs/report/preregistration.md]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pokercfr.experiments import registry as R  # noqa: E402

HYPOTHESES = {
    "H1": {
        "role": "primary",
        "claim": ("At equal wall-clock training budget, CFR+ reaches lower exploitability "
                  "than vanilla CFR on 2-player Leduc."),
        "tested_on": "leduc-2p-cap{1,2,3}; cfr vs cfr+ (all 8 corners also run, exploratory)",
        "measurement": ("exploitability at 100% of the 600 s training budget "
                        "(evaluation time excluded), paired by seed across 5 replicates"),
        "supported_if": ("at a given cap, CFR+ has lower exploitability than vanilla in at "
                         "least 4 of the 5 seed pairs AND the median ratio "
                         "expl(cfr+)/expl(cfr) is below 0.8"),
        "fails_if": ("at a given cap, CFR+ is lower in 3 or fewer of the 5 pairs, OR the "
                     "median ratio is 0.8 or above (a win smaller than 20% does not count)"),
        "overall_rule": ("Reported per cap and never averaged. Overall 'supported' needs all "
                         "three caps; 'partially supported' is some caps; 'refuted' is none. "
                         "A cap where H1 fails is reported as a failure, not dropped."),
    },
    "H2": {
        "role": "primary",
        "claim": ("Cost to reach a fixed epsilon grows polynomially with information-set "
                  "count on the 2-player ladder, and CFR+ has a smaller exponent than "
                  "vanilla CFR."),
        "tested_on": "kuhn, 4rank, 4rank-2suit, 4rank-2suit-2dealt; cfr vs cfr+",
        "measurement": ("slope of log10(iterations to reach epsilon=1e-2) ~ log10(info sets), "
                        "per algorithm, with a bootstrap 95% CI (Akhil's stats module); "
                        "wall-clock cost vs deals reported as the secondary fit"),
        "supported_if": ("the 95% CI of (slope_vanilla - slope_cfr+) excludes 0 with "
                         "slope_cfr+ the smaller"),
        "fails_if": ("that CI includes 0, or slope_cfr+ is the larger. With only four rungs "
                     "the CI will be wide: an inconclusive result is reported as "
                     "inconclusive, not as support."),
        "overall_rule": ("A rung that never reaches epsilon within budget is censored and "
                         "reported as such, not dropped."),
    },
    "H3": {
        "role": "primary",
        "claim": ("With the deck held fixed, CFR's per-seat NashConv at a fixed budget rises "
                  "with player count, i.e. the multiplayer gap is a property of N and not "
                  "of the number of cards."),
        "tested_on": ("Leduc cap 1, N = 2,3,4 (5 is a stretch cell) and the fixed-deck Kuhn "
                      "family 4rank-2suit, -3p, -4p; cfr+ is the primary variant"),
        "measurement": ("MEAN NashConv per seat (total / N) at the final checkpoint. The total "
                        "is never used to compare across N, because it grows with N by "
                        "construction. For N=2 the mean equals exploitability."),
        "supported_if": ("within each family, per-seat mean NashConv increases at every step "
                         "in N by at least 10% relative, AND stays above the multiplayer "
                         "epsilon of 5e-2 at the largest N reached"),
        "fails_if": ("mean NashConv is non-monotone in N within a family, OR falls below 5e-2 "
                     "at N >= 3 (CFR then converges in practice and there is no gap), OR the "
                     "step is under 10%"),
        "overall_rule": ("Per family. Seat-level gains and the max-deviator seat are reported "
                         "as an EXPLORATORY finding (does position decide who gains most "
                         "from deviating); no direction is predicted."),
    },
}

EXPLORATORY = [
    "the full 2^3 ablation cube (R1) beyond the CFR vs CFR+ contrast",
    "DCFR parameter grid and its unification with CFR / CFR+",
    "alternating vs simultaneous updates as the source of CFR+'s advantage",
    "MCCFR variance by sampling scheme (outcome vs external vs chance)",
    "alpha-manifold drift on Kuhn",
    "head-to-head cross-play (supporting evidence only; never a measure of equilibrium quality)",
]


def collect() -> tuple[dict[str, list], dict[str, dict]]:
    """Every spec by matrix, and every distinct spec by hash. Detects collisions."""
    by_matrix: dict[str, list] = {}
    by_hash: dict[str, object] = {}
    by_path: dict[str, str] = {}
    problems: list[str] = []
    members: dict[str, list[str]] = defaultdict(list)
    for name, build in R.PER_PERSON.items():
        specs = build()
        by_matrix[name] = specs
        for sp in specs:
            h = sp.spec_hash()
            if h in by_hash and by_hash[h] != sp:
                problems.append(f"HASH COLLISION {h}: {by_hash[h]} vs {sp}")
            by_hash.setdefault(h, sp)
            if sp.result_path in by_path and by_path[sp.result_path] != h:
                problems.append(f"PATH COLLISION {sp.result_path}")
            by_path.setdefault(sp.result_path, h)
            if name not in members[h]:
                members[h].append(name)
    if problems:
        raise SystemExit("\n".join(problems))
    return by_matrix, {h: {"spec": sp, "matrices": members[h]} for h, sp in by_hash.items()}


def cell_record(spec, matrices: list[str]) -> dict:
    budget = spec.extra.get("time_budget")
    return {
        "game": spec.game, "variant": spec.variant, "seed": spec.seed,
        "budget_type": "wall_clock_seconds" if budget else "iterations",
        "budget": budget if budget else spec.iterations,
        "iteration_ceiling": spec.iterations,
        "epsilon": R.epsilon_for(spec.game),
        "spec_hash": spec.spec_hash(), "result_file": spec.result_path,
        "matrices": matrices,
    }


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def build(by_matrix, distinct) -> dict:
    full_order = [s.spec_hash() for s in R.full_matrix()]
    assert set(full_order) == set(distinct), "full_matrix() disagrees with the per-matrix union"
    cells = [cell_record(distinct[h]["spec"], distinct[h]["matrices"]) for h in full_order]
    digest = hashlib.sha256("".join(sorted(full_order)).encode()).hexdigest()
    return {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_head_at_freeze": git_head(),
        "design_digest": digest,
        "amendments": getattr(R, "AMENDMENTS", []),
        "n_cells": len(cells),
        "n_collisions": 0,
        "epsilon": R.EPSILON,
        "constants": {
            "h1_caps": list(R.H1_CAPS), "h1_seeds": list(R.H1_SEEDS),
            "h1_time_budget_s": R.H1_TIME_BUDGET,
            "h3_leduc_cells_players_cap": [list(c) for c in R.H3_LEDUC_CELLS],
            "h3_leduc_stretch": [list(c) for c in R.H3_LEDUC_STRETCH],
            "h3_variants": list(R.H3_VARIANTS), "h3_seeds": list(R.H3_SEEDS),
            "h3_time_budget_s": R.H3_TIME_BUDGET,
            "h3_stretch_time_budget_s": R.H3_STRETCH_TIME_BUDGET,
            "sampled_seeds": list(R.SAMPLED_SEEDS),
            "max_hours_per_run": R.MAX_HOURS_PER_RUN,
            "iteration_budgets": R.BUDGET,
        },
        "hypotheses": HYPOTHESES,
        "exploratory": EXPLORATORY,
        "cells": cells,
    }


def render_md(prereg: dict, by_matrix: dict[str, list]) -> str:
    c = prereg["constants"]
    L = []
    w = L.append
    w("# Pre-registration\n")
    w(f"Frozen {prereg['generated_utc']} at commit `{prereg['git_head_at_freeze']}`. "
      f"Design digest `{prereg['design_digest'][:16]}`. "
      "Generated by `scripts/a4_preregister.py` from `pokercfr/experiments/registry.py`; "
      "the machine-readable cell list is `results/verify/preregistration.json`.\n")
    w("Everything below was fixed before the final sweeps were run. Anything not listed "
      "under a numbered hypothesis is exploratory and is reported as exploratory.\n")
    w("> Hypothesis numbering follows this document. `LADDER.md` carries an earlier "
      "numbering (H1 size / H2 players / H3 abstraction) that this supersedes.\n")

    w("## Hypotheses and decision rules\n")
    for name, h in prereg["hypotheses"].items():
        w(f"### {name} ({h['role']})\n")
        w(f"**Claim.** {h['claim']}\n")
        w(f"**Tested on.** {h['tested_on']}\n")
        w(f"**Measurement.** {h['measurement']}\n")
        w(f"**Supported if.** {h['supported_if']}\n")
        w(f"**Fails if.** {h['fails_if']}\n")
        w(f"**Reporting.** {h['overall_rule']}\n")

    w("## Epsilon targets (chips/hand)\n")
    w("| Class | epsilon | Applied to |\n|---|---|---|")
    w(f"| kuhn | {R.EPSILON['kuhn']:g} | exploitability, `kuhn` |")
    w(f"| default | {R.EPSILON['default']:g} | exploitability, other 2-player Kuhn-family rungs |")
    w(f"| leduc-2p | {R.EPSILON['leduc-2p']:g} | exploitability, 2-player Leduc |")
    w(f"| multiplayer | {R.EPSILON['multiplayer']:g} | per-seat MEAN NashConv, any N >= 3 |\n")

    w("## Seeds\n")
    w("- Deterministic full-traversal solvers ignore their seed, so they run once (seed 0) "
      "except where wall-clock noise matters.")
    w(f"- H1: seeds {list(c['h1_seeds'])} are independent timing replicates "
      f"({len(c['h1_seeds'])} per cell); the CFR vs CFR+ comparison is paired by seed.")
    w(f"- H3 Leduc cells: seeds {list(c['h3_seeds'])}; the 5-player stretch cell has 1.")
    w(f"- Sampled (MCCFR) algorithms: seeds {list(c['sampled_seeds'])}, "
      f"{len(c['sampled_seeds'])} per cell.\n")

    for i, a in enumerate(prereg.get("amendments", []), 1):
        if i == 1:
            w("## Amendments after registration\n")
            w("Changes made after the design was first registered, recorded so this "
              "document shows what changed rather than reading as though it was "
              "always the plan.\n")
        w(f"**{i}. {a['date']} — {a['change']}**\n")
        w(f"*Why:* {a['reason']}\n")
        w(f"*Effect:* {a['effect']}\n")

    w("## Raise caps and budgets\n")
    w(f"- H1: raise caps {list(c['h1_caps'])} on 2-player Leduc, "
      f"{c['h1_time_budget_s']:g} s of training per cell, all 8 ablation corners.")
    cells = ", ".join(f"{p}p cap{k}" for p, k in c["h3_leduc_cells_players_cap"])
    w(f"- H3 Leduc: {cells}, variants {c['h3_variants']}, "
      f"{c['h3_time_budget_s']:g} s each. Stretch: "
      f"{', '.join(f'{p}p cap{k}' for p, k in c['h3_leduc_stretch'])} at "
      f"{c['h3_stretch_time_budget_s']:g} s, expected to finish only a few hundred "
      "iterations and reported as partial.")
    w("- **Cut for feasibility, from the measured census** "
      "(`results/verify/leduc_sizes.json`): Leduc at 4 players with cap 2 and above, 5 "
      "players with cap 2 and above, and 6 players at any cap. They are not registered "
      "because they cannot finish, and that limit is itself reported.")
    w("- Iteration-budgeted cells use the per-rung iteration budgets in the JSON; every "
      f"run is capped at {c['max_hours_per_run']:g} h.\n")

    w("## Cells per matrix\n")
    w("| Matrix | Cells | Distinct games |\n|---|---:|---|")
    for name, specs in by_matrix.items():
        games = sorted({s.game for s in specs})
        w(f"| `{name}` | {len(specs)} | {', '.join(games)} |")
    kinds = Counter(x["budget_type"] for x in prereg["cells"])
    w(f"\nTotal distinct cells after de-duplication: **{prereg['n_cells']}** "
      f"({kinds['iterations']} iteration-budgeted, {kinds['wall_clock_seconds']} wall-clock-budgeted). "
      f"Spec-hash and result-filename collisions: **{prereg['n_collisions']}**.\n")

    w("## Exploratory (no prediction made)\n")
    for item in prereg["exploratory"]:
        w(f"- {item}")
    w("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "results" / "verify" / "preregistration.json"))
    ap.add_argument("--md", default=str(ROOT / "docs" / "report" / "preregistration.md"))
    args = ap.parse_args()

    by_matrix, distinct = collect()
    prereg = build(by_matrix, distinct)

    out, md = Path(args.out), Path(args.md)
    out.parent.mkdir(parents=True, exist_ok=True)
    md.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prereg, indent=2))
    md.write_text(render_md(prereg, by_matrix))

    kinds = Counter(c["budget_type"] for c in prereg["cells"])
    print(f"{sum(len(v) for v in by_matrix.values())} cells across {len(by_matrix)} matrices "
          f"-> {prereg['n_cells']} distinct ({kinds['iterations']} iteration-budgeted, "
          f"{kinds['wall_clock_seconds']} wall-clock-budgeted)")
    print("no spec-hash or result-filename collisions")
    print(f"design digest {prereg['design_digest'][:16]}  head {prereg['git_head_at_freeze'][:10]}")
    print(f"wrote {out}\nwrote {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
