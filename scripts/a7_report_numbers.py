#!/usr/bin/env python3
"""Generate every number the report cites into docs/report/numbers.md (Aarush, step 7).

Nothing in the report is typed by hand: each entry below names the file(s) in
results/ it was computed from. Hypothesis verdicts apply the DECISION RULES FROZEN
in results/verify/preregistration.json, not rules chosen after seeing the data.
Where data has not landed the entry says so, with the coverage, rather than
omitting the hypothesis.

    python scripts/a7_report_numbers.py [--results-dir results] [--out MD]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("a6", ROOT / "scripts" / "a6_results_table.py")
a6 = importlib.util.module_from_spec(_spec)
sys.modules["a6"] = a6
_spec.loader.exec_module(a6)

H1_MIN_WINS, H1_PAIRS, H1_RATIO = 4, 5, 0.8        # frozen in the pre-registration
H3_MIN_STEP, H3_EPS = 0.10, 5e-2


def h1_verdict(pairs: list[tuple[float, float]]) -> tuple[str, str]:
    """pairs = [(exploitability_cfr, exploitability_cfr+)] by seed. Frozen rule."""
    if len(pairs) < H1_PAIRS:
        return "INCOMPLETE", f"{len(pairs)} of {H1_PAIRS} seed pairs have landed"
    wins = sum(1 for v, p in pairs if p < v)
    median = statistics.median(p / v for v, p in pairs if v > 0)
    ok = wins >= H1_MIN_WINS and median < H1_RATIO
    return ("SUPPORTED" if ok else "FAILS"), f"CFR+ lower in {wins}/{len(pairs)}, median ratio {median:.3f}"


def h3_verdict(means_by_n: dict[int, float]) -> tuple[str, str]:
    """means_by_n = per-seat mean NashConv at a matched budget. Frozen rule."""
    ns = sorted(means_by_n)
    if len(ns) < 2:
        return "INCOMPLETE", "fewer than two player counts have landed"
    steps = [means_by_n[b] / means_by_n[a] - 1 for a, b in zip(ns, ns[1:], strict=False)]
    ok = all(s >= H3_MIN_STEP for s in steps) and means_by_n[ns[-1]] > H3_EPS
    return ("SUPPORTED" if ok else "FAILS"), \
        "steps " + ", ".join(f"{s:+.1%}" for s in steps) + f"; mean at N={ns[-1]} is {means_by_n[ns[-1]]:.4g}"


def last_point(rec: dict) -> dict:
    return rec["curve"][-1]


def at_iterations(rec: dict, T: int) -> dict | None:
    pts = [p for p in rec["curve"] if p["iterations"] <= T]
    return pts[-1] if pts else None


def section_verification(out: list[str], rd: Path) -> None:
    out.append("## Measurement instrument\n")
    p = rd / "verify" / "br_audit.json"
    if not p.exists():
        out.append("_br_audit.json has not been generated._\n")
        return
    a = json.loads(p.read_text())
    deltas = [float(c["detail"].split("delta")[1]) for c in a["checks"]
              if c["check"].startswith("openspiel") and "delta" in c["detail"]]
    out.append(f"- Best-response audit: **{a['n_checks'] - a['n_failed']}/{a['n_checks']}** "
               f"checks pass. Largest OpenSpiel delta: **{max(deltas):.2e}**. "
               f"_Source: `results/verify/br_audit.json`_")
    p = rd / "verify" / "leduc_crosscheck.json"
    if p.exists():
        out.append("- Leduc cross-check against OpenSpiel: see `results/verify/leduc_crosscheck.json` "
                   f"(all checks agree: {json.loads(p.read_text()).get('all_agree', 'unknown')}).")
    out.append("")


def section_coverage(out, good, prereg) -> None:
    out.append("## Coverage against the pre-registration\n")
    if prereg is None:
        out.append("_No pre-registration found._\n")
        return
    landed = {s.spec_hash(): bool(r["complete"]) for _, r, s in good}
    cov = a6.coverage(prereg, landed)
    out.append("| Matrix | Complete | Partial | Missing |\n|---|---:|---:|---:|")
    for m in sorted(cov["per_matrix"]):
        c = cov["per_matrix"][m]
        out.append(f"| `{m}` | {c['complete']} | {c['partial']} | {c['missing']} |")
    t = cov["totals"]
    out.append(f"| **All distinct cells ({len(prereg['cells'])})** | **{t['complete']}** | "
               f"**{t['partial']}** | **{t['missing']}** |")
    out.append("\n_Source: every `results/*.json` and `results/verify/preregistration.json`._\n")


def index(good):
    return {(s.game, s.variant, s.seed): (p, r) for p, r, s in good}


def section_h1(out, idx) -> None:
    out.append("## H1 -- CFR+ vs vanilla CFR at equal wall-clock (2-player Leduc)\n")
    out.append("Rule frozen in the pre-registration: CFR+ lower in >= 4/5 seed pairs AND median "
               "ratio < 0.8, judged per cap.\n")
    out.append("| Cap | Pairs | Verdict | Detail | Source |\n|---|---:|---|---|---|")
    verdicts = []
    for cap in (1, 2, 3):
        pairs, files = [], []
        for seed in range(H1_PAIRS):
            a, b = idx.get((f"leduc-2p-cap{cap}", "cfr", seed)), idx.get((f"leduc-2p-cap{cap}", "cfr+", seed))
            if a and b and a[1]["complete"] and b[1]["complete"]:
                pairs.append((last_point(a[1])["exploitability"], last_point(b[1])["exploitability"]))
                files += [a[0].name, b[0].name]
        v, d = h1_verdict(pairs)
        verdicts.append(v)
        out.append(f"| {cap} | {len(pairs)} | **{v}** | {d} | {len(files)} files `leduc-2p-cap{cap}__cfr*` |")
    out.append("")
    if all(v == "INCOMPLETE" for v in verdicts):
        out.append("**H1 has not been tested yet: no complete seed pair has landed.**\n")


def section_h2(out, idx) -> None:
    out.append("## H2 -- iterations to reach epsilon = 1e-2 (raw inputs to the scaling fit)\n")
    out.append("The slope and its bootstrap CI come from Akhil's `fit_scaling_exponent`; this "
               "table is its input only, and `censored` means epsilon was not reached.\n")
    out.append("| Rung | Variant | Iterations to eps | Source |\n|---|---|---:|---|")
    any_row = False
    for rung in ("kuhn", "4rank", "4rank-2suit", "4rank-2suit-2dealt"):
        for variant in ("cfr", "cfr+"):
            hit = idx.get((rung, variant, 0))
            if not hit:
                continue
            any_row = True
            reached = next((p["iterations"] for p in hit[1]["curve"]
                            if p["exploitability"] is not None and p["exploitability"] <= 1e-2), None)
            tag = "" if hit[1]["complete"] else " (partial run)"
            shown = reached if reached else "censored"
            out.append(f"| {rung} | {variant} | {shown}{tag} | `{hit[0].name}` |")
    if not any_row:
        out.append("| _no 2-player ladder cells have landed_ | | | |")
    out.append("")


def section_h3(out, idx) -> None:
    out.append("## H3 -- per-seat mean NashConv vs player count (deck held fixed)\n")
    out.append("Per-seat MEAN (total / N), never the total. Kuhn-family rungs are compared at the "
               "largest iteration count every N reached; Leduc cells are wall-clock budgeted so "
               "their iteration counts differ by N and are shown, which limits what the Leduc "
               "comparison can claim.\n")
    families = {
        "Kuhn family (4rank-2suit)": [(2, "4rank-2suit"), (3, "4rank-2suit-3p"), (4, "4rank-2suit-4p")],
        "Leduc, cap 1": [(n, f"leduc-{n}p-cap1") for n in (2, 3, 4, 5)],
    }
    for name, members in families.items():
        recs = {n: idx[(g, "cfr+", 0)] for n, g in members if (g, "cfr+", 0) in idx}
        out.append(f"### {name}\n")
        if len(recs) < 2:
            out.append("_Fewer than two player counts have landed._\n")
            continue
        kuhn = name.startswith("Kuhn")
        T = min(last_point(r)["iterations"] for _, r in recs.values()) if kuhn else None
        means, rows = {}, []
        for n, (path, rec) in sorted(recs.items()):
            pt = at_iterations(rec, T) if kuhn else last_point(rec)
            if pt is None:
                continue
            means[n] = pt["nashconv_mean"]
            note = "" if rec["complete"] else " (partial)"
            rows.append(f"| {n} | {pt['iterations']:,} | {pt['nashconv_mean']:.5g} | "
                        f"{pt['nashconv_max_seat']} | {', '.join(f'{g:.4g}' for g in pt['gains'])} | "
                        f"`{path.name}`{note} |")
        out.append("| N | Iterations | Mean NashConv | Max-gain seat | Per-seat gains | Source |\n"
                   "|---:|---:|---:|---:|---|---|")
        out += rows
        v, d = h3_verdict(means)
        out.append(f"\n**{v}** -- {d}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "report" / "numbers.md"))
    args = ap.parse_args()
    rd = Path(args.results_dir)

    good, errors = a6.load_results(rd)
    if errors:
        for e in errors:
            print(f"ERROR {e}", file=sys.stderr)
        return 2
    pp = rd / "verify" / "preregistration.json"
    prereg = json.loads(pp.read_text()) if pp.exists() else None
    idx = index(good)

    out = ["# Report numbers\n",
           "Generated by `scripts/a7_report_numbers.py`. Do not edit by hand; every entry "
           "names the `results/` file it came from.\n"]
    section_verification(out, rd)
    section_coverage(out, good, prereg)
    section_h1(out, idx)
    section_h2(out, idx)
    section_h3(out, idx)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))
    print(f"{len(good)} result files -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
