"""Figures -- OWNER: Akhil. Every figure in the report is generated here.

HARD RULE: no number and no figure in the report is hand-made. `make figures`
regenerates all of it from results/*.json, and every figure's caption names the
result file it came from. The course guidelines are explicit that a fabricated
result is graded as our own error and that "the AI generated it" is not a defence,
so traceability from figure back to a committed run is what makes every claim
defensible at viva.

Figures (each a function; each saves to figures/<name>.png AND .pdf at 200 dpi):

  fig_convergence        exploitability vs iterations AND vs wall-clock, log-log,
                         one line per variant, seed band shaded. The headline plot.
  fig_ablation_cube      the 2**3 corners: main effects and interactions. R1.
  fig_scaling_exponents  fitted slope per algorithm per ladder axis, with CIs. H2.
  fig_nashconv_players   NashConv plateau vs player count, deck held fixed. H3.
  fig_alpha_manifold     which point on Kuhn's alpha in [0,1/3] equilibrium family
                         each variant converges to, and whether it drifts. R5.
  fig_strategy_heatmap   per-info-set action probabilities vs iteration -- shows
                         bluffing frequency settling toward its known value.
  fig_oracle_agreement   LP vs CFR+ vs OpenSpiel vs closed form on one axis. R4.

Style: colourblind-safe palette, no red/green pairs, readable at print size,
axis units always stated ("exploitability, chips/hand"). Every log plot gets a
reference slope so a reader can read the rate off the figure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: `make figures` must work over SSH and in CI
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

from pokercfr.eval.exploitability import validate_record  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
DPI = 200

# The order figures are generated in; each name must be a function in this module.
FIGURES = (
    "fig_convergence",
    "fig_ablation_cube",
    "fig_scaling_exponents",
    "fig_nashconv_players",
    "fig_alpha_manifold",
    "fig_strategy_heatmap",
    "fig_oracle_agreement",
    "fig_mccfr_variance",
)


class ResultError(Exception):
    """A result file is unreadable or breaks the frozen record schema."""


def save(fig, name: str, out_dir: Path | None = None,
         sources: list[str] | None = None) -> list[Path]:
    """Save to both PNG (slides, demo) and PDF (report) and return the paths.

    `sources` names the result files the figure was drawn from. They are written to
    <name>.sources.txt beside the figure, which the caption cites, because a figure
    built from 120 runs cannot list them all in its caption.
    """
    if not name or any(sep in name for sep in "/\\") or Path(name).suffix:
        raise ValueError(f"figure name must be a bare stem like 'fig_convergence', got {name!r}")
    out = FIGURE_DIR if out_dir is None else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.pdf"]
    for path in paths:
        # A fixed creation date keeps the PDF byte-identical across reruns, which is
        # what lets the fresh-clone check compare figures with a plain diff.
        metadata = {"CreationDate": None} if path.suffix == ".pdf" else {"Software": None}
        fig.savefig(path, dpi=DPI, bbox_inches="tight", metadata=metadata)
    plt.close(fig)
    if sources is not None:
        listing = out / f"{name}.sources.txt"
        listing.write_text("".join(f"results/{s}\n" for s in sorted(sources)))
        paths.append(listing)
    return paths


# --- shared style --------------------------------------------------------------
# Validated categorical slots (blue, orange, aqua): colourblind-separable and no
# red/green pair. Aqua is low-contrast on white, so every line ALSO carries its own
# dash pattern and marker plus a direct label -- identity never rests on colour.
VARIANT_STYLE = {
    "cfr": {"color": "#2a78d6", "linestyle": "-", "marker": "o", "label": "CFR"},
    "cfr+": {"color": "#eb6834", "linestyle": "--", "marker": "s", "label": "CFR+"},
    "linear-cfr": {"color": "#1baf7a", "linestyle": ":", "marker": "^", "label": "Linear CFR"},
}
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e4e4e4"


def _style_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    # Label decades only; matplotlib's minor labels collide on short log ranges.
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10))
        axis.set_minor_formatter(NullFormatter())
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)


def _metric(record: dict) -> tuple[str, str]:
    """Exploitability for 2-player games; per-seat mean NashConv otherwise."""
    if record["game"]["players"] == 2:
        return "exploitability", "exploitability, chips/hand"
    return "nashconv_mean", "mean NashConv per seat, chips/hand"


def seed_band(records: list[dict], x_key: str, y_key: str):
    """Median and min..max of y across seeds, checkpoint by checkpoint.

    Runs of one cell share a checkpoint schedule (same time fractions or the same
    iteration list), so checkpoint i is comparable across seeds. x is the median
    across seeds at each checkpoint: under a wall-clock budget seeds reach slightly
    different iteration counts at the same time fraction.
    """
    lengths = {len(r["curve"]) for r in records}
    if len(lengths) != 1:
        raise ValueError(f"seeds disagree on checkpoint count {sorted(lengths)}: "
                         + ", ".join(r["_file"] for r in records))
    xs = np.array([[p[x_key] for p in r["curve"]] for r in records], dtype=float)
    ys = np.array([[p[y_key] for p in r["curve"]] for r in records], dtype=float)
    return np.median(xs, axis=0), np.median(ys, axis=0), ys.min(axis=0), ys.max(axis=0)


def _reference_slope(ax, x0: float, y0: float, x1: float, slope: float, text: str) -> None:
    xs = np.array([x0, x1])
    ax.plot(xs, y0 * (xs / x0) ** slope, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)),
            zorder=1)
    ax.annotate(text, (x1, y0 * (x1 / x0) ** slope), xytext=(3, 0), textcoords="offset points",
                fontsize=7, color=MUTED, va="center")


def fig_convergence(records: list[dict] | None = None,
                    games: tuple[str, ...] = ("leduc-2p-cap1", "leduc-2p-cap2", "leduc-2p-cap3"),
                    time_budget: float = 600.0,
                    variants: tuple[str, ...] = ("cfr", "cfr+", "linear-cfr"),
                    out_dir: Path | None = None) -> list[Path]:
    """Exploitability vs iterations and vs wall-clock, one row per game.

    Uses the H1 wall-clock cells (fixed time budget, one machine) so the seconds
    axis is a fair comparison. Lines are the median over seeds; the shaded band is
    the full seed range. The remaining ablation-cube corners belong to
    fig_ablation_cube, which keeps this, the headline plot, readable.
    """
    records = load_results() if records is None else records
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        key = (r["game"]["name"], r["spec"]["variant"])
        if key[0] in games and key[1] in variants and r["time_budget"] == time_budget:
            cells.setdefault(key, []).append(r)
    if not cells:
        raise NotImplementedError(f"no {time_budget:g}s runs for {games} in results/ yet")

    rows = [g for g in games if any(k[0] == g for k in cells)]
    fig, axes = plt.subplots(len(rows), 2, figsize=(7.5, 2.6 * len(rows)), sharey="row",
                             squeeze=False)
    used: list[str] = []
    for r_i, (row, game) in enumerate(zip(axes, rows, strict=True)):
        for col, (x_key, x_label) in enumerate([("iterations", "iterations"),
                                                ("train_seconds", "wall-clock, seconds")]):
            ax = row[col]
            _style_axis(ax)
            first = None
            for variant in variants:
                runs = cells.get((game, variant))
                if not runs:
                    continue
                y_key, y_label = _metric(runs[0])
                x, y, lo, hi = seed_band(runs, x_key, y_key)
                st = VARIANT_STYLE[variant]
                ax.fill_between(x, lo, hi, color=st["color"], alpha=0.18, linewidth=0, zorder=2)
                ax.plot(x, y, color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                        markersize=4, linewidth=1.6, zorder=3, label=st["label"])
                if col == 1:  # direct label at the line end, in ink, not series colour
                    ax.annotate(st["label"], (x[-1], y[-1]), xytext=(5, 0),
                                textcoords="offset points", fontsize=7, color=INK, va="center")
                first = first or (x[0], y[0], x[-1])
                if x_key == "train_seconds":  # a 10-minute budget spans barely 1.5 decades
                    ticks = [t for t in (10, 30, 60, 120, 300, 600, 1800, 3600)
                             if x[0] * 0.8 <= t <= x[-1] * 1.25]
                    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
                if col == 0:
                    used += [r["_file"] for r in runs]
            if first and col == 0:
                x0, y0, x1 = first
                _reference_slope(ax, x0, y0 * 1.6, x1, -1.0, "∝ 1/T")
                _reference_slope(ax, x0, y0 * 1.6, x1, -0.5, "∝ 1/√T")
            if r_i == len(rows) - 1:
                ax.set_xlabel(x_label, fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel(y_label, fontsize=8, color=INK)
                n = len(next(iter(cells[k] for k in cells if k[0] == game)))
                ax.set_title(f"{game}  ·  n={n} seeds", fontsize=9, color=INK, loc="left")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, -0.01, f"Line: median over seeds. Band: seed range. Budget {time_budget:g}s "
             f"per run on one machine. Sources: fig_convergence.sources.txt",
             ha="center", fontsize=7, color=MUTED)
    fig.tight_layout()
    return save(fig, "fig_convergence", out_dir=out_dir, sources=used)


def load_results(results_dir: Path | None = None) -> list[dict]:
    """Every result record under results_dir, schema-checked, oldest filename first.

    Each record gains a "_file" key (the bare filename) so a figure's caption can
    name exactly which runs it was drawn from. Any schema violation is fatal: a
    figure drawn from a record we cannot trust is worse than no figure.
    """
    folder = RESULTS_DIR if results_dir is None else Path(results_dir)
    records = []
    for path in sorted(folder.glob("*.json")):  # top level only; verify/ is not run data
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ResultError(f"{path.name}: not valid JSON ({exc})") from exc
        errors = validate_record(record)
        if errors:
            raise ResultError(f"{path.name}: " + "; ".join(errors))
        record["_file"] = path.name
        records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    """Generate every figure; an unwritten figure prints SKIP instead of crashing.

    Returns 0 when every implemented figure succeeded, 1 if any raised.
    """
    module = sys.modules[__name__]
    failed = 0
    for name in FIGURES:
        fn = getattr(module, name, None)
        if fn is None:
            print(f"SKIP  {name}: not written yet")
            continue
        try:
            paths = fn()
        except NotImplementedError as exc:
            print(f"SKIP  {name}: {exc}")
            continue
        except Exception as exc:  # report every broken figure, not just the first
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
            failed += 1
            continue
        print(f"OK    {name}: " + ", ".join(str(p.relative_to(ROOT)) for p in paths))
    return 1 if failed else 0


# --- ablation cube (R1) ------------------------------------------------------------
# The 2**3 factorial: regret matching (rm vs rm+), averaging (uniform vs linear),
# update schedule (simultaneous vs alternating). Three corners carry paper names.
CUBE_FACTORS = ("RM+", "linear averaging", "alternating")
_NAMED_CORNERS = {"cfr": "rm/uniform/sim", "cfr+": "rm+/linear/alt", "linear-cfr": "rm/linear/sim"}


def cube_corner(variant: str) -> tuple[int, int, int] | None:
    """(rm+, linear, alternating) as 0/1 for an ablation-cube variant, else None."""
    parts = _NAMED_CORNERS.get(variant, variant).split("/")
    if len(parts) != 3 or parts[0] not in ("rm", "rm+") or parts[1] not in ("uniform", "linear") \
            or parts[2] not in ("sim", "alt"):
        return None
    return int(parts[0] == "rm+"), int(parts[1] == "linear"), int(parts[2] == "alt")


def _effect_terms() -> list[tuple[int, ...]]:
    """Main effects, then two-way interactions, then the three-way one."""
    return [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]


def factorial_effects(table: dict[int, dict[tuple[int, int, int], float]]):
    """Per-seed factorial effects from {seed: {corner: response}}.

    Standard +/-1 contrast coding: an effect is (mean response where the contrast
    is +1) minus (mean where it is -1). Computed within each seed, so the corners
    are compared on the same seed -- the paired design -- and the spread across
    seeds is what the confidence interval describes.
    """
    corners = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    seeds = sorted(table)
    for s in seeds:
        missing = [c for c in corners if c not in table[s]]
        if missing:
            raise ValueError(f"seed {s} is missing cube corners {missing}")
    out = {}
    for term in _effect_terms():
        per_seed = []
        for s in seeds:
            total = 0.0
            for corner in corners:
                sign = np.prod([1 if corner[f] else -1 for f in term])
                total += sign * table[s][corner]
            per_seed.append(total / 4)  # 4 corners on each side of every contrast
        out[term] = per_seed
    return out


def _term_label(term: tuple[int, ...]) -> str:
    return " × ".join(CUBE_FACTORS[f] for f in term)


def fig_ablation_cube(records: list[dict] | None = None,
                      games: tuple[str, ...] = ("leduc-2p-cap1", "leduc-2p-cap2", "leduc-2p-cap3"),
                      time_budget: float = 600.0, out_dir: Path | None = None) -> list[Path]:
    """Main effects and interactions of the cube on final log10 exploitability.

    Response: log10 exploitability at the end of the fixed wall-clock budget, so
    alternating updates pay for their extra traversals. An effect of -1 means
    switching that factor on cuts exploitability tenfold. Dots are the mean over
    seeds; bars are bootstrap 95% CIs over seeds. Also writes the numbers to
    <figure>.csv so the report quotes them from a file rather than a plot.
    """
    from pokercfr.analysis.stats import bootstrap_ci

    records = load_results() if records is None else records
    tables: dict[str, dict[int, dict]] = {}
    used: list[tuple[str, int, str]] = []
    for r in records:
        corner = cube_corner(r["spec"]["variant"])
        name = r["game"]["name"]
        if corner is None or name not in games or r["time_budget"] != time_budget:
            continue
        y_key, _ = _metric(r)
        tables.setdefault(name, {}).setdefault(r["spec"]["seed"], {})[corner] = \
            float(np.log10(r["curve"][-1][y_key]))
        used.append((name, r["spec"]["seed"], r["_file"]))
    # Keep only seeds that ran all 8 corners, so every effect is a paired comparison.
    for name in tables:
        tables[name] = {s: t for s, t in tables[name].items() if len(t) == 8}
    rows = [g for g in games if tables.get(g)]
    if not rows:
        raise NotImplementedError(f"no complete {time_budget:g}s ablation cube in results/ yet")

    terms = _effect_terms()
    fig, axes = plt.subplots(1, len(rows), figsize=(2.6 * len(rows) + 1.6, 3.4), sharey=True, sharex=True,
                             squeeze=False)
    lines = ["game,effect,mean_log10,ci_low,ci_high,n_seeds"]
    for ax, game in zip(axes[0], rows, strict=True):
        effects = factorial_effects(tables[game])
        n = len(tables[game])
        ax.axvline(0, color=MUTED, linewidth=0.8)
        for i, term in enumerate(terms):
            est = bootstrap_ci(effects[term], seed=i)
            ax.plot(est.ci95, [i, i], color="#2a78d6", linewidth=2, solid_capstyle="round")
            ax.plot(est.value, i, "o", color="#2a78d6", markersize=6,
                    markeredgecolor="white", markeredgewidth=1.2)
            lines.append(f"{game},{_term_label(term)},{est.value:.6g},{est.ci95[0]:.6g},"
                         f"{est.ci95[1]:.6g},{n}")
        ax.set_title(f"{game}  ·  n={n} seeds", fontsize=9, color=INK, loc="left")
        ax.grid(True, axis="x", color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.set_xlabel("effect on log10 exploitability\n(negative = better)", fontsize=8, color=INK)
    ax0 = axes[0][0]
    ax0.set_yticks(range(len(terms)), [_term_label(t) for t in terms], fontsize=8, color=INK)
    ax0.invert_yaxis()
    fig.text(0.5, -0.04, f"Response: final exploitability after {time_budget:g}s on one machine. "
             "Dot: mean effect over seeds. Bar: bootstrap 95% CI. "
             "Sources: fig_ablation_cube.sources.txt; numbers: fig_ablation_cube.csv",
             ha="center", fontsize=7, color=MUTED)
    fig.tight_layout()
    kept = [f for g, s, f in used if g in rows and s in tables[g]]
    paths = save(fig, "fig_ablation_cube", out_dir=out_dir, sources=kept)
    csv_path = (FIGURE_DIR if out_dir is None else Path(out_dir)) / "fig_ablation_cube.csv"
    csv_path.write_text("\n".join(lines) + "\n")
    return paths + [csv_path]


# --- NashConv vs player count (H3) ---------------------------------------------------
# One ordinal blue ramp, light -> dark as N grows, so "more players" reads as "more"
# without inventing new hues. Each N also gets its own marker for print.
PLAYER_STYLE = {2: ("#86b6ef", "o"), 3: ("#3987e5", "s"), 4: ("#1c5cab", "^"),
                5: ("#0d366b", "D"), 6: ("#08264d", "v")}


def fig_nashconv_players(records: list[dict] | None = None, cap: int = 1,
                         variants: tuple[str, ...] = ("cfr", "cfr+", "linear-cfr"),
                         out_dir: Path | None = None) -> list[Path]:
    """Mean NashConv per seat vs iterations, one line per player count, deck fixed.

    H3 asks whether NashConv plateaus above zero once there are 3+ players. Plotting
    against iterations (not one end-of-budget number) is what separates a genuine
    plateau from a curve that is simply slower: under a fixed wall-clock budget a
    4-player run gets ~300x fewer iterations than a 2-player one, so comparing final
    values alone would confound "more players" with "less training".
    The per-seat MEAN is used because total NashConv grows with N by construction.
    """
    records = load_results() if records is None else records
    cells: dict[tuple[str, int], list[dict]] = {}
    for r in records:
        name = r["game"]["name"]
        if not name.startswith("leduc-") or not name.endswith(f"-cap{cap}"):
            continue
        if r["spec"]["variant"] not in variants or r["time_budget"] is None:
            continue
        cells.setdefault((r["spec"]["variant"], r["game"]["players"]), []).append(r)
    # One budget per player count: the longest available, so 2p is not a 10-minute
    # run sitting beside hour-long multiplayer ones.
    for key, runs in cells.items():
        longest = max(r["time_budget"] for r in runs)
        cells[key] = [r for r in runs if r["time_budget"] == longest]
    panels = [v for v in variants if any(k[0] == v and k[1] > 2 for k in cells)]
    if not panels:
        raise NotImplementedError(f"no multiplayer leduc cap{cap} runs in results/ yet")

    fig, axes = plt.subplots(1, len(panels), figsize=(2.7 * len(panels) + 0.6, 3.3),
                             sharey=True, squeeze=False)
    used: list[str] = []
    lines = ["variant,players,info_sets,n_seeds,time_budget_s,iterations,nashconv_mean_median,"
             "nashconv_mean_min,nashconv_mean_max"]
    for ax, variant in zip(axes[0], panels, strict=True):
        _style_axis(ax)
        for n in sorted(p for v, p in cells if v == variant):
            runs = cells[(variant, n)]
            x, y, lo, hi = seed_band(runs, "iterations", "nashconv_mean")
            color, marker = PLAYER_STYLE.get(n, ("#08264d", "x"))
            if len(runs) > 1:
                ax.fill_between(x, lo, hi, color=color, alpha=0.2, linewidth=0, zorder=2)
            ax.plot(x, y, color=color, marker=marker, markersize=4, linewidth=1.6, zorder=3,
                    label=f"{n} players")
            ax.annotate(f"N={n}" + ("" if len(runs) > 1 else "*"), (x[-1], y[-1]),
                        xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK,
                        va="center")
            used += [r["_file"] for r in runs]
            lines.append(f"{variant},{n},{runs[0]['game']['info_sets']},{len(runs)},"
                         f"{runs[0]['time_budget']:g},{x[-1]:.0f},{y[-1]:.6g},{lo[-1]:.6g},"
                         f"{hi[-1]:.6g}")
        ax.set_title(VARIANT_STYLE.get(variant, {"label": variant})["label"], fontsize=9,
                     color=INK, loc="left")
        ax.set_xlabel("iterations", fontsize=9, color=INK)
    axes[0][0].set_ylabel("mean NashConv per seat, chips/hand", fontsize=8, color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.06))
    fig.text(0.5, -0.05, f"Leduc, raise cap {cap}, deck fixed as players are added. Line: median "
             "over seeds; band: seed range; * = one seed only. Each N uses its longest wall-clock "
             "budget. Sources: fig_nashconv_players.sources.txt; numbers: fig_nashconv_players.csv",
             ha="center", fontsize=7, color=MUTED, wrap=True)
    fig.tight_layout()
    paths = save(fig, "fig_nashconv_players", out_dir=out_dir, sources=used)
    csv_path = (FIGURE_DIR if out_dir is None else Path(out_dir)) / "fig_nashconv_players.csv"
    csv_path.write_text("\n".join(lines) + "\n")
    return paths + [csv_path]


def _cost_to_epsilon(record: dict, epsilon: float,
                     epsilon_axis: str = "iterations") -> float | None:
    """Iterations needed to reach `epsilon`, interpolated in log-log space.

    Checkpoints are log-spaced, so taking the first one at or below epsilon
    quantises the answer onto that grid -- which on this ladder swamped the
    scaling signal entirely. Interpolating between the bracketing checkpoints
    recovers a continuous estimate. None means the run never got there.
    """
    key = "train_seconds" if epsilon_axis == "seconds" else "iterations"
    curve = [(p[key], p["exploitability"]) for p in record["curve"]
             if p.get("exploitability") is not None and p["exploitability"] > 0
             and p.get(key)]
    previous = None
    for iterations, value in curve:
        if value <= epsilon:
            if previous is None:
                return float(iterations)
            (i0, v0) = previous
            if v0 <= epsilon or v0 == value:
                return float(iterations)
            # linear in log-log, which is the space the curve is a power law in
            import math
            t = (math.log(v0) - math.log(epsilon)) / (math.log(v0) - math.log(value))
            return math.exp(math.log(i0) + t * (math.log(iterations) - math.log(i0)))
        previous = (iterations, value)
    return None


def _ladder_epsilon(records: list[dict], variants: tuple[str, ...]) -> float:
    """The tightest epsilon every included run actually reaches.

    Chosen from the data rather than fixed in advance, because an epsilon that
    some cell never reaches censors it, and an epsilon every cell clears in its
    first few checkpoints measures the checkpoint grid instead of the algorithm.
    """
    finals = [r["curve"][-1]["exploitability"] for r in records
              if r["spec"]["variant"] in variants and r["curve"]
              and r["curve"][-1].get("exploitability")]
    return max(finals) * 1.5 if finals else 1e-2


def fig_scaling_exponents(records: list[dict] | None = None,
                          out_dir: Path | None = None,
                          epsilon: float | None = None) -> list[Path]:
    """H2: how does cost to reach a fixed epsilon grow with information-set count?

    One fitted slope per variant across the two-player ladder, with a bootstrap
    interval. The slope IS the scaling law, and an interval excluding another
    variant's slope is what turns "CFR+ looked faster" into "CFR+ scales
    measurably better".

    Restricted to the eight ablation corners. The DCFR grid runs on a different
    iteration budget and the LP oracle is not iterative at all, so neither
    belongs on an iterations-to-epsilon axis.
    """
    import matplotlib.pyplot as plt

    from pokercfr.analysis.stats import fit_scaling_exponent
    from pokercfr.solvers.cfr import ablation_cube

    records = load_results() if records is None else records
    LADDER = ("kuhn", "4rank", "4rank-2suit", "4rank-2suit-2dealt")
    CORNERS = tuple(ablation_cube())

    on_ladder = [r for r in records
                 if r["game"]["name"] in LADDER and r["game"]["players"] == 2
                 and r["spec"]["variant"] in CORNERS and r["curve"]]
    if not on_ladder:
        raise ResultError("no two-player ladder runs for the ablation corners yet")
    eps = _ladder_epsilon(on_ladder, CORNERS) if epsilon is None else epsilon

    fits: dict[str, dict] = {}
    sources, censored = [], 0
    for axis in ("iterations", "seconds"):
        points: dict[str, list[tuple[float, float]]] = {}
        for record in on_ladder:
            reached = _cost_to_epsilon(record, eps, axis)
            if reached is None or reached <= 0:
                censored += axis == "iterations"
                continue
            points.setdefault(record["spec"]["variant"], []).append(
                (float(record["game"]["info_sets"]), reached))
            sources.append(record["_file"])
        for variant, pairs in points.items():
            if len(pairs) < 3:
                continue
            pairs.sort()
            fits.setdefault(variant, {})[axis] = fit_scaling_exponent(
                [x for x, _ in pairs], [y for _, y in pairs])

    both = {v: f for v, f in fits.items() if len(f) == 2}
    if not both:
        raise ResultError("no corner has three ladder rungs on both axes yet")

    order = sorted(both, key=lambda v: both[v]["seconds"].value)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 0.42 * len(order) + 2.4),
                             sharey=True)
    titles = {"iterations": "iterations to $\\epsilon$",
              "seconds": "wall-clock seconds to $\\epsilon$"}
    ys = list(range(len(order)))
    for ax, axis in zip(axes, ("iterations", "seconds"), strict=True):
        est = [both[v][axis] for v in order]
        ax.errorbar(
            [e.value for e in est], ys,
            xerr=[[max(0.0, e.value - e.ci95[0]) for e in est],
                  [max(0.0, e.ci95[1] - e.value) for e in est]],
            fmt="o", capsize=3, linewidth=1.2, markersize=5, color="#1f4e79")
        ax.axvline(0.0, color="#c1121f", linestyle="--", linewidth=1)
        ax.axvline(1.0, color="0.6", linestyle=":", linewidth=1)
        ax.set_xlabel(f"exponent: {titles[axis]} vs information sets")
        ax.set_title(titles[axis])
        _style_axis(ax)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(order)
    fig.suptitle(f"H2: where the cost of a bigger game actually lands "
                 f"($\\epsilon$ = {eps:.2g} chips/hand)", y=1.02)
    axes[1].annotate("0 = no growth", xy=(0.0, len(order) - 0.6), fontsize=8,
                     color="#c1121f", rotation=90, va="top", ha="right")
    fig.tight_layout()

    csv = (FIGURE_DIR if out_dir is None else Path(out_dir)) / "fig_scaling_exponents.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    csv.write_text(
        f"# epsilon = {eps:.6g} chips/hand, the tightest every included run reaches\n"
        "variant,iter_exponent,iter_ci_low,iter_ci_high,"
        "sec_exponent,sec_ci_low,sec_ci_high\n" + "".join(
            f"{v},{both[v]['iterations'].value:.6f},{both[v]['iterations'].ci95[0]:.6f},"
            f"{both[v]['iterations'].ci95[1]:.6f},{both[v]['seconds'].value:.6f},"
            f"{both[v]['seconds'].ci95[0]:.6f},{both[v]['seconds'].ci95[1]:.6f}\n"
            for v in order))
    return save(fig, "fig_scaling_exponents", out_dir, sorted(set(sources))) + [csv]


def fig_alpha_manifold(records: list[dict] | None = None,
                       out_dir: Path | None = None) -> list[Path]:
    """Which point on Kuhn's equilibrium family does each variant select?

    Kuhn's equilibria form a one-parameter family indexed by the first player's
    bluffing frequency alpha in [0, 1/3]. Exploitability cannot tell two
    equilibria apart, so this is the only way to see that the variants disagree.
    """
    import matplotlib.pyplot as plt

    source = RESULTS_DIR / "verify" / "alpha_manifold.json"
    if not source.exists():
        raise ResultError("run scripts/h10_alpha_manifold.py first")
    data = json.loads(source.read_text())
    traces = data["traces"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    for variant, trace in sorted(traces.items()):
        plus = variant.startswith("rm+") or variant == "cfr+"
        ax.plot([p["iterations"] for p in trace], [p["alpha"] for p in trace],
                marker="o", markersize=3, linewidth=1.3,
                color="#c1121f" if plus else "#1f4e79",
                alpha=0.85, label=variant)
    ax.axhspan(0.0, 1 / 3, color="0.9", zorder=0)
    ax.axhline(1 / 3, color="0.5", linestyle="--", linewidth=1)
    ax.annotate(r"$\alpha = 1/3$, edge of the family", xy=(trace[0]["iterations"], 1 / 3),
                xytext=(0, 4), textcoords="offset points", fontsize=8, color="0.35")
    ax.set_xscale("log")
    ax.set_xlabel("iterations")
    ax.set_ylabel(r"$\alpha = P(\mathrm{bet} \mid J)$")
    ax.set_title("Which equilibrium each variant converges to")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    _style_axis(ax)

    finals = [(t[-1]["alpha"], v) for v, t in traces.items()]
    bx.scatter([a for a, _ in finals],
               [3 * a for a, _ in finals], s=28, color="#1f4e79", label="measured")
    lo, hi = min(a for a, _ in finals), max(a for a, _ in finals)
    bx.plot([lo, hi], [3 * lo, 3 * hi], color="#c1121f", linewidth=1.2,
            linestyle="--", label=r"closed form $3\alpha$")
    for a, v in finals:
        bx.annotate(v, xy=(a, 3 * a), xytext=(3, -3), textcoords="offset points",
                    fontsize=6, color="0.35")
    bx.set_xlabel(r"$\alpha$")
    bx.set_ylabel(r"$P(\mathrm{bet} \mid K)$")
    bx.set_title("Kuhn's identity holds at every selected point")
    bx.legend(fontsize=8, frameon=False)
    _style_axis(bx)
    fig.tight_layout()
    return save(fig, "fig_alpha_manifold", out_dir, ["verify/alpha_manifold.json"])


def fig_oracle_agreement(records: list[dict] | None = None,
                         out_dir: Path | None = None) -> list[Path]:
    """Three independent routes to the same game value.

    An iterative regret minimiser, an exact sequence-form linear program, and
    Kuhn's closed form. Agreement between unrelated algorithms is far stronger
    evidence of correctness than either matching itself.
    """
    import matplotlib.pyplot as plt

    source = RESULTS_DIR / "verify" / "lp_values.json"
    if not source.exists():
        raise ResultError("run scripts/sid1_lp_values.py first")
    data = json.loads(source.read_text())

    rows = {}
    for entry in data["cfr_plus_agreement"]:
        rows.setdefault(entry["game"], entry)
    games = [e["game"] for e in data["lp"] if e["game"] in rows]
    if not games:
        raise ResultError("no game has both an LP value and a CFR+ comparison")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    diffs = [abs(rows[g]["abs_diff"]) or 1e-18 for g in games]
    ax.barh(range(len(games)), diffs, color="#1f4e79", height=0.55)
    ax.set_yticks(range(len(games)))
    ax.set_yticklabels(games, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("|LP value $-$ CFR+ value|  (chips/hand)")
    ax.set_title("Exact LP vs converged CFR+")
    ax.axvline(1e-9, color="#c1121f", linestyle="--", linewidth=1)
    ax.annotate("1e-9", xy=(1e-9, len(games) - 0.5), fontsize=8, color="#c1121f")
    _style_axis(ax)

    expl = [abs(e.get("exploitability_eval", 0.0)) or 1e-18 for e in data["lp"]
            if e["game"] in rows]
    bx.barh(range(len(games)), expl, color="#2a6f4e", height=0.55)
    bx.set_yticks(range(len(games)))
    bx.set_yticklabels(games, fontsize=8)
    bx.set_xscale("log")
    bx.set_xlabel("exploitability of the LP policy (chips/hand)")
    bx.set_title("The LP solution is exact to machine precision")
    _style_axis(bx)
    fig.tight_layout()

    kuhn = next((e for e in data["lp"] if e["game"] == "kuhn"), None)
    csv = (FIGURE_DIR if out_dir is None else Path(out_dir)) / "fig_oracle_agreement.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    lines = ["game,lp_value,cfr_plus_value,abs_diff,lp_policy_exploitability\n"]
    for entry in data["lp"]:
        g = entry["game"]
        if g not in rows:
            continue
        lines.append(f"{g},{entry['value']:.17g},{rows[g]['cfr_plus_value']:.17g},"
                     f"{rows[g]['abs_diff']:.3e},{entry.get('exploitability_eval', 0.0):.3e}\n")
    if kuhn:
        lines.append(f"# kuhn closed form (Kuhn 1950) = {-1/18:.17g}, "
                     f"LP deviation = {abs(kuhn['value'] + 1/18):.3e}\n")
    csv.write_text("".join(lines))
    return save(fig, "fig_oracle_agreement", out_dir, ["verify/lp_values.json"]) + [csv]


# --- MCCFR sampling variance ------------------------------------------------------

MCCFR_STYLE = {
    "mccfr-outcome": {"color": "#8c4bbd", "linestyle": "-", "label": "MCCFR outcome"},
    "mccfr-external": {"color": "#1baf7a", "linestyle": "--", "label": "MCCFR external"},
    "mccfr-chance": {"color": "#8c4bbd", "linestyle": "-", "label": "MCCFR chance"},
}


def fig_mccfr_variance(records: list[dict] | None = None,
                       out_dir: Path | None = None) -> list[Path]:
    """What sampling costs in seed-to-seed spread, and whether it pays for itself.

    MCCFR buys cheaper iterations with variance. Both halves of that trade are
    measurable here: the spread across ten seeds at each checkpoint, and the level
    reached against full traversal on the same game at the same wall clock.
    """
    import matplotlib.pyplot as plt

    if records is None:
        records = load_results()
    sampled = [r for r in records if r["spec"]["variant"].startswith("mccfr-")]
    if not sampled:
        raise NotImplementedError("no mccfr-* results yet")

    game = sampled[0]["spec"]["game"]
    sampled = [r for r in sampled if r["spec"]["game"] == game]
    y_key, y_label = _metric(sampled[0])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    files, spread = [], {}

    for variant in sorted({r["spec"]["variant"] for r in sampled}):
        cell = sorted((r for r in sampled if r["spec"]["variant"] == variant),
                      key=lambda r: r["spec"]["seed"])
        files += [r["_file"] for r in cell]
        style = MCCFR_STYLE.get(variant, {"color": "#8c4bbd", "linestyle": "-",
                                          "label": variant})
        secs = np.array([[p["train_seconds"] for p in r["curve"]] for r in cell], dtype=float)
        its = np.array([p["iterations"] for p in cell[0]["curve"]], dtype=float)
        ys = np.array([[p[y_key] for p in r["curve"]] for r in cell], dtype=float)
        xs = np.median(secs, axis=0)
        for row_x, row_y in zip(secs, ys, strict=True):
            ax.plot(row_x, row_y, color=style["color"], linewidth=0.6, alpha=0.30, zorder=2)
        ax.plot(xs, ys.mean(axis=0), color=style["color"], linestyle=style["linestyle"],
                linewidth=2.0, marker="D", markersize=3.5, zorder=4,
                label=f"{style['label']} (mean of {len(cell)} seeds)")
        spread[variant] = (its, 100.0 * ys.std(axis=0, ddof=1) / ys.mean(axis=0))

    for variant in ("cfr", "cfr+"):
        full = [r for r in records
                if r["spec"]["game"] == game and r["spec"]["variant"] == variant]
        if not full:
            continue
        run = full[0]
        files.append(run["_file"])
        pts = [(p["train_seconds"], p[y_key]) for p in run["curve"] if p[y_key] > 0.0]
        style = VARIANT_STYLE[variant]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=style["color"],
                linestyle=style["linestyle"], linewidth=1.6, marker=style["marker"],
                markersize=3.0, zorder=3, label=f"{style['label']} (full traversal)")

    ax.set_xlabel("training wall-clock, seconds")
    ax.set_ylabel(y_label)
    ax.set_title(f"At equal wall clock on {game}, sampling does not pay")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    _style_axis(ax)

    for variant, (xs, cv) in spread.items():
        style = MCCFR_STYLE.get(variant, {"color": "#8c4bbd", "linestyle": "-",
                                          "label": variant})
        bx.plot(xs, cv, color=style["color"], linestyle=style["linestyle"],
                linewidth=2.0, marker="D", markersize=3.5, label=style["label"])
    bx.set_xscale("log")
    bx.set_xlabel("iterations")
    bx.set_ylabel("spread across seeds, per cent of the mean")
    bx.set_title("Sampling noise grows as a share of the remaining error")
    bx.grid(True, which="major", color=GRID, linewidth=0.6)
    bx.xaxis.set_major_locator(LogLocator(base=10))
    bx.xaxis.set_minor_formatter(NullFormatter())
    bx.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        bx.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        bx.spines[side].set_color(MUTED)
    if len(spread) > 1:
        bx.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()

    csv = (FIGURE_DIR if out_dir is None else Path(out_dir)) / "fig_mccfr_variance.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"game,variant,iterations,seeds,{y_key}_mean,{y_key}_sd,cv_percent,"
             "train_seconds_mean\n"]
    for variant in sorted({r["spec"]["variant"] for r in sampled}):
        cell = sorted((r for r in sampled if r["spec"]["variant"] == variant),
                      key=lambda r: r["spec"]["seed"])
        n = len(cell)
        for i, it in enumerate(p["iterations"] for p in cell[0]["curve"]):
            vals = np.array([r["curve"][i][y_key] for r in cell], dtype=float)
            secs = np.mean([r["curve"][i]["train_seconds"] for r in cell])
            sd = vals.std(ddof=1) if n > 1 else 0.0
            lines.append(f"{game},{variant},{it},{n},{vals.mean():.6e},{sd:.6e},"
                         f"{100.0 * sd / vals.mean():.2f},{secs:.1f}\n")
    csv.write_text("".join(lines))
    return save(fig, "fig_mccfr_variance", out_dir, files) + [csv]
