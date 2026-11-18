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
