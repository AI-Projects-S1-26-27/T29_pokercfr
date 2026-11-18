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

matplotlib.use("Agg")  # headless: `make figures` must work over SSH and in CI
import matplotlib.pyplot as plt  # noqa: E402

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


def save(fig, name: str, out_dir: Path | None = None) -> list[Path]:
    """Save to both PNG (slides, demo) and PDF (report) and return the paths."""
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
    return paths


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
