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

from pathlib import Path

FIGURE_DIR = Path(__file__).resolve().parents[2] / "figures"
DPI = 200


def save(fig, name: str) -> list[Path]:
    """Save to both PNG (slides, demo) and PDF (report) and return the paths."""
    raise NotImplementedError("OWNER: Akhil -- see module docstring")
