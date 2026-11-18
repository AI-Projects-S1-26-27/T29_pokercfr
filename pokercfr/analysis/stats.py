"""Statistics -- OWNER: Akhil. This module exists because of specific feedback.

The proposal lost marks for having "no statistics other than seed variation".
Everything here answers that directly, and NO number reaches the report without
passing through this module.

Three jobs:

1. UNCERTAINTY. bootstrap_ci on anything measured across seeds. Exploitability
   across MCCFR seeds is not normally distributed, so a bootstrap percentile
   interval is the honest choice over mean +/- 1.96*sd.

2. PAIRED COMPARISON. Ablation corners run on the SAME seeds, so comparisons are
   paired and a paired test is both correct and far more powerful than unpaired.
   Wilcoxon signed-rank avoids assuming normality. Report the effect size
   (Cliff's delta) alongside p -- "CFR+ is better" is uninteresting without
   "by how much", and with n=10 seeds a p-value alone says little.

3. SCALING EXPONENTS -- this is H2's headline number. Regress
       log(iterations or seconds to reach epsilon)  ~  log(info sets)
   per algorithm per ladder axis, and report the slope with a bootstrap CI. The
   slope IS the scaling law. A CI that excludes the other algorithm's slope is
   what turns "CFR+ looked faster" into "CFR+ scales measurably better".

ACCEPTANCE (tests/test_stats.py):
  - bootstrap_ci on a known normal sample brackets the true mean ~95% of the time
  - fit_scaling_exponent recovers a planted exponent (y = 3*x^1.7) to within CI
  - paired_wilcoxon on identical inputs gives p == 1.0 and effect size 0
  - every function is deterministic given a seed
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Estimate:
    value: float
    ci95: tuple[float, float]
    n: int

    def __str__(self) -> str:
        return f"{self.value:.6g} [{self.ci95[0]:.6g}, {self.ci95[1]:.6g}] (n={self.n})"


def _as_1d(samples: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(samples, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def bootstrap_ci(samples: Sequence[float], statistic=None, resamples: int = 10_000,
                 seed: int = 0) -> Estimate:
    """Percentile bootstrap CI. `statistic` defaults to the mean."""
    x = _as_1d(samples, "samples")
    stat = np.mean if statistic is None else statistic
    value = float(stat(x))
    if x.size == 1:
        # One seed carries no information about spread; say so rather than invent a CI.
        return Estimate(value, (value, value), 1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(resamples, x.size))
    if statistic is None:
        boot = x[idx].mean(axis=1)  # vectorised fast path for the common case
    else:
        boot = np.array([stat(x[row]) for row in idx], dtype=float)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return Estimate(value, (float(lo), float(hi)), int(x.size))


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """P(a > b) - P(a < b) over all pairs; in [-1, 1], 0 means no difference."""
    x, y = _as_1d(a, "a"), _as_1d(b, "b")
    diff = x[:, None] - y[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """(p_value, cliffs_delta) for paired per-seed measurements."""
    x, y = _as_1d(a, "a"), _as_1d(b, "b")
    if x.size != y.size:
        raise ValueError(f"paired samples differ in length: {x.size} vs {y.size}")
    delta = cliffs_delta(x, y)
    if np.all(x == y):
        # scipy raises on all-zero differences; identical runs are the null exactly.
        return 1.0, delta
    p = float(stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue)
    return p, delta


def _ols_slope(lx: np.ndarray, ly: np.ndarray) -> float:
    xc = lx - lx.mean()
    denom = float(np.dot(xc, xc))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(xc, ly - ly.mean()) / denom)


def fit_scaling_exponent(sizes: Sequence[float], costs: Sequence[float],
                         resamples: int = 10_000, seed: int = 0) -> Estimate:
    """Slope of log(cost) ~ log(size), with a bootstrap CI. H2's headline number.

    The CI resamples (size, cost) pairs, so it reflects both seed noise at each rung
    and how well a single power law fits across rungs.
    """
    s, c = _as_1d(sizes, "sizes"), _as_1d(costs, "costs")
    if s.size != c.size:
        raise ValueError(f"sizes and costs differ in length: {s.size} vs {c.size}")
    if np.any(s <= 0) or np.any(c <= 0):
        raise ValueError("sizes and costs must be positive to take logs")
    lx, ly = np.log(s), np.log(c)
    if np.unique(lx).size < 2:
        raise ValueError("need at least two distinct sizes to fit a slope")
    slope = _ols_slope(lx, ly)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, s.size, size=(resamples, s.size))
    boot = np.array([_ols_slope(lx[row], ly[row]) for row in idx])
    boot = boot[np.isfinite(boot)]  # resamples that hit only one size have no slope
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return Estimate(slope, (float(lo), float(hi)), int(s.size))
