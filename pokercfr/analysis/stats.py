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


@dataclass(frozen=True)
class Estimate:
    value: float
    ci95: tuple[float, float]
    n: int

    def __str__(self) -> str:
        return f"{self.value:.6g} [{self.ci95[0]:.6g}, {self.ci95[1]:.6g}] (n={self.n})"


def bootstrap_ci(samples: Sequence[float], statistic=None, resamples: int = 10_000,
                 seed: int = 0) -> Estimate:
    """Percentile bootstrap CI. `statistic` defaults to the mean."""
    raise NotImplementedError("OWNER: Akhil -- see module docstring")


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """(p_value, cliffs_delta) for paired per-seed measurements."""
    raise NotImplementedError("OWNER: Akhil")


def fit_scaling_exponent(sizes: Sequence[float], costs: Sequence[float],
                         resamples: int = 10_000, seed: int = 0) -> Estimate:
    """Slope of log(cost) ~ log(size), with a bootstrap CI. H2's headline number."""
    raise NotImplementedError("OWNER: Akhil")
