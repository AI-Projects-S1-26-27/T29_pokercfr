"""Acceptance tests for pokercfr/analysis/stats.py (see its module docstring)."""
import numpy as np
import pytest

from pokercfr.analysis.stats import (
    Estimate,
    bootstrap_ci,
    cliffs_delta,
    fit_scaling_exponent,
    paired_wilcoxon,
)


def test_bootstrap_ci_coverage_near_95_percent():
    rng = np.random.default_rng(123)
    true_mean, trials, hits = 5.0, 200, 0
    for t in range(trials):
        sample = rng.normal(true_mean, 2.0, size=40)
        est = bootstrap_ci(sample, resamples=2000, seed=t)
        hits += est.ci95[0] <= true_mean <= est.ci95[1]
    # Percentile bootstrap slightly under-covers at n=40; 88-99% is the honest band.
    assert 0.88 <= hits / trials <= 0.99


def test_bootstrap_ci_custom_statistic_and_single_sample():
    est = bootstrap_ci([1.0, 2.0, 3.0, 100.0], statistic=np.median, resamples=500)
    assert est.value == 2.5 and est.ci95[0] <= est.value <= est.ci95[1]
    one = bootstrap_ci([4.2])
    assert one == Estimate(4.2, (4.2, 4.2), 1)


def test_fit_scaling_exponent_recovers_planted_exponent():
    rng = np.random.default_rng(7)
    sizes = np.repeat(np.geomspace(10, 1e5, 8), 5)
    costs = 3 * sizes**1.7 * rng.lognormal(0, 0.1, size=sizes.size)
    est = fit_scaling_exponent(sizes, costs, resamples=2000)
    assert est.ci95[0] <= 1.7 <= est.ci95[1]
    assert abs(est.value - 1.7) < 0.05


def test_fit_scaling_exponent_rejects_bad_input():
    with pytest.raises(ValueError):
        fit_scaling_exponent([10, 10, 10], [1, 2, 3])
    with pytest.raises(ValueError):
        fit_scaling_exponent([1, 2, 3], [1, 0, 3])


def test_paired_wilcoxon_identical_inputs():
    a = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert paired_wilcoxon(a, a) == (1.0, 0.0)


def test_paired_wilcoxon_detects_consistent_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.1, size=10)
    p, delta = paired_wilcoxon(a, a + 0.5)
    assert p < 0.01 and delta < -0.9
    with pytest.raises(ValueError):
        paired_wilcoxon([1, 2], [1, 2, 3])


def test_cliffs_delta_extremes():
    assert cliffs_delta([5, 6], [1, 2]) == 1.0
    assert cliffs_delta([1, 2], [5, 6]) == -1.0


def test_everything_deterministic_given_seed():
    x = np.random.default_rng(1).exponential(size=12)
    assert bootstrap_ci(x, seed=3) == bootstrap_ci(x, seed=3)
    s, c = [10, 100, 1000, 10000], [5, 60, 700, 9000]
    assert fit_scaling_exponent(s, c, seed=3) == fit_scaling_exponent(s, c, seed=3)
