"""Acceptance tests for the three Monte Carlo CFR sampling schemes."""

from __future__ import annotations

from statistics import mean, stdev

from pokercfr.eval.best_response import exploitability
from pokercfr.games.kuhn_family import game
from pokercfr.solvers.cfr import CFRSolver
from pokercfr.solvers.mccfr import SCHEMES, MCCFRSolver


def test_all_schemes_converge_on_kuhn() -> None:
    g = game("kuhn")
    for scheme in SCHEMES:
        policy = MCCFRSolver(g, scheme=scheme, seed=3).train(50_000).average_strategy()
        assert exploitability(g, policy) < 0.03, scheme


def test_seed_controls_the_complete_sampling_trace() -> None:
    g = game("kuhn")
    first = MCCFRSolver(g, scheme="external", seed=99).train(200).average_strategy()
    again = MCCFRSolver(g, scheme="external", seed=99).train(200).average_strategy()
    other = MCCFRSolver(g, scheme="external", seed=100).train(200).average_strategy()
    assert first == again
    assert first != other


def test_chance_sampling_regrets_are_unbiased_on_4rank_2suit() -> None:
    """Chance-sampled regret estimate is within two empirical SEs of full CFR.

    This is the meaningful unbiasedness comparison: regret updates are linear
    estimators, whereas exploitability after many nonlinear regret-matching
    updates is not itself an unbiased estimator.
    """
    g = game("4rank-2suit")
    full = CFRSolver(g)
    full.step()
    key, action = "Ah", "b"  # fixed non-degenerate information set
    target = full.regret_sum[key][action]
    samples = []
    for seed in range(500):
        sampled = MCCFRSolver(g, scheme="chance", seed=seed)
        sampled.step()
        samples.append(sampled.regret_sum[key][action])
    standard_error = stdev(samples) / len(samples) ** 0.5
    assert abs(mean(samples) - target) <= 2 * standard_error


def test_external_has_lower_observed_seed_variance_than_outcome() -> None:
    g = game("kuhn")

    def scores(scheme: str) -> list[float]:
        return [
            exploitability(g, MCCFRSolver(g, scheme=scheme, seed=seed).train(3_000).average_strategy())
            for seed in range(5)
        ]

    assert stdev(scores("external")) < stdev(scores("outcome"))
