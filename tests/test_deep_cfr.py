"""The optional neural baseline must remain close to its exact-game target."""

from __future__ import annotations

import pytest

from pokercfr.eval.best_response import exploitability
from pokercfr.games.kuhn_family import game
from pokercfr.solvers.deep_cfr import DeepCFRSolver
from pokercfr.solvers.mccfr import MCCFRSolver


def test_function_approximation_has_small_kuhn_cost() -> None:
    pytest.importorskip("torch")
    g = game("kuhn")
    tabular = MCCFRSolver(g, scheme="external", seed=4).train(100)
    neural = DeepCFRSolver(g, seed=4).train(100)
    approximation_cost = exploitability(g, neural.average_strategy()) - exploitability(
        g, tabular.average_strategy()
    )
    assert approximation_cost < 0.01
