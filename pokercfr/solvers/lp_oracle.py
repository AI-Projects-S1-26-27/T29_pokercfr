"""Sequence-form LP -- an INDEPENDENT exact Nash oracle. OWNER: Siddhant.

This is the project's strongest answer to "what does your code add beyond
re-running OpenSpiel". Everything else here is an iterative regret-minimiser that
*approaches* equilibrium; this solves for one exactly, by a completely different
route -- linear programming over the sequence form (Koller, Megiddo & von Stengel
1996; von Stengel 1996).

Two consequences, both load-bearing:

  1. VERIFICATION. A converged CFR policy and the LP solution must agree on the
     game value. Agreement across two unrelated algorithms is far stronger evidence
     of correctness than either one matching itself. Combined with Kuhn's closed
     form and OpenSpiel, headline numbers are checked three independent ways.

  2. A NEW ARTIFACT. It yields the EXACT equilibrium value of every 2-player rung
     of our ladder. Those games are ours, so those numbers exist nowhere else --
     and with exact values in hand, "iterations to reach epsilon" is measured
     against truth rather than against a long CFR run used as a proxy.

Formulation (2-player zero-sum, perfect recall). Realization plans live on
sequences, not behavioural probabilities, which makes the constraints LINEAR:
    maximise   v  subject to  payoff and information-set flow constraints
Build the sparse payoff matrix over terminal sequence pairs and each player's
constraint matrix (one row per info set: inflow from the parent sequence equals
outflow over that info set's actions), then hand it to scipy.optimize.linprog
with method="highs".

Scope honestly: this is for 2 players. N-player equilibrium computation is not an
LP (it is PPAD-complete), and the report must say so rather than imply the oracle
covers H3.

ACCEPTANCE (tests/test_lp_oracle.py):
  - Kuhn game value == -1/18 to 1e-9 (tighter than any CFR run reaches)
  - LP value agrees with 200k-iteration CFR+ to < 1e-4 on every 2-player rung
    up to `4rank-2suit-2dealt`
  - the LP policy's exploitability, measured by our own best_response, is ~0 (<1e-9)
  - raises a clear error for num_players > 2 rather than returning nonsense
"""
from __future__ import annotations

from pokercfr.core.protocols import Game, Policy


def solve_exact(game: Game) -> tuple[float, Policy]:
    """Exact equilibrium value (for seat 0) and an equilibrium policy."""
    raise NotImplementedError("OWNER: Siddhant -- see module docstring")
