"""Fixed-policy opponents -- OWNER: Shashwat.

The opponent pool from the proposal, weakest first. These are SUPPORTING evidence
only: head-to-head chips/hand is not a measure of equilibrium quality, and the
report must keep that distinction (a strategy can beat every baseline here and
still be wildly exploitable -- demonstrating that gap is itself a result worth
showing, and the demo makes it visible).

Each returns a Policy over the game's info sets, so the match runner, the
exploitability calculator and the demo all consume them identically.

ACCEPTANCE (tests/test_baselines.py):
  - every baseline yields a valid Policy: every reachable info set covered,
    probabilities non-negative and summing to 1
  - each is measurably exploitable (exploitability > 0) and a converged CFR+
    policy beats each one over seat-rotated duplicate deals
  - always_fold is the worst and uniform_random beats it
"""
from __future__ import annotations

from pokercfr.core.protocols import Game, Policy


def uniform_random(game: Game) -> Policy:
    """Mix uniformly over legal actions at every info set."""
    raise NotImplementedError("OWNER: Shashwat")


def always_pass(game: Game) -> Policy:
    """Always 'p' -- check when possible, fold when facing a bet. Maximally passive."""
    raise NotImplementedError("OWNER: Shashwat")


def always_bet(game: Game) -> Policy:
    """Always 'b' -- bet/call relentlessly. Maximally aggressive."""
    raise NotImplementedError("OWNER: Shashwat")


def threshold_rule(game: Game, threshold: float = 0.5) -> Policy:
    """Bet only with a hand in the top `threshold` fraction of possible holdings.
    A plausible human heuristic: never bluffs, so a solver should exploit it by
    folding to its bets -- a nice thing to show live."""
    raise NotImplementedError("OWNER: Shashwat")
