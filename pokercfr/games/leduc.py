"""Leduc Hold'em -- OWNER: Samanyu. Primary testbed for H1 and H3.

Leduc (Southey et al., UAI 2005) is the standard mid-size benchmark and the only
rung with a *mid-tree* chance node, which is why the frozen protocol has
is_chance_node() at all:

    deal 1 private card each  ->  betting round 1  ->  reveal 1 public card
                              ->  betting round 2  ->  showdown

Standard Leduc: 6 cards = 3 ranks x 2 suits, ante 1, bet size 2 in round 1 and 4
in round 2, raise cap 2 per round. A pair with the public card beats any unpaired
hand; otherwise the higher private rank wins; equal ranks split.

TWO THINGS THIS MUST GET RIGHT -- they are the project's novelty claim
---------------------------------------------------------------------
1. `raise_cap` is a free parameter (H1 tests across caps), not hardcoded to 2.
2. `players` goes 2..6 with the DECK HELD FIXED. OpenSpiel grows Leduc's deck as
   2*(N+1) cards with player count, which confounds "more players" with "different
   card distribution". Holding the deck fixed is precisely what lets H3 attribute
   a rising NashConv plateau to player count alone. If the deck is too small to
   deal N players one card each, raise a clear error rather than silently resizing.

ACCEPTANCE (tests/test_leduc.py):
  - satisfies the Game/GameState protocols; probabilities sum to 1 at both chance nodes
  - utilities() sums to 0 at every terminal state
  - 2-player standard Leduc info-set count matches OpenSpiel's (load_game("leduc_poker"))
  - CFR+ reaches exploitability < 0.01 chips/hand within 2000 iterations
  - exploitability cross-checks against OpenSpiel's on the same policy
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeducConfig:
    ranks: int = 3
    suits: int = 2
    players: int = 2
    raise_cap: int = 2
    bet_sizes: tuple[int, int] = (2, 4)  # per betting round

    @property
    def deck_size(self) -> int:
        return self.ranks * self.suits


class LeducGame:
    """Satisfies pokercfr.core.protocols.Game."""

    def __init__(self, config: LeducConfig = LeducConfig(), name: str | None = None):
        raise NotImplementedError("OWNER: Samanyu -- see module docstring")


def game(players: int = 2, raise_cap: int = 2) -> LeducGame:
    """Convenience constructor: leduc.game(players=3, raise_cap=1)."""
    raise NotImplementedError("OWNER: Samanyu")
