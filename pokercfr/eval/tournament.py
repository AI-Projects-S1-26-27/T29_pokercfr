"""Seat-rotated duplicate-deal match runner -- OWNER: Shashwat.

Card luck swamps skill in poker at small sample sizes. The fix is duplicate
scoring, as used in bridge: play every deal twice with the seats swapped, so both
agents face identical cards. Reported chips/hand then reflects decisions rather
than who got dealt the king. With `n` deals and `players!` rotations the variance
drop is large enough that a few thousand deals give useful CIs.

ACCEPTANCE (tests/test_tournament.py):
  - two identical policies score 0.000 +/- float noise over rotated deals
    (the sharpest possible check that rotation is actually cancelling card luck)
  - swapping the agent order negates the result
  - a converged CFR+ policy beats uniform_random with a CI excluding 0
  - identical seed reproduces an identical result
"""
from __future__ import annotations

from dataclasses import dataclass

from pokercfr.core.protocols import Game, Policy


@dataclass(frozen=True)
class MatchResult:
    chips_per_hand: list[float]
    ci95: list[tuple[float, float]]
    hands_played: int

    def __str__(self) -> str:
        parts = [f"seat{i}: {v:+.4f} [{lo:+.4f},{hi:+.4f}]"
                 for i, (v, (lo, hi)) in enumerate(zip(self.chips_per_hand, self.ci95, strict=True))]
        return f"{self.hands_played} hands | " + " | ".join(parts)


def play_match(game: Game, policies: list[Policy], deals: int = 2000,
               seed: int = 0, rotate_seats: bool = True) -> MatchResult:
    """Play `deals` deals, each replayed in every seat rotation when rotate_seats."""
    raise NotImplementedError("OWNER: Shashwat -- see module docstring")


def cross_play(game: Game, named: dict[str, Policy], deals: int = 2000,
               seed: int = 0) -> dict[tuple[str, str], float]:
    """Round-robin matrix. Pairs exploitability (what a BR would win) against what
    agents actually win off each other -- the two often disagree, which is the point."""
    raise NotImplementedError("OWNER: Shashwat")
