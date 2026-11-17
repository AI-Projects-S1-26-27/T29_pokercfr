# 1. Introduction

Counterfactual regret minimization (CFR) is the backbone of every strong computer poker agent,
from heads-up limit hold'em solvers to multiplayer play (Zinkevich et al., 2007; Bowling et al.,
2015; Brown & Sandholm, 2019). Much is known about how fast CFR variants converge on a given
game. Much less is known about *which property of a game makes CFR expensive*: the number of
information sets, the number of deals, the number of players, the depth of the betting tree, or
the raise cap. Published comparisons usually change several of these at once, so a result such as
"CFR+ is faster" cannot be attributed to any one of them.

This project isolates those properties on games small enough to solve exactly, so every reported
number is a true exploitability or NashConv rather than a sampled estimate. We build a ladder of
poker games that varies one property at a time (a Kuhn-family ladder with configurable deck and
player count, and Leduc Hold'em with a configurable raise cap and N players on a fixed deck), a
single solver with an eight-corner ablation cube over regret matching, averaging and update order,
and an exact best-response evaluator validated against OpenSpiel.

We pre-registered three hypotheses before running the final sweeps (see `preregistration.md`):

- **H1.** At equal wall-clock budget, CFR+ reaches lower exploitability than vanilla CFR on
  2-player Leduc, at every raise cap tested.
- **H2.** Cost to reach a fixed epsilon grows polynomially in information-set count on the
  2-player ladder, with a smaller exponent for CFR+.
- **H3.** With the deck held fixed, per-seat NashConv at matched budget rises with player count.

H3 matters because OpenSpiel's multiplayer Leduc grows the deck as 2(N+1) with N, which confounds
player count with card distribution; holding the deck fixed separates them.

_Results are in Section 4; limitations, including cells that did not finish, are in Section 5._
