"""Monte Carlo CFR -- OWNER: Vikranth.

Full-traversal CFR visits every deal every iteration. On the upper rungs that is
hopeless: `6rank-4suit-2dealt-4p` has 1.85 BILLION deals but only 8,832 info sets.
MCCFR (Lanctot et al., NeurIPS 2009) samples instead, trading per-iteration cost
for variance -- which is exactly why it needs >= 5 seeds and why seed variance is
a reported quantity rather than an afterthought.

Three sampling schemes, same regret bookkeeping, one class with a `scheme` switch
(same reasoning as the CFR ablation cube -- one code path so differences are the
sampling rule, not the implementation):

  "outcome"   sample ONE action at every node, importance-weight by 1/reach.
              Cheapest per iteration, highest variance.
  "external"  sample chance and the OPPONENTS' actions, enumerate the traverser's.
              The usual default: much lower variance, modest extra cost.
  "chance"    sample only the deal, enumerate all betting. Sits between full CFR
              and outcome sampling, and is the natural fit for our ladder, where
              deals explode but info sets stay small.

MUST reuse CFRConfig from pokercfr.solvers.cfr so the ablation flags compose with
sampling (an MCCFR+ corner is then free). Must accept an explicit seed and be
bit-for-bit reproducible given one.

ACCEPTANCE (tests/test_mccfr.py):
  - all three schemes drive Kuhn exploitability < 0.03 within 50k iterations
  - the same seed produces an identical policy twice; different seeds differ
  - chance-sampling on `4rank-2suit` lands within 2 std-errors of full CFR
  - external sampling shows lower seed variance than outcome sampling (the
    textbook claim -- verify it rather than assert it)
"""
from __future__ import annotations

import random

from pokercfr.core.protocols import Game, Policy
from pokercfr.solvers.cfr import VANILLA_CFR, CFRConfig

SCHEMES = ("outcome", "external", "chance")


class MCCFRSolver:
    """Satisfies pokercfr.core.protocols.Solver."""

    def __init__(self, game: Game, config: CFRConfig = VANILLA_CFR,
                 scheme: str = "external", seed: int = 0):
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
        self.game, self.config, self.scheme = game, config, scheme
        self.rng = random.Random(seed)
        self.iterations = 0
        raise NotImplementedError("OWNER: Vikranth -- see module docstring")

    def step(self) -> None: ...
    def average_strategy(self) -> Policy: ...
    def current_strategy(self) -> Policy: ...
