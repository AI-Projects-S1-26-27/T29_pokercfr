"""The results matrix -- OWNER: Aarush. One place that defines every run.

This is the pre-registration: epsilon targets, compute budgets, seeds and raise
caps are fixed HERE, committed, and then the sweeps are run. Fixing them in
advance is what stops the study drifting into "we tried things until a plot
looked good", and it is the difference between a sound experimental design and a
pile of runs.

Enumerating the matrix in code (rather than in a shell script per person) is also
what lets eight machines shard it without collisions: every run has a stable
identity from its RunSpec hash, so machine k simply takes every spec where
index % 8 == k and nobody coordinates anything.
"""
from __future__ import annotations

from pokercfr.core.config import RunSpec

# Pre-registered epsilon targets, chips/hand. Fixed before the final runs.
EPSILON = {"default": 1e-2, "kuhn": 1e-3, "leduc-2p": 1e-2}

SEEDS = tuple(range(10))  # 10 seeds; deterministic solvers ignore all but one

# Checkpoints for convergence curves: log-spaced so a log-log plot has even
# point spacing, which matters for reading a slope off the figure.
CHECKPOINTS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)

# R1: the 2**3 ablation, run on every rung small enough for full traversal.
ABLATION_RUNGS = ("kuhn", "4rank", "4rank-2dealt", "4rank-2suit", "4rank-2suit-2dealt")

# H3: player count with the deck held fixed.
PLAYER_RUNGS = ("kuhn-3p", "4rank-2suit-3p", "4rank-2suit-4p")


def ablation_matrix() -> list[RunSpec]:
    """R1 x the ladder. Deterministic variants need one seed, not ten."""
    from pokercfr.solvers.cfr import ablation_cube

    specs = []
    for rung in ABLATION_RUNGS:
        for label in ablation_cube():
            specs.append(RunSpec(game=rung, variant=label, seed=0,
                                 iterations=10_000, checkpoints=CHECKPOINTS))
    return specs


def full_matrix() -> list[RunSpec]:
    """Every run in the study, in a stable order so sharding is reproducible."""
    specs = list(ablation_matrix())
    # TODO(Aarush): + MCCFR x seeds, + Leduc raise caps (H1), + player counts (H3)
    return sorted(set(specs), key=lambda s: (s.game, s.variant, s.seed))
