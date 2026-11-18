"""The results matrix: every run in the study, pre-registered.

OWNER: Aarush (the design and the epsilon/budget decisions). Each named matrix
below belongs to the member who runs it, listed in its docstring.

WHY PRE-REGISTRATION. Epsilon targets, budgets, seeds and raise caps are fixed
HERE, committed, and only then are the sweeps run. That is what stops the study
drifting into "we tried things until a plot looked good", and it is the
difference between an experimental design and a pile of runs.

WHY THE MATRICES ARE THIS BIG. Measured single-core cost of one CFR+ iteration:

    kuhn                     0.00035 s     4rank-2suit-3p          0.083 s
    4rank                    0.00066 s     4rank-2suit-2dealt-3p   0.675 s
    4rank-2suit              0.00284 s     4rank-2suit-4p          1.620 s
    4rank-2suit-2dealt       0.02349 s

Cost is driven by the number of DEALS, not information sets, so it explodes along
the player-count axis: one 100k-iteration run on `4rank-2suit-4p` is ~45 CPU-hours.
Each matrix below is sized to roughly 100-150 CPU-hours, which is 13-19 hours of
wall-clock on one laptop using 8 cores. Nothing here is padding -- every
dimension buys something the study needs:

  more seeds       -> confidence intervals the sampled algorithms actually need
  more iterations  -> asymptotic-rate claims, alpha-drift, and tighter epsilon
  upper rungs      -> the large end of the size axis, without which H2's scaling
                      exponent is fitted on two orders of magnitude and means little
  more deals       -> head-to-head CIs that exclude zero

Run yours with:  python -m pokercfr.experiments.runner --matrix <name> --estimate
then drop --estimate to launch it. Runs checkpoint and resume, so a closed lid
costs minutes rather than hours.
"""
from __future__ import annotations

from pokercfr.core.config import RunSpec

# Pre-registered epsilon targets, chips/hand. Fixed before the final runs.
# For N >= 3 the target applies to the per-seat MEAN NashConv, never the total:
# the total grows with N by construction (see eval/exploitability.py).
EPSILON = {"default": 1e-2, "kuhn": 1e-4, "leduc-2p": 1e-2, "multiplayer": 5e-2}


def epsilon_for(game: str) -> float:
    """The pre-registered epsilon target for a rung name."""
    if game == "kuhn":
        return EPSILON["kuhn"]
    if game.startswith("leduc-"):
        players = int(game.split("-")[1].rstrip("p"))
        return EPSILON["leduc-2p"] if players == 2 else EPSILON["multiplayer"]
    if game.endswith(("-3p", "-4p")):
        return EPSILON["multiplayer"]
    return EPSILON["default"]


# H1 -- the primary hypothesis, stated in WALL-CLOCK terms. Deterministic solvers
# ignore their seed, but a wall-clock run's iteration count depends on machine
# timing, so seeds here are independent timing REPLICATES, and the CFR vs CFR+
# comparison is paired by seed.
H1_CAPS = (1, 2, 3)
H1_SEEDS = tuple(range(5))
H1_TIME_BUDGET = 600.0          # training seconds per cell, evaluation excluded

# H3 on Leduc, deck held fixed at 6 cards. Sized from the measured census in
# results/verify/leduc_sizes.json: 4p cap2 is ~105 s/iter and 5p cap2 ~20 min/iter
# (both projected), and 6p is infeasible -- so cap 2 stops at 3 players and the
# deep end uses cap 1. 5p cap1 (~38 s/iter, projected) is a registered STRETCH
# cell expected to finish only a few hundred iterations; it is reported as partial.
H3_LEDUC_CELLS = ((2, 1), (3, 1), (4, 1), (2, 2), (3, 2))      # (players, raise cap)
H3_LEDUC_STRETCH = ((5, 1),)
H3_VARIANTS = ("cfr", "cfr+", "linear-cfr")
H3_SEEDS = (0, 1, 2)
H3_TIME_BUDGET = 3600.0
H3_STRETCH_TIME_BUDGET = 21_600.0
LEDUC_MCCFR_TIME_BUDGET = 600.0

# Log-spaced so a log-log convergence plot has evenly spaced points -- which
# matters because the SLOPE is what we report, and uneven spacing biases the eye.
def checkpoints(upto: int) -> tuple[int, ...]:
    """Log-spaced, always ending exactly on `upto`.

    The budget itself has to be the final checkpoint: training stops at the last
    one, so without it a budget that is not {1,2,5}x10^n silently truncates --
    a declared 3,000 iterations would have run 2,000.
    """
    out, t = [], 1
    while t <= upto:
        out.extend([t, 2 * t, 5 * t])
        t *= 10
    out.append(upto)
    return tuple(c for c in sorted(set(out)) if c <= upto)


TWO_PLAYER_LADDER = ("kuhn", "4rank", "4rank-2suit", "4rank-2suit-2dealt")
MULTIPLAYER_LADDER = ("kuhn-3p", "4rank-2suit-3p", "4rank-2suit-2dealt-3p", "4rank-2suit-4p")

# Deterministic solvers ignore their seed, so running them on 10 seeds would
# produce 10 identical files. Only sampled algorithms get the seed sweep.
SAMPLED_SEEDS = tuple(range(10))

# Per-rung iteration budgets, chosen so each run lands in the 1-7 CPU-hour band.
BUDGET = {
    "kuhn": 2_500_000, "4rank": 1_000_000, "4rank-2suit": 500_000,
    "4rank-2suit-2dealt": 200_000, "kuhn-3p": 500_000, "4rank-2suit-3p": 65_000,
    "4rank-2suit-2dealt-3p": 8_000, "4rank-2suit-4p": 3_000,
    "13rank-4suit-2dealt-2p": 500, "6rank-4suit-2dealt-4p": 2_500,
}

MAX_HOURS_PER_RUN = 7.0

DCFR_GRID_ITERATIONS = 50_000



def _spec(game: str, variant: str, seed: int = 0, iterations: int | None = None,
          time_budget: float | None = None) -> RunSpec:
    """One cell. With `time_budget` the run is wall-clock-budgeted and `iterations`
    is only a ceiling; the budget lives in `extra`, where the runner reads it."""
    n = iterations or BUDGET.get(game, 10_000)
    extra = {"time_budget": time_budget} if time_budget else {}
    return RunSpec(game=game, variant=variant, seed=seed, iterations=n,
                   max_seconds=MAX_HOURS_PER_RUN * 3600, checkpoints=checkpoints(n),
                   extra=extra)


def cube_deep() -> list[RunSpec]:
    """OWNER: Himnish. R1 at depth, plus the DCFR parameter grid.

    The 8 cube corners on the two-player ladder, run far enough that the
    asymptotic rate is visible rather than inferred -- CFR+ is known to beat its
    own O(1/sqrt(T)) bound, and you cannot show that from 10k iterations.

    The DCFR grid tests Brown & Sandholm's claim that their parameter family
    UNIFIES CFR and CFR+: at the limits it should reproduce both measured curves.
    If it does, that is independent corroboration of our cube. If it does not, the
    discrepancy is the more interesting result.
    """
    from pokercfr.solvers.cfr import ablation_cube

    specs = [_spec(rung, label) for rung in TWO_PLAYER_LADDER for label in ablation_cube()]
    for alpha in (1.0, 1.5, 2.0, 3.0):
        for beta in (0.0, 0.5, 1.0):
            for gamma in (1.0, 2.0, 3.0):
                label = f"dcfr-a{alpha:g}-b{beta:g}-g{gamma:g}"
                for rung in ("4rank", "4rank-2suit", "4rank-2suit-2dealt"):
                    specs.append(_spec(rung, label, iterations=DCFR_GRID_ITERATIONS))

    # Every DCFR setting above uses simultaneous updates, because that is what
    # the discount family can express. No DCFR setting reached CFR+ in the
    # step-2 check, and the suspected reason is that CFR+'s advantage comes
    # mostly from alternating updates, which sits outside the family. Pairing
    # the discount schedule WITH alternating updates separates those two
    # explanations: if the gap closes, alternating updates was the cause.
    for alpha, beta, gamma in ((1.5, 0.0, 2.0), (2.0, 0.0, 1.0), (3.0, 0.0, 1.0)):
        label = f"dcfr-alt-a{alpha:g}-b{beta:g}-g{gamma:g}"
        for rung in ("4rank", "4rank-2suit", "4rank-2suit-2dealt"):
            specs.append(_spec(rung, label))   # the rung's own budget, as cfr+ uses
    return specs


def leduc_sweep() -> list[RunSpec]:
    """OWNER: Samanyu. H1 (raise caps) and H3 (player count, deck held fixed).

    H1 is the project's primary pre-registered hypothesis and it is stated in
    WALL-CLOCK terms, so these cells carry a time budget, not an iteration budget.
    Comparing at equal iterations would flatter the alternating corners, which do
    one traversal per player and cost more per iteration. The raise-cap axis
    exists because a claim that holds only at the standard cap of 2 is a claim
    about one game, not about the algorithm.

        H1: 3 caps x 8 corners x 5 timing replicates, 600 s each  (~20 CPU-h)

    The player-count axis holds the deck FIXED, which is the whole novelty claim --
    OpenSpiel grows Leduc's deck as 2(N+1) with player count, so its multiplayer
    results cannot separate "more players" from "different cards". Sized from the
    measured census (see H3_LEDUC_CELLS), not from what we would like to run.

        H3: 5 game cells x 3 variants x 3 replicates, 3600 s each (~45 CPU-h)
            + the 5-player cap-1 stretch cell, 1 replicate x 3 variants, 6 h
    """
    from pokercfr.solvers.cfr import ablation_cube

    specs = []
    for cap in H1_CAPS:
        for label in ablation_cube():
            for seed in H1_SEEDS:
                specs.append(_spec(f"leduc-2p-cap{cap}", label, seed=seed,
                                   iterations=1_000_000, time_budget=H1_TIME_BUDGET))
    for players, cap in H3_LEDUC_CELLS:
        for label in H3_VARIANTS:
            for seed in H3_SEEDS:
                specs.append(_spec(f"leduc-{players}p-cap{cap}", label, seed=seed,
                                   iterations=1_000_000, time_budget=H3_TIME_BUDGET))
    for players, cap in H3_LEDUC_STRETCH:
        for label in H3_VARIANTS:
            specs.append(_spec(f"leduc-{players}p-cap{cap}", label, seed=0,
                               iterations=1_000_000, time_budget=H3_STRETCH_TIME_BUDGET))
    return specs


def multiplayer() -> list[RunSpec]:
    """OWNER: Aarush. H3 on the Kuhn family, where exact NashConv stays computable.

    These are the most expensive rungs in the study (one 4-player run is hours),
    and they are what H2's scaling exponent needs at the large end: a slope fitted
    only over tiny games is an extrapolation dressed up as a measurement.

    Reports NashConv per SEAT, not just the total. The total necessarily grows
    with player count, so quoting it alone would make the multiplayer gap look
    worse simply because there are more deviators -- the per-seat decomposition is
    what makes the H3 claim honest, and it also answers a question nobody asks:
    whether position determines who has the most incentive to deviate.
    """
    from pokercfr.solvers.cfr import ablation_cube

    # The discounting variants are included here as well as on the two-player
    # ladder, because multiplayer is exactly where CFR has no convergence
    # guarantee -- so whether DCFR's discount schedule helps or hurts once the
    # guarantee is gone is an open question rather than a repeat measurement.
    variants = list(ablation_cube()) + [
        f"dcfr-a{a:g}-b{b:g}-g{g:g}"
        for a, b, g in ((1.5, 0.0, 2.0), (1.0, 0.5, 2.0), (2.0, 0.0, 1.0),
                        (3.0, 1.0, 3.0), (1.5, 0.5, 1.0), (1.0, 0.0, 3.0))
    ]
    return [_spec(rung, label) for rung in MULTIPLAYER_LADDER for label in variants]


def mccfr_variance() -> list[RunSpec]:
    """OWNER: Vikranth. Sampling schemes and the variance they actually cost.

    MCCFR trades per-iteration cost for variance, so it needs orders of magnitude
    more iterations than full-traversal CFR to reach the same exploitability --
    which is exactly the trade-off being measured, and why these budgets are the
    largest in the study. Ten seeds per cell because a single-seed MCCFR number is
    not a result, it is one draw from a distribution.

    The textbook claim is that external sampling has lower variance than outcome
    sampling. We verify it rather than cite it, on games where the truth is known.
    """
    specs = []
    for scheme in ("outcome", "external", "chance"):
        for rung in ("kuhn", "4rank", "4rank-2suit", "4rank-2suit-2dealt", "kuhn-3p"):
            for seed in SAMPLED_SEEDS:
                specs.append(_spec(rung, f"mccfr-{scheme}", seed=seed, iterations=10_000_000))
    # The sampled algorithms also run on Leduc (2-player, cap 2), wall-clock
    # budgeted like H1 so they are directly comparable with its CFR/CFR+ cells.
    for scheme in ("outcome", "external", "chance"):
        for seed in SAMPLED_SEEDS:
            specs.append(_spec("leduc-2p-cap2", f"mccfr-{scheme}", seed=seed,
                               iterations=100_000_000, time_budget=LEDUC_MCCFR_TIME_BUDGET))
    return specs


def top_rung() -> list[RunSpec]:
    """OWNER: Siddhant. The largest game we can solve, plus the exact LP oracle.

    `13rank-4suit-2dealt-2p` is a real 52-card deck: 5,304 information sets but
    1.6 MILLION deals, so full traversal costs an estimated ~90 s per iteration in
    pure Python -- around 25 hours for a single 1,000-iteration run. That number
    is the justification for vectorizing the traversal over hands with numpy, and
    the measured speedup is itself a reportable result.

    The honest sequence: measure the pure-Python baseline on a handful of
    iterations, vectorize, verify the vectorized traversal produces IDENTICAL
    regrets to the scalar one, then run the rung properly. Correctness first --
    a fast traversal that computes the wrong thing is worse than a slow one.
    """
    from pokercfr.solvers.cfr import ablation_cube

    specs = [_spec("13rank-4suit-2dealt-2p", label, iterations=1_000)
             for label in ablation_cube()]
    specs += [_spec(rung, "lp-oracle", iterations=1) for rung in TWO_PLAYER_LADDER]
    return specs


def matched_budget() -> list[RunSpec]:
    """OWNER: Shashwat. Cross-play at MATCHED compute, not matched iterations.

    The expensive half of a tournament is not playing the hands, it is producing
    the players: every agent has to be trained at each compute budget before any
    match happens. Six budgets spanning three orders of magnitude, so the
    cross-play matrix can answer a question exploitability cannot -- whether a
    lower-exploitability strategy actually WINS more against real opponents.

    It often does not, and that gap is the point. A strategy can beat every
    baseline in the pool and still be heavily exploitable; demonstrating that
    separation is what keeps head-to-head results honest as supporting evidence
    rather than passing them off as a measure of equilibrium quality.
    """
    from pokercfr.solvers.cfr import ablation_cube

    specs = []
    for budget in (1_000, 10_000, 100_000, 1_000_000, 5_000_000):
        for rung in ("kuhn", "4rank", "4rank-2suit", "4rank-2suit-2dealt"):
            for label in ablation_cube():
                specs.append(_spec(rung, label, iterations=budget))
    return specs


def alpha_manifold() -> list[RunSpec]:
    """OWNER: Akhil. Where on Kuhn's equilibrium continuum does each variant land?

    Kuhn's Nash equilibria are not a point but a one-parameter family, indexed by
    the first player's bluffing frequency alpha in [0, 1/3]. Every algorithm here
    converges to SOME equilibrium, so exploitability alone cannot distinguish
    them -- but they need not pick the same one.

    Open questions this answers: does each variant lock onto a single alpha or
    drift along the manifold forever? Do the sampled algorithms land at a
    different alpha than the deterministic ones, and with how much seed spread?
    Nobody reports this, it needs very long runs to see drift rather than
    transient wandering, and it makes an unusually good live visual.
    """
    from pokercfr.solvers.cfr import ablation_cube

    # 50M iterations: drift along the manifold is a slow effect, and at 1M what
    # you see is still the transient. The run length IS the experiment here.
    specs = [_spec(rung, label, iterations=50_000_000)
             for rung in ("kuhn", "4rank") for label in ablation_cube()]
    specs += [_spec("kuhn", f"mccfr-{scheme}", seed=seed, iterations=50_000_000)
              for scheme in ("outcome", "external") for seed in SAMPLED_SEEDS]
    return specs


def determinism_audit() -> list[RunSpec]:
    """OWNER: Sai Aashish. Reproducibility as a measurement, not a promise.

    Re-runs a cross-section of every other matrix on a different machine so the
    results can be diffed bit for bit. Two things must hold, and both are claims
    the report makes: a deterministic solver gives identical output regardless of
    seed, and an identical RunSpec gives identical numbers across machines up to
    float associativity. Verifying that is inherently as expensive as the runs
    being verified, which is why this is a full workload rather than a checkbox.

    Also owns the feasibility frontier: push each rung until it exceeds the time
    budget and record where. "How far can each method go on this hardware" was
    promised as a result in the proposal, and it only exists if someone actually
    walks the ladder until it breaks.
    """
    specs = []
    for matrix in (cube_deep, multiplayer, mccfr_variance):
        specs.extend(matrix()[::4])  # every 4th cell, across the whole range
    return specs


def determinism_audit_4h() -> list[RunSpec]:
    """A bounded deterministic-only audit sized for about four hours wall time.

    Keeps a paired CFR/CFR+ sample across the two-player ladder and one small
    multiplayer rung. MCCFR is omitted while its solver is unimplemented. Budgets
    are deliberately below the full audit so this matrix can run locally.
    Re-estimate on the target machine before launching; CPU contention changes
    elapsed wall time.
    """
    cells = (
        ("kuhn", 22_500_000), ("4rank", 9_000_000),
        ("4rank-2suit", 4_500_000), ("4rank-2suit-2dealt", 1_800_000),
        ("kuhn-3p", 4_500_000),
    )
    variants = ("cfr", "cfr+")
    return [_spec(rung, variant, iterations=iterations)
            for rung, iterations in cells for variant in variants]


# Keyed by matrix name; `full` is added after full_matrix is defined, and is
# deliberately NOT a member of this dict -- it iterates over it.
PER_PERSON = {
    "cube-deep": cube_deep,            # Himnish
    "leduc-sweep": leduc_sweep,        # Samanyu
    "multiplayer": multiplayer,        # Aarush
    "mccfr-variance": mccfr_variance,  # Vikranth
    "top-rung": top_rung,              # Siddhant
    "matched-budget": matched_budget,  # Shashwat
    "alpha-manifold": alpha_manifold,  # Akhil
    "determinism-audit": determinism_audit,  # Sai Aashish
}


def full_matrix() -> list[RunSpec]:
    """Every run in the study, deduplicated and in a stable order."""
    seen, out = set(), []
    for build in PER_PERSON.values():
        for spec in build():
            key = spec.spec_hash()
            if key not in seen:
                seen.add(key)
                out.append(spec)
    return sorted(out, key=lambda s: (s.game, s.variant, s.seed, s.iterations))


MATRICES = {**PER_PERSON, "determinism-audit-4h": determinism_audit_4h,
            "full": full_matrix}


def print_estimate(specs: list[RunSpec]) -> None:
    """Measure s/iter on each rung present, then project CPU-hours for these runs.

    Measured rather than guessed, because the projection is what people plan their
    night around. Rungs that do not exist yet (Leduc) are reported as unknown
    instead of being filled in with a number nobody checked.
    """
    import os
    import time

    from pokercfr.experiments.runner import build_solver

    # Probe each (rung, variant) pair. Probing one variant per rung would
    # understate every alternating-update corner by ~2x, since those do one
    # traversal per player per iteration -- and these projections are what people
    # plan an overnight run around.
    per_iter: dict[tuple[str, str], float | None] = {}
    for key in sorted({(s.game, s.variant) for s in specs}):
        probe = next(s for s in specs if (s.game, s.variant) == key)
        try:
            _, solver = build_solver(probe)
            solver.step()  # warm up: the first iteration allocates the tables
            n = 3
            t0 = time.perf_counter()
            for _ in range(n):
                solver.step()
            per_iter[key] = (time.perf_counter() - t0) / n
        except Exception:
            per_iter[key] = None

    total, unknown = 0.0, 0
    by_rung: dict[str, float] = {}
    not_built: dict[str, int] = {}
    for spec in specs:
        cost = per_iter.get((spec.game, spec.variant))
        if cost is None:
            not_built[spec.game] = not_built.get(spec.game, 0) + 1
            unknown += 1
            continue
        # A wall-clock-budgeted cell costs its budget by definition; an iteration
        # cell costs iterations x measured s/iter, capped at the per-run limit.
        budget = spec.extra.get("time_budget")
        hours = (budget if budget else min(spec.iterations * cost, MAX_HOURS_PER_RUN * 3600)) / 3600
        by_rung[spec.game] = by_rung.get(spec.game, 0.0) + hours
        total += hours

    print(f"{'rung':<26}{'runs':>6}{'CPU-hours':>12}")
    for rung in sorted(set(by_rung) | set(not_built)):
        n = len([s for s in specs if s.game == rung])
        if rung in by_rung:
            print(f"{rung:<26}{n:>6}{by_rung[rung]:>12.1f}")
        else:
            print(f"{rung:<26}{n:>6}{'not built':>12}")
    cores = os.cpu_count() or 8
    print(f"\n{len(specs)} runs, {total:.0f} CPU-hours measured"
          + (f" (+{unknown} runs not yet buildable)" if unknown else ""))
    print(f"on this machine's {cores} cores: ~{total / cores:.1f} h wall-clock "
          f"(capped at {MAX_HOURS_PER_RUN:g} h per run)")
