"""The CFR family as ONE traversal with orthogonal switches.

CFR+ is not a single algorithm -- Tammelin (2014) bundles three independent
changes on top of vanilla CFR:

    1. regret-matching+   clamp cumulative regret at zero after every update,
                          so an action that was bad early can be re-tried the
                          moment it stops being bad
    2. linear averaging   weight iteration t by t in the average strategy, so
                          early (bad) iterates are discounted
    3. alternating        update one player per traversal instead of all at once
       updates

The literature reports the bundle. This module exposes the three as independent
booleans, so the 2**3 = 8 corners of the cube are all reachable from one code
path -- (F,F,F) is exactly vanilla CFR and (T,T,T) is exactly CFR+. That is the
factorial ablation (R1): because there is one traversal, any measured difference
between corners is the update rule and cannot be an implementation artifact.

Discounted CFR (Brown & Sandholm 2019) is the same traversal with its own
discount schedule; set `dcfr` to enable it.

IMPORTANT -- what one `step()` means
------------------------------------
A step always updates EVERY player once, so iteration counts stay comparable
across variants. Simultaneous does that in 1 traversal; alternating needs
`num_players` traversals and so costs ~N x more per step. That is exactly why
every result is reported against wall-clock as well as iterations: measuring
alternating updates per-iteration only would flatter them.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from pokercfr.core.protocols import Game, GameState, Policy

AVERAGING_SCHEMES = ("uniform", "linear", "quadratic")


@dataclass(frozen=True)
class CFRConfig:
    """One corner of the ablation cube (plus optional DCFR)."""

    regret_matching_plus: bool = False
    averaging: str = "uniform"
    alternating: bool = False
    # Discounted CFR (alpha, beta, gamma). Paper's recommended values are
    # (1.5, 0.0, 2.0). None disables discounting entirely.
    dcfr: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.averaging not in AVERAGING_SCHEMES:
            raise ValueError(f"averaging must be one of {AVERAGING_SCHEMES}, got {self.averaging!r}")

    @property
    def label(self) -> str:
        """Stable short name used in result filenames and plot legends."""
        if self.dcfr is not None:
            a, b, g = self.dcfr
            return f"dcfr(a={a:g},b={b:g},g={g:g})"
        for name, cfg in NAMED_VARIANTS.items():
            if cfg == self:
                return name
        rm = "rm+" if self.regret_matching_plus else "rm"
        alt = "alt" if self.alternating else "sim"
        return f"{rm}/{self.averaging}/{alt}"


VANILLA_CFR = CFRConfig()
CFR_PLUS = CFRConfig(regret_matching_plus=True, averaging="linear", alternating=True)
LINEAR_CFR = CFRConfig(averaging="linear")
DCFR = CFRConfig(averaging="linear", dcfr=(1.5, 0.0, 2.0))

# The paper's stated limits of the discount family. DCFR cannot express
# alternating updates, so its CFR+ limit is the rm+/linear/sim corner of the
# cube rather than cfr+ itself.
DCFR_AS_CFR = CFRConfig(averaging="uniform", dcfr=(math.inf, math.inf, 0.0))
DCFR_AS_CFR_PLUS = CFRConfig(averaging="linear", dcfr=(math.inf, -math.inf, 0.0))

NAMED_VARIANTS = {"cfr": VANILLA_CFR, "cfr+": CFR_PLUS, "linear-cfr": LINEAR_CFR}


def ablation_cube() -> dict[str, CFRConfig]:
    """All 8 corners of the (rm+, averaging, alternating) cube, keyed by label.

    Averaging is binary here (uniform vs linear) so the design is a clean 2**3
    factorial; "quadratic" stays available for a separate one-factor sweep.
    """
    cube = {}
    for rm_plus in (False, True):
        for averaging in ("uniform", "linear"):
            for alternating in (False, True):
                cfg = CFRConfig(
                    regret_matching_plus=rm_plus, averaging=averaging, alternating=alternating
                )
                cube[cfg.label] = cfg
    return cube


def _discount(t: int, exponent: float) -> float:
    """Brown & Sandholm's t^x/(t^x+1) discount, with the paper's limits exact.

    alpha=beta=+inf means no discounting and beta=-inf clamps negative regret to
    zero, so both are returned exactly rather than approximated by a large
    finite exponent -- which overflows.
    """
    if exponent == math.inf:
        return 1.0
    if exponent == -math.inf:
        return 0.0
    try:
        p = float(t) ** exponent
    except OverflowError:
        return 1.0
    return 1.0 if math.isinf(p) else p / (p + 1.0)


def _float_dict() -> defaultdict:
    """Module-level factory so the regret/strategy tables stay PICKLABLE.

    `defaultdict(lambda: defaultdict(float))` is the obvious spelling and it
    cannot be pickled -- the lambda is a local object. That matters because runs
    on the upper rungs are 10+ hours and checkpoint by pickling the solver, so
    an unpicklable table silently costs a whole night's work.
    """
    return defaultdict(float)


def regret_matching(regrets: dict[str, float], actions: list[str]) -> dict[str, float]:
    """Strategy proportional to positive regret; uniform if nothing is positive."""
    positive = {a: max(regrets.get(a, 0.0), 0.0) for a in actions}
    total = sum(positive.values())
    if total <= 0.0:
        return {a: 1.0 / len(actions) for a in actions}
    return {a: positive[a] / total for a in actions}


class CFRSolver:
    """Tabular CFR over any Game satisfying the frozen protocol."""

    def __init__(self, game: Game, config: CFRConfig = VANILLA_CFR):
        self.game = game
        self.config = config
        self.num_players = game.num_players
        self.iterations = 0
        self.regret_sum: dict[str, dict[str, float]] = defaultdict(_float_dict)
        self.strategy_sum: dict[str, dict[str, float]] = defaultdict(_float_dict)
        self._nodes_touched = 0

    # ---- the public Solver interface -------------------------------------

    def step(self) -> None:
        """One iteration: every player updated exactly once.

        Regret deltas accumulate into a scratch buffer and are committed only
        once the traversal finishes. That is load-bearing: CFR requires the
        iteration's strategy sigma^t to be derived from R^(t-1) and held FIXED
        for the whole iteration. Writing regrets mid-traversal would let the
        second deal of an iteration play a different strategy from the first
        (our root enumerates every deal), which still converges but is not the
        algorithm and measurably slows it down.
        """
        self.iterations += 1
        self._apply_dcfr_discount()
        root = self.game.new_initial_state()
        players = range(self.num_players) if self.config.alternating else [None]
        for update_player in players:
            deltas: dict[str, dict[str, float]] = defaultdict(_float_dict)
            self._traverse(root, [1.0] * self.num_players, 1.0, update_player, deltas)
            self._commit(deltas)

    def _commit(self, deltas: dict[str, dict[str, float]]) -> None:
        """Fold one traversal's regret deltas into the cumulative sums.

        Regret-matching+ clamps the CUMULATIVE regret (Tammelin 2014), not the
        per-iteration delta, so the clamp belongs here and not in the traversal.
        """
        for key, delta in deltas.items():
            regrets = self.regret_sum[key]
            for a, dv in delta.items():
                total = regrets[a] + dv
                regrets[a] = max(total, 0.0) if self.config.regret_matching_plus else total

    def train(self, iterations: int) -> CFRSolver:
        for _ in range(iterations):
            self.step()
        return self

    def average_strategy(self) -> Policy:
        """Time-averaged strategy -- the one with the Nash guarantee (2p 0-sum)."""
        return {key: _normalize(sums) for key, sums in self.strategy_sum.items()}

    def current_strategy(self) -> Policy:
        """The latest iterate. Reported separately because it can cycle forever
        even while the average converges -- a gap we measure rather than hide."""
        return {
            key: regret_matching(regrets, sorted(regrets))
            for key, regrets in self.regret_sum.items()
        }

    # ---- internals --------------------------------------------------------

    @property
    def averaging_weight(self) -> float:
        t = self.iterations
        if self.config.averaging == "uniform":
            return 1.0
        if self.config.averaging == "linear":
            return float(t)
        return float(t * t)  # quadratic

    def _apply_dcfr_discount(self) -> None:
        """DCFR (Brown & Sandholm 2019): discount accumulated regret and strategy
        before iteration t, positive and negative regret at different rates."""
        if self.config.dcfr is None:
            return
        alpha, beta, gamma = self.config.dcfr
        t = self.iterations
        if t <= 1:
            return
        pos_factor = _discount(t, alpha)
        neg_factor = _discount(t, beta)
        strat_factor = ((t - 1) / t) ** gamma
        for regrets in self.regret_sum.values():
            for a, r in regrets.items():
                regrets[a] = r * (pos_factor if r > 0 else neg_factor)
        for sums in self.strategy_sum.values():
            for a in sums:
                sums[a] *= strat_factor

    def _traverse(
        self,
        state: GameState,
        reach: list[float],
        chance_reach: float,
        update_player: int | None,
        deltas: dict[str, dict[str, float]],
    ) -> list[float]:
        """Returns the expected utility vector at `state`.

        `update_player` is None for simultaneous updates (update everyone) or a
        seat index for alternating updates (update only that seat).
        """
        self._nodes_touched += 1

        if state.is_terminal():
            return state.utilities()

        if state.is_chance_node():
            total = [0.0] * self.num_players
            for child, prob in state.chance_outcomes():
                values = self._traverse(child, reach, chance_reach * prob, update_player, deltas)
                for i in range(self.num_players):
                    total[i] += prob * values[i]
            return total

        player = state.current_player()
        actions = state.legal_actions()
        key = state.information_set_key(player)
        strategy = regret_matching(self.regret_sum[key], actions)

        action_values: dict[str, list[float]] = {}
        for a in actions:
            child_reach = list(reach)
            child_reach[player] *= strategy[a]
            action_values[a] = self._traverse(
                state.apply_action(a), child_reach, chance_reach, update_player, deltas
            )

        node_value = [
            sum(strategy[a] * action_values[a][i] for a in actions) for i in range(self.num_players)
        ]

        if update_player is None or update_player == player:
            # Counterfactual reach: chance x everyone *except* the acting player.
            # This is what makes the regret "counterfactual" -- it weights by the
            # probability of reaching this info set if the player had tried to.
            cf_reach = chance_reach * math.prod(r for i, r in enumerate(reach) if i != player)
            delta = deltas[key]
            for a in actions:
                delta[a] += cf_reach * (action_values[a][player] - node_value[player])
            weight = self.averaging_weight * reach[player]
            sums = self.strategy_sum[key]
            for a in actions:
                sums[a] += weight * strategy[a]

        return node_value


def _normalize(sums: dict[str, float]) -> dict[str, float]:
    total = sum(sums.values())
    if total <= 0.0:
        return {a: 1.0 / len(sums) for a in sums}
    return {a: v / total for a, v in sums.items()}


def policy_expected_values(game: Game, policy: Policy) -> list[float]:
    """Each seat's expected payoff when every seat follows `policy`.
    Used for the Kuhn -1/18 ground-truth check and for zero-sum assertions."""

    def walk(state: GameState) -> list[float]:
        if state.is_terminal():
            return state.utilities()
        if state.is_chance_node():
            total = [0.0] * game.num_players
            for child, prob in state.chance_outcomes():
                values = walk(child)
                for i in range(game.num_players):
                    total[i] += prob * values[i]
            return total
        player = state.current_player()
        actions = state.legal_actions()
        probs = policy[state.information_set_key(player)]
        total = [0.0] * game.num_players
        for a in actions:
            values = walk(state.apply_action(a))
            for i in range(game.num_players):
                total[i] += probs[a] * values[i]
        return total

    return walk(game.new_initial_state())


__all__ = [
    "CFRConfig", "CFRSolver", "VANILLA_CFR", "CFR_PLUS", "LINEAR_CFR", "DCFR",
    "DCFR_AS_CFR", "DCFR_AS_CFR_PLUS", "NAMED_VARIANTS", "ablation_cube",
    "regret_matching", "policy_expected_values",
]
