"""Monte Carlo CFR with outcome, external, and chance sampling.

All schemes share tabular CFR's regret tables and ``CFRConfig`` switches.
``step`` updates every player once and is reproducible for a fixed seed.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from pokercfr.core.protocols import Game, GameState, Policy
from pokercfr.solvers.cfr import VANILLA_CFR, CFRConfig, _float_dict, regret_matching

SCHEMES = ("outcome", "external", "chance")
_EXPLORATION = 0.06  # full support is essential for outcome-sampling unbiasedness


class MCCFRSolver:
    """Tabular MCCFR over any frozen-protocol game."""

    def __init__(self, game: Game, config: CFRConfig = VANILLA_CFR, scheme: str = "external", seed: int = 0):
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
        self.game, self.config, self.scheme = game, config, scheme
        self.rng, self.num_players, self.iterations = random.Random(seed), game.num_players, 0
        self.regret_sum: dict[str, dict[str, float]] = defaultdict(_float_dict)
        self.strategy_sum: dict[str, dict[str, float]] = defaultdict(_float_dict)

    def train(self, iterations: int) -> MCCFRSolver:
        for _ in range(iterations):
            self.step()
        return self

    def step(self) -> None:
        self.iterations += 1
        self._discount()
        if self.config.alternating:
            for player in range(self.num_players):
                delta: dict[str, dict[str, float]] = defaultdict(_float_dict)
                self._update_player(player, delta)
                self._commit(delta)
        else:
            delta = defaultdict(_float_dict)
            for player in range(self.num_players):
                self._update_player(player, delta)
            self._commit(delta)

    def average_strategy(self) -> Policy:
        return {key: self._normalise(sums) for key, sums in self.strategy_sum.items()}

    def current_strategy(self) -> Policy:
        return {key: regret_matching(regrets, sorted(regrets)) for key, regrets in self.regret_sum.items()}

    @property
    def averaging_weight(self) -> float:
        if self.config.averaging == "uniform":
            return 1.0
        if self.config.averaging == "linear":
            return float(self.iterations)
        return float(self.iterations * self.iterations)

    def _update_player(self, player: int, delta: dict[str, dict[str, float]]) -> None:
        root = self.game.new_initial_state()
        if self.scheme == "chance":
            self._chance(root, player, [1.0] * self.num_players, delta)
        elif self.scheme == "external":
            self._external(root, player, 1.0, 1.0, delta)
        else:
            self._outcome(root, player, delta)

    def _pick_chance(self, state: GameState) -> tuple[GameState, float]:
        outcomes = state.chance_outcomes()
        point, total = self.rng.random(), 0.0
        for child, probability in outcomes:
            total += probability
            if point < total:
                return child, probability
        return outcomes[-1]

    def _pick(self, items, probabilities):
        r, total = self.rng.random(), 0.0
        for item, _ in items:
            total += probabilities[item]
            if r < total:
                return item, probabilities[item]
        item, _ = items[-1]
        return item, probabilities[item]

    def _action(self, actions: list[str], sigma: dict[str, float], explore: bool) -> tuple[str, float]:
        probs = (
            {a: (1 - _EXPLORATION) * sigma[a] + _EXPLORATION / len(actions) for a in actions}
            if explore
            else sigma
        )
        return self._pick([(a, probs[a]) for a in actions], probs)

    # Chance sampling: sample every chance node, enumerate all betting actions.
    def _chance(self, state: GameState, target: int, reach: list[float], delta) -> float:
        if state.is_terminal():
            return state.utilities()[target]
        if state.is_chance_node():
            child, _ = self._pick_chance(state)
            return self._chance(child, target, reach, delta)
        player, actions = state.current_player(), state.legal_actions()
        key, sigma = (
            state.information_set_key(player),
            regret_matching(self.regret_sum[state.information_set_key(player)], actions),
        )
        values = {}
        for action in actions:
            next_reach = list(reach)
            next_reach[player] *= sigma[action]
            values[action] = self._chance(state.apply_action(action), target, next_reach, delta)
        value = sum(sigma[a] * values[a] for a in actions)
        if player == target:
            # Chance is sampled according to its true distribution, but the
            # opponent's *prefix* reach still belongs in counterfactual regret.
            cf_reach = math.prod(r for i, r in enumerate(reach) if i != target)
            for action in actions:
                delta[key][action] += cf_reach * (values[action] - value)
            self._strategy(key, sigma, reach[player])
        return value

    # External sampling: sample chance/opponents; enumerate target's actions.
    def _external(self, state: GameState, target: int, sample: float, own: float, delta) -> float:
        if state.is_terminal():
            return state.utilities()[target]
        if state.is_chance_node():
            child, prob = self._pick_chance(state)
            return self._external(child, target, sample * prob, own, delta)
        player, actions = state.current_player(), state.legal_actions()
        key, sigma = (
            state.information_set_key(player),
            regret_matching(self.regret_sum[state.information_set_key(player)], actions),
        )
        if player != target:
            action, prob = self._action(actions, sigma, False)
            return self._external(state.apply_action(action), target, sample * prob, own, delta)
        values = {
            a: self._external(state.apply_action(a), target, sample, own * sigma[a], delta) for a in actions
        }
        value = sum(sigma[a] * values[a] for a in actions)
        for action in actions:
            delta[key][action] += values[action] - value
        self._strategy(key, sigma, own / sample)
        return value

    # Outcome sampling estimator.  ``suffix`` is the likelihood correction for
    # all sampled decisions after an information set.
    def _outcome(self, root: GameState, target: int, delta) -> None:
        state, sample, cf, own = root, 1.0, 1.0, 1.0
        events = []
        while not state.is_terminal():
            if state.is_chance_node():
                state, p = self._pick_chance(state)
                sample *= p
                cf *= p
                events.append((None, 1.0))
                continue
            player, actions = state.current_player(), state.legal_actions()
            key = state.information_set_key(player)
            sigma = regret_matching(self.regret_sum[key], actions)
            action, q = self._action(actions, sigma, True)
            if player == target:
                events.append(((key, actions, sigma, action, q, sample, cf, own), sigma[action] / q))
                own *= sigma[action]
            else:
                events.append((None, sigma[action] / q))
                cf *= sigma[action]
            sample *= q
            state = state.apply_action(action)
        suffix, utility = 1.0, state.utilities()[target]
        for visit, ratio in reversed(events):
            if visit is not None:
                key, actions, sigma, chosen, q, prefix_sample, prefix_cf, prefix_own = visit
                weight = utility * prefix_cf / prefix_sample * suffix / q
                for action in actions:
                    # The sampled estimate of the node value is weighted by
                    # sigma of the *sampled* action, not sigma[action].
                    delta[key][action] += weight * ((action == chosen) - sigma[chosen])
                self._strategy(key, sigma, prefix_own / prefix_sample)
            suffix *= ratio

    def _strategy(self, key: str, sigma: dict[str, float], reach: float) -> None:
        for action, probability in sigma.items():
            self.strategy_sum[key][action] += self.averaging_weight * reach * probability

    def _commit(self, delta) -> None:
        for key, values in delta.items():
            for action, value in values.items():
                total = self.regret_sum[key][action] + value
                self.regret_sum[key][action] = max(total, 0.0) if self.config.regret_matching_plus else total

    def _discount(self) -> None:
        if self.config.dcfr is None or self.iterations <= 1:
            return
        alpha, beta, gamma = self.config.dcfr

        def factor(exp):
            if exp == math.inf:
                return 1.0
            if exp == -math.inf:
                return 0.0
            value = float(self.iterations) ** exp
            return 1.0 if math.isinf(value) else value / (value + 1.0)

        pos, neg = factor(alpha), factor(beta)
        for values in self.regret_sum.values():
            for action, value in values.items():
                values[action] = value * (pos if value > 0 else neg)
        scale = ((self.iterations - 1) / self.iterations) ** gamma
        for values in self.strategy_sum.values():
            for action in values:
                values[action] *= scale

    @staticmethod
    def _normalise(values: dict[str, float]) -> dict[str, float]:
        total = sum(values.values())
        return (
            {a: value / total for a, value in values.items()}
            if total > 0
            else {a: 1 / len(values) for a in values}
        )


__all__ = ["SCHEMES", "MCCFRSolver"]
