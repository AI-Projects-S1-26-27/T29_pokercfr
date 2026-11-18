"""Exact best response, exploitability and NashConv -- the project's ground truth.

This is the measurement instrument for every hypothesis. Nothing here samples or
approximates: given a policy, we compute the *exact* most profitable deviation by
enumerating the tree. That is only possible because every rung of our ladder is
small enough to enumerate, and it is the whole reason the study reports true
exploitability instead of self-play win rates.

Why best response needs two passes
----------------------------------
A responder must play the SAME action at every state inside one information set --
that is what "imperfect information" means. So we cannot just maximize node by
node. Instead:

  pass 1  walk the tree and bucket every responder decision node by its info-set
          key, carrying the *counterfactual* reach probability: chance x the other
          players' action probabilities, deliberately EXCLUDING the responder's own
          (they are choosing those; weighting by them would beg the question).

  pass 2  decide info sets deepest-first. Every responder node inside the subtree
          of an info set is strictly deeper, so by the time we choose an action for
          info set I, every decision it depends on is already fixed. Perfect recall
          plus a public betting history means all states in one info set share a
          history length, so history length is a valid depth order.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from pokercfr.core.protocols import Game, GameState, Policy


@dataclass(frozen=True)
class NashConvResult:
    """How far a policy is from equilibrium, decomposed per seat.

    For 2-player zero-sum, exploitability = total / 2 and is the headline metric.
    For N >= 3 the total is still well-defined but grows with the number of
    deviators, so `mean` and `max` are reported alongside it -- a bare total can
    rise purely because there are more players, which would be a misleading way
    to report the multiplayer gap (H3).
    """

    policy_values: list[float]
    best_response_values: list[float]

    @property
    def gains(self) -> list[float]:
        """Each seat's gain from unilaterally switching to its best response."""
        return [br - v for br, v in zip(self.best_response_values, self.policy_values, strict=True)]

    @property
    def total(self) -> float:
        return sum(self.gains)

    @property
    def mean(self) -> float:
        return self.total / len(self.gains)

    @property
    def max(self) -> float:
        return max(self.gains)

    @property
    def max_seat(self) -> int:
        """The seat with the largest deviation gain (lowest index on an exact tie).

        First-class rather than recomputed by callers, because "does position
        determine who has the most incentive to deviate" is a reported finding.
        """
        gains = self.gains
        return gains.index(max(gains))

    @property
    def exploitability(self) -> float:
        """Two-player zero-sum only: half the total deviation gain, chips/hand."""
        if len(self.gains) != 2:
            raise ValueError(
                f"exploitability is defined for 2 players; this is {len(self.gains)}-player. "
                "Use .total / .mean / .max (NashConv) instead."
            )
        return self.total / 2.0


def _full_policy(game: Game, policy: Policy) -> Policy:
    """Fill in uniform behaviour at any info set the policy never visited.

    A solver's average strategy can legitimately omit info sets it never reached
    (probability-zero branches). A best response will still probe them, so an
    unspecified info set must mean something explicit rather than raise KeyError.
    Uniform is the neutral choice and we record how many were filled.
    """
    filled = dict(policy)

    def walk(state: GameState) -> None:
        if state.is_terminal():
            return
        if state.is_chance_node():
            for child, _ in state.chance_outcomes():
                walk(child)
            return
        actions = state.legal_actions()
        key = state.information_set_key(state.current_player())
        if key not in filled:
            filled[key] = {a: 1.0 / len(actions) for a in actions}
        for a in actions:
            walk(state.apply_action(a))

    walk(game.new_initial_state())
    return filled


def best_response(game: Game, policy: Policy, responder: int) -> tuple[float, Policy]:
    """The responder's exact best-response value and the deviation achieving it.

    Everyone except `responder` plays `policy`; `responder` plays optimally.
    """
    policy = _full_policy(game, policy)

    # ---- pass 1: bucket responder nodes by info set, with counterfactual reach
    buckets: dict[str, list[tuple[GameState, float]]] = defaultdict(list)

    def collect(state: GameState, cf_reach: float) -> None:
        if state.is_terminal() or cf_reach == 0.0:
            return
        if state.is_chance_node():
            for child, prob in state.chance_outcomes():
                collect(child, cf_reach * prob)
            return
        player = state.current_player()
        actions = state.legal_actions()
        if player == responder:
            # Do NOT scale by the responder's own probabilities -- that is the
            # "counterfactual" in counterfactual value.
            buckets[state.information_set_key(responder)].append((state, cf_reach))
            for a in actions:
                collect(state.apply_action(a), cf_reach)
        else:
            probs = policy[state.information_set_key(player)]
            for a in actions:
                if probs[a] > 0.0:
                    collect(state.apply_action(a), cf_reach * probs[a])

    collect(game.new_initial_state(), 1.0)

    # ---- pass 2: choose actions deepest-first
    br_action: dict[str, str] = {}
    # Keyed by the state VALUE, not id(). States are created fresh by every
    # apply_action call and immediately collected, so CPython recycles their ids
    # -- an id-keyed cache silently returns one action's value for another and
    # the argmax below degenerates to "always pick the first action".
    memo: dict[GameState, float] = {}

    def value(state: GameState) -> float:
        """Responder's expected utility from `state`, using br_action where fixed."""
        cached = memo.get(state)
        if cached is not None:
            return cached
        if state.is_terminal():
            result = state.utilities()[responder]
        elif state.is_chance_node():
            result = sum(p * value(c) for c, p in state.chance_outcomes())
        else:
            player = state.current_player()
            actions = state.legal_actions()
            if player == responder:
                key = state.information_set_key(responder)
                # Deepest-first ordering guarantees this is already decided.
                result = value(state.apply_action(br_action[key]))
            else:
                probs = policy[state.information_set_key(player)]
                # Skip zero-probability branches, exactly as `collect` does.
                # Those contribute nothing to the value, and walking them here
                # while pass 1 pruned them means br_action has no entry for the
                # info sets inside -- a KeyError that only a PURE opponent
                # strategy triggers, which is why CFR policies never hit it and
                # an exact LP solution does.
                result = sum(probs[a] * value(state.apply_action(a))
                             for a in actions if probs[a] > 0.0)
        memo[state] = result
        return result

    # All states in one info set share a public history, so len(history) is depth.
    def depth_of(key: str) -> int:
        return len(buckets[key][0][0].history)

    for key in sorted(buckets, key=depth_of, reverse=True):
        nodes = buckets[key]
        actions = nodes[0][0].legal_actions()
        best, best_value = actions[0], float("-inf")
        for a in actions:
            total = sum(reach * value(state.apply_action(a)) for state, reach in nodes)
            if total > best_value:
                best, best_value = a, total
        br_action[key] = best
        memo.clear()  # a newly fixed action invalidates cached subtree values

    br_policy: Policy = {key: {a: 1.0 if a == act else 0.0
                               for a in buckets[key][0][0].legal_actions()}
                         for key, act in br_action.items()}
    return value(game.new_initial_state()), br_policy


def nash_conv(game: Game, policy: Policy) -> NashConvResult:
    """Every seat's exact gain from deviating. The core measurement."""
    from pokercfr.solvers.cfr import policy_expected_values

    values = policy_expected_values(game, _full_policy(game, policy))
    br_values = [best_response(game, policy, p)[0] for p in range(game.num_players)]
    return NashConvResult(policy_values=values, best_response_values=br_values)


def exploitability(game: Game, policy: Policy) -> float:
    """Two-player zero-sum exploitability in chips/hand."""
    return nash_conv(game, policy).exploitability


__all__ = ["best_response", "nash_conv", "exploitability", "NashConvResult"]
