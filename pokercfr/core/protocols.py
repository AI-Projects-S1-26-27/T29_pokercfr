"""THE FROZEN CONTRACT. Every game, solver and evaluator in this project talks
through the two protocols below. Do not change this file without a PR reviewed
by the integration lead -- eight people are coding against it in parallel.

Design note (why there is an explicit chance node)
--------------------------------------------------
Our Kuhn-family games put all randomness at the root: deal private cards, then
pure betting. Leduc does not -- it deals, runs a betting round, then reveals a
public card mid-tree, then runs a second round. A solver that only understood
"enumerate the root deal" would therefore work on one engine and not the other.

So chance is a *node type*, exactly as in OpenSpiel. A traversal handles chance
generically and every solver then works unchanged on both engines. This is the
seam that lets the game owners and the solver owners work without talking.

State lifecycle -- exactly one of these is true at any point:
    is_terminal()      -> utilities() is valid
    is_chance_node()   -> chance_outcomes() is valid
    otherwise          -> current_player() / legal_actions() / apply_action() valid
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

# An information-set key. Must encode everything the acting player knows and
# nothing they don't -- their own private cards plus all public information.
# Two states with the same key are indistinguishable to that player.
InfoSetKey = str

# A behavioural strategy: info-set key -> action -> probability (sums to 1).
Policy = dict[InfoSetKey, dict[str, float]]


@runtime_checkable
class GameState(Protocol):
    """One node of an extensive-form game tree. Implementations are immutable:
    apply_action returns a new state and never mutates self."""

    def is_terminal(self) -> bool: ...

    def is_chance_node(self) -> bool: ...

    def chance_outcomes(self) -> list[tuple[GameState, float]]:
        """Every outcome with its probability; probabilities sum to 1.
        Only valid when is_chance_node() is True."""
        ...

    def current_player(self) -> int:
        """Seat index of the player to act. Only valid at a decision node."""
        ...

    def legal_actions(self) -> list[str]:
        """Actions available at a decision node, in a deterministic order.
        Determinism matters: solvers index regret arrays by position."""
        ...

    def apply_action(self, action: str) -> GameState: ...

    def utilities(self) -> list[float]:
        """Payoff per seat at a terminal node, net of contributions. Only valid
        when is_terminal() is True. Must sum to 0 (all our games are zero-sum)."""
        ...

    def information_set_key(self, player: int) -> InfoSetKey: ...


@runtime_checkable
class Game(Protocol):
    """A game is a factory for initial states plus its own metadata. A 'rung' of
    the ladder is one Game instance -- a configuration, not a new codebase."""

    @property
    def name(self) -> str:
        """Stable identifier used in result filenames. Never change one silently:
        it would orphan already-committed results."""
        ...

    @property
    def num_players(self) -> int: ...

    def new_initial_state(self) -> GameState: ...

    def num_info_sets(self) -> int:
        """Exact count, for the scaling regressions (R2). Computed, not guessed."""
        ...


class Solver(Protocol):
    """An equilibrium-finding algorithm. All solvers are single-step-driven so the
    experiment runner, the convergence logger and the live demo share one loop:

        while not done:
            solver.step()
            if checkpoint: log(exploitability(game, solver.average_strategy()))
    """

    iterations: int

    def step(self) -> None:
        """Advance exactly one iteration."""
        ...

    def average_strategy(self) -> Policy:
        """The time-averaged strategy. THIS is what converges to Nash in
        two-player zero-sum games -- not the current iterate."""
        ...

    def current_strategy(self) -> Policy:
        """The latest iterate. Can cycle forever even as the average converges;
        we report that gap as a result rather than hiding it."""
        ...
