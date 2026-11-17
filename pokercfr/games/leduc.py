"""Leduc Hold'em -- OWNER: Samanyu. Primary testbed for H1 and H3.

Leduc (Southey et al., UAI 2005) is the standard mid-size benchmark and the only
rung with a *mid-tree* chance node, which is why the frozen protocol has
is_chance_node() at all:

    deal 1 private card each  ->  betting round 1  ->  reveal 1 public card
                              ->  betting round 2  ->  showdown

Standard Leduc: 6 cards = 3 ranks x 2 suits, ante 1, bet size 2 in round 1 and 4
in round 2, raise cap 2 per round. A pair with the public card beats any unpaired
hand; otherwise the higher private rank wins; equal ranks split.

Betting matches OpenSpiel's `leduc_poker` exactly (checked in
scripts/s5_openspiel_leduc_crosscheck.py): actions are 'f' fold, 'c' call (a
check when nothing is owed), 'r' raise. Fold is only legal when facing a bet, the
opening bet counts toward the raise cap, and round 2 opens with the lowest seat
still in the hand. A round ends once every player still in has acted since the
last raise.

TWO THINGS THIS MUST GET RIGHT -- they are the project's novelty claim
---------------------------------------------------------------------
1. `raise_cap` is a free parameter (H1 tests across caps), not hardcoded to 2.
2. `players` grows with the DECK HELD FIXED. OpenSpiel grows Leduc's deck as
   2*(N+1) cards with player count, which confounds "more players" with "different
   card distribution". Holding the deck fixed is precisely what lets H3 attribute
   a rising NashConv plateau to player count alone. If the deck is too small to
   deal N private cards plus the public card, we raise a clear error rather than
   silently resizing -- on the standard 6-card deck that caps N at 5.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from functools import cached_property

from pokercfr.games.kuhn_family import GameConfig as _CardNames

FOLD, CALL, RAISE = "f", "c", "r"
ROUND_BREAK = "/"  # appended to the history when round 1 closes


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

    def validate(self) -> None:
        if self.players < 2:
            raise ValueError(f"Leduc needs at least 2 players, got {self.players}")
        if self.raise_cap < 1:
            raise ValueError(f"raise_cap must be >= 1, got {self.raise_cap}")
        if self.players + 1 > self.deck_size:
            raise ValueError(
                f"a {self.deck_size}-card deck cannot deal {self.players} private cards plus a "
                f"public card; the deck is held fixed across player counts by design, so use "
                f"fewer players or a larger deck explicitly (ranks/suits)")

    def card_name(self, card: int) -> str:
        return _CardNames(ranks=self.ranks, suits=self.suits).card_name(card)


STANDARD = LeducConfig()


@dataclass(frozen=True, slots=True)
class LeducState:
    """Immutable game state. The derived betting fields (contrib, folded, ...) are
    stored rather than recomputed from `history` so apply_action stays O(players)."""

    config: LeducConfig
    private: tuple[int, ...] = ()  # empty -> the deal chance node
    public: int | None = None
    history: str = ""              # e.g. "rc/cr": round 1, a break, round 2
    contrib: tuple[int, ...] = ()
    folded: tuple[bool, ...] = ()
    to_act: int = 0
    pending: int = 0               # players still to act before the round closes
    raises: int = 0                # raises so far in the current round
    terminal: bool = False

    # -- node type ---------------------------------------------------------------

    def is_terminal(self) -> bool:
        return self.terminal

    def is_chance_node(self) -> bool:
        if self.terminal:
            return False
        return not self.private or (self.public is None and self._round() == 1)

    def _round(self) -> int:
        return 1 if ROUND_BREAK in self.history else 0

    def _require_decision(self) -> None:
        if self.terminal or self.is_chance_node():
            raise ValueError("not a decision node")

    # -- chance ------------------------------------------------------------------

    def chance_outcomes(self) -> list[tuple[LeducState, float]]:
        if not self.is_chance_node():
            raise ValueError("chance_outcomes() called on a non-chance node")
        cfg = self.config
        if not self.private:
            deals = list(itertools.permutations(range(cfg.deck_size), cfg.players))
            n = cfg.players
            return [(LeducState(cfg, deal, None, "", (1,) * n, (False,) * n, 0, n, 0, False),
                     1.0 / len(deals)) for deal in deals]
        # The public card comes from the deck minus every private card.
        remaining = [c for c in range(cfg.deck_size) if c not in self.private]
        return [(replace(self, public=c), 1.0 / len(remaining)) for c in remaining]

    # -- decisions ---------------------------------------------------------------

    def current_player(self) -> int:
        self._require_decision()
        return self.to_act

    def legal_actions(self) -> list[str]:
        self._require_decision()
        actions = []
        if self.contrib[self.to_act] < max(self.contrib):
            actions.append(FOLD)
        actions.append(CALL)
        if self.raises < self.config.raise_cap:
            actions.append(RAISE)
        return actions

    def apply_action(self, action: str) -> LeducState:
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action!r} at history {self.history!r}")
        p, n = self.to_act, self.config.players
        contrib, folded = list(self.contrib), list(self.folded)
        pending, raises = self.pending, self.raises
        if action == FOLD:
            folded[p] = True
            pending -= 1
        elif action == CALL:
            contrib[p] = max(contrib)
            pending -= 1
        else:
            contrib[p] = max(contrib) + self.config.bet_sizes[self._round()]
            raises += 1
            pending = n - sum(folded) - 1  # everyone else still in must respond

        history = self.history + action
        live = [i for i in range(n) if not folded[i]]
        terminal, to_act = False, p
        if len(live) == 1:
            terminal = True
        elif pending == 0:
            if self._round() == 0:
                # Round 1 closes: the next state is the public-card chance node.
                history += ROUND_BREAK
                to_act, pending, raises = live[0], len(live), 0
            else:
                terminal = True  # showdown
        else:
            to_act = next(i % n for i in range(p + 1, p + n) if not folded[i % n])
        return LeducState(self.config, self.private, self.public, history,
                          tuple(contrib), tuple(folded), to_act, pending, raises, terminal)

    # -- payoffs and information -------------------------------------------------

    def utilities(self) -> list[float]:
        if not self.terminal:
            raise ValueError("utilities() called on a non-terminal state")
        n, suits = self.config.players, self.config.suits
        live = [i for i in range(n) if not self.folded[i]]
        if len(live) == 1:
            winners = live
        else:
            pub_rank = self.public // suits

            def strength(i: int) -> tuple[bool, int]:
                rank = self.private[i] // suits
                return rank == pub_rank, rank  # a pair beats any unpaired hand

            best = max(strength(i) for i in live)
            winners = [i for i in live if strength(i) == best]
        share = sum(self.contrib) / len(winners)
        return [(share if i in winners else 0.0) - self.contrib[i] for i in range(n)]

    def information_set_key(self, player: int) -> str:
        if not self.private:
            raise ValueError("the deal chance node has no information set")
        cards = self.config.card_name(self.private[player])
        if self.public is not None:
            cards += self.config.card_name(self.public)
        return f"{cards}:{self.history}"


class LeducGame:
    """Satisfies pokercfr.core.protocols.Game."""

    def __init__(self, config: LeducConfig = STANDARD, name: str | None = None):
        config.validate()
        self.config = config
        self._name = name or _config_name(config)

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_players(self) -> int:
        return self.config.players

    def new_initial_state(self) -> LeducState:
        return LeducState(self.config)

    def num_info_sets(self) -> int:
        return self._info_set_count

    @cached_property
    def _info_set_count(self) -> int:
        """Exact, without enumerating deals. The betting tree is the same for every
        deal, so an info set is (own card, public card if shown, decision history).
        Any own card can face any round-1 history, and any (own, public) pair of
        distinct cards can face any round-2 history -- validate() guarantees the
        other players' cards always fit. tests/test_leduc.py checks this against a
        full tree walk and OpenSpiel's count."""
        d = self.config.deck_size
        round1, round2 = _decision_histories(self.config)
        return d * round1 + d * (d - 1) * round2

    def __repr__(self) -> str:
        return f"LeducGame({self._name!r}, info_sets={self.num_info_sets():,})"


def _decision_histories(config: LeducConfig) -> tuple[int, int]:
    """Count decision histories in each betting round, using one fixed deal."""
    deal = LeducState(config).chance_outcomes()[0][0]
    counts = [0, 0]

    def walk(state: LeducState) -> None:
        if state.terminal:
            return
        if state.is_chance_node():
            walk(state.chance_outcomes()[0][0])
            return
        counts[state._round()] += 1
        for a in state.legal_actions():
            walk(state.apply_action(a))

    walk(deal)
    return counts[0], counts[1]


def _config_name(config: LeducConfig) -> str:
    deck = "" if (config.ranks, config.suits) == (STANDARD.ranks, STANDARD.suits) \
        else f"r{config.ranks}s{config.suits}-"
    return f"leduc-{deck}{config.players}p-cap{config.raise_cap}"


def game(players: int = 2, raise_cap: int = 2, ranks: int = 3, suits: int = 2) -> LeducGame:
    """Convenience constructor: leduc.game(players=3, raise_cap=1). The runner
    resolves rung names `leduc-<N>p-cap<M>` through exactly this signature."""
    return LeducGame(LeducConfig(ranks=ranks, suits=suits, players=players, raise_cap=raise_cap))
