"""Configurable one-round poker: Kuhn generalized along deck size, cards dealt,
and player count, so every rung of the game-size ladder is one config.

Rules (N-player Kuhn, Abou Risk & Szafron 2010): everyone antes 1. Players act
in seat order with 'p' (check) / 'b' (bet 1). Once someone bets, each other
player responds exactly once, in order after the bettor: 'p' fold / 'b' call.
No raises. Showdown among players still in; ties split the pot.

Hand strength (no community cards): pairs/trips/quads and two pair beat high
cards, compared by rank-count pattern then ranks.
# ponytail: no straights/flushes — add when a rung deals 5 cards
"""
from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

RANK_NAMES = "23456789TJQKA"
SUIT_NAMES = "shdc"


@dataclass(frozen=True)
class GameConfig:
    ranks: int = 3  # 3 -> J Q K (Kuhn), 4 -> J Q K A, more extend downward: 6 -> 9 T J Q K A
    suits: int = 1
    cards_dealt: int = 1
    players: int = 2

    def __post_init__(self) -> None:
        for name in ("ranks", "suits", "cards_dealt", "players"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.ranks < 1 or self.ranks > len(RANK_NAMES):
            raise ValueError(f"ranks must be between 1 and {len(RANK_NAMES)}")
        if self.suits < 1 or self.suits > len(SUIT_NAMES):
            raise ValueError(f"suits must be between 1 and {len(SUIT_NAMES)}")
        if self.cards_dealt < 1:
            raise ValueError("cards_dealt must be positive")
        if self.players < 2:
            raise ValueError("players must be at least 2")
        if self.cards_dealt * self.players > self.ranks * self.suits:
            raise ValueError("not enough cards in deck for all players")

    @property
    def deck(self) -> list[int]:
        """Cards as ints: rank * suits + suit."""
        return list(range(self.ranks * self.suits))

    def card_name(self, card: int) -> str:
        if isinstance(card, bool) or not isinstance(card, int) or card < 0 or card >= self.ranks * self.suits:
            raise ValueError(f"card index out of range: {card!r}")
        names = "JQKA"[: self.ranks] if self.ranks <= 4 else RANK_NAMES[-self.ranks:]
        rank = names[card // self.suits]
        return rank + (SUIT_NAMES[card % self.suits] if self.suits > 1 else "")

    def num_deals(self) -> int:
        n, k = len(self.deck), self.cards_dealt
        return math.perm(n, k * self.players) // math.factorial(k) ** self.players

    def num_info_sets(self) -> int:
        return math.comb(len(self.deck), self.cards_dealt) * len(_decision_histories(self.players))


KUHN = GameConfig()

# The game-size ladder, smallest first. Sizes: `python -m poker.game`.
LADDER = {
    "kuhn": KUHN,
    "4rank": GameConfig(ranks=4),
    "4rank-2dealt": GameConfig(ranks=4, cards_dealt=2),
    "4rank-2suit": GameConfig(ranks=4, suits=2),
    "4rank-2suit-2dealt": GameConfig(ranks=4, suits=2, cards_dealt=2),
    "kuhn-3p": GameConfig(players=3, ranks=4),
    "4rank-2suit-3p": GameConfig(ranks=4, suits=2, players=3),
    "4rank-2suit-2dealt-3p": GameConfig(ranks=4, suits=2, cards_dealt=2, players=3),
    "4rank-2suit-4p": GameConfig(ranks=4, suits=2, players=4),
    "6rank-4suit-2dealt-4p": GameConfig(ranks=6, suits=4, cards_dealt=2, players=4),
    "13rank-4suit-2dealt-2p": GameConfig(ranks=13, suits=4, cards_dealt=2),
}


def _hand_strength(cards: tuple[int, ...], suits: int) -> tuple:
    if not cards:
        raise ValueError("a hand must contain at least one card")
    if suits < 1:
        raise ValueError("suits must be positive")
    counts = Counter(c // suits for c in cards)
    groups = sorted(counts.items(), key=lambda rc: (rc[1], rc[0]), reverse=True)
    return tuple(c for _, c in groups), tuple(r for r, _ in groups)


def _decision_histories(players: int) -> list[str]:
    """Every non-terminal betting history."""
    out, frontier = [], [""]
    while frontier:
        h = frontier.pop()
        if _history_is_terminal(h, players):
            continue
        out.append(h)
        frontier += [h + "p", h + "b"]
    return out


def _history_is_terminal(history: str, players: int) -> bool:
    bet = history.find("b")
    if bet < 0:
        return len(history) == players
    return len(history) - bet - 1 == players - 1


@dataclass(frozen=True)
class PokerState:
    config: GameConfig
    hands: tuple[tuple[int, ...], ...]
    history: str = ""

    def __post_init__(self) -> None:
        if len(self.hands) != self.config.players:
            raise ValueError("hands must contain exactly one hand per player")
        if any(len(hand) != self.config.cards_dealt for hand in self.hands):
            raise ValueError("each hand must contain exactly cards_dealt cards")
        cards = [card for hand in self.hands for card in hand]
        if any(isinstance(card, bool) or not isinstance(card, int) or card not in self.config.deck
               for card in cards):
            raise ValueError("hands contain a card outside the configured deck")
        if len(set(cards)) != len(cards):
            raise ValueError("a card cannot be dealt to more than one player")
        if any(action not in "pb" for action in self.history):
            raise ValueError("history may contain only 'p' and 'b'")
        for i in range(1, len(self.history)):
            if _history_is_terminal(self.history[:i], self.config.players):
                raise ValueError("history continues after the hand is terminal")

    def _bet_index(self) -> int:
        return self.history.find("b")

    def is_chance_node(self) -> bool:
        return False  # all chance lives at the root (see ChanceRoot below)

    def chance_outcomes(self) -> list[tuple[PokerState, float]]:
        raise ValueError("PokerState is never a chance node; chance is at the root")

    def current_player(self) -> int:
        return len(self.history) % self.config.players  # seats act in strict rotation

    def legal_actions(self) -> list[str]:
        return [] if self.is_terminal() else ["p", "b"]

    def apply_action(self, action: str) -> PokerState:
        if self.is_terminal():
            raise ValueError("apply_action() called on a terminal state")
        if action not in ("p", "b"):
            raise ValueError(f"illegal action: {action!r}")
        # The parent has already passed full validation and action was checked;
        # avoid rescanning the unchanged deal and every history prefix per node.
        child = object.__new__(PokerState)
        object.__setattr__(child, "config", self.config)
        object.__setattr__(child, "hands", self.hands)
        object.__setattr__(child, "history", self.history + action)
        return child

    def is_terminal(self) -> bool:
        return _history_is_terminal(self.history, self.config.players)

    def utilities(self) -> list[float]:
        if not self.is_terminal():
            raise ValueError("utilities() called on a non-terminal state")
        n, h, i = self.config.players, self.history, self._bet_index()
        contrib = [1] * n
        if i < 0:
            live = list(range(n))
        else:
            live = [i % n]
            contrib[i % n] += 1
            for offset, a in enumerate(h[i + 1:], start=1):
                if a == "b":
                    p = (i + offset) % n
                    live.append(p)
                    contrib[p] += 1
        strength = {p: _hand_strength(self.hands[p], self.config.suits) for p in live}
        best = max(strength.values())
        winners = [p for p in live if strength[p] == best]
        pot = sum(contrib)
        return [(pot / len(winners) if p in winners else 0) - contrib[p] for p in range(n)]

    def utility(self, player: int) -> float:
        _check_player(player, self.config.players)
        return self.utilities()[player]

    def information_set_key(self, player: int) -> str:
        _check_player(player, self.config.players)
        cards = "".join(self.config.card_name(c) for c in sorted(self.hands[player], reverse=True))
        return f"{cards}{self.history}"


def _check_player(player: int, players: int) -> None:
    if isinstance(player, bool) or not isinstance(player, int) or not 0 <= player < players:
        raise ValueError(f"player index out of range: {player!r}")


def deal_outcomes(config: GameConfig = KUHN) -> list[tuple[PokerState, Fraction]]:
    """Every deal (ordered by seat, unordered within a hand) at exact probability."""
    k = config.cards_dealt

    def deals(remaining: frozenset[int], seats: int):
        if seats == 0:
            yield ()
            return
        for hand in itertools.combinations(sorted(remaining), k):
            for rest in deals(remaining - set(hand), seats - 1):
                yield (hand, *rest)

    all_deals = list(deals(frozenset(config.deck), config.players))
    prob = Fraction(1, len(all_deals))
    return [(PokerState(config, d), prob) for d in all_deals]


def sample_deal(config: GameConfig, rng: random.Random) -> PokerState:
    cards = rng.sample(config.deck, config.cards_dealt * config.players)
    k = config.cards_dealt
    return PokerState(config, tuple(tuple(cards[p * k:(p + 1) * k]) for p in range(config.players)))


if __name__ == "__main__":
    print(f"{'rung':26} {'players':>7} {'info sets':>12} {'deals':>16}")
    for name, cfg in LADDER.items():
        print(f"{name:26} {cfg.players:>7} {cfg.num_info_sets():>12,} {cfg.num_deals():>16,}")


@dataclass(frozen=True)
class ChanceRoot:
    """The root of a Kuhn-family game: a single chance node whose outcomes are
    every possible deal. Betting states that follow are plain PokerStates.

    This exists so one solver traversal serves both engines. Leduc has a second
    chance node mid-tree (the public card); here there is exactly one, at depth
    zero. Either way a solver just asks `is_chance_node()`.
    """

    config: GameConfig

    def is_terminal(self) -> bool:
        return False

    def is_chance_node(self) -> bool:
        return True

    def chance_outcomes(self) -> list[tuple[PokerState, float]]:
        return [(state, float(p)) for state, p in deal_outcomes(self.config)]

    def current_player(self) -> int:
        raise ValueError("ChanceRoot is a chance node, not a decision node")

    def legal_actions(self) -> list[str]:
        raise ValueError("ChanceRoot is a chance node, not a decision node")

    def apply_action(self, action: str):
        raise ValueError("ChanceRoot is a chance node; use chance_outcomes()")

    def utilities(self) -> list[float]:
        raise ValueError("ChanceRoot is not terminal")

    def information_set_key(self, player: int) -> str:
        raise ValueError("ChanceRoot has no information set")


class KuhnFamilyGame:
    """One rung of the ladder. A rung is a GameConfig, not a new codebase."""

    def __init__(self, config: GameConfig = KUHN, name: str | None = None):
        self.config = config
        self._name = name or _config_name(config)

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_players(self) -> int:
        return self.config.players

    def new_initial_state(self) -> ChanceRoot:
        return ChanceRoot(self.config)

    def num_info_sets(self) -> int:
        return self.config.num_info_sets()

    def num_deals(self) -> int:
        return self.config.num_deals()

    def __repr__(self) -> str:
        return f"KuhnFamilyGame({self._name!r}, info_sets={self.num_info_sets():,})"


def _config_name(config: GameConfig) -> str:
    for name, cfg in LADDER.items():
        if cfg == config:
            return name
    return f"r{config.ranks}s{config.suits}d{config.cards_dealt}p{config.players}"


def game(rung: str) -> KuhnFamilyGame:
    """Look up a ladder rung by name: game("kuhn"), game("4rank-2suit-3p")."""
    if rung not in LADDER:
        raise KeyError(f"unknown rung {rung!r}; known rungs: {', '.join(LADDER)}")
    return KuhnFamilyGame(LADDER[rung], name=rung)
