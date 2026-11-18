"""Vectorized CFR traversal over hands. OWNER: Siddhant.

WHY. On the Kuhn-family ladder every deal shares ONE public betting tree -- the
cards change who wins, never which actions exist. The scalar traversal in
solvers/cfr.py nevertheless walks that tree once per deal: on the 52-card rung
`13rank-4suit-2dealt-2p` that is 1,624,350 deals x 9 betting nodes per traversal,
measured at ~95 s per iteration. But an information set is just (own hand,
public history), so the same computation can be organised the other way round:
walk the betting tree ONCE, carrying for every node a vector over all hands.

WHAT IS VECTORISED
  * reach probabilities: one length-H vector per seat per betting node (H = number
    of possible private hands; 1,326 on the 52-card rung)
  * regret matching: one (H, A) array per betting node, matched row by row
  * terminal values: at each terminal history z, a dense tensor U_z[i] over hand
    tuples holds chance(deal) x u_i(deal, z), with zero for hand tuples that
    share a card (card removal). Seat i's counterfactual value vector is U_z[i]
    contracted with every OTHER seat's reach vector -- for two players, one
    matrix-vector product. Chance and card removal live entirely inside U_z.

WHAT IS NOT CHANGED -- the update rules are solvers/cfr.py's, line for line:
  * sigma^t is regret-matched from R^(t-1) and held fixed for the traversal;
    deltas are committed only after the traversal (the bug-1 fix)
  * RM+ clamps the CUMULATIVE regret at commit time
  * alternating updates do one traversal + commit per seat, so seat 1 sees seat
    0's freshly committed regrets; simultaneous does one traversal for everyone
  * DCFR discounts before each iteration with cfr._discount
  * averaging weight 1 / t / t^2, read after the iteration counter is incremented

ONE NON-OBVIOUS DETAIL, reproduced deliberately: the scalar solver adds
`weight * own_reach * sigma` to strategy_sum once PER DEAL (per state), not once
per information set. Its raw strategy sums are therefore scaled, per info set, by
the number of deals in that info set -- on the 52-card rung, by C(50,2) = 1,225.
Normalisation cancels the factor, so average strategies are unaffected, but we
multiply by the same per-hand deal count so the RAW tables are comparable
entry-for-entry with the scalar solver's, which is what the equivalence tests
check.

EQUIVALENCE. The only intended difference from the scalar solver is summation
order (a matrix-vector product instead of a running sum over deals), which moves
results by a few ulps relative to their magnitude. tests/test_vectorized.py
compares regret sums, strategy sums, average and current strategies and
exploitability, across every corner of the ablation cube plus DCFR, and states
the tolerance it uses and why.

SCOPE. Kuhn-family games only (games/kuhn_family.py): a single root chance node
dealing private hands, then a deal-independent betting tree. Leduc's mid-tree
public card breaks that structure and is rejected with a clear error rather than
handled approximately. N-player rungs work while the dense payoff tensor fits
(H^N entries per terminal per seat); the 4-player 24-card rung would need ~5.8e9
entries per tensor and is refused by the memory guard.

The module also provides `nash_conv`, an exact vectorised best response /
NashConv for the same games. It is cross-validated against eval/best_response.py
(tests/test_vectorized.py) and exists because exact evaluation on the 52-card rung
through the generic tree walker takes ~8 minutes and ~3.8 GB per checkpoint.
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from pokercfr.core.protocols import Game, Policy
from pokercfr.games import kuhn_family
from pokercfr.games.kuhn_family import ChanceRoot, GameConfig, PokerState
from pokercfr.solvers.cfr import VANILLA_CFR, CFRConfig, _discount

MAX_TENSOR_BYTES = 4 * 2**30   # refuse to build payoff tensors larger than this
UTILITY_SAMPLE = 2_000         # deals re-checked against PokerState.utilities() at build


@dataclass(frozen=True)
class _Node:
    history: str
    player: int
    actions: tuple[str, ...]
    children: tuple[str, ...]  # child history per action


class HandSpace:
    """Everything about a Kuhn-family game that does not change between iterations.

    Built once per game configuration (and cached), shared by every solver and
    evaluator on that game. All of it is derived from the game's own objects --
    deals from ChanceRoot.chance_outcomes(), info-set keys from
    PokerState.information_set_key(), payoffs from PokerState.utilities() -- so a
    change to the game engine flows through rather than being shadowed by a copy.
    """

    def __init__(self, config: GameConfig, max_tensor_bytes: int = MAX_TENSOR_BYTES):
        self.config = config
        n, k = config.players, config.cards_dealt
        self.num_players = n
        self.hands: list[tuple[int, ...]] = list(itertools.combinations(config.deck, k))
        H = self.H = len(self.hands)
        hand_index = {h: i for i, h in enumerate(self.hands)}

        # ---- betting tree (identical for every deal; walked on one representative)
        rep = kuhn_family.sample_deal(config, random.Random(0))
        self.nodes: list[_Node] = []      # pre-order: parents before children
        self.terminals: list[str] = []
        stack = [""]
        while stack:
            h = stack.pop()
            state = PokerState(config, rep.hands, h)
            if state.is_terminal():
                self.terminals.append(h)
                continue
            actions = tuple(state.legal_actions())
            self.nodes.append(_Node(h, state.current_player(), actions,
                                    tuple(h + a for a in actions)))
            stack.extend(reversed([h + a for a in actions]))
        self.node_of = {node.history: i for i, node in enumerate(self.nodes)}

        tensor_bytes = n * H ** n * len(self.terminals) * 8
        if tensor_bytes > max_tensor_bytes:
            raise MemoryError(
                f"vectorized payoff tensors for this rung need {tensor_bytes / 2**30:.1f} GiB "
                f"({len(self.terminals)} terminals x {n} seats x {H}^{n} hand tuples); limit is "
                f"{max_tensor_bytes / 2**30:.1f} GiB. Use the scalar solver or MCCFR here.")

        # Only now enumerate the deals: the guard above must run first, because on
        # the rungs it refuses the enumeration itself is what does not fit.
        outcomes = ChanceRoot(config).chance_outcomes()

        # ---- deals as hand indices, with their exact chance probability
        D = len(outcomes)
        deals = np.empty((D, n), dtype=np.int64)
        probs = np.empty(D, dtype=np.float64)
        for d, (state, p) in enumerate(outcomes):
            deals[d] = [hand_index[hand] for hand in state.hands]
            probs[d] = float(p)
        self.num_deals = D
        # Per seat, how many deals give that seat hand h: the scalar solver's
        # per-state strategy_sum scaling (see module docstring).
        self.deal_count = [np.bincount(deals[:, i], minlength=H).astype(np.float64)
                           for i in range(n)]

        # ---- payoff tensors U[z][i][h_0, ..., h_{n-1}] = chance x u_i
        # Kuhn-family utilities depend on the hands only through each seat's hand
        # strength (kuhn_family._hand_strength), so we ask the engine for the payoff
        # once per distinct strength pattern instead of once per deal, then spot-
        # check a sample of deals against PokerState.utilities() directly below.
        strengths = [kuhn_family._hand_strength(hand, config.suits) for hand in self.hands]
        classes = {s: i for i, s in enumerate(sorted(set(strengths)))}
        sid = np.array([classes[s] for s in strengths], dtype=np.int64)
        codes = np.ravel_multi_index(tuple(sid[deals[:, i]] for i in range(n)),
                                     (len(classes),) * n)
        uniq, first, inverse = np.unique(codes, return_index=True, return_inverse=True)
        index = tuple(deals[:, i] for i in range(n))
        self.U: dict[str, np.ndarray] = {}
        for z in self.terminals:
            table = np.array([PokerState(config, outcomes[d][0].hands, z).utilities()
                              for d in first], dtype=np.float64)          # (classes, n)
            per_deal = table[inverse]                                       # (D, n)
            U = np.zeros((n,) + (H,) * n, dtype=np.float64)
            for i in range(n):
                U[(i,) + index] = probs * per_deal[:, i]
            self.U[z] = U

        rng = np.random.default_rng(0)
        for d in rng.choice(D, size=min(D, UTILITY_SAMPLE), replace=False):
            hands = outcomes[d][0].hands
            for z in self.terminals:
                want = PokerState(config, hands, z).utilities()
                got = [self.U[z][(i,) + tuple(deals[d])] / probs[d] for i in range(n)]
                if not np.allclose(got, want, rtol=0, atol=1e-12):
                    raise AssertionError(f"payoff tensor disagrees with utilities() at deal "
                                         f"{hands}, history {z!r}: {got} vs {want}")

        # ---- info-set keys, straight from the engine, for every (seat, hand, node)
        holder = [dict() for _ in range(n)]   # seat -> hand index -> one deal's hands
        for d in range(D):
            for i in range(n):
                holder[i].setdefault(int(deals[d, i]), d)
            if all(len(holder[i]) == H for i in range(n)):
                break
        self.keys: list[list[str]] = []      # node index -> H keys of the acting seat
        for node in self.nodes:
            p = node.player
            self.keys.append([
                PokerState(config, outcomes[holder[p][h]][0].hands, node.history)
                .information_set_key(p) for h in range(H)])
        del outcomes

    def counterfactual_values(self, z: str, reach: list[np.ndarray], seat: int) -> np.ndarray:
        """Seat `seat`'s counterfactual value vector at terminal z: U_z[seat]
        contracted with every other seat's reach (chance is already inside U)."""
        T = self.U[z][seat]
        for j in reversed(range(self.num_players)):   # highest axis first keeps indices valid
            if j != seat:
                T = np.tensordot(T, reach[j], axes=([j], [0]))
        return T


@lru_cache(maxsize=2)
def hand_space(config: GameConfig) -> HandSpace:
    return HandSpace(config)


def _space_for(game: Game) -> HandSpace:
    config = getattr(game, "config", None)
    if not isinstance(config, GameConfig) or not isinstance(game.new_initial_state(), ChanceRoot):
        raise TypeError(
            f"vectorized traversal supports Kuhn-family games (games/kuhn_family.py) only, got "
            f"{getattr(game, 'name', game)!r}: it relies on a single root deal followed by a "
            "deal-independent betting tree. Leduc's mid-tree public card breaks that; use "
            "solvers.cfr.CFRSolver there.")
    return hand_space(config)


def _regret_match(R: np.ndarray) -> np.ndarray:
    """Row-wise regret matching, uniform where no regret is positive (= cfr.regret_matching)."""
    pos = np.maximum(R, 0.0)
    total = pos.sum(axis=1, keepdims=True)
    uniform = np.full_like(R, 1.0 / R.shape[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0.0, pos / np.where(total > 0.0, total, 1.0), uniform)


class VectorizedCFRSolver:
    """Drop-in for solvers.cfr.CFRSolver on Kuhn-family games: same config, same
    Solver interface, same numbers up to summation order."""

    def __init__(self, game: Game, config: CFRConfig = VANILLA_CFR):
        self.game = game
        self.config = config
        self.num_players = game.num_players
        self.iterations = 0
        space = _space_for(game)
        self._space = space
        self._R = [np.zeros((space.H, len(nd.actions))) for nd in space.nodes]
        self._S = [np.zeros((space.H, len(nd.actions))) for nd in space.nodes]

    # The HandSpace is derived data (the payoff tensors are ~140 MB on the 52-card
    # rung), so it is rebuilt on unpickle rather than written into every runner
    # checkpoint.
    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_space", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._space = _space_for(self.game)

    # ---- Solver interface -------------------------------------------------------

    def step(self) -> None:
        self.iterations += 1
        self._apply_dcfr_discount()
        seats = ([[p] for p in range(self.num_players)] if self.config.alternating
                 else [list(range(self.num_players))])
        for update in seats:
            deltas = self._traverse(update)
            for i, delta in deltas.items():
                total = self._R[i] + delta
                self._R[i] = np.maximum(total, 0.0) if self.config.regret_matching_plus else total

    def train(self, iterations: int) -> VectorizedCFRSolver:
        for _ in range(iterations):
            self.step()
        return self

    def average_strategy(self) -> Policy:
        out: Policy = {}
        for i, node in enumerate(self._space.nodes):
            S = self._S[i]
            total = S.sum(axis=1)
            for h, key in enumerate(self._space.keys[i]):
                if total[h] <= 0.0:
                    out[key] = {a: 1.0 / len(node.actions) for a in node.actions}
                else:
                    out[key] = {a: float(S[h, j] / total[h]) for j, a in enumerate(node.actions)}
        return out

    def current_strategy(self) -> Policy:
        out: Policy = {}
        for i, node in enumerate(self._space.nodes):
            sigma = _regret_match(self._R[i])
            for h, key in enumerate(self._space.keys[i]):
                out[key] = {a: float(sigma[h, j]) for j, a in enumerate(node.actions)}
        return out

    # Raw tables in the scalar solver's dict shape, for equivalence checks.
    @property
    def regret_sum(self) -> dict[str, dict[str, float]]:
        return self._table(self._R) if self.iterations else {}

    @property
    def strategy_sum(self) -> dict[str, dict[str, float]]:
        return self._table(self._S) if self.iterations else {}

    def evaluate(self, policy: Policy):
        """Exact NashConvResult of `policy` on this solver's game (vectorised)."""
        return nash_conv(self.game, policy)

    # ---- internals ----------------------------------------------------------------

    def _table(self, arrays: list[np.ndarray]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for i, node in enumerate(self._space.nodes):
            A = arrays[i]
            for h, key in enumerate(self._space.keys[i]):
                out[key] = {a: float(A[h, j]) for j, a in enumerate(node.actions)}
        return out

    @property
    def averaging_weight(self) -> float:
        t = self.iterations
        if self.config.averaging == "uniform":
            return 1.0
        if self.config.averaging == "linear":
            return float(t)
        return float(t * t)

    def _apply_dcfr_discount(self) -> None:
        if self.config.dcfr is None:
            return
        alpha, beta, gamma = self.config.dcfr
        t = self.iterations
        if t <= 1:
            return
        pos, neg = _discount(t, alpha), _discount(t, beta)
        strat = ((t - 1) / t) ** gamma
        self._R = [np.where(R > 0, R * pos, R * neg) for R in self._R]
        self._S = [S * strat for S in self._S]

    def _traverse(self, update: list[int]) -> dict[int, np.ndarray]:
        """One traversal with sigma fixed from the current regrets. Updates
        strategy sums for the seats in `update` and returns their regret deltas."""
        sp, n = self._space, self.num_players
        ones = np.ones(sp.H)
        reach: dict[str, list[np.ndarray]] = {"": [ones] * n}
        sigma: list[np.ndarray] = []
        for i, node in enumerate(sp.nodes):                 # forward: reach
            s = _regret_match(self._R[i])
            sigma.append(s)
            r = reach[node.history]
            for j, child in enumerate(node.children):
                rc = list(r)
                rc[node.player] = r[node.player] * s[:, j]
                reach[child] = rc

        cfv: dict[str, dict[int, np.ndarray]] = {
            z: {i: sp.counterfactual_values(z, reach[z], i) for i in update}
            for z in sp.terminals}
        deltas: dict[int, np.ndarray] = {}
        weight = self.averaging_weight
        for i in reversed(range(len(sp.nodes))):              # backward: values
            node, s = sp.nodes[i], sigma[i]
            p = node.player
            child_v = [cfv[c] for c in node.children]
            values = {}
            for seat in update:
                stacked = np.stack([cv[seat] for cv in child_v], axis=1)   # (H, A)
                values[seat] = (stacked * s).sum(axis=1) if seat == p else stacked.sum(axis=1)
                if seat == p:
                    deltas[i] = stacked - values[seat][:, None]
                    own = reach[node.history][p]
                    self._S[i] = self._S[i] + (sp.deal_count[p] * weight * own)[:, None] * s
            cfv[node.history] = values
        return deltas


# ---- exact vectorised evaluation ------------------------------------------------

def _policy_arrays(space: HandSpace, policy: Policy) -> list[np.ndarray]:
    """Policy dict -> per-node (H, A) arrays; missing info sets are uniform,
    matching eval.best_response._full_policy."""
    out = []
    for i, node in enumerate(space.nodes):
        arr = np.empty((space.H, len(node.actions)))
        for h, key in enumerate(space.keys[i]):
            row = policy.get(key)
            arr[h] = ([row[a] for a in node.actions] if row is not None
                      else [1.0 / len(node.actions)] * len(node.actions))
        out.append(arr)
    return out


def _forward(space: HandSpace, probs: list[np.ndarray], skip: int | None) -> dict:
    """Reach vectors per node; seat `skip` (the responder) keeps reach 1."""
    reach = {"": [np.ones(space.H)] * space.num_players}
    for i, node in enumerate(space.nodes):
        r = reach[node.history]
        for j, child in enumerate(node.children):
            rc = list(r)
            if node.player != skip:
                rc[node.player] = r[node.player] * probs[i][:, j]
            reach[child] = rc
    return reach


def nash_conv(game: Game, policy: Policy):
    """Exact per-seat deviation gains of `policy`, as an eval NashConvResult.

    Best response for seat i: walk the betting tree bottom-up with counterfactual
    value vectors (opponents' reach and chance folded in, seat i's own reach
    excluded) and take the per-hand max at seat i's nodes. Because an info set IS
    (hand, history) and the tree is shared, the per-hand max is exactly the
    info-set-consistent best response that eval/best_response.py computes with its
    two-pass bucket construction.
    """
    from pokercfr.eval.exploitability import NashConvResult

    space = _space_for(game)
    probs = _policy_arrays(space, policy)
    n = space.num_players

    full = _forward(space, probs, skip=None)
    values = []
    for seat in range(n):
        values.append(float(sum(full[z][seat] @ space.counterfactual_values(z, full[z], seat)
                                for z in space.terminals)))

    br_values = []
    for seat in range(n):
        reach = _forward(space, probs, skip=seat)
        v = {z: space.counterfactual_values(z, reach[z], seat) for z in space.terminals}
        for i in reversed(range(len(space.nodes))):
            node = space.nodes[i]
            stacked = np.stack([v[c] for c in node.children], axis=1)
            v[node.history] = stacked.max(axis=1) if node.player == seat else stacked.sum(axis=1)
        br_values.append(float(v[""].sum()))
    return NashConvResult(policy_values=values, best_response_values=br_values)


def exploitability(game: Game, policy: Policy) -> float:
    return nash_conv(game, policy).exploitability


def tensor_bytes(config: GameConfig) -> int:
    """Memory the payoff tensors of a rung would need, without building them."""
    H = math.comb(config.ranks * config.suits, config.cards_dealt)
    rep = kuhn_family.sample_deal(config, random.Random(0))
    count, stack = 0, [""]
    while stack:
        h = stack.pop()
        if PokerState(config, rep.hands, h).is_terminal():
            count += 1
        else:
            stack.extend(h + a for a in ("p", "b"))
    return config.players * H ** config.players * count * 8


__all__ = ["HandSpace", "VectorizedCFRSolver", "exploitability", "hand_space", "nash_conv",
           "tensor_bytes"]
