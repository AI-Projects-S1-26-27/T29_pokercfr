"""Sequence-form LP -- an INDEPENDENT exact Nash oracle. OWNER: Siddhant.

This is the project's strongest answer to "what does your code add beyond
re-running OpenSpiel". Everything else here is an iterative regret-minimiser that
*approaches* equilibrium; this solves for one exactly, by a completely different
route -- linear programming over the sequence form (Koller, Megiddo & von Stengel
1996; von Stengel 1996).

Two consequences, both load-bearing:

  1. VERIFICATION. A converged CFR policy and the LP solution must agree on the
     game value. Agreement across two unrelated algorithms is far stronger evidence
     of correctness than either one matching itself. Combined with Kuhn's closed
     form and OpenSpiel, headline numbers are checked three independent ways.

  2. A NEW ARTIFACT. It yields the EXACT equilibrium value of every 2-player rung
     of our ladder. Those games are ours, so those numbers exist nowhere else --
     and with exact values in hand, "iterations to reach epsilon" is measured
     against truth rather than against a long CFR run used as a proxy.

Formulation (2-player zero-sum, perfect recall). Realization plans live on
sequences, not behavioural probabilities, which makes the constraints LINEAR:
    maximise   v  subject to  payoff and information-set flow constraints
Build the sparse payoff matrix over terminal sequence pairs and each player's
constraint matrix (one row per info set: inflow from the parent sequence equals
outflow over that info set's actions), then hand it to scipy.optimize.linprog
with method="highs".

Scope honestly: this is for 2 players. N-player equilibrium computation is not an
LP (it is PPAD-complete), and the report must say so rather than imply the oracle
covers H3.

ACCEPTANCE (tests/test_lp_oracle.py):
  - Kuhn game value == -1/18 to 1e-9 (tighter than any CFR run reaches)
  - LP value agrees with 200k-iteration CFR+ to < 1e-4 on every 2-player rung
    up to `4rank-2suit-2dealt`
  - the LP policy's exploitability, measured by our own best_response, is ~0 (<1e-9)
  - raises a clear error for num_players > 2 rather than returning nonsense
"""
from __future__ import annotations

import time
from array import array
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from pokercfr.core.protocols import Game, GameState, Policy

# ---------------------------------------------------------------------------
# IMPLEMENTATION NOTES
#
# Sequence indexing
#   A *sequence* of seat p is the list of p's own (info set, action) pairs on the
#   path from the root. Under perfect recall it is identified by its LAST pair,
#   so seat p's sequences are: index 0 = the empty sequence (root), then one
#   index per (info-set key, action) in the order the tree walk first meets
#   them. `parent[I]` is the sequence p had played on reaching info set I; perfect
#   recall means it is the same for every state in I, and the walk checks that.
#
# Payoff matrix (sparse, never dense)
#   A[s0, s1] = sum over terminal z with seat sequences (s0, s1) of
#               chance_reach(z) * u_0(z).
#   Chance is folded in here, so x^T A y is seat 0's expected payoff for
#   realization plans x and y. Only sequence pairs that actually end at a
#   terminal are stored; duplicates are summed by the COO -> CSR conversion.
#
# Flow constraints (one row per info set, plus the root)
#   E x = e:   x[empty] = 1,   and for each info set I of seat 0:
#              sum_a x[I, a] - x[parent(I)] = 0.
#   F y = f is the same for seat 1. x, y >= 0.
#
# The LP pair (seat 0 maximises, seat 1 minimises x^T A y)
#   For fixed x, seat 1's best response  min_y x^T A y  s.t. Fy = f, y >= 0  has
#   the LP dual  max_q f^T q  s.t.  F^T q <= A^T x.  Substituting it in gives one
#   LP for seat 0's maximin plan:
#       (P0)  max_{x, q}  q[root]   s.t.  F^T q - A^T x <= 0,  E x = e,  x >= 0
#   and symmetrically for seat 1's minimax plan:
#       (P1)  min_{y, p}  p[root]   s.t.  A y - E^T p <= 0,    F y = f,  y >= 0.
#   (P0) and (P1) are LP duals of each other, so their optima are equal -- the
#   game value. We solve BOTH rather than read y off (P0)'s marginals: each
#   seat's plan then comes from its own optimisation, and the equality of the two
#   optima is a free consistency check on the whole construction.
#
# Back to behaviour
#   pi(a | I) = x[I, a] / x[parent(I)]. If x[parent(I)] is (numerically) zero the
#   info set is unreachable under the plan, every action is equally consistent
#   with equilibrium play, and we return uniform -- the same convention
#   eval/best_response._full_policy uses for info sets a policy omits.
# ---------------------------------------------------------------------------

# Tolerances. HiGHS's defaults (1e-7) are looser than the 1e-9 acceptance bar
# on the Kuhn value, so feasibility is tightened. They are not set lower: past
# ~1e-10 HiGHS starts rejecting its own optimal bases on rounding noise.
DEFAULT_HIGHS_OPTIONS = {
    "primal_feasibility_tolerance": 1e-10,
    "dual_feasibility_tolerance": 1e-10,
}
ZERO_SUM_TOL = 1e-9        # |u0 + u1| allowed at a terminal
CHANCE_SUM_TOL = 1e-9      # |sum of chance probabilities - 1| allowed
VALUE_AGREEMENT_TOL = 1e-7  # |value(P0) - value(P1)| before we refuse the answer
UNREACHABLE_TOL = 1e-12    # parent realization weight below this = unreachable


class LPOracleError(RuntimeError):
    """The LP could not be solved, or its two halves disagree."""


@dataclass
class SequenceForm:
    """The sequence-form representation of a two-player zero-sum game."""

    # seq_index[p][(key, action)] -> column of seat p; the empty sequence is 0.
    seq_index: list[dict[tuple[str, str], int]]
    # infosets[p][key] -> (parent sequence index, actions in legal order)
    infosets: list[dict[str, tuple[int, list[str]]]]
    payoff: sparse.csr_matrix          # A, shape (n0, n1), seat-0 payoff
    E: sparse.csr_matrix               # seat 0 flow constraints
    F: sparse.csr_matrix               # seat 1 flow constraints
    terminals: int                     # terminal histories visited
    build_seconds: float

    @property
    def num_sequences(self) -> tuple[int, int]:
        return (len(self.seq_index[0]) + 1, len(self.seq_index[1]) + 1)

    @property
    def num_infosets(self) -> tuple[int, int]:
        return (len(self.infosets[0]), len(self.infosets[1]))


@dataclass
class LPSolution:
    value: float                       # seat 0's equilibrium value, from (P0)
    value_check: float                 # the same value, from (P1)
    policy: Policy                     # behavioural, both seats, every info set
    plans: list[np.ndarray]            # realization plans x, y
    form: SequenceForm
    unreachable: list[str] = field(default_factory=list)
    solve_seconds: float = 0.0
    max_flow_residual: float = 0.0     # max |E x - e|, |F y - f| after solving

    def stats(self) -> dict:
        n0, n1 = self.form.num_sequences
        i0, i1 = self.form.num_infosets
        return {
            "value": self.value, "value_check": self.value_check,
            "sequences": [n0, n1], "infosets": [i0, i1],
            "payoff_nnz": int(self.form.payoff.nnz), "terminals": self.form.terminals,
            "unreachable_infosets": len(self.unreachable),
            "max_flow_residual": self.max_flow_residual,
            "build_seconds": self.form.build_seconds, "solve_seconds": self.solve_seconds,
        }


# ---- 1. matrix construction ----------------------------------------------------

def build_sequence_form(game: Game) -> SequenceForm:
    """Walk the tree once and build A, E, F. Raises ValueError on any structure the
    two-player zero-sum sequence form cannot represent."""
    if game.num_players != 2:
        raise ValueError(
            f"the sequence-form LP oracle solves TWO-player zero-sum games only; "
            f"{getattr(game, 'name', game)!r} has {game.num_players} players. "
            "N-player Nash equilibrium is not a linear program (it is PPAD-complete), "
            "so use NashConv from eval/exploitability for N >= 3 instead.")

    started = time.perf_counter()
    seq_index: list[dict[tuple[str, str], int]] = [{}, {}]
    infosets: list[dict[str, tuple[int, list[str]]]] = [{}, {}]
    rows, cols, vals = array("q"), array("q"), array("d")
    terminals = 0

    def register(p: int, key: str, parent: int, actions: list[str]) -> None:
        known = infosets[p].get(key)
        if known is None:
            if key in infosets[1 - p]:
                raise ValueError(f"info-set key {key!r} is used by both seats; keys must "
                                 "be unique per seat for the sequence form")
            infosets[p][key] = (parent, list(actions))
            for a in actions:
                seq_index[p][(key, a)] = len(seq_index[p]) + 1
            return
        if known[0] != parent:
            raise ValueError(f"perfect recall violated at info set {key!r} (seat {p}): it "
                             "is reached after two different own-action sequences")
        if known[1] != list(actions):
            raise ValueError(f"info set {key!r} (seat {p}) has different legal actions in "
                             "different states")

    # Iterative DFS: the 52-card rung has 1.6M deals, so no recursion per deal.
    stack: list[tuple[GameState, float, int, int]] = [(game.new_initial_state(), 1.0, 0, 0)]
    while stack:
        state, reach, s0, s1 = stack.pop()
        if state.is_terminal():
            u = state.utilities()
            if len(u) != 2:
                raise ValueError(f"terminal returned {len(u)} utilities for a 2-player game")
            if abs(u[0] + u[1]) > ZERO_SUM_TOL:
                raise ValueError(f"game is not zero-sum: utilities {u} at a terminal")
            terminals += 1
            if reach > 0.0 and u[0] != 0.0:
                rows.append(s0)
                cols.append(s1)
                vals.append(reach * float(u[0]))
            continue
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            probs = [float(p) for _, p in outcomes]
            if any(p < 0.0 for p in probs) or abs(sum(probs) - 1.0) > CHANCE_SUM_TOL:
                raise ValueError(f"chance probabilities must be >= 0 and sum to 1, got sum "
                                 f"{sum(probs)!r}")
            for (child, _), p in zip(outcomes, probs, strict=True):
                if p > 0.0:
                    stack.append((child, reach * p, s0, s1))
            continue
        p = state.current_player()
        if p not in (0, 1):
            raise ValueError(f"current_player() returned seat {p} in a 2-player game")
        actions = state.legal_actions()
        if not actions:
            raise ValueError("decision node with no legal actions")
        key = state.information_set_key(p)
        register(p, key, s0 if p == 0 else s1, actions)
        for a in actions:
            seq = seq_index[p][(key, a)]
            child = state.apply_action(a)
            stack.append((child, reach, seq, s1) if p == 0 else (child, reach, s0, seq))

    n0, n1 = len(seq_index[0]) + 1, len(seq_index[1]) + 1
    payoff = sparse.coo_matrix(
        (np.frombuffer(vals, dtype=np.float64),
         (np.frombuffer(rows, dtype=np.int64), np.frombuffer(cols, dtype=np.int64))),
        shape=(n0, n1)).tocsr()  # sums duplicate (s0, s1) entries
    E = _flow_matrix(seq_index[0], infosets[0])
    F = _flow_matrix(seq_index[1], infosets[1])
    return SequenceForm(seq_index, infosets, payoff, E, F, terminals,
                        time.perf_counter() - started)


def _flow_matrix(seq_index: dict, infosets: dict) -> sparse.csr_matrix:
    """Row 0: x[empty] = 1. Row r >= 1 (info set I): sum_a x[I,a] - x[parent(I)] = 0."""
    r, c, v = [0], [0], [1.0]
    for row, (key, (parent, actions)) in enumerate(infosets.items(), start=1):
        r.append(row)
        c.append(parent)
        v.append(-1.0)
        for a in actions:
            r.append(row)
            c.append(seq_index[(key, a)])
            v.append(1.0)
    return sparse.coo_matrix((v, (r, c)), shape=(len(infosets) + 1, len(seq_index) + 1)).tocsr()


# ---- 2. optimisation -----------------------------------------------------------

def _solve_side(A: sparse.csr_matrix, own: sparse.csr_matrix, other: sparse.csr_matrix,
                options: dict) -> tuple[float, np.ndarray]:
    """max_{x, q} q[0]  s.t.  other^T q - A^T x <= 0,  own x = e0,  x >= 0, q free.

    With A = seat-0 payoff this is (P0). (P1) is the same problem for seat 1 with
    A replaced by -A^T (seat 1 maximises -x^T A y), so one routine serves both.
    Returns (optimal objective, realization plan of the maximiser).
    """
    n_own, m_other = own.shape[1], other.shape[0]
    c = np.zeros(n_own + m_other)
    c[n_own] = -1.0  # maximise q[root]
    A_ub = sparse.hstack([-A.T, other.T], format="csr")
    A_eq = sparse.hstack([own, sparse.csr_matrix((own.shape[0], m_other))], format="csr")
    b_eq = np.zeros(own.shape[0])
    b_eq[0] = 1.0
    bounds = [(0.0, None)] * n_own + [(None, None)] * m_other
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]), A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs", options=options)
    if res.status != 0 or res.x is None:
        raise LPOracleError(f"HiGHS failed (status {res.status}): {res.message}")
    return -float(res.fun), np.asarray(res.x[:n_own])


def _behavioural(plan: np.ndarray, form: SequenceForm, p: int,
                 policy: Policy, unreachable: list[str]) -> None:
    for key, (parent, actions) in form.infosets[p].items():
        weights = np.array([max(plan[form.seq_index[p][(key, a)]], 0.0) for a in actions])
        total = weights.sum()
        if plan[parent] <= UNREACHABLE_TOL or total <= UNREACHABLE_TOL:
            unreachable.append(key)
            policy[key] = {a: 1.0 / len(actions) for a in actions}
        else:
            # Normalise by the children's sum rather than by plan[parent]: the two
            # are equal up to feasibility tolerance, and this way the row sums to
            # 1 exactly.
            policy[key] = {a: float(w / total) for a, w in zip(actions, weights, strict=True)}


def solve(game: Game, options: dict | None = None) -> LPSolution:
    """Build the sequence form, solve (P0) and (P1), and convert to behaviour."""
    form = build_sequence_form(game)
    opts = {**DEFAULT_HIGHS_OPTIONS, **(options or {})}
    started = time.perf_counter()
    value0, x = _solve_side(form.payoff, form.E, form.F, opts)
    neg_value0, y = _solve_side(-form.payoff.T.tocsr(), form.F, form.E, opts)
    value_check = -neg_value0
    solve_seconds = time.perf_counter() - started
    if abs(value0 - value_check) > VALUE_AGREEMENT_TOL:
        raise LPOracleError(f"primal and dual LPs disagree on the game value: {value0!r} vs "
                            f"{value_check!r}; the sequence form is inconsistent")
    e = np.zeros(form.E.shape[0])
    e[0] = 1.0
    f = np.zeros(form.F.shape[0])
    f[0] = 1.0
    residual = max(float(np.abs(form.E @ x - e).max()), float(np.abs(form.F @ y - f).max()))

    policy: Policy = {}
    unreachable: list[str] = []
    _behavioural(x, form, 0, policy, unreachable)
    _behavioural(y, form, 1, policy, unreachable)
    return LPSolution(value0, value_check, policy, [x, y], form, unreachable,
                      solve_seconds, residual)


def solve_exact(game: Game) -> tuple[float, Policy]:
    """Exact equilibrium value (for seat 0) and an equilibrium policy."""
    sol = solve(game)
    return sol.value, sol.policy


class LPOracleSolver:
    """Solver-shaped wrapper so the LP oracle runs through the normal harness.

    It is not iterative -- one `step()` solves the game exactly and every later
    step is a no-op. Wrapping it this way means the runner, the result format and
    the figures treat the exact answer as just another row, so "LP vs CFR+" is a
    comparison the existing plotting code can already make.
    """

    def __init__(self, game):
        self.game = game
        self.iterations = 0
        self._policy = None
        self.value: float | None = None

    def step(self) -> None:
        if self._policy is None:
            self.value, self._policy = solve_exact(self.game)
        self.iterations += 1

    def train(self, iterations: int) -> LPOracleSolver:
        """The runner's wall-clock loop calls train(); one step is the whole solve."""
        for _ in range(iterations):
            self.step()
        return self

    def average_strategy(self):
        if self._policy is None:
            self.step()
        return self._policy

    def current_strategy(self):
        return self.average_strategy()


__all__ = ["LPOracleError", "LPOracleSolver", "LPSolution", "SequenceForm",
           "build_sequence_form", "solve", "solve_exact"]
