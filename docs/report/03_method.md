# 3. Method

_Drafted by Aarush (evaluation, design) and Himnish (solver, runner). Sections marked TODO are
for Himnish._

## 3.1 The game ladder and the frozen contract
Every game implements one `Game`/`GameState`/`Solver` protocol (`pokercfr/core/protocols.py`),
so solvers and evaluators never depend on a particular game. Games: the Kuhn-family ladder
(`pokercfr/games/kuhn_family.py`) and Leduc with configurable raise cap and N players on a fixed
deck (`pokercfr/games/leduc.py`). _TODO (Himnish/Samanyu): table of rungs with measured info sets,
deals and seconds per iteration, from `results/verify/leduc_sizes.json`; label every projected
figure as projected._

## 3.2 The 2^3 ablation cube
One solver, three switches: regret matching vs regret matching+, uniform vs linear averaging,
simultaneous vs alternating updates; the corners include vanilla CFR and CFR+. One code path means
differences between corners are the switch, not the implementation. _TODO (Himnish)._

## 3.3 Exact best response and NashConv
Best response uses two passes (`pokercfr/eval/best_response.py`). Pass 1 buckets the responder's
nodes by information set, carrying counterfactual reach (chance times the other players' action
probabilities, excluding the responder's own). Pass 2 fixes information sets deepest-first, valid
because every responder node inside an information set's subtree is strictly deeper and all
states in one information set share a history length. Values are memoised on the state value
rather than `id(state)`, since CPython recycles ids of garbage-collected states.

Per-seat gains are reported with total, mean, max and the max-deviator seat
(`pokercfr/eval/exploitability.py`). Exploitability is half the total and exists for two players
only. For comparisons across player counts we use the **mean**, because the total grows with N by
construction.

## 3.4 Three validation layers
1. Internal invariants: gains are never negative, the best-response value is realised by playing
   the best-response policy, uniform Kuhn exploitability is 11/24, and CFR+ reaches -1/18
   (`scripts/a1_br_audit.py`).
2. Independent implementation: agreement with OpenSpiel on Kuhn and on Leduc
   (`results/verify/br_audit.json`, `results/verify/leduc_crosscheck.json`).
3. Exact oracle: sequence-form LP Nash solution. _TODO (Siddhant): not yet landed at time of
   writing; do not claim until it has._

## 3.5 Pre-registration and the runner
Epsilon targets, budgets, seeds, raise caps and decision rules were fixed and committed before the
final sweeps (`docs/report/preregistration.md`, `results/verify/preregistration.json`). H1 is
stated in wall-clock terms, so its cells are wall-clock budgeted with evaluation time excluded
from the training clock. Each run writes one JSON record in a versioned schema, named by the hash
of its spec; `scripts/a6_results_table.py` rejects any file whose hash does not match.
