# Contributing

This repo backs a course project, so commit history is graded for accountability —
follow these conventions so contributions are traceable to issues and reviewable.

## Branch naming

`<type>/<issue-number>-<short-description>`, all lowercase, words hyphenated.

- `type` is one of: `feat`, `fix`, `test`, `docs`, `infra`, `research`
- Examples: `feat/19-kuhn-poker-env`, `docs/18-contributing-guide`, `fix/24-leduc-showdown-ties`

Branch off `main`. One issue per branch.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/) style:

```
<type>: <short summary>

<optional body explaining why, not what>
```

- Types: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`
- Reference the issue in the body or footer (`Refs #19`) when the commit addresses one
- Summary in imperative mood ("add", not "added"), under 72 chars

## Pull requests

- One issue per PR. Title mirrors the branch's intent.
- Description must state: what changed, why, and how it was verified (test output,
  example run, plot, etc.)
- Link the issue it closes (`Closes #19`)
- Every PR gets at least a self-review comment before merge — call out any known
  gaps or follow-up issues filed
- **Request review from `@Samanyu-dev` or `@mrlegendary25` and get an approval
  before merging.** This is enforced by branch protection on `main` — a PR
  can't be merged without at least one approving review from a code owner.
- Squash-merge into `main` once approved, keeping the PR title as the merge commit
  summary

## Full workflow

1. Pick up (or get assigned) an issue. If the work doesn't map to an existing
   issue, open one first — no unowned or untracked work.
2. Branch off `main` using the naming convention above.
3. Implement, with a test per non-trivial piece of logic (see existing
   `tests/` for the pattern — plain `unittest`/`pytest`, no framework beyond
   that unless a task genuinely needs one).
4. Open a PR against `main`: description states what/why/how-verified, links
   `Closes #N`, includes a short self-review comment.
5. Request review from `@Samanyu-dev` or `@mrlegendary25`. Address feedback.
6. Once approved, squash-merge. Don't delete the branch (kept for the
   course's commit-history accountability record).

## Team and ownership

Eight members, one owner per file. **Nobody edits a file they do not own** — that
is what lets eight people work in parallel on the same repo without merge
conflicts, and it is enforced by [`.github/CODEOWNERS`](.github/CODEOWNERS).
Cross-cutting changes get an issue first.

Ownership mirrors the role table submitted in the project proposal, so the final
report's Individual Contributions section matches what each member committed to.

| Member | GitHub | Owns exclusively | Scope |
|---|---|---|---|
| Himnish Lalchandani | [@mrlegendary25](https://github.com/mrlegendary25) | `pokercfr/solvers/cfr.py`, `pokercfr/experiments/` | CFR / CFR+ / DCFR and the ablation cube; experiment runner; code review lead |
| Samanyu Reddy Allipuram | [@Samanyu-dev](https://github.com/Samanyu-dev) | `pokercfr/games/leduc.py` | Leduc Hold'em: public card, two betting rounds, configurable raise cap, N players at a fixed deck |
| Aarush Kanipakam | [@Aarush-Kanipakam](https://github.com/Aarush-Kanipakam) | `pokercfr/eval/best_response.py`, `pokercfr/eval/exploitability.py`, `pokercfr/experiments/registry.py` | Exact best response, exploitability, NashConv; experiment pre-registration and results matrix; report lead |
| Sai Aashish Juryala | [@SaiAashishJ](https://github.com/SaiAashishJ) | `pokercfr/games/kuhn_family.py`, `pokercfr/core/config.py` | Kuhn-family ladder engine, environment tests, seeding, determinism and reproducibility |
| Vikranth Reddy Vanga Reddy | `Vik217` (invite pending) | `pokercfr/solvers/mccfr.py`, `pokercfr/solvers/deep_cfr.py` | Outcome / external / chance-sampled MCCFR; Deep CFR (gated stretch) |
| Siddhant Narlawar | _handle TBD_ | `pokercfr/solvers/lp_oracle.py`, `pokercfr/solvers/vectorized.py` | Sequence-form LP exact Nash oracle; vectorized traversal for large decks |
| Shashwat Kumar | _handle TBD_ | `pokercfr/agents/`, `pokercfr/eval/tournament.py` | Baseline opponent pool; seat-rotated duplicate-deal match runner and cross-play |
| Akhil Thindi | _handle TBD_ | `pokercfr/analysis/`, `figures/` | Bootstrap CIs, paired tests, scaling-exponent fits; every figure in the report; fresh-clone reproducibility check |

Members whose handle reads _TBD_: open a one-line PR adding it to this table and
to `.github/CODEOWNERS`. Until then CODEOWNERS routes review of those paths to
the two leads, so review is never blocked — but per-file ownership is not
enforced by GitHub for them.

### The frozen contract

[`pokercfr/core/protocols.py`](pokercfr/core/protocols.py) defines the
`Game` / `GameState` / `Solver` interfaces that every game, solver and evaluator
codes against. Changing it affects all eight members at once, so it requires a PR
reviewed by **both** leads. Everything else is designed so it should not need one.

### Working in parallel

1. **Stub first, fill second.** Every module ships a type-correct stub whose
   docstring states its task and acceptance criteria. An unfinished module still
   imports, and the experiment runner prints `SKIP` rather than crashing, so no
   member is ever blocked waiting on another.
2. **Read your module's docstring before writing code.** It is the brief: what to
   build, why it matters to the study, and the tests that must pass.
3. **One branch and one PR per module**, under your own name. Commit history is a
   graded accountability record — push early and often rather than in one batch.

### Review

`@Samanyu-dev` and `@mrlegendary25` are the two required reviewers (see
[Pull requests](#pull-requests)). Review load follows the architecture: each
mainly reviews PRs adjacent to their own layer, but either can review any PR.

### Reproducibility rule

Every number and every figure in the final report must be produced by a committed
script and traceable to a file in `results/`. Nothing is hand-copied, estimated,
or remembered. If a result cannot be regenerated from a fresh clone, it does not
go in the report.
