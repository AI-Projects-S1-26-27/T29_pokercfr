# Contributing

This repo backs a course project, so commit history is graded for accountability —
follow these conventions so contributions are traceable to issues and reviewable.

## Branches (optional)

Branches are optional now that everyone pushes to `main` directly — use one only
if you want to park work in progress. If you do:

`<type>/<issue-number>-<short-description>`, all lowercase, words hyphenated.

- `type` is one of: `feat`, `fix`, `test`, `docs`, `infra`, `research`
- Examples: `feat/19-kuhn-poker-env`, `fix/24-leduc-showdown-ties`

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/) style:

```
<type>: <short summary>

<optional body explaining why, not what>
```

- Types: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`
- Summary in imperative mood ("add", not "added"), under 72 chars
- Say *why* in the body, not *what* — the diff already shows what
- Reference the issue in the body or footer (`Refs #19`) when the commit addresses one

Commit history is a graded accountability record, and every member is asked about
their own work in the individual viva. So commit as you go rather than in one
batch at the end: many small commits under your own name are the evidence that
you did the work.

## Pushing

`main` is unprotected and there are no pull requests — push straight to it. With a
deadline this close, waiting on review costs more than it catches.

```bash
git pull --rebase origin main     # before you push, every time
pytest -q                          # 35 tests; takes ~20s
git push origin main
```

Two things follow from having no PR gate, and they matter more than the gate did:

- **Run `pytest -q` before you push.** CI now runs *after* the fact, so a broken
  push breaks `main` for seven other people. Twenty seconds beats an hour of
  someone else debugging your commit.
- **`git pull --rebase` before pushing**, so history stays linear and your commits
  stay attributable to you.

If you want a second pair of eyes on something, just ask in the group chat and
push anyway — nothing is blocked on an approval.

## Workflow

1. Read your module's docstring. It is the brief: what to build, why the study
   needs it, and the acceptance criteria that count as done.
2. Write the tests from those acceptance criteria, then make them pass.
3. `git pull --rebase`, `pytest -q`, push.
4. Launch your long experiment matrix as soon as your module works — the runs take
   12–17 hours, so they gate everything downstream of them.

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
| Akhil Thindi | @hippiegeneral | `pokercfr/analysis/`, `figures/` | Bootstrap CIs, paired tests, scaling-exponent fits; every figure in the report; fresh-clone reproducibility check |

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

There is no required review — `main` is unprotected and anyone can push. The
ownership table above is about *avoiding collisions*, not about permission: if you
need a change inside someone else's file, message them rather than editing it, so
two people are never fixing the same thing twice.

`.github/CODEOWNERS` is kept as documentation of who to ask about what. With PRs
switched off GitHub no longer enforces it.

### Reproducibility rule

Every number and every figure in the final report must be produced by a committed
script and traceable to a file in `results/`. Nothing is hand-copied, estimated,
or remembered. If a result cannot be regenerated from a fresh clone, it does not
go in the report.
