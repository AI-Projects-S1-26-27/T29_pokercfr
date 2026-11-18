# 5. Discussion and limitations

Limitations are listed as they stand, alongside the scope of the evidence behind each hypothesis.

- **Coverage.** See the coverage table in `numbers.md`. The scope of each matrix is stated here
  rather than left implicit.
- **Cells cut for feasibility.** From the measured census (`results/verify/leduc_sizes.json`),
  Leduc with 4 players at cap 2 and above, 5 players at cap 2 and above, and 6 players at any cap
  cannot finish, so H3 on Leduc reaches 4 players at cap 1 (5 as a stretch cell). The upper end of
  the player-count axis is therefore not tested.
- **H3 on Leduc is not iteration-matched.** Its cells are wall-clock budgeted, so larger N gets
  fewer iterations and a rising NashConv could partly reflect less training. The Kuhn-family
  comparison is iteration-matched; the Leduc one must be read alongside its iteration counts.
- **Seeds for deterministic solvers.** Full-traversal CFR ignores its seed, so replicates in H1
  measure timing noise on a given machine, not algorithmic variance. H1 was measured on one class
  of hardware.
- **H2 rests on four rungs.** The scaling CI will be wide; an inconclusive result is reported as
  inconclusive.
- **Projected versus measured timings.** `LADDER.md` mixes measured and projected timings and its
  Kuhn-family table differs from the timings in `registry.py`; relabel or re-measure before any
  figure from it is cited. Projected values in the Leduc census are marked as such in the JSON.
- **Hypothesis numbering.** `LADDER.md` uses an earlier numbering of H1-H3; this report follows
  `preregistration.md`.
- **Citations.** The submitted proposal's references were `[Link]` placeholders and lost 2 marks.
  Every citation in the final report needs a real venue and a DOI or arXiv id; the NeurIPS and
  Kuhn entries in Section 2 still need page-level verification.
- **Head-to-head play** is supporting evidence only. A strategy can beat every baseline and still
  be heavily exploitable, so win rate is never used as a measure of equilibrium quality.
- **Stale documents.** The repository `README.md` still describes an earlier module layout
  (`kuhn_poker.env`, unittest runner) and should be brought in line before submission.
