# 2. Background

**Extensive-form games and CFR.** In an imperfect-information game a player must act identically
at all states in one information set. CFR (Zinkevich et al., 2007) runs regret matching at every
information set and, in two-player zero-sum games, the time-averaged strategy converges to a Nash
equilibrium. Exploitability, the best-response value against the average strategy, measures
distance from equilibrium. For three or more players no such guarantee exists, so we measure
NashConv, the sum over seats of each seat's gain from deviating to a best response, instead of
assuming convergence.

**CFR+.** Tammelin (2014) clamps negative regrets to zero (regret matching+), averages linearly,
and updates players alternately. It converges far faster than vanilla CFR in practice. Our
ablation cube separates these three ingredients, because the published algorithm changes all
three at once.

**Discounted CFR.** Brown & Sandholm (2019) generalise the family with discount parameters
(alpha, beta, gamma) and show CFR and CFR+ sit at its limits. We test whether our cube
reproduces that unification.

**Monte Carlo CFR.** Lanctot et al. (2009) trade per-iteration cost for variance by sampling
chance and actions (outcome, external and chance sampling), which is what makes large decks
tractable and why sampled results need multiple seeds.

**Leduc Hold'em.** Introduced by Southey et al. (2005) as a small poker game with two betting
rounds and a public card, and Kuhn poker (Kuhn, 1950) before it, whose equilibria are known in
closed form (game value -1/18 for the first player). OpenSpiel (Lanctot et al., 2019) provides
independent implementations; we use it as a ground-truth check, matching its exploitability and
NashConv to about 1e-16 on Kuhn.

**Multiplayer.** Brown & Sandholm (2019) reached superhuman six-player no-limit play without a
convergence guarantee; we measure the corresponding gap on games where exact NashConv is
computable.

## References

- Bowling, M., Burch, N., Johanson, M., Tammelin, O. Heads-up limit hold'em poker is solved. *Science* 347(6218):145-149, 2015. doi:10.1126/science.1259433
- Brown, N., Sandholm, T. Solving imperfect-information games via discounted regret minimization. *AAAI* 2019. arXiv:1809.04040
- Brown, N., Sandholm, T. Superhuman AI for multiplayer poker. *Science* 365(6456):885-890, 2019. doi:10.1126/science.aay2400
- Kuhn, H. W. A simplified two-person poker. In *Contributions to the Theory of Games, Vol. I*, Annals of Mathematics Studies 24, pp. 97-103. Princeton University Press, 1950.
- Lanctot, M., Waugh, K., Zinkevich, M., Bowling, M. Monte Carlo sampling for regret minimization in extensive games. *NeurIPS* 22, 2009.
- Lanctot, M. et al. OpenSpiel: a framework for reinforcement learning in games. 2019. arXiv:1908.09453
- Southey, F., Bowling, M., Larson, B., Piccione, C., Burch, N., Billings, D., Rayner, C. Bayes' bluff: opponent modelling in poker. *UAI* 2005. arXiv:1207.1411
- Tammelin, O. Solving large imperfect information games using CFR+. 2014. arXiv:1407.5042
- Zinkevich, M., Johanson, M., Bowling, M., Piccione, C. Regret minimization in games with incomplete information. *NeurIPS* 20, 2007.

_Citation check owed before submission: the NeurIPS and Kuhn entries have no DOI/arXiv id, so
confirm volume and page numbers against the publisher pages. Verify every entry above against its
source rather than trusting this list._
