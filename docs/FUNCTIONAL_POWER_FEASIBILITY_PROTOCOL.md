# R-C1 power-proposal feasibility v1

2026-10-10. Analytical design sensitivity, not a confirmation or report experiment.
The attribution endpoint, alpha/effect proposals and correlation assumptions are
unfrozen. Human rubric review and factual-audit qualification remain required.
No report, rating, training run, success gate or historical verdict is added.

## Question and model

R-C1 proposes a 25-point paired effect, a one-sided exact McNemar test at 0.05 and
roughly 48 paired episodes. The earlier self-label protocol uses 16 episodes per
each of three model pairs. Test how independence assumptions alter the rejection
rate of that proposed design; do not treat the older observed rates as fitted
parameters for a new endpoint.

For one pair, define Y=+1 if only the target condition is positive, Y=-1 if only the
comparison is positive and Y=0 if concordant. Use p01=0.05/0.10 and p10=p01+delta,
with delta=0 or 0.25. Concordance has probability 1-p10-p01. These assumed joint
distributions do not supply human ratings; marginals alone cannot determine power.

For each independent model, with probability rho all its episode pairs share one
categorical outcome; otherwise pairs are independent. Use rho=0,0.1,0.3,1. This
illustrative mixture preserves pair marginals and yields Corr(Y_i,Y_j)=rho within
a model by total covariance. It is not an estimate of actual model/report/rater
dependence and does not cover every distribution with the same correlation.

Enumerate within-model multinomial/shared counts and convolve across independent
models. Evaluate the proposed pooled fair-binomial discordance test on those
counts. Also evaluate an illustrative whole-model sign test, using one sign of
the paired-count difference per model and dropping ties. This latter test concerns
model-level sign balance, not automatically the population mean difference; its
null is symmetric under this illustrative model. It is not selected for confirmation.

A one-sided whole-model sign/swap test with K independent models has smallest
possible p-value 2^-K (larger with ties). Three models cannot reach 0.05 by that
test, regardless of episodes per model. This is test resolution, not proof that
three models cannot yield useful component evidence or conditional inference.

## Fixed grid, calculations and retention

Use model/episode budgets (3,16), (6,8), (12,4), (6,16): 48 pairs in the first
three designs and 96 in the fourth. Evaluate all 64 design/joint-rate/effect/rho
combinations, retaining null rejection rates as well as assumed-effect power.
Independent-model signs and paired counts use exact discrete probability sums;
floating-point evaluation is checked for probability mass and independence limits.
There is no Monte Carlo sampling or seed selection.

Retain every within-model, pooled and sign-count probability array, exact sources,
all numerical rows and deterministic compressed bytes. Reserve the analytical
attempt and refuse overwrite/retry. Replay the entire grid. Prepared sources are
saved locally before calculation; network restrictions prevent publication, so
this is explicitly not a published preregistration.

Interpret higher null rejection as a sensitivity to the assumed correlation and
population target, not a retroactive failure of the registered self-label study.
Independent episodes conditional on fixed checkpoints are a different estimand
from generalization over new training models. Additional models, cluster-aware
analysis, agreement, multiplicity and endpoint validity must be decided together
after review. Numerical power for a generic grounding feature does not supply
consciousness-report correspondence or achieve the source-of-qualia objective.
