# Power depends on the population target and model dependence

2026-10-10. Exact analytical sensitivity of R-C1's sample-size proposal, following
the [local preparation protocol](FUNCTIONAL_POWER_FEASIBILITY_PROTOCOL.md).
No reports, human ratings, training, empirical correlation estimates or success
gate are added. The endpoint and numerical criteria remain unfrozen. Preparation
and results are saved as isolated local commits; publication is unavailable under
the current environment restrictions.

## What 48 independent pairs would establish

For an assumed 25-point paired difference, with p01=0.05/0.10 and p10=p01+0.25,
a one-sided exact McNemar test at 0.05 has 90.722% / 80.636% power at 48 independent
pairs. Concordant pairs can be both positive or both negative; their split does
not affect this test. These assumed discordance rates are not fitted human ratings.
The corresponding null rejection rates are 1.166% / 2.275%, below nominal because
of discreteness. This supports a conditional feasibility calculation, not a frozen
sample size or evidence that the endpoint is consciousness-related.

## Illustrative dependence within three models

The mixture assumes each independent model's episodes share one outcome with
probability rho; otherwise they are independent. It preserves pair marginals and
gives within-model paired-difference correlation rho. This is one explicit sensitivity
family, not an estimate or comprehensive model of actual dependence. Each range
below covers the two assumed p01 values, with three models and 16 episodes each.

| Assumed within-model correlation | Pooled test null rejection | Pooled test power for assumed effect |
|---|---|---|
| 0 | 1.166–2.275% | 80.636–90.722% |
| 0.1 | 2.441–4.895% | 77.208–86.615% |
| 0.3 | 4.911–9.632% | 70.903–78.446% |
| 1, perfect shared outcome | 12.875–22.000% | 59.625–59.938% |

The hand-checkable null case rho=1, p10=p01=0.1 has rejection probability
0.192 + 0.024 + 0.001 + 0.003 = 0.22 when each model outcome is repeated 16 times
and the test treats the repeats as independent. This is a synthetic population-null
sensitivity, not a reanalysis that overturns the registered self-label verdict.

## Independent models and test resolution

An illustrative one-sided whole-model sign test uses one sign per model's paired
count difference and drops ties. Its smallest possible p-value is 2^-K (larger
with ties); three models have minimum 0.125 and can never reach 0.05 by this test.
At least five non-tied models are necessary for that resolution, not sufficient
for adequate power. Other analyses require their own assumptions and validation;
this does not say three models cannot supply useful mechanistic or conditional
evidence.

For zero assumed correlation, the exact whole-model sign-test powers are:

| Independent models × episodes/model | Total pairs | Whole-model sign power |
|---|---|---|
| 3 × 16 | 48 | 0% |
| 6 × 8 | 48 | 51.058–64.649% |
| 12 × 4 | 48 | 62.048–77.485% |
| 6 × 16 | 96 | 77.998–89.344% |

The sign test discards magnitudes and tests model-level sign balance, not automatically
the population mean difference. Its null is symmetric in the assumed family. These
powers are not a recommendation to select that test or a threshold to pass. At
larger assumed correlation many models become tied, reducing effective evidence;
all such cases are retained in the archive. More independent models do not make
naive episode pooling valid by themselves.

## Interpretation and next design decisions

Independent episodes conditional on fixed checkpoints target those checkpoints;
generalization to newly trained systems targets a model population. Define that
estimand before selecting a unit, test, model/episode budget or multiplicity rule.
No correlation of actual human report judgments has yet been estimated, and
factual-audit qualification and independent definition review are still pending.

The next protocol must justify its report correspondence and endpoint, qualify its
measurement, assess feasible dependence on development data, then evaluate power
for the chosen clustered estimand. A generic grounding feature with adequate power
still does not establish source-of-qualia evidence. No earlier gate, report, rating
or failed verdict is changed by these calculations.

## Reproduction and archive

Four meaningful mathematical tests pass: exact-tail/degenerate limits, derived
mixture variance, independent convolution and the hand-calculated clustered null.
All 64 design/rate/effect/correlation rows and every within-model, pooled and sign
probability array replay exactly, including compressed bytes. There is no Monte
Carlo seed or uncertainty; floating-point mass/independence checks pass to 1e-11.

```bash
.venv/bin/python -m unittest tests.test_functional_power_feasibility
.venv/bin/python scripts/functional_power_feasibility.py --stage verify
```

Local artifacts are `audits/functional_power_feasibility_v1/manifest.json`,
`summary.json` and `distributions.json.gz` (1,275,889 bytes), with exact source
snapshots. Archive SHA-256:
`2483244576345bdc4ddb5248921c6727592f4deeb6e20615374a36681bdd0227`.
