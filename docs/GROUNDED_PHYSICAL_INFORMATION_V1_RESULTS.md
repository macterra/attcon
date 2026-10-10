# Grounded physical information v1: retention established, appearance unqualified

2026-10-10. The frozen attention model's risk hidden state retains information
about objective properties of its latest delivered images. A held-out linear
readout recovers that information better than current/refresh forecasts alone;
upstream sensory features support stronger recovery. This supplies a concrete
source for reporting image-property information, not a qualified interpretation
of represented appearance or evidence that attention generates qualia. The
Controller and Reporter remain distinct; no language reports were collected.

The [protocol](GROUNDED_PHYSICAL_INFORMATION_V1_PROTOCOL.md),
[registration](../audits/grounded_physical_information_v1/manifest.json),
[complete results](../audits/grounded_physical_information_v1/results.json),
[exact replay](../audits/grounded_physical_information_v1/verification.json) and
[independent numerical review](../audits/grounded_physical_information_v1/results_review/review.md)
retain the complete attempt and limitations. This was a descriptive development
diagnostic with no appearance acceptance gate.

## Measurement and results

Each of three previously frozen visual/attention-model pairs receives 512 fresh
training, 256 calibration and 512 evaluation worlds. Eight natural observation
commands populate latest sensory memory. Only seen locations enter decoder fitting,
penalty selection and scoring; unseen placeholders remain archived. Evaluation
contains 1,834, 1,832 and 1,828 seen location rows. Worlds, not rows, are the split
and bootstrap units. Neither the sensory encoder nor attention model is retrained.

Two physical targets are population standard deviation over all RGB/spatial image
values and RMS differences between adjacent pixels. The first includes differences
between channel means; neither is a human clarity rating. Noise remains present at
nominal zero contrast. Linear ridge penalties are selected using calibration
physical-target loss; evaluation targets never select the decoder.

| Hidden/readout source | 1301 R²: value SD / neighbor RMS | 1311 | 1321 |
|---|---:|---:|---:|
| Attention-model risk hidden | .615 / .608 | .840 / .760 | .867 / .783 |
| Upstream sensory hidden | .946 / .888 | .948 / .879 | .937 / .876 |
| Current/refresh forecasts | .291 / .383 | .330 / .414 | .336 / .419 |
| Raw logit plus forecasts | .291 / .385 | .329 / .414 | .337 / .419 |
| Joint hidden, location-specific readouts | .277 / .278 | .281 / .304 | .355 / .390 |
| Route hidden, location-specific readouts | .001 / .000 | −.008 / −.008 | .001 / −.002 |
| Observed-status-only | 0 / 0 | 0 / 0 | 0 / 0 |

R² uses the **training seen-target mean** baseline, including within-stratum results;
it does not use evaluation or stratum means. Risk-hidden RMSE is
`.003096/.003427`, `.001989/.002645`, and `.001783/.002465` in normalized image
units. Auxiliary reconstruction of stored raw-logit sign reaches 96.78%, 97.38%
and 98.80%; this is fidelity to a stored decision, not actual orientation correctness.

All six pooled outcome-only minus risk-hidden MSE bootstrap intervals exclude
zero in the favorable direction. All six sensory minus risk intervals favor sensory.
Intervals condition on frozen models and fitted readouts, with 256 whole-world
resamples. They do not establish generalization to a population of trained models.
All locations, orientations, contrast levels and success bins are retained.

Pooled superiority is not uniform: neighbor-RMS MSE favors outcome-only at seed1301
contrasts3 and4, and seed1311 contrast0. For the last condition, risk-hidden MAE is
`.001801` versus `.001586` for outcome-only. Per-location exceptions and empty-bin
intervals remain in the full results and audit. No difficult data were discarded.

## Source controls and verification

Rotating sensory-memory rows while retaining actual command history moves local
risk features and their physical readouts with the addressed content. This is a
synthetic memory/history conflict, not a pure binding intervention holding every
marginal fixed. Inverse restoration is bitwise exact. A separate represented-route
patch changes delivery and recomposed post-command forecasts while local sensory,
risk and physical readouts remain fixed; this does not intervene on joint neural
hidden state or simulate new observations.

A pre-output numerical fixture exposed floating-point row-permutation differences.
The original fixture, failure, source and corrections were saved before registration.
Source equivariance uses fixed `atol=1e-6`, `rtol=1e-5`; physical-readout tolerance
is the propagated actual feature-roundoff envelope plus `1e-12`. This is a numerical
bound, not an appearance margin. Maximum physical permutation discrepancy is
`5.88e-8`; all source checks and exact restoration checks pass.

Eleven final fixtures passed before execution. One reserved production attempt
completed; fresh collection, inference, decoder selection and interventions replay
exactly with all75 archive hashes matching. An independent implementation refits
all39 decoders and checks target association, metrics, bootstrap intervals and
intervention arithmetic. It produces12,672 checks. Its504 initial flags concern
empty-bin `null` interval serialization and are explicitly adjudicated; every
original flag, checker and log is preserved. No unresolved numerical mismatch
remains. Maximum physical-prediction difference is `6.88e-15`; maximum auxiliary
score difference is `2.00e-12`.

The independent checker does not reconstruct neural inference or the pixel renderer;
those are covered by production replay, with its shared-code limits. Independent
AI review also shares model training and project commissioning. Integrity checks
do not prove historical absence of every unrecorded action.

## Interpretation and next step

Risk hidden consumes sensory memory. Its extra recoverability over a two-number
forecast projection can therefore be inherited visual information. Unequal feature
dimensions, numbers of fitted readouts and calibration choices limit architectural
comparisons. Correlations with forecasts/logits and a linear raw-logit control do
not establish independence from confidence, much less attention-specific origin.

These external physical decoders have not been qualified as the model's own
presentation estimates. Do not call their outputs clarity or consciousness by
definition. Next qualify the factual reporting interface and any strict presentation
counterpart separately, then preregister neutral fresh reports and matched source
interventions. The [qualitative design](QUALITATIVE_REPORT_CORRESPONDENCE_V1_DESIGN.md)
keeps access structure distinct from presentation and attention relations.
The source-of-qualia goal remains unachieved.
