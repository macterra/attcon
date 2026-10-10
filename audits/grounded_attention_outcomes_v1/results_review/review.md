# Independent archived-result reconstruction

Reviewer: an unblinded AI subagent commissioned separately from implementation.
This is a code/data audit, not human review or an independent research-group
replication. Shared AI capabilities and the common source material limit
epistemic independence.

The first recorded audit attempt completed successfully. All 177 raw archives
were hash-checked; 44,856 recursively checked values reconstruct with maximum
numerical discrepancy 1.1102230246251565e-16. Every primary model retains all
56 registered gates. Nonprimary method verdicts remain null rather than vacuous
passes. No frozen source, target, fitted study model or result was changed.

## What was independently reconstructed

The audit uses standard Python and PyTorch functions without importing production
collection, model, assessment or Controller modules. It independently rebuilds
world labels, aperture geometry, phase, contrast and actual command permutation
from the documented RNG streams; all historical and branch noise/pixels; frozen
sensor convolution/readout computations; latest delivered memory and route
history; actual current, refreshed and post-command binary correctness; and all
physical-remap observations and states. Raw future images and latent world truth
remain outside the reconstructed prediction inputs.

Training-only normalization is reconstructed after the specified information
masks. Frozen network weights are forwarded independently, and calibration-only
temperatures/signed-logit mappings are refitted independently. Frozen sensory and
outcome forecasts, including internal activations, reconstruct exactly. The audit
then independently reconstructs all probability reliability bins, actual
Bernoulli Brier/NLL scores, baseline constants, status cells, latent-truth
diagnostics, whole-world intervals, Controller query/action choices and actual
success, represented route corruption, equivariance/restoration, physical remap
revision and all qualification gates.

The outcome networks are not retrained by this independent audit. Training seed,
budget, parameter count, archived trace and summary consistency are checked;
intermediate training traces are archive cross-checks rather than independently
regenerated optimization trajectories. The separate exact full-budget replay
refits all outcome networks and has its own successful terminal receipt. This
distinction must remain explicit.

Freeze provenance verifies that commit `59c0271` contains the final manifest,
all matching frozen source/review hashes and no attempt, results or model archive.
The reserved attempt matches that manifest. Commits `3fdb9e4` and `2fc3def`
preserve the initial audit implementation and its corrected diagnostic coverage
in verified ancestry order. The initial audit omitted the subsequently frozen
latent-diagnostic summary and review hashes; that audit-only omission was
corrected before reviewing production results. The first version and correction
record remain available. Production study sources were not changed.

## Findings and limits

| Visual model | Unseen-query gain over random | Original minus route-corrupted success | Final remap top-1 | Registered gates |
|---|---:|---:|---:|---:|
| 1301 | 22.28 percentage points | 28.30 percentage points | 100% | 56/56 |
| 1311 | 22.18 percentage points | 28.76 percentage points | 100% | 56/56 |
| 1321 | 21.29 percentage points | 29.01 percentage points | 100% | 56/56 |

Whole-world lower confidence bounds for both primary Controller contrasts exceed
zero in each model. The acquisition bridge is functionally qualified for the
registered synthetic family. Coarse status-cell calibration is a bounded point
estimate: model1321's maximum post-command residual is .07278, close to its .075
gate. Seen zero-contrast current overprediction remains .1181/.1375/.1367 and
refresh overprediction .1661/.1679/.1823. These visible failures by sensory
condition are diagnostic rather than secretly added or removed gates. They
constrain subsequent use of the forecasts.

History plus observed-status achieves approximately the same Controller gain
(22.55/22.18/21.67 percentage points). For a selected unseen query, the structured
risk inputs are identical zeros with observed=0, so risk estimates are constants;
the decision reduces to routing when refresh exceeds current. Richer sensory
evidence improves current Brier relative to that ablation, but this Controller
endpoint does not establish a necessary role for bound sensory contents,
phenomenal appearance, or the proposed quale substrate. The equal-information
joint network has a larger parameter count and a different loss; its descriptive
performance does not isolate architectural necessity.

Route rotation is an off-distribution consumption intervention with actual task
cost, not a selective appearance-source intervention. Physical remap changes
actual deliveries and tests revision from new observations; intermediate mixed
histories are outside the static training family. Restoration verifies saved
model inputs and forecasts, not a newly executed physical restoration trajectory.
The Controller is a separate consumer. The Reporter was not run.

The synthetic geometry/RGB transfer has no qualified participant-stimulus
equivalence. No appearance endpoint or human trial rating was supplied. All
three models share an architecture and synthetic family. Previous sensory
confidence failures remain unchanged. This result supplies a functional
prerequisite and does not establish consciousness-related correspondence,
subjective experience, theory-facing power or broader replication.

The results document's numerical tables, branch counts, intervals, calibration
limits, comparator claims and scope have been checked against the reconstructed
results. Its final verification wording and reviewed file hash are recorded in
the accompanying metadata after the successful receipts are incorporated.
