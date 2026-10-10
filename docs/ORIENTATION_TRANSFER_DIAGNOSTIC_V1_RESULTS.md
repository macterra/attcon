# Synthetic paired-orientation diagnostic: results

2026-10-10. All three registered frozen visual checkpoints pass the narrowly
specified **synthetic sensory-feasibility** gates. Highest-contrast paired hidden
readout accuracy is 100% for every model; all zero-contrast readouts are exactly
50% with identical images/features. Full empirical stimulus transfer, participant
appearance and the attention-source hypothesis remain untested.

This follows a new [archive acquisition and transfer review](../audits/appearance_reference_acquisition_v1/transfer_review/review.md).
It is an ancillary prerequisite; **orientation task accuracy is not the project
objective or consciousness evidence**. Reporter and Controller were not run.
The [protocol](ORIENTATION_TRANSFER_DIAGNOSTIC_V1_PROTOCOL.md), code, tests and
checkpoints were frozen and committed before the single reserved attempt.

## What was tested

Three existing visual encoders, each with 64 development and 64 evaluation base
units. Six contrast conditions and two counterfactual orientations per unit give
192 evaluation units and 2,304 dependent images. Gray binary gratings sit within
circular/square apertures. The source paper's light/dark luminance values inform
an explicitly uncalibrated 16-pixel illustration; frequency, spatial conversion,
noise, gamma and temporal equivalence are unverified. No original trial stimuli or
participant ratings were acquired or substituted.

A fixed development-only ridge readout fits the highest-contrast pairs. Lower
contrasts are held out from fitting. Base units are independently drawn between
splits, but quantized texture classes and aperture masks can repeat. Paired
centering sees both test orientations and shared nuisance noise partly cancels.
This asks whether features distinguish a controlled orientation transformation,
not whether the apparatus classifies an isolated image or performs the human task.

## All held-out accuracies

Percent correct, preserving every checkpoint, contrast and diagnostic source:

### Checkpoint 1301

| Source | 0% | 2.2% | 3.9% | 5.0% | 5.5% | 6.9% |
|---|---:|---:|---:|---:|---:|---:|
| Visual hidden (64) | 50.00 | 90.62 | 96.88 | 100.00 | 100.00 | 100.00 |
| Visual logits (8) | 50.00 | 81.25 | 85.94 | 87.50 | 87.50 | 87.50 |
| Category probabilities (8) | 50.00 | 68.75 | 70.31 | 75.00 | 78.12 | 79.69 |
| Category confidence (4) | 50.00 | 78.12 | 78.12 | 79.69 | 79.69 | 79.69 |
| Directional edge control (1) | 50.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |

### Checkpoint 1311

| Source | 0% | 2.2% | 3.9% | 5.0% | 5.5% | 6.9% |
|---|---:|---:|---:|---:|---:|---:|
| Visual hidden (64) | 50.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Visual logits (8) | 50.00 | 89.06 | 96.88 | 96.88 | 96.88 | 96.88 |
| Category probabilities (8) | 50.00 | 85.94 | 93.75 | 95.31 | 95.31 | 95.31 |
| Category confidence (4) | 50.00 | 81.25 | 76.56 | 78.12 | 78.12 | 81.25 |
| Directional edge control (1) | 50.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |

### Checkpoint 1321

| Source | 0% | 2.2% | 3.9% | 5.0% | 5.5% | 6.9% |
|---|---:|---:|---:|---:|---:|---:|
| Visual hidden (64) | 50.00 | 89.06 | 100.00 | 100.00 | 100.00 | 100.00 |
| Visual logits (8) | 50.00 | 70.31 | 78.12 | 78.12 | 79.69 | 82.81 |
| Category probabilities (8) | 50.00 | 65.62 | 71.88 | 78.12 | 81.25 | 75.00 |
| Category confidence (4) | 50.00 | 46.88 | 46.88 | 45.31 | 43.75 | 45.31 |
| Directional edge control (1) | 50.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |

The physical edge control passes at every positive contrast. Hidden readout
accuracy at the lowest positive contrast ranges from 89.06% to 100%; those lower
contrasts have no engineering success threshold and remain visible. The highest
contrast reaches 100% in 64 independent base units per checkpoint. Conditional
bootstrap intervals there are 100–100%, reflecting no observed failures; this
is not proof of perfect population accuracy. At the lowest positive contrast,
hidden 95% percentile intervals are 84.38–96.29%, 100–100%, and 82.81–96.29%
for checkpoints 1301, 1311 and 1321. Intervals condition on each fixed checkpoint
and fitted readout; all intervals remain in
[results.json](../audits/orientation_transfer_diagnostic_v1/results.json).

Category probability/confidence readouts are less consistent. Checkpoint 1321's
confidence readout is below chance for several positive contrasts, including
45.31% at the highest; it is not dropped or reversed after observing the result.
The native color argmax is **blue in every evaluation image**, despite the input
having an underlying grayscale target with RGB sensor noise. Blue is what this
out-of-domain model classifies, not verified
physical color. No gray category or native orientation task is thereby supplied.

## Source interpretation

Missing an orientation head did not mean missing all orientation information:
upstream hidden features distinguish these paired transformations. Production
binding discards those features/logits and retains only color/shape probabilities.
Category outputs carry some orientation information under an added probe, but do
not constitute the original orientation answer or a calibrated appearance state.
No attention history mapping or attention-state intervention is measured here.

The independent transfer review distinguishes three questions: numerical behavioral
reference qualification, sensory-task conversion, and causal attention-model source
identification. Passing this synthetic component advances the second question
partly. It cannot replace the first or third. Appearance fitting must not inject
human visibility/confidence into acquisition quality or relabel recovery as clarity.

## Archive access and next valid experiment

The [retrieval review](../audits/appearance_reference_acquisition_v1/retrieval_review/review.md)
found documented maintainer CSV/JSON exports and advertised archive provenance.
Raw trial and native stimulus downloads fail through available access methods.
No participant rows or program bytes were inspected; no source-paper dissociation
was reproduced. Package metadata is not local checksum, data-license or schema
verification. Human-only review is removed; empirical data remain necessary for
the proposed reference comparison.

Next separately register **isolated-image sensory task validation**, before an
attention/acquisition bridge consumes this added readout. Use independently drawn
noise/phase and explicit null/uncertainty controls; do not supply counterfactual
partners at task execution. Ground any new bridge in generated orientation and
observed synthetic acquisition outcomes, without human appearance targets. Such
an adapter is a new apparatus component and must preserve old frozen results.

For theory-facing testing, qualify actual participant files/stimuli and a matched
experiment independently varying attention and sensory strength. The contrast-only
reference can ground a visibility/confidence distinction, but predicting its
ratings alone does not identify their source. Preserve the identified bound-model
locus and alternatives, and keep Controller action selection separate from Reporter
instrumentation. Only then evaluate accurate source-dependent reports against a
frozen independently grounded consciousness-related criterion. Credible source-of-
qualia evidence, rather than generic task accuracy, remains the full goal.

## Verification and review

Three tests pass: identical zero-contrast pairs/chance, a directional physical
signal on separate development/evaluation draws, and invariant features failing
to classify orientation. Exact replay verifies all retained inputs, extracted
features, coefficients, predictions, bootstrap draws and scores, together with
source/checkpoint hashes. Raw archive:
[states.pt.gz](../audits/orientation_transfer_diagnostic_v1/states.pt.gz).
Pre-output review retained a holdout wording correction; corrected protocol and
exposure boundary are archived separately. Subagents share training/commissioning
context and do not constitute human empirical validation.

A [separate post-output raw-tensor audit](../audits/orientation_transfer_diagnostic_v1/results_review/review.md)
independently regenerates images/features and recomputes coefficients, predictions,
every score and interval without importing the diagnostic build. All match exactly.
It identifies all four discrete texture classes in both development/evaluation
splits for each model, plus 9/7/10 overlapping aperture masks. Its separate audit
source and hash are archived in metadata. This unblinded AI audit verifies the
limited computation and preserves its scope; it does not validate appearance.

The scientific goal remains **unachieved**; this positive ancillary result is not
rescoring of previous reporting, control or physical-source outcomes.
