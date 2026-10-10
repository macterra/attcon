# Orientation-preserving sensory transfer diagnostic

2026-10-10. Prospective ancillary engineering diagnostic; no participant data or
consciousness endpoint. The appearance-reference transfer review recommends
testing sensory information before importing human judgments into this apparatus.
Original archives remain inaccessible. This uses an explicitly **synthetic
illustration**, not reconstructed participant trials or verified native stimuli.

## Fixed question

Can frozen visual hidden features distinguish horizontal from vertical grating
texture under a documented grayscale conversion? Native color/shape outputs do
not supply the human orientation task. Hidden decoding may still be possible.
Either result leaves the attention-origin appearance estimand untested. The
Controller and Reporter are not run; no appearance/qualia labels enter fitting.

## Synthetic inputs and unknowns

Use the three existing frozen visual checkpoints 1301, 1311 and 1321. For each,
generate 64 development and 64 evaluation base units using seeds
`890000000 + visual_seed + 10000 * split`, split 0/1. Each unit has a paired
horizontal/vertical texture at six contrast conditions, common geometry, phase
and sensor noise. Circle/square apertures alternate. Centers and radii vary within
the original renderer's approximate range. All variants of a base unit stay in
one split. Evaluation has 192 base units and 2,304 dependent images.

The [source article](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2016.00591/full)
reports grayscale targets with six light/dark luminance pairs:
38.2/38.2, 39.0/37.3, 39.5/36.5, 40.0/36.2, 40.2/36.0 and 41.1/35.8 cd/m²,
against a 35.5 cd/m² background. Use these numerical levels divided by 50 in all
RGB channels. This division is an unvalidated illustration conversion, not monitor
calibration. Use four binary texture cycles across a 16-pixel field and Gaussian
noise SD 0.015. These frequency, resolution and noise choices are synthetic; the
native grating frequency, gamma calibration, trial-level aperture and rendering
are unverified. The source's visual angle and temporal exposure are not modeled.
These limitations prohibit enrolling the result as same-stimulus human transfer.

## Readout and gates

Extract upstream visual hidden (64), logits (8), category probabilities (8),
category max probabilities/entropies (4), and a physical directional-edge control
(1). Hidden features and logits are extra instrumentation discarded by production
binding. Center each feature across the two orientations within unit and contrast.
Fit one fixed ridge classifier, lambda 0.01 with development RMS scaling, only on
the highest-contrast development pairs. Targets are generated orientation -1/+1.
Evaluate every contrast on all evaluation units. Positive scores predict vertical;
ties predict horizontal. This is a paired transformation diagnostic, not isolated
image classification. No fitting, tuning or selection uses human ratings.

Report per-source accuracy at every contrast and 95% percentile intervals from
256 base-unit bootstrap draws, seed `900000000 + visual_seed`. These condition on
checkpoint/readout. The five lower contrasts are held out from fitting, and all
evaluation base units are independently drawn with different seeds. Discrete
texture patterns and quantized aperture masks can repeat across splits; this is
not held-out texture-class or spatial-frequency generalization. Do not select
successful models. Paired centering has both counterfactual test orientations;
success does not establish isolated-image or human-task orientation ability.

Require exact image/feature equality across the zero-contrast orientation pair
and exactly 50% accuracy there. Require the physical edge control and hidden
probe each to achieve at least 90% on the highest-contrast evaluation condition,
in all three checkpoints, for **synthetic sensory feasibility**. Preserve failed
gates. These engineering minima cannot qualify empirical stimulus equivalence,
appearance correspondence or consciousness; full reference transfer remains false
regardless of their outcome. Category/logit/confidence scores are descriptive.

Pin code/protocol/tests/checkpoint hashes before the single reserved attempt.
Retain all base images, orientation features, coefficients, predictions, generated
labels, bootstrap indices and scores. Exact replay and zero-information controls
must pass. No retry, replacement, encoder retraining or report generation.

## Consequences

If successful, this permits planning a synthetic orientation task without
presuming that a missing output head implies missing information. It does not
repair unverified archive acquisition, photometric/temporal transfer, the absent
human attention manipulation, or the ungrounded attention-history mapping. Those
remain separate scientific requirements. A failure restricts this conversion and
probe, not all possible representations or the theory.

The broader goal remains credible source-of-qualia evidence through accurate
attention-model reports with independently grounded consciousness-related
correspondence. Orientation accuracy is an ancillary prerequisite, not that goal.
