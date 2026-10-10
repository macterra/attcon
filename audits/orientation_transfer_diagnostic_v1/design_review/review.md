# Pre-output review: synthetic orientation transfer diagnostic

**Disposition: suitable for a narrow paired sensory-information diagnostic, with a required scope clarification about phase/geometry holdout.** No collection or checkpoint evaluation was performed during this review. This is not approval of empirical appearance transfer or an attention-origin experiment.

## Tensor, fitting and control logic

Static inspection finds stimulus broadcasting consistent with `[base, contrast, orientation, RGB, y, x]`. Shared geometry, phase and pixel noise supply an orientation counterfactual. Horizontal varies along y and vertical along x. Identical light/dark values at contrast zero remove orientation from the pixels; the shared noise preserves exact pair equality. Images retain the orientation distinction rather than replacing the grating with a native color/shape task.

The extracted hidden vector has 64 coordinates; logits/categories have eight; maxima and entropies have four. Centering over the orientation axis, flattening unit/orientation, repeating -1/+1 labels, development-only RMS scaling and fixed ridge solve are consistent. The classifier evaluates only positive versus nonpositive scores; exact ties become horizontal, producing exactly half correct on a zero-information pair. The directional-edge control's x-minus-y convention is aligned with vertical-positive labels. No participant visibility/confidence, correctness or attention labels enter the functions.

Base-index resampling preserves the dependence of two orientations and six contrasts. Intervals are conditional on one checkpoint/readout, rather than treating 2,304 images as independent observations. All three specified checkpoints are retained. High-contrast thresholds apply to observed accuracy, not the lower confidence limit; this is a declared engineering minimum, not statistical confirmation. Keeping lower-contrast outcomes descriptive avoids hiding failed transfer beneath the high-contrast gate.

## Required wording correction

The protocol says evaluation geometries/phases/noise are held out. The RNG seeds establish independently drawn development/evaluation base units, but **do not hold out phase classes or spatial frequencies**. Four-cycle binary textures quantize continuous phase into a small set of pixel patterns, so evaluation may repeat development texture classes. Rasterized aperture masks can also repeat. Revise the wording to independent base-unit draws, with no guaranteed phase-pattern, geometry-mask or frequency-disjoint split. This is not a code leak; it limits generalization. A future strict texture/frequency holdout would be a different registered diagnostic.

## Paired inference boundary and shared noise

Evaluation centering consumes both orientations' feature vectors. Thus it uses the counterfactual partner at evaluation. A successful result establishes that hidden features distinguish paired transformations under this conversion, not that a readout classifies an isolated unseen image or performs the human task. This boundary is already named in the protocol and should remain prominent in results.

Common noise is sampled independently of orientation labels, so it does not label one orientation. It does make the paired comparison easier: shared nuisance largely cancels under subtraction, including all linear directional comparisons. Nonlinear hidden responses can retain interactions with the shared nuisance. Shared phase and geometry also enable this paired advantage. They are appropriate for controlling nuisance in this diagnostic, but are not ordinary independent human presentations. The physical edge-control gate is consequently a renderer/readout sanity check, not strong empirical validation of perception.

No direct development/evaluation data sharing is visible. Each split uses a separate seeded base generator; RMS and ridge fit only high-contrast development features. Shared noise across contrasts within a base is declared and bootstrap preserves that dependence. Both input pairs at zero and equality of each evaluated feature pair are asserted. CPU numerical replay is expected only under the pinned environment; the manifest records torch version and threads but does not identify all BLAS/hardware details, so exact replay outside this environment is not guaranteed.

## Gate scope and retention

Passing all gates supports only synthetic sensory feasibility at this rendering and paired information boundary. It does not validate photometry, native grating frequency, temporal sampling, human input equivalence, an orientation acquisition bridge, visibility prediction or attention-origin qualia. The result fields correctly keep empirical transfer readiness, attention-source testing and goal attainment false. Hidden features are extra upstream instrumentation rather than production-bound source fields; this distinction must survive any later reporting experiment.

The single reserved attempt, source/checkpoint hashes, replay and retained tensors protect against output-guided retry. The attempt file remains reserved after completion, with a separate completion record; together they identify terminal outcome. A failed attempt should stay reserved and documented rather than erased. All outputs, including negative checkpoint outcomes and descriptive categorical/entropy scores, should remain visible.

## Review limitations

This is static code/protocol/test review, not execution-based certification. The test source was read, but tests were not run by this reviewer. No checkpoint bytes, model outputs, diagnostic result files, human archive or reports were inspected. The reviewer previously evaluated the primary article and architectural transfer question, and knows the parent-provided hypothesis and intended gates. This is output-unexposed engineering review, not blind theory adjudication. Shared model training limits subagent independence; no human qualification was supplied or required for this ancillary step.
