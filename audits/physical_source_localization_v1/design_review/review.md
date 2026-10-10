# Pre-freeze design and code review

Date: 2026-10-10. Reviewer: `/root/frozen_source_assay_review`, a separate AI subagent. Final pre-freeze source hashes and exposure boundary are recorded in `metadata.json`. These hashes identify the corrected version reread after the first-pass recommendations; the initial pre-correction bytes were not separately archived or hashed by this reviewer. No new diagnostic model outputs were encountered or collected. The supplied four unit tests passed.

## Assessment

The bounded synthetic physical-information diagnostic is coherent. No critical tensor-shape or readout-math error was identified in the inspected implementation. This assessment concerns source localization for objective image transformations, not participant appearance or consciousness-related correspondence. The protocol correctly keeps those latter requirements open.

The script centers features and targets separately within each base-scene/view/object's six variants, fits development-only feature scales and a fixed ridge penalty, and evaluates all held-out variants without selecting category-retained examples. Grouping bootstrap draws by base scene respects the dependence induced by its related objects and variants. Intervals condition on the frozen checkpoint and fitted readout; they are not model-population, human, or readout-training uncertainty estimates. Centering evaluation features across fixed groups makes this a transformation-response diagnostic rather than isolated-image deployment prediction.

## Source boundaries

`PredictiveAttention` receives only allocation, acquisition, and command histories. At identical histories and initial hidden state, neither its GRU hidden vector nor forecast heads can distinguish changed pixels. Measuring equality across independently recomputed forwards, together with a separate changed-acquisition nondegeneracy control, tests this actual path; it does not test all possible properties of the broader bound model.

The upstream encoder's 64-dimensional hidden vector may retain physical image information. It is discarded by production binding, which retains eight category probabilities. `bind()` additionally masks unremembered contents to uniform and establishes an explicit location/object relation. The functional categorical buffer is a separate unmasked path. The implementation and protocol correctly distinguish these sources.

The bridge is exactly `q * category + .25 * (1-q)`. The category/q/interaction baseline contains its complete deterministic reconstruction. Probe differences against that baseline can arise from feature basis, standardization, dimensionality, and regularization. They do not show new information created by attention. Category argmax retention also does not establish matched category certainty; the planned confidence-drift measurements are necessary.

## First-pass issues and verified pre-freeze corrections

1. **Retention mismatch (R1, medium).** The inspected collector retains derived features, physical targets, categories, and scalar checks, but not original pixel scenes, process observations/allocation, or direct attenuation/binding/restoration tensors. The protocol promises scene retention and causal integrity checks. Preserve base pixels and process inputs, plus original/intervened/restored state tensors, to make those checks directly auditable, or narrow the retention promise before registration. Seed-based regeneration is valuable but does not replace direct source retention.
2. **Nonzero-change gates (R2, low).** Access attenuation and binding-roll changes are measured but not asserted nonzero. If these are intended as functioning-intervention controls, require nonzero changes before freeze and retain any failed attempt. Equality/restoration checks alone can pass a degenerate intervention.

These recommendations were sent to the parent before registration and model output collection. They are first-pass findings, not post-output revisions. The parent corrected both before freeze: source inputs and last-variant causal tensors are retained, and nonzero access/binding/target effects are asserted. The corrected implementation was reread and its exact hash recorded. No critical issue remains identified in that reviewed version; later bytes require separate review.

## Interpretation limits

An upstream hidden-vector advantage localizes physical information to the encoder. Category, bound, or buffered decoding can inherit that signal. Attenuating q at fixed pixels changes the engineered bridge; a changed probe estimate would establish instrument dependence or calibration sensitivity, not a changed physical stimulus or felt appearance. Binding intervention/restoration locates a represented relation while remaining compatible with ordinary computation.

No physical-only outcome completes the project goal. Participant-level appearance, task, and confidence data, valid stimulus conversion, and independent reference qualification remain separate work. Exact core invariance cannot establish absence of experience or falsify every relational interpretation of the attention model.

## AI review limits

This review has procedural separation from the parent implementation and no exposure to this diagnostic's new outputs. It shares model training and commissioning context; inherited context included prior project results. It is not independent human review, a naive assessment of historical findings, or validation of a human appearance reference.
