# Frozen grounded-model physical-information diagnostic v1

2026-10-10. **Pre-output development registration.** This is a source-sufficiency
diagnostic, not a consciousness-report, appearance, psychophysical-transfer or
powered theory-facing confirmation experiment. The Reporter is not run and the
Controller is not the Reporter. No subjective or human-rating targets are used.

## Question and fixed sources

Does the newly grounded attention-model hidden state retain objective information
about the sampled content that the fixed outcome-only probe recovers less well?
This is a comparison of specified readouts, not information-theoretic or semantic
independence from task-success forecasts. The old history-only
attention model's image invariance cannot answer this question for the new sensory-
conditioned model. Physical decoding alone will not establish qualitative appearance,
attention origin, or qualia. The decoder is external diagnostic instrumentation.

Freeze visual checkpoints and raw sensory adapter weights for seeds1301,1311,1321,
and the structured/joint calibrated instruments from grounded attention outcomes v1.
Use only weights/normalization/calibration from those instruments: no old per-world
forecasts, labels or selected examples. Freeze code, this protocol, fixture tests,
pre-output design review and all dependencies/input hashes before the one reserved
production attempt. Existing frozen protocols/verdicts remain unchanged.

## Fresh observations and targets

For each frozen model, sample 512 training, 256 calibration and 512 evaluation
worlds in separate new RNG ranges (exact constants in registered code). Each world
has four independent orientation/contrast locations and a random command permutation,
with the existing static mask/phase/noise renderer. Execute eight uniform random
commands. Only the naturally reached snapshot8 latest-memory/history enters inference.
Do not select correct, high-contrast, extreme-risk or reference-like examples.

Archive world truth separately, all eight delivered images/noise/decoded observations,
commands, actual delivered locations, latest memory/history and latest-image target
correspondence. Unseen rows remain zero observations with no image target and are
excluded using their observed flag, not hidden truth. Targets for seen rows are:

1. Image-value standard deviation over the stored delivered image's 3x16x16 values,
   using population variance, in normalized image units. This pools channels as
   well as spatial positions; it is not a purely spatial luminance measure.
2. Root mean of the two directional nearest-neighbour squared differences (each
   direction mean separately, then average and square root), same units.
3. Auxiliary fixed orientation decision `(stored_raw_logit > 0)`, not actual world
   correctness, confidence, or perceptual content certification.

Targets1/2 characterize objective delivered-image variation, not felt clarity or
the original nominal stimulus contrast. Include zero/low contrast and clipping/noise
effects. True contrast/orientation are evaluator-only strata, never probe inputs.

## Frozen external probes

Infer structured risk_hidden32 and route_hidden64, joint_hidden64, sensory hidden64,
current/refresh forecasts and raw orientation logits from the frozen sources.
Views are risk_hidden, joint_hidden, sensory_hidden, outcomes_only (current/refresh),
rawlogit_outcomes, route_hidden and observed_only. The global joint/route vectors
are evaluated with separate location readouts. Local risk/sensory/outcome/observed
views use one shared pooled-seen-row readout, preserving memory-address equivariance.
Report dimensions, fitting units, rows and parameters; their differing information
and probe budgets preclude an architectural-necessity/superiority conclusion.

Fit a linear ridge decoder with unpenalized intercept. Standardize features and
both physical targets using training seen-row mean/population RMS, replacing feature
RMS<=1e-6 by1. Keep all transformations training-only. Candidate penalties are
.001,.01,.1,1,10. Choose the first smallest calibration two-physical-target
standardized MSE for each shared or location readout. Evaluation never selects a
penalty. Auxiliary raw-decision target shares that physical-selected fit penalty;
evaluate its continuous linear score threshold .5, not a calibrated probability.
No attention/source network is trained, and no prose/appearance labels are fitted.

Evaluate held-out seen rows: R2 against the training-target-mean baseline, MAE,
RMSE, target means/variances, raw-decision agreement and counts. Report per model,
view, location, true contrast/orientation and coarse current-success bins. Keep
empty groups explicit. Whole-world bootstrap256 with a frozen independent RNG
gives conditional-model95% intervals; do not treat correlated rows as independent
trials or generalize three frozen checkpoints to a model population. Report physical
readout versus risk associations descriptively. No pass threshold, appearance
acceptance or theory verdict is inferred from these development estimates.

## Source interventions

Apply native sensory-memory row rotation at fixed observed route history and infer
the primary structured model again. Risk hidden, current/refresh and logits must permute within fixed source tolerance
rtol1e-5/atol1e-6; raw stored content and delivery must stay bitwise fixed/permuted
as appropriate. Shared local physical decodes must stay within the analytic
propagated feature-roundoff bound below. Address-specific risks change, so
this is not a pure binding perturbation holding all per-location marginals fixed.
The per-location global joint readout is not required to permute under this operation.

Patched memory can disagree with unchanged delivery history. This is a synthetic
source intervention, not a natural-state or physical-observation claim. Preserve
both values; patched memory has precedence only for current represented contents,
while history continues to describe actual past deliveries. No inference that the
patch physically occurred is licensed.

Separately patch represented delivery-command rows by rotation at fixed sensory,
history, current/refresh and recompute `after=delivery*refresh+(1-delivery)*current`.
Risk hidden/current/refresh/physical decodes remain fixed by architecture. This
tests declared source export composition, not attention-independent experience or
an intervention on joint_hidden. Preserve original/patched/restored matrices and
source invariants. Exact inverse restoration is checked for inputs and every
structured prediction/readout. No physical world or future observation is changed.

For each shared local decoder, separately check feature permutation against the
same fixed source tolerance. The elementwise physical-output error allowance is
`abs(delta_features)/feature_scale @ abs(W_physical) * target_scale + 1e-12`.
This is linear error propagation of the actually measured source roundoff, plus a
float64 rounding floor; it is not a physical-quality or theory acceptance margin.
Archive feature/output discrepancies and propagated bounds. Bitwise inverse
restoration remains required for sources, forecasts and readouts. An independent
calibrated toy fixture exposed the initial bitwise-permutation mistake before any
production registration/inference; its failing output and original sources remain
preserved in the design review. The correction uses no production outcomes.

## Verification, stopping and interpretation

One exclusive attempt is reserved before collection; errors and partial artifacts
remain. No silent retry, favorable subset or fresh protocol chosen from evaluation
results. Archive all frozen fits/normalizers/penalty scores, features, predictions,
bootstrap indices, source interventions and result JSON, each raw archive <100MB.
Replay re-collects observations, re-infers frozen models, re-fits train/cal probes
and compares all saved tensors/results. Pre-output synthetic fixtures check actual
image target computation, split/seen handling, penalty leakage and local permutation.
Separate-context review inspects implementation before production.

Positive retention is candidate source information, potentially inherited from
upstream sensory features or generic risk representations. Weak/absent linear
decoding limits this probe/source/interface, not all possible architecture semantics
or experience. A physical adjective does not automatically satisfy the qualitative
report endpoint. Any later physical estimates in Reporter input need explicit
diagnostic provenance and held-out error bounds; no new appearance head is inserted
into the source. Next qualify the [qualitative reporting design](QUALITATIVE_REPORT_CORRESPONDENCE_V1_DESIGN.md)
with factual/source and blind semantic tests before new prose. The full source-of-
qualia goal remains unachieved regardless of this diagnostic's result.

Commit registration, executed results, verification and documentation milestones,
attempting authenticated GitHub CLI publication each time. Do not imply a successful
push when network access fails.
