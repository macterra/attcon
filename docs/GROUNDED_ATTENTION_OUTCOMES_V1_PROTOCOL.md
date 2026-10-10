# Grounded attention outcomes v1: prospective qualification

This is a fresh functional prerequisite to the consciousness-related interpretation
plan. It tests predictions of actual sensory acquisition and decision outcomes,
with a separate Controller consuming the forecasts. It supplies no appearance
labels, phenomenological reports, human trial data, or proof of consciousness.
The Reporter is neither the Controller nor the attention model.

## Fixed sensory boundary and worlds

Freeze visual checkpoints 1301, 1311 and 1321 and the corresponding all-condition
orientation classifiers. Consume only the frozen hidden representation and raw
orientation logits. The sensory adapter's failed confidence calibrators are not
used. Its failed v1 verdict remains unchanged. Raw logits are uncalibrated task
coordinates; a new model must qualify its own actual outcome forecasts.

For each visual checkpoint, generate 512 training, 256 calibration and 512
evaluation worlds. The four locations have independent Bernoulli orientations,
independent six-level contrasts, and the existing synthetic circle/square, phase,
RGB conversion and Gaussian noise. These are independent synthetic worlds, not
validated reproductions of participant stimuli. World orientation, contrast,
mask, phase, command mapping and correctness belong to the evaluator archive.
Only delivered RGB pixels enter the frozen sensor. No paired orientation,
future image, future hidden state, world label or correctness enters inference.

World seeds are 960000000 + visual_seed*1000 + split*100000 + world_index;
world nuisance, orientation and command permutations use the unchanged world
primitive's documented offsets. Eight uniformly random exploration commands
use 970000000 + visual_seed*1000 + split*100000. History noise uses the same base
plus 980000000. The command-to-delivery mapping is a random permutation of four
locations, initially unknown. All 24 permutations belong to the same training
and evaluation family; this is no claim of unseen mapping-class generalization.

The latest delivered hidden64/rawlogit is stored by location, with an observed
flag. Unseen logits are zero, giving a horizontal guess. Static memory has no
invented age-dependent degradation. Each command row retains its most recent
observed delivered location or all zeros. At pre-command times 0, 2, 4 and 8,
clone this evidence, execute all four commands with independent noise, and
record actual current correctness, fresh-glimpse correctness by delivered
location, command delivery and post-command correctness at all locations. Use
two noise repetitions in training/calibration and eight in evaluation. Branch
noise seeds are 990000000 + visual_seed*1000 + split*100000 + snapshot_index*1000.
Each branch updates only its actually delivered memory location. Current targets
are not replicated as new independent observations. Whole worlds remain the
split and bootstrap units. Raw pixels, noise, decoded evidence, world truth,
pre-command inputs and all branch outcomes are archived separately and hashed.
Frozen sensor inference uses ordered chunks of 256; torch CPU threads = 2.

## Models and fitting

Primary `structured` model: a shared location-wise 66→32 ReLU→2 risk network
predicts current and fresh-glimpse success, and a 16→64 ReLU→16 route network
predicts a four-location distribution per command. Its post-command forecast is
exactly `delivery * refresh + (1-delivery) * current` at each location. This
composition assumes the static world/latest-glimpse update rule and separates
route effects from sensory risk; its adequacy must be tested on actual outcomes.

Equal-information `joint` comparison: 280→64 ReLU→40, producing 16 route logits,
four current logits, four refresh logits and sixteen direct post-command logits.
Its post-command head is independently calibrated and is not constrained to the
structured identity. The joint network has more parameters (20,584 versus 4,338),
and no primary gate demands that it lose. `sensory_only` masks all route history;
`history_only` masks hidden features and raw logits but retains observed-location
status. Those are information ablations, not proof of architecture necessity.
Masks are applied before training-only normalization.

Float32 training-only feature mean/RMS with scales ≤1e-6 replaced by 1. Standard
Linear initialization in an isolated RNG scope, model seed 1010000000 + visual_seed
+ kind_index*10000, kind order structured/joint/sensory_only/history_only. Fixed
1500 full-batch Adam steps, rate .003, decay .001, betas (.9,.999), epsilon 1e-8,
amsgrad false, foreach false. Structured loss is route cross-entropy plus half
the sum of current BCE and refresh BCE on branch-mean targets. Joint uses route
cross-entropy plus the mean of current, refresh and post-command BCE. No
checkpoint selection, early stopping or evaluation tuning. Archive loss traces
at 0/500/1000/1500, weights, statistics, raw logits and internal activations.

Calibration only: delivery temperature in (.5,1,2,4), smallest calibration
cross-entropy (first tie). Separate signed-logit two-parameter calibration for
current and refresh; also post-command for joint. Calibration-only score
mean/RMS with same floor, intercept/slope L2=.001, 100 Newton steps, final max
gradient ≤1e-8. Calibration of structured source forecasts precedes composition;
there is no independent structured after calibrator. Fractional branch means
represent observed repeated binary outcomes. Preserve fitted network weights.

## Frozen evaluation and qualification gates

Ten probability bins, signed probability-minus-outcome residual, Brier scores,
reliability counts, paired Brier-difference and Controller-gain whole-world
bootstrap intervals. Calibration ECE/residual/top-1 gates use point estimates,
not confidence-bound guarantees. 256 bootstrap resamples of whole worlds (512 draws for the all-world set,
subset-size draws for the preselected unseen-query set),
seed 1020000000 + visual_seed for assessment; Controller seed below. Scalar
pooled calibration means are binary constant baselines. Delivery baseline is
calibration mean location distribution by command, pooled over snapshots.

Primary per-visual-model gates:

- Current and refresh: seen/unseen strata ECE ≤.10 and absolute residual ≤.075;
  pooled Brier no worse than calibration constant.
- Post-command success: four query-seen × command-known cells use the same ECE
  and residual limits; pooled Brier no worse than calibration constant.
- Delivery: known-command top-1 ≥.95; known/unknown top-confidence ECE ≤.10,
  each of sixteen command/location binary cells ECE ≤.10 and absolute residual
  ≤.075 within each known/unknown status; multiclass
  Brier and NLL no worse than calibration constant. Zero-history top-1 within
  .075 of .25 is a label-leak/null integrity control, not generalization evidence.
- Controller gain, represented route corruption degradation, route equivariance
  and final physical-remap revision/restoration gates defined below.

Unavailable true future contrast and orientation are diagnostic strata only,
never qualification gates: inference cannot predict unobserved latent truth.
Seen/command-known status is observable. A known command necessarily routes to
an already seen location, so unseen-query/known-command is a no-refresh control;
exploration benefit lies in unseen-query/unknown-command and permutation
exclusion. Empty required strata fail qualification. Calibration is evaluated,
not asserted from fitting. Broad aggregate success cannot override a failed gate.

## Separate Controller and causal interventions

At snapshot 4 choose each world's query uniformly from unseen locations if any,
otherwise uniformly from all four, using generator 1030000000 + visual_seed.
Selection uses no label, model score or branch outcome. Controller selects argmax
post-command success for that query (first tie). Compare actual archived branch
success to the exact uniform-random policy expectation over all four commands,
using eight noise repetitions. Retain all 512 worlds and all ties/zero-contrast
worlds. Report all-world gain and the fixed unseen-query subset; primary
unseen-query paired-world bootstrap 95% lower gain bound must exceed zero.
Bootstrap uses the Controller generator's documented draws in source snapshots.

For structured only, rotate represented command rows of the delivery forecast
by +1, holding stored sensory evidence, physical mapping, histories and executed
branches fixed. Recompute composed post-command forecast and Controller action.
Unique winners with top-two gap >1e-6 must select the +1 rotated command; ties
stay in outcome analysis but are excluded only from this equivariance identity.
Restoration must be tensor exact. Primary unseen-query original-minus-corrupted
actual success 95% lower bound must exceed zero. Changed actions alone are only
a consumption check. This intervention corrupts route estimates off distribution;
it does not alter sensory appearance or establish a qualia source.

Physical remap: use the first 64 evaluation worlds at snapshot 8 without outcome
selection. Change actual command mappings to (old+1)%4, leave represented state
initially unchanged, then execute commands 0,1,2,3 with fresh real rendered inputs
(noise seed 1000000000 + visual_seed). Archive forecasts before and after every
observation. Intermediate contradictory old/new rows are outside static training.
Final structured delivery top-1 ≥.95 and exact restoration of original sensory
memory and all history rows are gates. This tests observed-state revision, not
online weight learning or a general comparator mechanism. All methods retain
revision diagnostics; the qualification gates apply to the primary model.

## Freeze, review and claim boundary

Independent unblinded AI subagent source review and synthetic fixture checks
precede registration. Commit the manifest/source snapshots before any sensory
checkpoint inference for this study. One reserved run; terminal failures retained,
no silent rerun or revised gates. Exact replay is verification only. Audit inputs,
target construction, model consumption and results independently, then commit
and attempt GitHub publication at every milestone. Human review is optional.

Passing would qualify a functional acquisition-outcome bridge on these worlds.
It would not establish independent conscious-appearance correspondence, complete
state-report fidelity, selective appearance-source causality, power or replication.
The ultimate attention-model-as-qualia hypothesis remains unresolved either way.
