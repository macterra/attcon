# Spatial native task v1: development contract

2026-10-10. New task/source development, not confirmation, presentation
qualification or evidence of experienced spatial form. No attention model A or
Reporter is introduced here. This contract implements the next step in the
[content-form plan](ATTENTION_CONTENT_FORM_DEVELOPMENT_V1_PLAN.md) and
[semantic design review](../audits/attention_content_form_semantics_design_v1/review.md).

## Question and source boundary

Can an existing frozen spatial encoder support a constituent-location field
that a native consumer actually uses to target an action? If so, do internal
allocation operations change that field while sometimes preserving its coarse
content relation? The answer can establish task/content semantics. A later
independent review must still determine whether any property describes manner
of representation beyond additional content, task access and confidence.

Retain all three frozen encoders, seeds 1301, 1311 and 1321. Extract the 24×4×4
map after `VisualEncoder.net[:6]`; parameters remain frozen. Retain original
color/shape logits through `net[6:]` and the frozen orientation head through
`net[6:9]`. The latter is an off-task control, never constituent truth. Preserve
untransformed maps, pixels and original logits alongside each completed W.
An encoder site has an overlapping receptive field, not an observed object cell.

## New task and native consumer

Each world contains four delivered 16×16 RGB patches at outer addresses l0–l3.
Each patch contains one red and one blue token on a neutral background, in
distinct cells of a 4×4 stimulus grid. The two roles occupy opposite left/right
halves. Balance coarse role order and fine grid placement across the declared
batch, with independently varied square/cross token shapes, intensity and noise.
Exact rendering, balancing and nuisance parameters are fixed in the reviewed
source before generation. Task truth and occupancy labels are evaluator-only
during inference; they enter the training loss as task supervision only.

A shared 24→32→3 ReLU head H has 899 trainable parameters. At each W site it
predicts background/red/blue occupancy logits. Fit cross-entropy on the exact
4×4 task grid, using only untransformed fit-world maps. For each role normalize
its role-minus-background logits over all sixteen sites using softmax. Identical
sites therefore give uniform evidence. H is frozen before diagnostic operators.

For every item and both role queries the Controller selects the maximum-evidence
grid cell, choosing the first row-major cell on exact ties. The coarse relation
uses each role's expected column: red-left-of-blue if red's is lower, the reverse
if higher, and unknown within absolute difference 1e-6. Unknown counts as an
incorrect coarse answer. This deliberately replaces the earlier design's half-mass
comparison: it tests order of role centroids, not proof of occupancy in an intended
half. Preserve the possibly wrong native relation separately from physical truth.
The spatial softmax is conditional forced localization under the task's one-token-
per-role assumption, not presence detection or calibrated confidence. Keep full
occupancy logits alongside location fields. Record ties, including T4's uniform field. The Controller
uses only H's spatial evidence and the address/role query, with no pixel, retained
identity, raw-logit or renderer-truth bypass. No claim of attentional prediction
or conscious control is made for this targeting Controller.

## Fixed development envelope

For each encoder use 128 fit worlds and 64 diagnostic worlds, four items each.
World generation namespaces are `210000000 + encoder_seed*1000 + split*100000`,
where fit split=0 and diagnostic split=1. Initialization seed is
`220000000 + encoder_seed`; the raw head uses that seed plus 1000000. Whole worlds
define splitting and uncertainty. Deterministic position layouts overlap across
splits; fresh nuisance realizations do not establish generalization to unseen
layout combinations. This task does not claim such generalization.
There are 576 worlds total, 768 diagnostic items and 1,536 native role queries.
This is development only, with no reserved confirmation split.

Fit H once per encoder for exactly 500 full-batch Adam updates: learning rate
.003, betas (.9,.999), eps 1e-8, weight decay zero, amsgrad false, foreach false.
Use float32 CPU tensors and four CPU threads; no stopping/selection by diagnostic
results. Normalize channels using fit-only mean and population standard deviation,
replacing scales below 1e-6 with one. Preserve those coordinates for every operator.
No additional initialization, hyperparameter sweep or layer search in this version.

As an upstream feasibility control, fit the same shared 3→32→3 architecture on
raw pixels averaged into the 4×4 grid, with the same worlds, loss, optimizer and
update budget. It has 227 parameters, so it is an information/capacity alternative,
not an equal-information source-specificity control.

## Operators, provenance and measurements

For every diagnostic item execute all T1/T2/T4 alternatives from identical saved
V using the existing finite block-average primitive. These are 2,304 item/operator
opportunities and 4,608 role-query actions. Retain all opportunities, wrong answers,
erased coarse relations, unchanged fields and ties. No coarse-preserving subset
is the primary denominator. T acts before H and any standardization. T4 removes
all within-map variation; loss of coarse order is a valid outcome.

Record full role-indexed fields, selected cells, coarse relation and truth,
occupancy loss, role targeting accuracy, entropy/max/margin and expected columns.
Expose red and blue targeting accuracy separately as well as their pooled mean.
Also record per-channel V/W means, norms and standard deviations, category logits,
orientation logits, and changes from preserved originals. Report per-model and
per-world aggregates, with conditional world-bootstrap intervals (256 draws,
seed `230000000 + encoder_seed`). Such intervals are not uncertainty over model
families or people.

Include sham and exact saved restoration. Matched endogenous/exogenous drivers
use the same numerical T; driver provenance is audit-only and equal W entails
equal H/action outputs. Include consistent outer-address permutations. A synthetic
within-map site permutation preserves channel means/norms and, for this shared
head, the role field's entropy/max/margin while changing its spatial placement.
Keep its synthetic provenance explicit; it is not a naturally reached attention
state. These controls establish sensitivity or invariance under stated inputs,
not phenomenal-source specificity. Neither scalar omission nor pixel omission
is an equally informed ordinary-estimator comparison.

The fixed duplicate-control ledger, over all three encoders, is:

| Control | Schedule | Item opportunities | Role queries |
|---|---|---:|---:|
| Sham | T1 compared with the saved initial state | 768 | 1,536 |
| Driver parity | Exogenous T2 compared with endogenous T2 | 768 | 1,536 |
| Restoration | Restore the saved initial state after T2 | 768 | 1,536 |
| Outer-address permutation | Roll all four items +1 at every primary operator | 2,304 | 4,608 |
| Synthetic site permutation | Roll flattened spatial sites +1 before H at every primary operator | 2,304 | 4,608 |

Consistent address remapping must carry native evidence and actions with the item.
Site permutation must carry native logits/fields within declared tolerance and
preserve their entropy/max/margin. First-row-major action tie breaking is not
permutation equivariant; test action equivariance only for unique maxima, retaining
all tied and ineffective cases. Same-W drivers compare actual emitted actions as
well as fields. Preserve full control outputs or deterministic reconstruction
inputs and per-opportunity checks, rather than only aggregate passing booleans.

T2 repeats each site's vector across a 2×2 block, so every maximum ties at least
four ways. Its targeting policy selects only block top-left cells and is capped
at .25 accuracy under balanced fine placement. T4 is uniform with sixteen ties,
.0625 targeting accuracy and unknown coarse order. These imposed effects are not
evidence of experienced blur. A top-two-site margin is zero under T2 regardless
of separation between blocks.

Define a changed role field by maximum absolute T1/T2 difference greater than
1e-6. Correctly retained coarse order requires both T1 and T2 native answers to
match evaluator order. Report the conjunction over all 768 diagnostic items;
T1-wrong/T2-correct cases do not count as retention. Bootstrap per-world means
using 256 resamples of all 64 worlds with replacement and .025/.975 quantiles.

## Engineering endpoint and interpretation

Before execution, the engineering feasibility requirements are per-encoder T1
mean role targeting accuracy at least .50 and coarse order accuracy at least .75,
plus raw-grid targeting accuracy at least .95. These margins concern utility above
chance (.0625 role targeting, .50 binary coarse order); they are chosen engineering
criteria, not psychophysical or presentation thresholds. All models must pass.
Software invariants require unchanged frozen weights, finite outputs, exact
restoration, driver parity and query/input isolation. Numerical permutation checks
use atol=1e-6, rtol=1e-5; exact equality is used for copies and repeated T blocks.

Native role/task feasibility does not qualify presentation manner. Preserve
technical-only, failed, mixed, or erased-content endpoints. Report the unconditional
frequency of T2 changes with correctly retained coarse order descriptively, without
promoting it to semantic qualification. No phenomenal adjective, form bin, human
rating, reference wording or report text is a training target.

## Execution and preservation

Implement and independently review the source, runner, software first pass and
this exact contract before any development checkpoint load, world generation or
fit. Tiny software fixtures and random toy encoders are distinct from development
execution. Preserve every review issue and first-run failure before repair.
Bind the reviewed executable and checkpoint hashes in a committed registration
before the one development run. Archive actual inputs, targets, fitted H, native
outputs, actions and all operator results; a separate reviewer must reconstruct
the retained numerical results. Preserve first-run logs even on failure.

No automatic fallback to the first pooled layer is allowed. If this stage fails,
the explicitly proposed alternative needs a new version and fresh review. If only
position decoding, scalar uncertainty or task access is defensible, retain no
qualified presentation counterpart. Even a successful new content-bearing source
does not identify A as its source or establish qualia. The full project goal
remains unchanged and unachieved. Commit and attempt authenticated `gh` publication
at every milestone.
