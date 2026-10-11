# Attention content/form semantic development design v1

2026-10-10. Static source review and prospective DEVELOPMENT design by
`/root/content_form_semantics_development_design`. No checkpoints or tensors were
loaded, no worlds were generated, and no inference, fits, reports, ratings,
network operations or Git mutations were performed. This is not a registration,
reserved evaluation, qualified presentation counterpart or positive qualia result.

**Verdict: the current frozen source does not meet the semantic preconditions.
The smallest useful next extension is a native spatial constituent task whose
spatial representation guides an executed task action.** Exposing a convolutional
map and averaging it is mechanically compatible with the code. That fact does
not supply content-specific form semantics. The existing strict negative remains.

## What the existing code actually supports

The [development plan](../../docs/ATTENTION_CONTENT_FORM_DEVELOPMENT_V1_PLAN.md)
correctly separates mechanical feasibility, semantic qualification, process
prediction and reporting. The [critical review](../attention_source_bridge_design_v1/critical_review/review.md)
correctly requires a literal addressed relation and independent native role.

The exact primary extraction candidate is `encoder.net[:6](flat_pixels)` from
[VisualEncoder](../../src/attcon/bound_content.py#L45): after the second
`MaxPool2d(2)`, before `Flatten`, shape `[N,24,4,4]` for `[N,3,16,16]` input.
Preserve its float32 channel units and item address. The unchanged original
consumer `encoder.net[6:](W)` produces eight color/shape logits. The orientation
consumer instead applies `encoder.net[6:9](W)` to obtain hidden64, then the frozen
standardizer and 64→32→1 head from
[FrozenSensor](../../scripts/grounded_outcome_data_v1.py#L44). These are distinct
native downstream pathways, and both must be retained as controls in any future
extension. The eight-logit classifier was trained for whole-patch color/shape;
the later orientation head was trained for whole-patch horizontal/vertical labels.

The 4×4 map is not a 4×4 observed object layout. From the kernel/pool sequence,
an interior output site's nominal receptive field is 10×10 input pixels with
stride four; neighboring sites overlap substantially. There is no code-level
channel-to-contour, channel-to-bar or cell-to-constituent interpretation. The
grating stimuli have a four-pixel period and are off the original color/shape
training task. Spatial resolution and semantic transfer must therefore be tested,
not inferred from tensor shape.

The grounded memory stores hidden64, raw orientation logit and observed status,
not these maps. Its current answer uses logit sign. In the structured outcome
network, `risk_hidden` is a function of sensory rows alone; `route_hidden` is a
function of delivery history alone. The two pathways compose forecasts without
changing a represented local pattern. There is no existing native task that
requires within-patch constituent placement or compares an internally transformed
spatial pattern with a content model.

[Mechanical primitives](../../scripts/attention_content_form_primitives_v1.py)
operate on copied addressed maps and preserve source V, literal identity record,
raw logit and delivery time. T applies bandwidth 1/2/4 nonoverlapping block means
to saved V, repeats the means into the original shape, and rounds float64
arithmetic back to input dtype. No padding or partial blocks are used. On 4×4,
bandwidth four leaves only each channel's global mean. Restoration copies the
saved state. These are appropriate mechanics. Their toy `left-low/right-high`
identity is an assigned numerical relation, not a learned semantic relation.
Their prospective export is an analytic known-operator result, not learned A.

Thus the native suffix is useful for checking task effects, but a change in
color/shape logits, orientation margin, entropy, correctness or pixel-property
decoding cannot by itself qualify content-specific manner. There is no justified
existing-source proposal for that endpoint based on static inspection alone.

## Minimal extension: spatial content used to target an action

Introduce a separate new task/source version; do not amend historical frozen
sources. Initially retain each frozen encoder and add one shared local task
consumer. This is a new task head and state pathway, not discovery of an existing
native presentation counterpart. No attention model A or Reporter is needed for
this feasibility stage.

Use four outer item addresses with delivered 16×16 patches. Each observed patch
contains one red constituent and one blue constituent on a neutral background,
at two distinct positions of a declared 4×4 stimulus grid. Balance their positions,
color roles and left/right order; vary intensity, noise and token shape independently.
Include red-left-of-blue and blue-left-of-red worlds, vertical-order crossings,
role swaps and translated layouts. Renderer truth stays evaluator-only at
inference. Red/blue refer to task-trained categories, not externally claimed
experience. An outer address names a delivered memory item, not a true hidden
world object inferred from default state.

The literal relation is **“at retained item address j, the represented red
constituent is left of the represented blue constituent.”** Its reverse is an
equally frequent negative. Preserve identity as the ordered role pair and a
separate coarse relation record; never substitute supplied physical labels for
the consumer's possibly erroneous current content.

The new native consumer H applies the *same* 24→32→3 ReLU MLP at every W site,
producing local red/blue/background evidence. This candidate has 899 trainable
parameters. Normalize each role's sixteen scores over spatial sites to yield
native addressed spatial evidence `p_j(role,row,column)`. Exact normalization,
zero-evidence behavior and tie policy must be written before execution. A learned
occupancy task objective gives the evidence fields literal content semantics;
no form bins, adjectives or report targets are supplied.

Use those fields in the operational task: a query specifies retained address j
and role red or blue, and the Controller executes a grid-targeting action chosen
from that role's location scores. Score the actual targeted cell against task
truth, while retaining wrong choices and all ties. A coarse left/right query
uses spatial evidence summed over declared left/right halves. The coarse relation
and fine action paths must both consume W through H, with no bypass from saved
pixels, raw logits, identity records or physical truth. This creates a native
content role beyond attaching an external diagnostic label to a tensor.

The **candidate representational form**, stated literally, is the location-indexed
arrangement of red/blue evidence *within* the coarse relation: separately located
constituent evidence versus evidence repeated over the sites in a spatial block.
Record the entire `p_j` field and role-specific pairwise site contrasts, not a
pooled “detail” scalar. Under T2, H's shared-site computation yields identical
evidence within each averaged 2×2 block. That equality is architectural; the
semantic question is whether those positions carried constituent evidence that
the native consumer used before averaging and whether the same coarse relation
remains represented afterwards. Such retention is not guaranteed.

This is a candidate *model representational* manner. Additional spatial detail
can also be classified as additional content or task access. Choosing a coarse
content definition does not settle that interpretive issue. If development
supports only objective position decoding or a drop in fine action accuracy,
keep that technical scope and terminate semantic qualification. Do not rename
errors “blur” or manufacture experienced indistinctness.

## Development checks and controls

Fit H on untransformed task worlds for native occupancy and action utility, then
freeze H before same-V operator checks. Do not train H to recognize bandwidth or
reconstruct desired form labels. A separate whole-world development check uses
new layouts/noise and no A prediction loss. Native consumer performance supports
the interpretation of local content; intervention dependence supports its role.
Neither alone supplies phenomenal truth.

1. **Content and address.** Cross role identity, coarse order, fine placement and
   outer address. Permuting outer items and consistently relabelling addresses
   must move the evidence/action with the item. Relabelling coordinates without
   moving W must not move its native spatial evidence. No seen/unseen default
   answer counts as represented constituent content. Patch role fields between
   reachable, provenance-labelled items only as secondary causal diagnostics;
   such edits are synthetic unless realized by the declared task/operation.
2. **Same V and restoration.** Apply all T1/T2/T4 alternatives from each saved V
   at each addressed item, with a sham and exact saved-state restoration. Keep
   pixels, original category evidence, raw orientation logit and observed status
   fixed. Include a matched exogenous driver producing exactly the same W.
   Record every opportunity, including erased coarse relations, unchanged maps
   and uninformative transitions. No selection of only coarse-preserving cases.
3. **Spatial evidence versus energy.** Compare full role-indexed fields with
   per-channel means, norms, variance and global activation energy. Within-block
   permutations preserve per-channel sums and energy while changing arrangement;
   across layouts, test whether content/action follows coordinate placement.
   Post-encoder permutations are labelled synthetic and are not assumed to be
   naturally reachable. T itself changes energy, so preserved channel means
   alone are insufficient control.
4. **Scalar confidence and W-derived access.** Keep unchanged original logits
   separate from logits recomputed through W's native suffix and orientation
   head. Also retain H role margins, maxima, entropy, predicted positions, coarse
   correctness and fine targeting success. Compare layouts with the same coarse
   relation and matching scalar summaries. Spatial permutations of H output can
   give exact entropy/max/margin matches as synthetic controls; naturally
   matched overlap must be reported, not invented. A scalar baseline cannot
   recover position by definition, but that is an information ablation, not
   evidence for a special attention source. The decisive question is whether
   claimed form remains more than loss of content or fine-task access.
5. **Raw pixels and upstream causes.** A raw-pixel task consumer supplies a
   sensory alternative on the same task and split budget. Hold actual pixels
   constant during internal T, and separately vary pixels/layout at T1 to show
   inheritance. An upstream filtered image generally will not produce exactly
   T(V) through nonlinear conv/ReLU/max-pooling. Archive the mismatch if this
   arm is explored; do not claim upstream equivalence without measurement.
   Same-W exogenous parity is expected and limits attention-provenance claims.

Show role-specific local content validity and native action dependence before
giving the fields semantic names in a future export. The native consumer is
introduced to solve the targeting task, not solely to restate a desired adjective.
Only if coarse content survives in a bounded nonselected distribution and the
content-bound spatial arrangement has an independent native role is a next
semantic review warranted. Full fields remain available even where summarized
checks fail. Freeze tolerances before a future run; none is registered here.

## Small budget candidates, explicitly unregistered

One proposed initial DEVELOPMENT envelope is 128 fit worlds plus 64 diagnostic
worlds for **each** fixed encoder 1301/1311/1321: 576 worlds total, each with four
outer items. Fit one 899-parameter H per encoder for at most 500 full-batch Adam
updates, one initialization and one fixed learning-rate candidate (.003). Select
all seed/split namespaces before generation; whole worlds, not correlated items,
define splitting and uncertainty. There is no heldout confirmation split. The
192 diagnostic worlds supply 768 addressed opportunities and 2,304 item/operator
alternatives for T1/T2/T4, excluding separately declared duplicate controls.
These are budget suggestions, not a power argument, acceptance threshold or
permission to execute. Preserve failures across all three encoders.

If the 24×4×4 extraction cannot support native targeting, one **optional explicit
source revision** is `encoder.net[:3]`, the first pooled 12×8×8 map, with a shared
12→32→3 head (515 parameters) and an 8×8 targeting grid. Use fresh development
worlds and repeat the entire semantic review; this is not an undocumented
favorable-layer search. A candidate ceiling is another 128+64 worlds per encoder,
500 updates and one initialization. If that also fails, stop this frozen-encoder
extension. Training a new spatial encoder would be a separate architecture
project and requires a new design, not further retries under this version.

The cheapest appropriate first action before those candidates is writing the
new task/input/consumer contract and reviewing that static source, still with no
model inference. This review itself has performed only that design work.

## Negative endpoints and later implications

Terminate this semantic stage if local constituent semantics fail, native actions
can bypass the local field, T changes only generic statistics, coarse content
disappears whenever candidate form changes, or the only justified endpoint is
correctness/confidence/fine-task access. Also retain mixed results and absent
scalar-matched support; do not choose a successful encoder or invent a binning
scheme after looking. The outcome then remains **no qualified presentation
counterpart**, accompanied by whatever task/statistical feasibility is justified.

Even a positive native field/action result establishes a newly engineered
content-bearing spatial process. It does not establish A as its source, attention
as psychological attention, or qualia. Later A_current/A_next models would need
separate timestamp contracts and equally informed ordinary/analytic comparators;
their targets must be independent of these semantic checks. A's forecasts must
not be relabelled current W. The Controller executes operations; the Reporter,
if later introduced, is separate diagnostic instrumentation. Report accuracy or
consciousness-report structure cannot replace this gate.

This reviewer is a separate commissioned agent in the same shared-model team,
with inherited context and theory exposure. This is useful independent task
allocation, not an independent model family, human endorsement, empirical
replication or institutional review. No human input is required for the proposed
bounded technical development. The user's ultimate source-of-qualia goal remains
unachieved, and engineered compatibility cannot be converted into proof by naming.
