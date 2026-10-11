# Spatial native task v1: independent first contract review

2026-10-10. Static review by `/root/spatial_native_task_contract_review`. This
first pass reviews the [development contract](../../../docs/SPATIAL_NATIVE_TASK_V1_PROTOCOL.md)
against the [semantic design](../../attention_content_form_semantics_design_v1/review.md),
the development plan, existing encoder/readout source and copied spatial
primitives. The new task module, runner and their fixtures were **not inspected**
in this pass. No checkpoint, tensor archive or dataset was loaded; no world,
fit, model inference, Reporter call, network operation or Git operation occurred.

**Verdict: a useful, bounded native-content development proposal, with unresolved
pre-execution specification and measurement obligations. It does not qualify
representational manner or provide differential evidence that the attention-control
model is the source of qualia.** Proceed with implementation and independent
source review. Do not treat this document as certification of an executable or
as permission to skip the contract's own review/registration sequence. Preserve
this first pass when addressing its findings.

## What the contract supports

The extraction site agrees with `VisualEncoder`: `net[:6]` ends at the second
pool and produces 24×4×4 maps for 16×16 inputs. `net[6:]` is the original
color/shape suffix. The orientation pathway requires **both** `net[6:9]` and
the separately frozen hidden64 standardizer and 64→32→1 readout in
`FrozenSensor`; that slice alone is not an orientation head. These original
consumers are controls on off-task inputs, not constituent truth. The overlapping
10×10 nominal receptive fields make grid occupancy a learned task association,
not a direct interpretation of output sites as observed cells.

The proposed shared 24→32→3 head has 899 parameters; the 3→32→3 raw-grid
alternative has 227. Freezing the encoder and fitting H only on original fit
maps gives a newly engineered content decoder. Using its addressed role field
to emit and score the selected cell supplies an operational native task role
beyond external annotation, provided the actual Controller implements the stated
input isolation. H need not be trained on action loss for the proposed occupancy
training and argmax policy to have that role. This is an inference-time targeting
consumer, not a learned attention predictor, a Modeler-refinement mechanism or
an experiencing locus. The separate external Reporter is absent.

The documented primitive reads saved V, applies finite nonoverlapping means in
float64 and returns the input dtype; T1 copies, T2 repeats 2×2 means, and T4
repeats a channel mean. It preserves unselected addressed items. The current
restoration source copies immutable buffers directly, consistent with the
correction proposed in the preserved earlier failed mechanical review. This
inspection is not a new execution or verification of that repair. The new runner
must demonstrate that it actually uses these paths rather than merely duplicating
their names.

The main counts are internally consistent: 576 worlds, 768 diagnostic items,
1,536 T1 role queries, 2,304 T1/T2/T4 item alternatives and 4,608 corresponding
role actions. Whole-world uncertainty, all three encoders, fixed fit budget,
no diagnostic selection, no automatic layer fallback and unconditional operator
denominators are appropriate. No successful subset can replace those totals.

## First findings and required dispositions

**C1 — The Markdown contract is not yet the complete pre-output freeze.** Lines
32–33 explicitly delegate exact rendering and balancing to reviewed source.
That is acceptable if the completed source is independently reviewed and bound
before any development generation or checkpoint load. That review must establish
token pixel masks and cell coordinates, neutral background, palette/intensity/noise
ranges and clipping, nuisance sampling independence, balance across split/address/
role/coarse-order/fine-placement, and the exact world-index-to-seed rule. Namespace
formulas alone do not specify those details. It must also bind H initialization,
raw-head RNG handling, channel-standardization axes, occupancy class encoding and
loss reduction, raw-grid averaging, queried address handling and archive schema.
The runner must freeze bootstrap resampling/aggregation and interval quantiles,
the definition of a changed role field, and what “correctly retained coarse order”
requires at both T1 and T2. Do not resolve these after diagnostic outputs. Hashes
must bind the inspected executable, configuration, software receipts and exact
encoder/orientation-readout checkpoint artifacts. This first pass has not verified
any of those executable obligations.

**C2 — The coarse relation has changed from the semantic proposal.** The semantic
design proposes a coarse query using evidence summed over left/right halves;
the contract instead compares expected columns, with an absolute 1e-6 unknown
band. This is an explicit executable definition, but not an equivalent readout:
two distributions can have the same half masses and different expected columns,
or vice versa. Expected-column order is a relation between role centroids, not
proof that each represented constituent occupies its intended half. Record this
as a deliberate development change before execution. Keep the consumer's
possibly erroneous relation separate from evaluator order and from any retained
identity record. “Correctly retained” must not include a T1-wrong/T2-correct
transition merely because T2 agrees with truth. No redesign is demanded here;
the source/results must state which relation was actually tested.

**C3 — Spatial softmax is conditional localization, not role detection.**
Cross-entropy supplies three-class occupancy logits at each site. The subsequent
role-minus-background spatial softmax is mathematically consistent: its exponent
is a local role/background odds coordinate before normalization. It does not
equal local occupancy probability and does not test whether a role is present.
It always produces unit total mass for each queried role, including when every
site favors background or both roles select the same cell. Uniform evidence is
well defined, but is not an explicit absence response. One-red/one-blue rendering
makes this a legitimate forced-location task; limit the content claim to that
distribution. Entropy, maxima and margins describe conditional spatial evidence,
not calibrated correctness or experienced confidence. Retain the native occupancy
logits and loss inputs so a numerical reviewer can reconstruct both objectives
and fields. A mean occupancy loss over fourteen background sites and two token
sites can conceal role failures; report role-separated targeting and errors rather
than relying on the mean alone. No additional detection task is required for this
bounded version.

**C4 — T2 action degradation is partly an architectural tie effect.** T2 makes
each block's feature vectors identical. Fit-only channel standardization and the
shared H preserve that equality. Each role's argmax therefore ties across at least
four cells; the first-row-major policy can choose only a block's top-left cell.
With exactly balanced fine placement, T2 fine targeting is structurally capped
at .25 even if the correct block is perfectly identified. T4 makes all sixteen
sites equal, selects cell zero for both roles, and makes coarse order unknown;
under balanced role positions its mean fine accuracy is .0625. These are expected
mechanical consequences, not semantic discoveries or implementation failures.
Record tie multiplicity and policy as well as selected actions. A conventional
top-two-site margin is zero under T2 regardless of separation between blocks.
Do not translate that margin or lost fine accuracy into blurred presentation.

Synthetic site permutations preserve the multiset of H outputs and thus role
entropy/max/margin, within numerical tolerance. Their selected action follows
the permuted unique maximum; **first-row-major tie breaking is not equivariant**
to arbitrary site permutations. Control assertions must reflect that distinction.
A permutation of a uniform T4 map cannot change spatial placement at all; retain
that ineffective opportunity rather than claiming every permutation changes form.

**C5 — Duplicate-control opportunities need an explicit ledger.** The main
T1/T2/T4 counts exclude sham, saved restoration, exogenous repeats, outer-address
permutations and synthetic site permutations. Their exact number, selected
permutations, application point and schedule are not written here. Freeze these
in the reviewed runner. Identify each control by encoder/world/item/operator/
role as applicable; retain every failed, unchanged, tied or erased case and
record denominators separately from the 2,304 main alternatives. Same-W driver
parity should compare native logits, fields and actual emitted actions; it is
expected by construction when provenance is audit-only. Demonstrate query/input
isolation with changed evaluator truth/retained metadata and consistent address
remapping, rather than relying on a comment asserting no bypass. Preserve
nonselected items, original pixels/maps/logits, source weights and exact restored
state. Independent numerical reconstruction needs sufficient retained native
outputs, not only rounded aggregate summaries.

## Interpretation and the source-of-qualia goal

The per-encoder T1 .50 targeting/.75 coarse and raw-grid .95 targeting requirements
are chosen engineering margins. They do not define psychophysical presentation,
power a theory test or establish semantics for every role separately. A pooled
role mean can hide asymmetry; expose both roles without inventing new acceptance
thresholds after outputs. Raw-grid failure is a task/control failure; encoder-only
failure is a source/decoder/budget limitation; an invariant failure is software or
provenance failure. Preserve mixed and failed first runs. Passing every engineering
requirement warrants only native targeting feasibility in this new task.

T1-to-T2 changes with correct coarse order can demonstrate loss of fine spatial
information at fixed delivered pixels. Fine placement is itself content, and
the tested consumer uses it precisely to perform the fine action. The existing
design offers no additional criterion proving that a surviving coarse relation
and altered fine field are manner rather than less detailed content/task access.
The contract correctly defers that distinction. A successful new source should
receive a later independent semantic assessment that can still conclude **no
qualified presentation counterpart**. Do not replace repeated negative source
qualification with a coarser content definition, a favorable subset, a new form
bin, or a favorable encoder/layer chosen after results.

Synthetic scalar-matched permutations test arrangement beyond those scalars;
they do not prove naturally reachable scalar-matched support. The raw-grid arm
has different information and capacity. Sensory/scalar omissions are ablations,
not equally informed rivals. Endogenous/exogenous labels select the same T and
thus cannot discriminate attention provenance or phenomenal source. Deferring
ordinary/analytic process predictors is reasonable **because this stage contains
no A prediction or reporting assay**; none of its results may be promoted to
source specificity. A later A_current/A_next mechanism needs timestamped targets
and matched ordinary/analytic information and budgets. Modeler comparison and
refinement is also absent; executing a queried grid action does not supply it.

The ultimate goal remains credible evidence that the attention-control model is
the source of qualia. This task can repair one prerequisite—content-bearing native
spatial utility—but neither identifies A as the source nor distinguishes an
ordinary implementation of the same computation. Future source/report fidelity,
independent content/manner separation, observable differential predictions,
confirmation and replication remain separate requirements. The prior strict
negative and this proposal's technical-only endpoints remain valid results.

## Exposure and preservation

This is a separately commissioned agent in the same model team, with inherited
project/theory context and the previous negative reviews available. It supplies
independent task allocation, not an independent model family, human endorsement,
institutional approval or empirical replication. Theory statements are taken
from the supplied MSTC role audit; its external paper was not retrieved. Only
`review.md` and `metadata.json` in this owned directory were written. First findings
C1–C5 remain preserved; later source review or revisions must be separately labelled.
