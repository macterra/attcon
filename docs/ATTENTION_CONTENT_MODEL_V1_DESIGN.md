# Learned attention-content model v1: prospective design

2026-10-11. Architecture/mechanism development proposal. This document authorizes
no generation, checkpoint load, fitting, inference, Reporter call or reserved
evaluation. Independent design and executable review, a committed registration,
and satisfaction of the native reconstruction dependency remain required. The
user's broader authorization is retained; this clarification does not introduce
a new user-approval step. No source semantics or
presentation counterpart is claimed qualified here. No result is asserted.

Amendment v1 resolves C1–C4 of the preserved
[independent design review](../audits/attention_content_model_v1/design_review/review.md)
before A implementation or outputs. The original design record and review remain
preserved. Architecture, world envelope, optimizer budget and native gates are
unchanged; no post-fit performance or manner threshold is added.

This is the next implementable bridge in the
[content-form development plan](ATTENTION_CONTENT_FORM_DEVELOPMENT_V1_PLAN.md),
using the exact source and frozen native consumer in the
[spatial native contract](SPATIAL_NATIVE_TASK_V1_PROTOCOL.md). It does not alter
that contract's thresholds, splits, controls, first-pass preservation or negative
endpoints. Its aim is a learned, content-bound attention-control state that can
be accurately described and subsequently tested for independently recognized
report structure. It neither requires access to private experience nor claims
that a task must be impossible without this state.

## The concrete bridge and its dependency

The previous mechanical milestone implements T but contains no learned A. The
native spatial task asks whether H consumes constituent-location evidence in W.
This extension learns how addressed commands change those maps, keeps separately
scored current and prospective estimates, and makes a native Controller choose
an operation using the prospective content. An external Reporter is not a
Controller, does not generate W, and is absent from this development run.

Retain the three frozen encoders, seeds 1301/1311/1321, their exact 24×4×4
extraction, the independently fitted native H and its own standardizer. Reuse
their frozen artifacts only after the native development's existing gates have
actually passed and its retained outputs have been independently reconstructed.
This proposal has not inspected those outputs. A failed native stage terminates
this version; no fallback layer, replacement H or changed gate is permitted.
The reported diagnosis of all 106 original reconstruction failures as source/
reference-kernel arithmetic is an explanation, not corrected full revalidation.
That full revalidation and independent reconstruction must complete before actual
A execution. This amendment has not loaded checkpoints or numerical results.

Native utility establishes task/content semantics. The potential manner claim
concerns the within-item representation of the red/blue location relation;
T2 can change its fine arrangement while sometimes retaining its coarse order.
Fine location can also be additional content, access, or uncertainty. Independent
semantic review must examine the exact frozen source, address/content crossings
and scalar/access alternatives before any presentation qualification. There is
no assurance of qualification. A learned map predictor can be developed and
reported technically if native task feasibility passes but manner fails; that
outcome retains **no qualified presentation counterpart** and does not open the
phenomenal correspondence stage. Existing native study outcomes remain intact.

## Actual state, commands and timestamps

Each fresh world supplies four patches, V[4,24,4,4], and initial completed W₀=V.
V remains immutable throughout all common histories and forks. Actual operations
use the existing finite block-average primitive, including float64 accumulation,
return-to-input dtype, exact T1 copying, and no cumulative degradation:
Wₜ₊₁[address] = T_bandwidth(V[address]). Other addresses remain unchanged.
T1/T2/T4, exogenous parity and saved restoration retain their current meanings.
Restore an immutable saved snapshot; never describe lossy averaging as invertible.

Commands are (address, symbol), with addresses l0–l3 and opaque symbols c0–c2.
Each address has an independently assigned permutation from symbols to bandwidths
{1,2,4}, fixed within its world. The evaluator/executor knows the assignment.
Neither A, the ordinary predictor nor the analytic mapping estimator receives it,
bandwidth labels, task truth, pixels, driver labels, or a mapping-derived cue.
The operator family is publicly declared for all comparators; the uncertainty
concerns its command assignment, not the existence of T. Mapping is not inferred
from an address name, world identifier, renderer seed or evaluation metadata.

At decision t the executor has completed Wₜ. The model has V, issued commands
through t−1, and full completed-map feedback only through Wₜ₋₁. Wₜ itself is
withheld until the decision, prediction scoring and all forks at t are finished.
At t=0 the observed initial W₀=V is available; it is an exact initial state, not
a learned-current prediction. At t≥1, A_current(t) predicts Wₜ from the last
revealed snapshot and the last issued command. A_next(t,address,c) predicts the
completed Wₜ₊₁ for that command without receiving Wₜ or any fork outcome.
Wₜ and each fork Wₜ₊₁ remain separate actual records. Never call A_next current
presentation or a completed event. Wrong estimates remain source contents.

After scoring at t, reveal Wₜ and its already recorded command cₜ₋₁; update the
corresponding recurrent state once. The fixed common command cₜ is executed to
construct Wₜ₊₁, which remains concealed at the next decision. Feedback at t=0
has no transition and performs no recurrent update. The terminal t=15 reveal is
archived, with no additional scored decision. Assert this ordering in the runner;
training target tensors are not inference arguments. Completion feedback includes
all four maps, but only the changed item's V, completed W and command enter its
shared recurrent update; all comparators may access the identical full history.

## Exact proposed trained architecture

Use CPU float32 PyTorch with four threads. Fit a separate A per frozen encoder.
Fit-only channel mean and population standard deviation normalize both V and W;
scales below 1e−6 become one. Compute these coordinates from this version's fit V
only. Keep H's separately frozen native coordinates; invert A normalization before
feeding a predicted map through H. No orientation or original category logits
enter A. Those pathways remain recorded evaluator controls.

The candidate implementation is a `torch.nn.Module` composed of:

* Four addressed recurrent states h[4,48], initially zero; shared
  `nn.GRUCell(input_size=771, hidden_size=48)`. For revealed transition (a,c,W),
  its input is concatenated normalized V[a].reshape(384), normalized completed
  W[a].reshape(384), and one-hot c[3]. Update h[a] only. There is no global
  recurrence, address embedding, mapping label or private side channel.
* A shared residual decoder
  `nn.Sequential(nn.Linear(435,96), nn.ReLU(), nn.Linear(96,384))`.
  Its input concatenates normalized V[a][384], h[a][48], and c[3]. Its output
  is added to normalized V[a], reshaped to [24,4,4], and denormalized. This is
  a learned numerical transform, with no block-average call, bandwidth classifier,
  hardcoded spatial smoothing or fitted adjective inside A's forward path.
* An explicit assembly function. For current estimates, copy the last observed
  completed snapshot and replace only the address of cₜ₋₁ with the decoder's
  prediction using pre-reveal h. For next estimates, copy A_current and replace
  only the candidate address with the decoder prediction. At t=0 use V as
  A_current. Nonselected next addresses can inherit uncertain current estimates;
  score them separately rather than silently using concealed actual Wₜ.

The GRU has 118,224 parameters and decoder 79,104: **197,328 learned parameters**.
Inputs per batch are V[B,4,24,4,4], last-revealed W[B,4,24,4,4], h[B,4,48],
issued address[B] and symbol[B], with explicit time/history masks. Outputs are
A_current[B,4,24,4,4] and the three alternatives
A_next[B,3,4,24,4,4] for a supplied candidate address. Latent h and content maps
are distinct exports; the native report candidate is the estimated addressed
content map plus frozen-H local logits/role fields computed from that map.
Do not substitute actual H(W) for H(A) while labelling it attention-model content.

No code is implemented by this design milestone. The later implementation must
use these actual trainable modules and declare any departure as a new reviewed
version, not leave A as a prose name for an analytic operator.

## Fixed development envelope and split boundary

The proposed envelope is 96 fit, 96 development and 96 heldout worlds per encoder:
864 worlds total. This is bounded mechanism development, not powered confirmation.
Whole-world splitting includes all four items, histories, operations and donor
forks. Different nuisance realizations use the existing renderer and its exact
128-layout orbit; this does not establish unseen-layout generalization. Mapping
permutations also recur across splits, as intended for inference of an unknown
assignment in a fresh world. No row, item or temporal-window split is allowed.

Use renderer namespaces `310000000 + encoder_seed*1000 + split*100000`, where
split=0/1/2 for fit/dev/heldout. For each address, start with sixteen copies of
each of the six lexicographic bandwidth permutations and independently shuffle
the 96 entries using a local CPU generator seeded by
`330000000 + encoder_seed*1000 + split*100000 + address_index`.
World indices and seed metadata stay evaluator-only. This fixes balanced
per-address marginals without forcing addresses to share an assignment.

Every common episode has fifteen transitions. For the first twelve, each address
receives each symbol once. At each address shuffle [c0,c1,c2] with a local generator
seeded by `340000000 + encoder_seed*1000 + split*100000 + world_index*4 + address_index`.
Visit addresses in row-major order within each of the three rounds. This schedule
is independent of the mapping. The final three query addresses are
`(world_index+j) mod 4`, roles red/blue by `(world_index+j) mod 2`, for j=0,1,2.
Their common-spine symbols are `(world_index+j) mod 3`. Every decision t=0..14
has all three same-V prospective command forks for its scheduled address.
Only the final three decisions carry native task queries. Common histories remain
fixed across A and every rival, independently of their policy decisions.

Initialize A with seed `320000000 + encoder_seed`, the ordinary predictor with
that seed plus 1000000. Fit each once for exactly 300 full-batch Adam updates,
learning rate .003, betas (.9,.999), eps 1e−8, weight decay zero, amsgrad false,
foreach false. Reset recurrent states for every world on every update; unroll the
complete fifteen-transition episode with backpropagation through all feedback
updates. Apply global gradient clipping at norm 1.0 after backward. No dropout,
early stopping, restart, hyperparameter sweep, additional encoder or layer search.

The loss is .5 times mean normalized-map MSE for selected-address current
predictions at t=1..14, plus .5 times mean selected-address prospective MSE across
all fifteen decisions and three symbols. Each term averages its own declared
world/time/command opportunities and all 384 map elements. Targets are actual
operator outputs, not cells, mapping labels, phenomenal adjectives, report prose,
human ratings or invented clarity bins. Report each term independently. The
assembled full-state errors, H-derived fields and task outcomes are independent
diagnostics, not replacement losses.

The one exclusive execution has the following strict release sequence. A sealed
split means no generation, feature extraction, map/target construction, source
forward calls or output inspection from that split, even if its seeds are public.

1. Before any generation or fit, commit the reviewed input registration. Bind
   SHA-256 hashes of the exact frozen encoder and H weight files, H's native
   normalizer and orientation-control inputs, extraction `net[:6]`, source/module
   dependency closure, primitives, renderer, collector, runner, verifier, this
   contract and permission schema. Specify which files are model inputs, evaluator
   truth, audit-only metadata and allowed feedback. Record source shapes/dtypes,
   native correction/reconstruction acceptance and all split namespaces. These
   **input hashes** are existing frozen artifacts, not nonexistent fitted A.
2. Release fit only. Generate its 96 worlds per encoder, fit its normalizer and
   train A/ordinary exactly once for 300 updates each. Dev and heldout stay sealed.
   Record first outputs and all update counts. Seal an immutable fit receipt
   containing hashes of each fitted A/ordinary checkpoint, fit-only normalizer,
   optimizer terminal state, training log, fit manifest and input registration;
   include initialization seeds and evidence that no dev/heldout forward occurred.
   Append/commit that receipt to the registration before releasing dev. These are
   **fitted hashes**, distinct from the input hashes and never retrospectively
   substituted for them.
3. With fitted bytes immutable, release dev for its single fixed diagnostic pass
   and inspection. No optimization or recurrent-history transfer from dev to
   another world/split is allowed. Run the predeclared literal-content diagnostics
   below and source/observation invariants; preserve failures and the independent
   assessment without selecting a checkpoint, query, donor or threshold. Seal
   an immutable dev receipt linking the same fitted/input hashes, dev outputs,
   reconstruction checks and semantic disposition. Append/commit it before
   heldout release. Low predictive or semantic utility does not authorize refitting.
4. Release heldout only after the registered input and fitted receipts exist and
   the software/observation checks pass. Use those exact fitted bytes for one
   heldout pass; no further fit, new initialization or dev-selected analysis.
   Archive a heldout receipt and independent reconstruction. If manner is absent
   but technical bookkeeping is valid, retain the predetermined technical run
   and its negative semantic endpoint. Reporting remains separately gated.

These receipts implement a predeclared sealing procedure within one authorized
execution, not new performance-dependent permission loops. A hard software/input
failure stops the run with its first outputs retained; correction requires a new
reviewed version. Any additional fit or architectural change requires preserved
first outputs, fresh split namespaces and independent review. Do not inspect
heldout outcomes to choose report queries, semantic subsets, donors, stopping
rules or tolerances. All heldout calls must be reconstructable from permission
manifests, including exact per-call histories rather than inferred nonleakage.

## Native Controller use and fair alternatives

At each of the final three query decisions, compute H(A_next) for all three
symbols on the queried address. For the queried role, choose the symbol with
largest predicted role-field maximum; exact ties choose c0 before c1 before c2.
Execute that command in its actual same-V fork. The existing NativeController
then uses H(actual completed W) alone to issue the actual grid-cell action,
including its first-row-major tie rule. Archive the predicted field/selected
symbol and the emitted actual field/cell separately. The chosen branch is real
execution and is not substituted for the independently fixed common spine.

This is attention-control use of a content prediction. The utility is also a
scalar sharpness/access proxy; scalar policy parity is expected to be possible.
Record a confidence-only choice rule supplied the same per-symbol maxima. It
must make the identical symbol choice; that identity bounds what command choice
alone establishes. The full spatial predictions remain necessary for the stated
content-error and correspondence tests, not alleged universally necessary for
task performance. Compare task accuracy descriptively; never use performance
necessity as a narrow reporting gate.

The ordinary process predictor has the identical trainable architecture, inputs,
targets, history, feedback, optimizer, update count and parameter count, separately
initialized as above. Its output can replace A in the same Controller, eliminating
a usage/instruction advantage. Also run an exact deterministic alias check on the
same weights: renaming this computation an ordinary state estimator must change
nothing. Independent-initialization differences are not evidence for the name or
for qualia; architectural ordinary parity is a legitimate explanation. No rival
is asked to reproduce outputs while being denied A's computation or observations.

The analytic alternative receives the same V, full delayed feedback, command
history and declared operator family. For each address maintain a uniform posterior
over the six mappings, eliminating a permutation only when its actual T(V) is
inconsistent with revealed W at atol=1e−6, rtol=1e−5. Preserve all surviving
mappings, including ambiguity on uninformative V; zero survivors is an invariant
failure. Before a reveal, predict each map by the posterior mean of the respective
T(V). Assemble current/next states with the same visibility contract as A, and
apply H to that predicted mean for the primary policy comparison. Record posterior
entropy and predictions integrating H over mappings as secondary alternatives.
At every decision record revealed-symbol count and posterior survivor count/
entropy per address. When T(V) outputs are distinct, two revealed symbols can
identify the permutation; the final policy queries can therefore be fully
calibrated. Do not call those choices online resolution of an unknown mapping.
Publish early versus calibrated errors descriptively, retaining ambiguous cases
and the complete primary denominators.
This mapping estimator is not an oracle and does not receive bandwidth truth.
It has analytic numerical knowledge rather than matched trainable capacity;
publish that difference. An evaluator-known-mapping ceiling, if implemented,
must be explicitly oracle-only, absent from model/report inputs and primary
evidence. Ordinary or analytic superiority/parity can establish a concrete
prediction/access explanation without demanding supernatural source specificity.

Sensory-only V→field and W-derived identity, orientation, confidence/access
readouts are secondary explanation controls. They omit information or introduce
additional information and must not be labelled equal-information rivals.
Preserve original logits and calculate transformed-W controls using the frozen
pathways. Report address/content crossings and nuisance statistics regardless of
whether they are informative. No strong specificity conclusion follows solely
from beating an information-omission baseline.

## Fixed causal controls and reachability

Run sham, exact saved restoration, and exogenous driver parity at each of the
three queried decision snapshots. Compare identical numerical operators from
identical V; provenance is audit-only and identical W must yield identical native
actions. These branches never update A's common history. Test consistent address
permutation by rolling all four items +1 with V, last feedback, h, commands, queries,
mapping/executor state and outputs; shared local modules imply equivariance.
Invalid/nonfinite tensors, unknown symbols/addresses, unseen addresses, wrong
timestamps and target leakage are hard failures. Copy/restoration use exact
equality; numerical encoder/readout/permutation checks use atol=1e−6, rtol=1e−5.

For reachable A/W mismatches use two separately counted schedules at each final
decision: a current-estimate patch at the most recently issued command's address,
and a prospective-control patch at the queried address. Run both even when their
addresses coincide; retain their duplicate status. The donor for each is fixed
before execution. Keep that world's V and all other address mappings unchanged;
replace the selected address's mapping by the next
lexicographic permutation modulo six. At decision t∈{12,13,14}, execute donor
common commands with indices 0..t−1 to construct donor Wₜ, but recurrently update
only from donor completions W₁..Wₜ₋₁, paired with command indices 0..t−2. Thus a
prefix has t executor transitions and t−1 permitted feedback updates. Donor Wₜ
and all donor forks remain concealed from the pre-reveal h and prediction call.
Save the exact update list (command index, address, symbol, completion timestamp,
map hash) and per-call pre/post-reveal visibility masks. A's and ordinary's donor
states use their own immutable fitted weights and the same donor observations;
the analytic source replays posterior updates from those same permitted reveals.
The resulting h[address]
is an actually reached state, not an invented vector or arbitrary output override.
Patch only that h (or the analytic source's addressed posterior) into the
recipient before its current/next decoding, retaining
recipient V, actual W, last revealed snapshot, commands, time and native H.
Do not refresh recipient feedback in response to the patch. Because recurrence
is per-address, other h and address-local decoder contents are unchanged; verify
this exactly. A current patch can change that addressed estimate inherited by a
full A_next assembled for another queried address. This is intended inheritance,
not leakage; compare nonpatched address-local decoder outputs, then separately
verify the declared current/next assembly dependency. Do not incorrectly require
the inherited patched address in A_next to stay fixed.
Archive the complete donor-prefix provenance, donor/recipient times and mappings
in evaluator-only records, plus both decoded states and actual fields.

For each source and patch, save its recipient pre-reveal state and unpatched
estimates/policy outputs first. Restore the original recipient h (or analytic
posterior) before any feedback update, decode and select again, and require exact
equality to those baseline states, estimates and policy outputs. Retain the saved
last-revealed snapshot, commands, time and normalization throughout. A restoration
pair is this source patch→restoration comparison; it is not the separate actual-W
saved-restoration control and does not replay a new recipient observation.

The donor state is reachable under a different recorded history; it is not
claimed normally reachable from the recipient's unchanged observation history.
Its substitution is a selective synthetic internal intervention. Every predetermined donor
counts even if h/maps/actions are equal, wrong, or uninformative. Donor h alone
does not guarantee a valid completed whole-world state; decoding retains the
recipient's explicit assembly contract. Report current/next addressed content
changes, scalar changes, Controller symbol changes, emitted actions and factual
agreement with fixed W separately. Patch/restoration shows the source's causal
role in computation/export, not changed experience. Do not select donors after
finding an attractive dissociation or describe synthetic site permutations as
reached attention states. A frozen-H site-permutation sensitivity check may be
retained with explicitly synthetic provenance, outside this reachable ledger.

## Named denominators and endpoints

For each split the all-encoder denominator is 288 complete worlds (96 per encoder)
and 1,152 items. The table below gives heldout counts; fit/dev have the same fixed
opportunity schedule. A and the ordinary source fit on fit only; analytic outputs
have no trainable fit. Use explicit source index s∈{A, ordinary, analytic}; the
analytic source means its posterior-mean-map primary prediction. A ledger marked
per-source repeats for each s, sharing the same worlds and actual targets. The
three sources never multiply the independent sampling denominator. The primary
ledgers are:

| Ledger | Scope | Per world | Heldout all encoders |
|---|---|---:|---:|
| Common-spine actual transitions | Shared actual history | 15 | 4,320 |
| Actual prospective alternative targets | Shared forks | 45 | 12,960 |
| Current selected-map estimates, t=1..14 | Per source | 14 | 4,032 |
| Prospective selected-map estimates | Per source | 45 | 12,960 |
| Full prospective item records, including inherited estimates | Per source | 180 | 51,840 |
| Queried actual operator/role actions | Shared subset of actual forks | 9 | 2,592 |
| Prediction-selected executed policy branches | Per source, duplicate actual fork | 3 | 864 |
| Reachable current donor patch/restoration pairs | Per source | 3 | 864 |
| Reachable prospective donor patch/restoration pairs | Per source | 3 | 864 |
| Actual-W sham, driver-parity or saved-restoration snapshots, each | Shared actual controls | 3 | 864 |
| Address-remapping source comparisons | Per source | 3 | 864 |
| Address-remapping actual snapshots | Shared actual control | 3 | 864 |

Across three primary sources these produce 12,096 current selected-map records,
38,880 next selected-map records, 155,520 full-next item records, 2,592
prediction-selected policy branch records, and 5,184 donor patch/restoration
pairs. Each donor schedule contributes 2,592 of the last total. Restored outputs
are paired duplicate baseline records, not additional source opportunities.
Confidence-only policies are derived from each primary source's three maxima:
864 duplicate choice/action records per source, with no extra execution or
independent evidence. The same-weight alias is an exact duplicate of A's ledgers,
not a fourth independently fitted source. Analytic expectations of H over mapping
hypotheses are derived secondary records with the same 12,960 alternatives and
864 query decisions, not another actual/world denominator.

Do not deduplicate the two donor schedules, including coincident addresses. Each
world entails 2×(12+13+14)=78 donor executor transitions and
2×(11+12+13)=72 permitted donor feedback updates. Across heldout these are
**22,464 shared donor-prefix transitions** and **20,736 feedback events**.
Execute those donor prefixes once, share their actual observations across the
three sources, and replay 20,736 recurrent/posterior updates per source (62,208
source update operations total). These workload/provenance counts are not the
1,728 donor pairs per source or added independent worlds. Serialize the common
and donor prefixes once with references from per-source outputs. The common
spine's 4,320 transitions and shared 12,960 actual alternatives remain separately
labelled, including their matching-command duplicates. Prediction-selected
baseline branches execute again and retain their duplicate fork references.
Donor policy outputs instead select/reference an already executed shared query
fork, so their 1,728 records per source add no actual executor transitions.

Publish analogous fit/dev ledgers, all failures, retained ambiguous mappings,
uninformative transitions and all opportunities with unchanged fields or ties.
No coarse-preserving or changed-map subset is the primary denominator. Frozen-H
role fields are conditional forced localization, not calibrated presence evidence.
Report red/blue separately. Do not overwrite wrong native coarse order with truth.

Primary outcomes are current and next selected-map MSE; inherited-map/full-state
MSE; frozen-H field L1 error, coarse-order disagreement and cell disagreement;
and actual targeting accuracy under each prediction-selected policy. Publish the
complete distribution and paired differences against the ordinary and analytic
alternatives. Use per-encoder whole-world paired bootstrap intervals, 256 resamples,
seed `350000000 + encoder_seed`, .025/.975 quantiles. Model seeds are three fixed
engineering sources, not a model-family sample; worlds share layouts and operator
families. No population, human or theory-facing significance claim follows.

Causal outcomes use the full predetermined intervention ledger: how many patches
change addressed source maps, H fields, chosen commands and actual actions; how
many leave nonselected addresses intact; and how often source-vs-W assertions
would disagree. Separately tabulate scalar changes and W-derived task/access
alternatives. Sham, driver parity, exact restoration, extraction isolation and
absence of forbidden inputs must all pass; any hard invariant failure invalidates
the run pending a preserved new-version repair. No new performance threshold
qualifies manner or consciousness. Loss improvements and task changes are bounded
mechanism measurements, including null/negative outcomes.

## Independent literal-content and export qualification

H was validated on actual native maps, not arbitrary decoder outputs. H(A), H(W)
and evaluator stimulus truth are three separately attributed quantities; an
analytic posterior-mean map can also be outside the executed T-state family.
Before any future reporting-source gate, an independent evaluator examines the
fixed dev split's 4,032 current and 12,960 next selected-map records per source,
without fitting H/readouts again, changing A, selecting report queries or using
heldout outcomes. This is the declared independent-world diagnostic split, not
independent model-family replication. Preserve its entire ledger, including all
wrong/ambiguous estimates and uniform/tied fields.

Freeze these diagnostics now: normalized map MSE; full role-indexed H(A)/H(W)
fields and their L1 discrepancy; red/blue cell and coarse-order disagreement;
separate agreement of H(A) and H(W) with evaluator truth; entropy/max/margin and
expected columns; W-derived category/orientation/access alternatives; the existing
consistent address crossing and source-local content dependence checks; and each
predicted selected map's distance to every actual T1/T2/T4(V) alternative using
the same normalized 384-element mean-square metric. Report nearest-family distance
continuously with the three distances, not as a newly chosen support cutoff.
Analytic posterior means retain their hypothesis/weight provenance, even when
they do not equal any executed map. Map distance or low loss alone cannot qualify
literal constituent meaning or manner.

The independent literal-content assessment asks whether the frozen coordinates
and H(A) fields retain a defensible addressed red/blue location interpretation
under those errors and crossings, rather than merely producing arbitrary logits
read by a head trained on another distribution. Erroneous but identifiable model
contents may qualify as estimates; they are not replaced by H(W) or renderer truth.
No new numeric threshold is introduced. Preserve a documented qualified, limited,
unresolved or failed disposition for each source/quantity, with its evidence and
permitted wording. Insufficient evidence leaves literal content unqualified; it
does not license post-fit training or a favorable subset. The subsequent heldout
technical run remains fixed and can test that disposition without retuning it.

Two export gates remain distinct from the manner gate: (1) literal-source-content
qualification from these independent diagnostics, and (2) exact bookkeeping/
fidelity of maps, fields, source identity, addresses and current/future timestamps
to the saved forward calls. A technical export can truthfully expose an
unqualified tensor or erroneous estimate with that status explicit. It cannot
promote H(A) to qualified constituent content by accurate wording alone. The
separate presentation-manner assessment still must distinguish form from extra
content, confidence and access. Passing either export gate does not pass manner;
passing native H(W) utility does not pass the H(A) literal-content gate. Until the
required source-content, bookkeeping and manner dependencies qualify, retain the
technical-only endpoint and do not release phenomenal-correspondence claims.

## Subsequent neutral reporting, only under a new registration

If estimated literal source content, bookkeeping and presentation manner qualify
under the separate gates above, separately register a
reporting version and an output-independent world/query schedule. It must export
native A_current maps/fields as estimates and A_next as forecasts with explicit
timestamps, not silently turn estimates into actual internal events. Matched
conditions are A-only, actual-W-only, and exactly joinable A+W with equal addresses,
times, precision, fields, metadata and token budget. Ordinary/analytic narrators
receive the equivalent exports from their sources and identical instructions.
When a narrator is given W as well as A, so is every comparator; when only A is
given, factual checking against W happens in the audit and does not become a
hidden informational advantage. Keep driver/mapping/donor labels audit-only.

Cross selective A changes at fixed W with W changes at fixed exported A, including
reachable donor mismatches and matched drivers. Source-aware factual auditors
score faithfulness to the explicitly exported source, current/next attribution
and unsupported claims. Separate blinded raters assess correspondence to a
predeclared consciousness-report reference structure, with the reference, theory,
desired differences, examples and phenomenal vocabulary withheld from narrators.
Retain technical-only, refused, inaccurate and incomplete reports. Fidelity,
P_access, P_presentation, model-presentation scope, attention relation and
unauditable experiential wording stay separate. Source-only fidelity does not
qualify semantics; reports need not assert private experience to count as faithful
content descriptions. Reference resemblance can fail despite precise reporting.

Faithful reports that independently resemble the reference and follow selective
source changes would support the restricted claim that this learned attention-
control state supplies reportable content form. Ordinary/analytic parity would
bound the explanation to shared predictive/state-estimation computation. Technical
prediction with no qualified manner, or qualified contents with absent report
correspondence, are concrete negative cruxes. Reporting-source causality alone
does not establish A as a quale-generating medium. Identical computation under an
ordinary name cannot be excluded through naming or instructions.

## Stronger separate Modeler/comparator extension

The [MSTC role audit](MSTC_REPORTING_ROLE_AUDIT_V1.md) distinguishes Controller
correction from refinement of a content-constructing Modeler. The frozen encoder
and H are not refined in this version; its errors and GRU feedback do not meet
that stronger comparison-medium role. Do not make this extra role a mandatory
gate for the narrower accurate-content/report-correspondence test.

A separate extension would introduce a trainable Modeler and a fixed common
representation coordinate system, compare predicted bound contents with newly
constructed contents, and use the content-specific discrepancy to update that
Modeler. Predeclare later independent-observation outcomes. Compare blocked,
reversed and world-yoked discrepancies, Controller-only correction and equally
informed ordinary predictive updates under a fixed sensory schedule, with exact
lesion/restoration and source provenance. This is a stronger causal refinement
claim, not phenomenal certification. It requires its own implementation, budget,
review and registration; no refinement is smuggled into this frozen-source run.

## Required next steps and preservation

Independently review this exact design before implementation or execution. The
review must check timing/leakage, attainable donor provenance, split namespaces,
parameter/loss arithmetic, equal-information comparators, Controller use, source
qualification dependency and reporting parity. Then implement and review the
trained module, collector, runner and reconstruction verifier against this exact
contract. Tiny toy fixtures can check software without loading development
artifacts. Freeze input hashes, seeds, source-indexed denominators, optimizer,
controls, split release/receipt procedure and failure policy in a committed
registration before the one new development run; append fitted/dev hashes only
at their predeclared immutable phase boundaries. Corrected full native numerical
revalidation and independent reconstruction remain necessary execution inputs.

Archive actual V/W, permitted observation histories, evaluator targets/mappings,
fitted weights/normalizers, source estimates, native fields/actions, all alternatives,
donor inputs and per-opportunity checks. Prefer one reconstructable tensor bundle,
compact JSON manifest/aggregate tables and first-run logs per encoder over many
duplicated files or report archives. Preserve first passes before repairs and
independent reconstruction. Commit and attempt authenticated `gh` publication at
each later authorized milestone; this design-only drafting performs no Git change.

This proposal moves from label qualification to a trained prediction-and-control
mechanism whose contents and failures can be inspected. Semantic qualification,
neutral correspondence, powered fresh-system confirmation and another-task or
architecture replication remain separate work. The full qualia-source goal is
unchanged and unachieved.
