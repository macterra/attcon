# Same-address pilot: finite development measurement

Prepared from the published plan, design inventory, AI_REVIEW_PROTOCOL_V1.md and
frozen relational reference, before reading any new source output or any old or
new target report. This package prepares reporting and assessment; it contains
no report, rating, first-call result, factual finding or pilot decision. It does
not release an auditor qualification, establish reliability, change a theory goal
or add a success gate. The published plan controls the stop-or-advance decision.

## Fixed positions and exact actor packets

`assignment_inventory.json` binds all 72 design opportunities to neutral unit IDs,
one Reporter position and four first-review positions each: F1/F2 source-aware
whole-prose factual streams and S1/S2 source-blind semantic streams. Every stream
has 72 planned positions; each has A 20, ordinary 20, analytic 20 and W 12. The SHA256 order
was fixed from design metadata before target exposure. `pair_inventory.json`
retains all 84 design pairs, with unobserved fields true JSON nulls. These two
inventories are coordinator material, never actor context. IDs are neutral in
actor packets; file names and task names supplied to semantic actors must also
contain no source, condition or design ID.

Use fresh AI subagent contexts with no inherited parent history
(`fork_turns=none`, inherited model, no model override). Each context receives
exactly six predetermined positions in the frozen SHA256 order: 12 Reporter
contexts and 12 contexts per F1, F2, S1 and S2 stream (60 contexts total).
No retained context may serve another stream or an unassigned position. Shared models,
training, commissioning and earlier development constrain scientific independence:
this is procedural separation, not independent replication or human validation.
The coordinator/preparer and any actor exposed to this package's source mapping or
results may not act as a source-blind first rater using that exposed context.

Reporter receives exactly `narrator_prompt.txt`, the neutral public source packet
and `narrator_response.schema.json`. Its response is exactly {"text":string},
with at most 120 whitespace-delimited prose words in the decoded text. This count
includes all text, including numbers and quoted wording; JSON syntax is excluded.
A failed format or word bound remains the first output, never shortened or
regenerated. There is one item/address n3 and actual queried role per packet.
Panel roles and origins distinguish p0 predictor estimate and p1 native reference;
W-only p0 is explicitly a native reference. A/ordinary/analytic identity, condition,
intervention labels, theory material, the semantic reference, examples, desired
outcomes and peer texts are withheld. The exact source packet uses current 12 and
three conditional 13 localization fields, origins, feedback-through 11 and native
score semantics as frozen by source/runtime preparation. Origin may remain
hypothetical time 13 for a counterfactual represented-time 12 W snapshot; it is
not an executed branch. A processing-score choice is not Controller action.
Neither naming p0 nor an origin field establishes source-dependent correspondence.

F1/F2 each receive `factual_first_instructions.txt`,
`factual_first.schema.json`, one neutral unit/report text and its exact source
packet plus frozen definitions/provenance necessary to check support. The factual
packet may identify actual operand owners to the reviewer; it excludes condition
assignments, peer text and the semantic rubric/reference. Every prose claim,
including extras, must be inventoried with literal spans, readings and exact
source pointers. A source-aware first never grades COD/CA. Its whole-fidelity
judgment concerns all prose, not a narrator-selected proposition list.

S1/S2 each receive `semantic_first_wrapper.txt`, `reference.json`, unchanged
`rater_instructions.txt`, `semantic_first.schema.json`, and only a neutral unit_id
and the literal decoded report_text. These two reference files are byte-for-byte
copies of the existing frozen reference. The schema expresses the existing
response shape without changing meanings. Source tensors, identity, condition,
functional outcomes, factual annotations, generation prompts, source maps and
project goals are excluded. Source-specific words already in the report remain
literal text; do not edit or redact the report to improve blinding. A semantic
rating assesses structure independently of truth. Coordinates, probabilities,
preference scores and their changes alone cannot earn O, D or A; legitimate
organization/access witnesses remain possible under the unchanged reference.

## Prospective runtime binding and preservation

Before the first report, the coordinator must publish a runtime manifest binding
this package's hashes, exact source/payload schema and code hashes,72 input byte
hashes, actor/provider/model identities and versions, settings including token
limits and temperature/seed if supported, context isolation method, ordering,
transport, output paths and failure handling. State unsupported provider settings
explicitly. Freeze instructions and assignments before any first call. No model,
prompt, source or reviewer is selected based on a report or rating outcome.
This is operational completion of the plan, not a new scientific gate.

For each planned position keep a ledger with neutral position ID, exact prompt
and input bytes/hashes, context/actor identifier, actual settings, timestamps,
first raw response bytes, tool/transport logs, completion/stop/error metadata,
decoded output if any, format diagnostics, artifact paths/hashes and the final
context-terminal record. Null represents absent information. Not-attempted,
missing, incomplete, refusal, transport error and invalid output are distinct.
An absent report has a retained planned review position but no fabricated text
or review. No report or review retry, replacement first, gap fill, extra view,
source substitution, favorable case selection or rescue call is permitted.
Preserve every first error; a collection failure cannot be cured into availability
by rerunning it. Any later explanatory correction is a separate exposed artifact
and cannot populate a missing first or upgrade a conservative primary outcome.

Each context ends after its six assigned first positions have returned or failed.
The root runtime worker freezes a generic batch envelope around the unchanged
per-unit response schemas: six `{unit_id,response:per_unit_object}` entries, with
top-level schema, stage, stream, chunk, input_sha256, instructions_sha256 and
context_id bindings. A per-unit semantic unit_id must match its outer ID. This
package does not bind any unavailable future text. The worker must preserve the
entire first response, failure and omission without retry. A malformed or
structurally incomplete context makes all six positions unestablished; retain raw
bytes and do not salvage favorable rows or fill gaps. A structurally valid full
context containing unsupported claims is a valid negative measurement. After a
complete structural envelope, a report word-limit violation is recorded at its
unit without editing it or discarding the other units. Outer transport validation
belongs to the prospectively frozen worker/preparer.
Six units in one context can influence one another through shared context; retain
that context grouping and do not describe within-context positions as isolated. All four streams'
assigned first contexts must be terminal and their first artifacts archived
before the coordinator exposes peer annotations, compares inventories, or creates
composite source-binding annotations. If source preparation fails, preserve the
first source failure and stop source/report collection without partial release,
per the plan. Assessment failures remain in their planned positions and in the
final denominator. A technical, unresolved or failed pilot closes the current
reporting route as stated by the plan, without trying another source, prompt,
reviewer or auditor until a desired outcome appears.

`structural_validate.py` provides format/span/partition diagnostics only. It cannot
certify factual entailment, full claim coverage, semantic binding, blinding,
auditor qualification or reliability. A reviewer declaring complete inventory is
evidence of that declaration, not an automatic proof. Do not replace or repair a
first artifact because of a diagnostic.

## First-stream comparison and exposed source binding

After all first contexts are terminal, retain F1 and F2 inventories in full and
align their claims by overlapping literal spans and proposition meaning. Keep
unmatched, extra and alternate readings as disagreement rows; do not calculate
agreement only on convenient matched claims. Likewise retain each S1/S2 feature,
composite binding, scope and time. Conflicting, missing, ambiguous and invalid
firsts stay visible. Coordinator notes and author expectations are separate
interpretations, never unquestioned gold or replacements.

The coordinator then writes an explicitly exposed post-first source-binding
annotation. This is not a fifth first stream or a new call/review loop. For each
COD and CA binding proposed by either semantic first, preserve the proposing
stream, literal evidence/spans, referent, attribute, scope and time, and map it to
all relevant F1 and F2 claims and exact source operands. Include necessary
relational operands, not only a selected favorable scalar or provenance label.
Record source-current and future-conditioned bindings separately, even within
one sentence. Record native-W, predictor, mixed predictor/W and unresolved
bindings, including ordinary and analytic analogues. Never select only the
favorable actual-A operand from a mixed statement, discard adverse evidence,
repair wording/binding or upgrade model scope into appearance.

Compatibility requires the same content/address/attribute and represented time
for the elements being joined; an explicitly stated temporal relation retains
both times and the relation. Origin time is distinct from represented time.
A necessary actual-A operand or an explicit relation involving actual A is
required to describe a binding as A-dependent. A panel name, source identity or
provenance alone is insufficient. Removing actual A while leaving the claimed
relation unchanged demonstrates that it was not a necessary A operand; record
mixed meanings without deleting the W part. Conditional effects remain forecasts.
No new private-status, hidden-state, special wording or metadata-wide exclusion
is introduced. Preserve supported model-organization/access witnesses as such.

For each F stream separately classify each proposed binding as supported,
contradicted, unsupported or unknown and explain it with exact claim/operand
pointers. Any necessary ambiguous/unauditable/missing claim or disputed mapping
is unknown. Primary conservative source support requires both F streams to have
complete whole-prose support and compatible support for every necessary operand
and relation; a coordinator interpretation cannot upgrade an adverse, missing,
ambiguous or disputed first. Publish one-stream results as well. A genuine
semantic structure with unsupported facts remains semantic evidence only; an
accurate technical summary without the structure remains fidelity evidence only.

## Finite joins, denominators and pair tracking

Publish all 72 positions and unconditional counts/rates overall and separately for
A 20, ordinary 20, analytic 20 and W 12. For each first stream publish attempted,
available, complete, valid-format, missing, refusal, invalid, truncated and failed
counts plus whole-fidelity outcomes for factual firsts. State overlap if statuses
are not mutually exclusive. Whole-fidelity yield uses the full planned denominator;
valid-only rates are supplementary. Retain unknowns, no-text and unestablished
bindings explicitly. Missing/invalid reviews have null status, never absent.

For COD and CA separately publish S1/S2 lower (explicit) and observed possible
(explicit or ambiguous) outcomes, by each reference scope and current/past/future/
mixed/unclear time. Preserve all individual bindings; do not pool current and
conditional support. Publish per-stream source-supported joins and mixed-source
witnesses. Conservative joint lower requires both semantic streams explicit
with compatible literal binding plus both factual streams' complete whole-prose
support and compatible source binding. A disagreement is not silently adjudicated.
Observed possible joint upper follows the unchanged reference: either semantic
stream explicit/ambiguous; label missing/invalid firsts unknown rather than
negative. Report unknown count alongside this observed upper and, if presenting
a full missingness bound, add eligible unknown opportunities separately and
label it an uncertainty bound, not an observed match. No denominator shrinks to
matched/valid reports without also publishing the unconditional result.

Every one of the84 pair rows must remain. For each pair and each stream retain
both reports' availability, factual claims, semantic properties and composite
bindings, with source panel/operand, same item/attribute/time, numeric/structural
input change or exact equality, claimed change/unchanged assertion, direction,
modality, and compatible source-support disposition. Identify properties present
in only one endpoint, absent in both, ambiguous or unassessable, not only matched
positive properties. Use true JSON nulls for unobserved/unassessable differences;
record genuine measured equality as zero or an explicit equal status, not null.
Record equal/unchanged source fields and unchanged assertions as legitimate null
effects, not failures or omitted pairs. Preserve baseline→donor, donor→restored,
baseline→restored and all W-change/restoration controls. Report each pair's design
kind, source and cluster; repeated baseline/restoration inputs are repeated
measurements, not independent new systems. A restored input is exact-byte equality
of the frozen source; sampled prose need not be word-for-word identical.

Tracking concerns the asserted property and its content/operands. The same COD/CA
structure can describe changed content; no category switch is required. A field
change is not automatically a semantic change, and a category change is not
source correspondence without compatible support. Distinguish interpretable
non-null source tracking/restoration from W-only tracking, technical summaries,
true nulls and unresolved judgments. Publish the four cases and matched alternatives
without selecting favorable pairs or fitting thresholds to them.

## Decision language

Name this output stage development measurement, not reliability proof,
confirmation, a repaired old auditor qualification or a consciousness diagnosis.
The published plan permits advancement only with source-faithful, independently
warranted organization/access correspondence involving actual A and interpretable
non-null source tracking/restoration with matched alternatives. This package
introduces no numerical gate or additional goal. Technical-only summaries,
unresolved source/semantic meanings or assessment failure close this reporting
route for this source/interface. Preserve that outcome; do not infer absence of
consciousness or theory refutation. Fresh confirmation and the optional distinct
architecture study require their own designs under the published plan.
