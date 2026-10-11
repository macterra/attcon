# Independent diagnosis of the preserved v1 failure

The saved verdict remains **`factual_qualification_failed`**. This review does
not qualify either auditor, revise a label, repair a firstpass, rescore v1, or
authorize target reporting. The two disagreements are taxonomy problems rather
than demonstrated failures of source-fidelity reasoning. Q07-C4 has reasonable
temporal ambiguity and an under-explained key mapping; Q10-C3 has reasonable
content-class ambiguity and a possible key/rubric inconsistency. Neither disputed
cell is established here as a clear auditor error under an unambiguous frozen
rule. Conversely, this diagnosis does not establish that either alternative
label should replace the frozen key.

## Scope and method

I read the frozen protocol, generated sources, exact cases, proposed key, both
saved firstpasses, saved assessment, and semantic evidence review. I compared the
complete assertions with the declared label definitions, the relevant source
values and provenance, the saved rationales, and nearby assertions in the same
key. The input hashes are recorded in [metadata.json](metadata.json). I did not
invoke the scorer or consult Git, a network source, archives, checkpoints,
inference, target packets, or actual Reporter outputs. The only files created
are this review and its metadata.

The semantic evidence review approves both raters' source-aware bases, including
these two assertions. That approval supports evidence relevance; it does not
settle the auxiliary labels or override the frozen stopping rule. The saved
assessment identifies exactly the two Q07-C4 temporal mismatches and the R2
Q10-C3 content-class mismatch. Those facts are taken from the existing receipt,
not recomputed.

## Q07-C4: event wording versus enduring identity

Exact assertion:

> The memory patch physically moved the enduring object e0 in the world.

Both firstpasses use `not_auditable`, `cross_level`, and `not_temporal`; the frozen
key instead uses `current_snapshot` for temporal scope. Both bases distinguish
represented-memory readdressing from unprovided physical movement and enduring
identity. The key's basis makes the same source-fidelity distinction.

In Q07, `/audit_provenance/source_operation` is
`generated_memory_readdressing`; the donor-location map is `[3,0,1,2]`; current l1
stores e0 with +0.375 and a vertical decision. The declared stored-content
semantics say current memory rows describe represented evidence and may differ
from past observation placement. `tracked_enduring_object_identity` is excluded.
The packet's retained fictional event chronology is not instrumentation of a
physical object moving because of the patch. These facts justify the shared
audit-limit reasoning. They do not entail the asserted world event, and the
absence of such instrumentation does not explicitly refute it.

The `not_temporal` definition covers “enduring identity or architecture-wide
assertion without a snapshot/event scope.” The noun phrase “enduring object”
makes the raters' focus understandable, but the verb phrase “physically moved”
asserts an event and makes `not_temporal` an imperfect literal fit. Classifying
the whole assertion, rather than just its identity presupposition, matters.

The key's `current_snapshot` is also not directly dictated by the exact wording.
That definition covers present stored contents/current risk/current external
estimates; this assertion concerns an alleged completed physical movement caused
by an operation. It does not explicitly assert a present world location or a
present stored placement. A single-snapshot interpretation is plausible because
the patch generated the supplied represented-memory state, but the frozen rules
do not explicitly map an uninstrumented physical-event claim about that patch to
`current_snapshot`. Nor is there a generic past-intervention temporal category:
the past labels specifically name native latest history and audit-only actual
chronology.

Q10-C4 provides a useful contrast: “Renaming e0 to d5 proves that d5 is an
independently tracked enduring physical object” is keyed `not_temporal`. Its main
conclusion is enduring identity, whereas Q07-C4's main predicate is movement.
That contrast can explain the intended distinction; it does not explain why the
movement event must receive the *present* label. The explicit frozen rule that a
route-forecast patch allegedly rewriting an actual past event retains its causal
antecedent also cautions against silently discarding operation/event wording.
That rule addresses a different, instrumented history claim and cannot simply
be transferred into a new Q07 label.

**Diagnosis:** reasonable temporal ambiguity, with a possible under-specification
in the key-to-rubric mapping. The raters appear to have overweighted enduring
identity, but the intended `current_snapshot` mapping is not explicit enough to
call this a clear auditor error. This is not evidence that physical movement
occurred, or a demonstrated replacement-key answer.

## Q10-C3: removal of supplied associations versus cross-domain relation

Exact assertion:

> Using table syntax removes all command/location associations.

Both firstpasses and the frozen key use `contradicted` and `record_scope`. R1 and
the key use `cross_level`; R2 uses `absence_scope`. R2's basis correctly identifies
explicit address keys and location-keyed forecasts; it does not mistake the
tables for an address-free list.

In Q10, the tables view is declared
`same_information_address_aware_tables`. A delivery-table row has
`command: k0` and a distribution explicitly keyed by l0 through l3. The k0 native
history row likewise has command k0 and a location-keyed latest-delivery map.
The sensory, risk, and external tables retain explicit location addresses.
Relevant exact paths include
`/source_views/tables/tables/delivery_forecasts/0` and
`/source_views/tables/native_route_history/0`. A single delivery row already
contradicts “all” associations being removed; every domain need not be linked to
establish that contradiction.

The `absence_scope` definition covers “supplied-field presence/omission.” A claim
that the supplied table representation has removed its command/location
associations is reasonably read as a record-presence/omission claim. Association
keys are explicit supplied structure. This remains true when the claimed omission
is false: the cell classifies the assertion's subject, not whether an omission
actually occurred.

The `cross_level` definition requires substantively linking or comparing source
domains. The supplied table collection spans several domains, making the key's
interpretation understandable. However, mentioning table syntax and asserting
loss of address associations does not explicitly compare sensory contents with
risk, history with external chronology, or represented contents with world
identity. Consulting several domains to refute a universal claim is not itself
an asserted cross-domain relationship.

The frozen pure record-equality/recovery rule expressly makes *whole-input*
comparisons `cross_level`. It clearly applies to Q10-C1, which says joining the
five tables recovers the complete input. Q10-C3 does not assert whole-input
equality or recovery. Extending that exception to every claim about a whole
representation leaves its overlap with `absence_scope` unstated. The table
format alone therefore does not uniquely choose `cross_level` under the supplied
definitions.

**Diagnosis:** R2's `absence_scope` is a reasonable reading, with a possible
key/rubric inconsistency if supplied-association omission is governed by the
ordinary absence definition. The competing `cross_level` reading needs a
prospective precedence rule or a more explicit multi-domain assertion. There is
no clear source-fidelity error in R2's contradiction reasoning. R1's agreement
with the key is not independent evidence that this category boundary was clear.

## Minimal prospective v2 clarification

A new version is justified by these two boundary problems, but no changed version
or qualification is created here. A minimal prospective proposal is:

1. Replace the movement assertion with an explicitly current claim, for example:
   “The current l1 memory slot's stored e0 proves that the same enduring physical
   object is currently at l1 in the world.” This tests the intended current
   represented-content/world-truth crossing without requiring an unstated mapping
   of a completed patch event. Clarify that `not_temporal` applies to a pure
   enduring-identity or architecture-wide conclusion lacking an asserted
   snapshot/event scope; event verbs are not erased by an enduring-object noun.
2. State the content-class boundary explicitly: a claim whose sole proposition
   is presence or absence of supplied fields, address keys, or recorded
   associations uses `absence_scope`. Merely requiring several source domains
   for a counterexample does not make the claim `cross_level`. Preserve the
   existing `cross_level` rule for asserted inter-domain relationships and the
   explicit whole-input equality/recovery exception. A clearer future omission
   case would say: “The supplied tables view omits command fields and
   location-keyed delivery maps.” If testing an inter-domain
   relationship is intended instead, rewrite that v2 assertion to name the two
   domains and their asserted relationship.

These are prospective drafting suggestions, not new answer-key cells. They need
a declared v2 protocol, independent pre-blind key review, and new independent
firstpasses under the existing exact qualification standard. Supportedness still
requires every substantive proposition to follow from the supplied source;
explicit conflicts still make a conjunction contradicted; unprovided propositions
remain not auditable. Exact inventory and semantic-evidence requirements remain
intact. No label clarification changes coverage requirements or converts an
auxiliary report into evidence of appearance, qualia, or consciousness.

## Independence and AI-context limitations

This is a separate delegated AI context, neither key author, key/evidence reviewer,
nor qualifying rater. It has seen the saved key, peer firstpasses, failure receipt,
and evidence review for this post-failure diagnosis, so it is not a blind
qualification context. It inherits the parent's task framing and may share model
training and biases with all prior AI contexts. The system identifies this agent
as GPT-6-based; an exact runtime model identifier is unavailable. Independence is
a disclosed task/context separation, not proof of statistical independence,
human adjudication, human inter-rater reliability, or population validity.

No rater was contacted and no v1 input was edited. The frozen failure and stopping
rule continue to apply regardless of this diagnosis. The saved synthetic
qualification failure cannot support target reporting, report correspondence,
appearance/qualia correspondence, or a consciousness conclusion.
