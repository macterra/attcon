# Attention-content model v1: source permissions and phase records

2026-10-11. Draft executable schema implementing the approved design. This is
not a release receipt, a passing reconstruction or semantic qualification. Freeze
its exact bytes with the complete reviewed source closure before registration.

At each scored decision t=0..14, a source prediction receives only immutable
V[B,4,24,4,4], the last revealed full map snapshot W[max(0,t−1)], the last issued
address/symbol when t>0, an all-observed Boolean mask, its own pre-reveal recurrent
state and the scheduled prospective address. Source/state identity and timestamps
are audit metadata. They cannot alter numerical predictions. No source receives
pixels, world identifiers, physical cells, command-to-bandwidth assignments,
concealed current W(t), fork outcomes or driver labels.

After scoring t>0, feedback reveals W(t) and command t−1 exactly once. Only the
changed address's V, completed map and command enter a recurrent update. At t=0
the initial W=V is observed and there is no update. Actual W15 is archived after
the fifteen transitions; it introduces no scored decision15 or extra recurrent
update. Target tensors enter fitting losses and evaluator calculations only.

All maps and learned/native weights are finite CPU float32. Addresses/symbols are
CPU long tensors; visibility masks are CPU Boolean tensors. Map normalizer
coordinates are fitted from fit V only. The native H retains its separately frozen
normalizer. Current selected-map records index decisions1..14; prospective records
index decisions0..14 and all three symbols, in symbol order0/1/2. Actual W, estimated
A and the ordinary/analytic estimates keep distinct source identities.

The fitting manifest contains both learned sources, exact initialization seeds,
197,328 parameters each, 300 update records per source, current/next/combined loss,
gradient norms, immutable fit coordinates, initial/final checkpoints and terminal
optimizer states. Every update resets addressed state. The fixed prediction/
feedback observation recipe is hash-bound for all 300 repetitions, with its scope
and reconstruction counts explicit. Recorded terminal states/logs do not prove an
unobserved optimizer trajectory; independent trajectory proof requires replay.

Evaluation retains actual per-call input hashes and timing masks, selected maps,
H logits/fields, pre-reveal states, source-specific outcomes and shared actual
fork/action records. Full current/next states can be compactly reconstructed from
the last revealed snapshot and the explicitly indexed selected predictions. Actual
states can be compactly reconstructed from saved V/operator-family maps and the
actual command/mapping selection ledger. Exact reconstruction checks precede
compaction. Evaluation errors, unchanged interventions, ties and ordinary parity
are descriptive outcomes, never replaced by hard software gates.

Donor prefixes have t executor transitions and t−1 permitted feedback updates at
t=12,13,14. Preserve command/completion hashes and update lists. Patch only the
specified reached h/posterior, without revealing a new recipient observation;
restore source state and map/field/policy outputs exactly before feedback. Keep
current and prospective donor schedules even when they coincide. Confidence-only
choices and same-weight aliases remain labelled duplicates.

Phase release requires an immutable input registration, then a committed fit
receipt before dev, then a committed dev receipt and independent reconstruction/
semantic assessment before heldout. Every receipt binds the same registration,
source closure and input archives. Dev/heldout use the fit checkpoint bytes and
normalizer without optimization. A semantic negative permits the fixed technical
heldout run when hard invariants pass; it cannot release phenomenal reporting.

The dev acceptance JSON uses schema `attention_content_model_v1_dev_acceptance`
and binds `dev_receipt_sha256`, `fitted_artifact_sha256`,
`reconstruction_report_path` and `reconstruction_report_sha256`. That independent
report must be committed with the acceptance, identify the registered verifier
bytes, pass its complete dev check inventory, and use the accepted registration
and fit commits. Its dev artifact commit precedes the final acceptance commit.
An assertion of `hard_invariants_pass` without that report cannot release heldout.
The assessment additionally records `source_semantic_dispositions` for A,
ordinary and analytic: each has `literal_content_disposition` equal to
`qualified` or `unqualified` and Boolean `presentation_manner_qualified`.
`reporting_released` remains false. Negative semantic dispositions are retained
and permit the fixed technical heldout phase when hard invariants pass.

The exact engine gate inventory and opportunity shapes are recorded in
`audits/attention_content_model_v1/experiment_software/metadata.json`, with
the same names/shapes enforced by the phase runner and independently reconstructed
by the verifier. Per encoder, each source has fourteen primary feedback updates,
seventy-two donor replay updates and twenty-seven separately labelled decoder
probes. Prediction-call counts are forty-five for A (including fifteen same-weight
alias duplicates) and thirty for each rival; they do not multiply world counts.
Actual compaction retains sixty-one per-world reconstruction checks, including
initial W0. Failed reconstruction retains the original actual arrays.

The complete source review must freeze named hard invariants, exact source-indexed
shapes/counts and reconstruction-verifier inputs. Generic Boolean success flags,
caller-supplied timestamps or a split-name argument alone cannot satisfy this
schema. First failures retain their original traceback and available partial
outputs; a retention error is recorded separately. No automatic retry, additional
fit, alternate checkpoint or undeclared phase release is authorized.
