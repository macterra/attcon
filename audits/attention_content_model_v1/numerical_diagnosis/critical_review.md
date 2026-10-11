# First fit reconstruction: independent critical diagnosis

2026-10-11. Separate same-AI-family task with inherited context, not human or
independent-family endorsement. Source and retained JSON/text evidence only;
no tensor/model/checkpoint/trace-archive loads, imports, execution, fits, source
edits, Git operations or future phase release. Decompressed only the published
first reconstruction JSON, checking its lossless transport hashes.

**The retained failure supports a narrow auxiliary-reference correction and
subsequent full reconstruction of the same fitted bytes. It does not establish
a passing fit reconstruction or approve an unreviewed corrected execution.**
The first report remains failed and incomplete:27616 emitted checks pass, then
one `verification_exception` terminates the planned inventory. No original
scientific criterion, tolerance or production observation guard should change.

## What the first evidence says

The [first report manifest](../fit_reconstruction/first_report_manifest.json)
binds raw SHA256
`75559ba487323ab1a59b0f6ce3ebec23876895b198c59ece6755812f567cb37e`
and compressed SHA256
`7901896463b8a7619f49210f52af5f614b8d809a9eec4944d10fda602a6e52c5`.
Both match inspected retained JSON transport. The CLI record/log have terminal
exit1 and failed status. The report used registered verifier
`11b821bea207f4ef8250fa589148777df9d600bb3dffd01ef600cf4f78a423ab`,
registration commit `304a052e537115e04b641e4fca3b3c3136fbd739` and fit artifact
commit `28406d63c053b810c0fd19348cad84b147f03690`.

The traceback reaches `verify_production_replay` line1020, the first A t0 locality
decoy call, from encoder1301 fit diagnostics. Its four immediately preceding
state/selected-next/copied-current/inherited-next exact checks pass. The exception
is the core's `ValueError: initial state is exactly observed V without a command`.
Later replay, sources and encoders were not completed.27616 is the first emitted
prefix, not a promised full inventory or evidence that all six fits are validated.
The fit receipt records completed first execution, unchanged registered artifact
hashes and no retry; this review did not deserialize those artifacts or inspect
optimizer trajectories.

## Cause and legitimate correction

[ObservationBatch.validate](../../../scripts/attention_content_model_v1.py)
correctly requires, at t0, no previous command and exact `last_revealed_w == v`.
The initial observed state is V. The verifier constructs that valid observation,
then creates `decoy_v` by changing only addresses outside the candidate/previous
selected decoder set. At t0 there is no previous selected address. It passes
`replace(obs, v=decoy_v)` while preserving the original last-revealed snapshot.
Those irrelevant-address changes make the initial pair inconsistent. The core
therefore rejects before the intended decoder-isolation comparison.

The minimal reference change is to construct a valid auxiliary observation:
change V to decoy_v, and set last_revealed_w to decoy_v only when t==0; for t>0
retain the original last-revealed snapshot. Keep time, commands, observed mask,
candidate address, relevant V/h/posterior values, fitted weights and normalizer
unchanged. Continue perturbing the same irrelevant addressed state and checking
the original selected-next equality at every t and selected-current equality
when t>0. The full initial current copy will differ at deliberately perturbed
nonselected addresses; it is not the selected-output isolation assertion.

This decoy remains an explicitly synthetic input-locality probe, not an additional
world, reached recipient state, actual episode or fit observation. It must not
alter the archived baseline manifests or run data. The correction fixes its
invalid test input without granting the model concealed targets or relaxing the
production W0=V invariant. Do not skip t0, catch this error and call it a pass,
remove the decoy checks, modify the core guard or change continuous tolerances.

Before full corrected reconstruction, review exact source diff and bounded
fixtures proving: valid t0 paired decoys reach unchanged selected comparisons;
t>0 preserves the original feedback snapshot; the original inconsistent t0 pair
still fails. Cover learned and analytic APIs so correcting one branch does not
hide a later branch failure. Preserve the correction source, fixtures and first
outcomes separately. This review approves that correction direction only.

## Minimum explicit versioned acceptance amendment

The original registration binds the original verifier inside its fourteen-file
closure, and both source files must remain unchanged. A derived reference program
cannot simply replace that file: the existing closure digest checks would reject
it. The frozen heldout runner also requires the dev report's executed verifier
hash to equal the registered original hash. A corrected report cannot satisfy
that check by falsely writing the original hash, editing registration or rewriting
the first failed report.

Append one committed, narrowly scoped reference/acceptance amendment that binds:

1. The exact original registration path/hash/commit, complete original science
   closure and complete-approval record, accepted native inputs, fit receipt,
   original fit artifact hash inventory and preservation commit.
2. Original verifier path/hash, first failed CLI/log/report/manifest hashes and
   failed/incomplete disposition; retain all27616 emitted checks as the original
   prefix and retain the traceback without relabeling.
3. A separately named corrected verifier path/hash/commit, exact diff explaining
   valid-initial-decoy construction, independent static approval, bounded fixture
   sources/first outcomes, and any separately versioned acceptance implementation.
4. Separate fields for the original registered verifier SHA and actual executed
   corrected verifier SHA; an exact allowed pair, never an arbitrary accepted hash
   or generic latest-verifier exception. The corrected reference must independently
   bind original registration/source/current/committed bytes and additionally bind
   its own approved correction bytes before loading artifacts.
5. One exclusive separately labeled full revalidation of the unchanged fit
   artifacts: all originally planned comparison/check call sites across every
   encoder/source/donor/summary, same gates/tolerances and opportunity inventories,
   plus explicit correction provenance checks. Compare the preserved first prefix
   by names/outcomes and unchanged numeric evidence; disclose any necessary
   provenance-check change rather than silently changing its expected hash.

The amendment changes reference provenance acceptance only. It authorizes no
new fit, optimizer update, generation, selected checkpoint, donor choice, metric,
semantic gate or science closure. Later phase/assessment records must reference
the unchanged original registration and fit bytes plus this amendment.

For corrected future-dev reports, an explicitly reviewed versioned acceptance
entry point must recognize this exact original/corrected pair while retaining
all other report status/inventory/registration/fit/ancestry/disposition checks.
The original runner may remain frozen; any separate compatibility implementation
must have its own immutable review/binding and disclose the executed entry point.
A generic substitution, monkeypatch bypass, report hash relabeling or rerun under
a silently rewritten registration is not an acceptable amendment. No compatibility
code or future phase is approved here; exact proposed source/manifest must be
reviewed before use. This is a narrow implementation/provenance dependency, not a
new performance gate or human approval requirement.

## Limits and verdict

Source and retained traceback explain the observed stop without suggesting a
wrong fitted map or a production observation violation. They do not guarantee
that all unexecuted comparisons will pass, prove the full optimizer trajectory
or qualify literal content/manner. Further reference defects or actual numerical
mismatches remain possible and must retain their own first evidence. A corrected
failure remains failed; a passing full revalidation can establish only its bounded
reconstruction claim. No existing first failure is erased by it.

Proceed to prepare and review the narrow correction plus explicit acceptance
amendment against unchanged original artifacts. This review authorizes no runtime,
future phase release or Reporter. The source/report scientific goal remains open.
