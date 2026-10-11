# Experiment engine E1–E5: prospective corrections

2026-10-11. Source-only correction after preserving the first engine at `8bb7e99`
and first skeptical review at `98d6dd4`, as reported by the parent. No first
engine execution occurred before these corrections. The original engine
design.md and both experiment_review files remain unchanged. No checkpoints,
archives, world generation, imports/forward calls, fits, tests, Reporter or Git
operations were performed. Syntax compilation alone passed.

Corrected [engine](../../../scripts/attention_content_model_v1_experiment.py)
SHA-256: `09a5cbd3e6c7f3c37c65d2bdde35e1db40d65d35904c832d24d7c1533d887878`.
The [first review](../experiment_review/review.md) remains the record of the
original deficiencies. Disposition: E1–E5 have prospective source resolutions,
pending independent review and first software fixtures; no empirical outcome or
semantic qualification is asserted.

E1: caller-owned fit partial state now retains `source_states[A|ordinary]` with
both not-yet-started and active weights, optimizer and completed-update count,
refreshing after each successful step. Before handoff the completed source's
weights and terminal optimizer are retained, so an ordinary failure cannot erase
A. Original logs/initial weights/exception behavior remain. Shared control checks,
emitted partial per-world outputs and provenance, plus every completed donor
prefix, are attached to evaluation partial state before subsequent risky work.

E2: analytic nonselected-state isolation covers survivors and revealed-symbol
masks together. The selected patch must exactly transfer both masks from the
reached donor; restoration covers both. Donor records now add
`donor_revealed_symbols_at_patched_address` [96,3,3], matching the existing
survivors [96,3,6]. Analytic terminal state retains survivors, revealed symbols,
revealed-through14 and update indices0..13. W15 remains archive-only.

E3: prediction manifests are recorded before each actual call, including candidate
address and source h/posterior hashes, source identity, permitted input hashes,
timestamps/history, returned current/next hashes and post-call input/state hashes.
The existing prediction inventory remains 45 A calls and 30 calls for each rival.
New `metadata.source_update_visibility[s]` records fourteen actual primary feedback
calls and seventy-two actual donor replay calls per source, with completion/command/
V hashes, exact indices, pre/post state hashes and reveal masks. Donor records
also identify their schedule and originating decision. Training retains its
explicit first-update permission recipe and 300 resets, without pretending to
archive or independently prove every intermediate fitting state.

E4: named hard checks now cover exact initial V, current nonselected copies,
next noncandidate inheritance, observed histories/timestamps, the last revealed
snapshot, and prediction input/state nonmutation. Each patch additionally checks
selected-state transfer, nonselected decoder outputs and intended next inheritance.
Address-local decoder probes are a separate diagnostic inventory: baseline and
both patched states at the three query times, three opaque symbols each, flattened
over four known addresses. This is 27 direct decoder batches per source, not
additional typed prediction calls, worlds, policy actions or primary opportunities.
Probe inputs/state/output hashes are in `metadata.source_decoder_visibility[s]`;
the verifier can reconstruct outputs from retained source states and fitted bytes
without archiving duplicate full probe tensors.

The shared `actual_compaction_reconstruction_exact` check is [96,61]: initial W,
45 forks, fifteen common transitions. It uses the existing compact recipe. The
runner must require that named gate and exact equality to compact episode's
`actual_reconstruction_exact`, plus status `exact_reconstruction_bound`, after
archiving outputs and before any completed receipt. Failed compaction still
retains original raw W/forks. All original gates remain, with the following
additions to the runner's exact shape registry:

| Scope | Name | Shape |
|---|---|---|
| Shared | actual_compaction_reconstruction_exact | [96,61] |
| Every source | initial_current_exact_v | [96,1] |
| Every source | current_nonselected_copy_exact | [96,14] |
| Every source | next_noncandidate_inheritance_exact | [96,15] |
| Every source | observation_history_exact | [96,15] |
| Every source | last_revealed_snapshot_exact | [96,15] |
| Every source | prediction_inputs_unchanged_exact | [96,15] |
| Every source | prediction_state_unchanged_exact | [96,15] |
| Every source, each schedule | patch_{schedule}_selected_state_transfer_exact | [96,3] |
| Every source, each schedule | patch_{schedule}_nonselected_decoder_exact | [96,3] |
| Every source, each schedule | patch_{schedule}_next_inheritance_exact | [96,3] |

E5: the same saved 256 paired whole-world bootstrap draws now cover selected,
full-state and inherited next MSE; selected/full field L1; selected/full cell and
coarse disagreements; role-specific cell agreements/disagreements; and pooled,
red and blue policy targeting. Current learned-state reductions exclude exact
initial t0; all prospective/inherited opportunities remain unconditional. The
inherited denominator is fifteen×three symbols×three noncandidate items/world.

Every predetermined donor pair retains current selected H fields and next policy
candidate H fields, full-current/full-next field change magnitudes by address,
exact numerical field-change flags, entropy/max/margin/expected-column changes,
full cell/coarse disagreement and normalized map error against fixed actual W,
and selected literal disagreements/agreements against actual fields and truth.
`metadata.donor_summaries[s][schedule]` tabulates all 288 pairs per schedule/source;
no changed-only subset or support cutoff is added. Both schedules retain their
coincident-address flags even when numerical donor states coincide. Causal nulls,
wrong estimates and ordinary/analytic parity remain measurements, not hard failures.

Architecture, optimizer/300-update budget, split counts, original typed prediction/
update opportunities, donor replay counts, scalar-policy identity, native gates
and reporting/semantic dependencies are unchanged. The expanded evidence is
instrumentation and diagnostics; it neither changes the Modeler-refinement scope
nor qualifies source manner. Independent complete-source review, software first
pass and committed phase registration remain required before an actual study.
