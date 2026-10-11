# Experiment engine: first static integration review

2026-10-11. Separate skeptical review task within the same AI family, with
inherited project/hypothesis context; no human or independent-family endorsement.
Reviewed engine SHA-256 `5c50f7d7107c4e5cf6571311abc3963a9ffefeb6eca66218cba58389f041e7ef`
and [engine design](../experiment_software/design.md) SHA-256
`6fb3c7a87f5d5839de981e3211748b75aba9f6fb61db292a39515482e746f7da`
against approved amended [contract](../../../docs/ATTENTION_CONTENT_MODEL_V1_DESIGN.md)
SHA-256 `b64062f3c07aa1e29b8fd5d3f50bcc04b2188edfe2a8d6b26cafb4a9d04ff2af`.
Source/text inspection only: no import, compilation, test, world/schedule
generation, checkpoint/archive load, forward, fit, Reporter or Git operation.
The coordinator reports passing native reconstruction; this review did not load
that evidence and does not certify its runner binding.

**Disposition: revise the first engine draft before actual phase execution.**
The core mechanism follows the approved architecture, but the evidence and hard
checks below are incomplete. These are prospective software defects, not actual
A-study failures. Preserve this first source/review before separately labeled
corrections. No performance threshold, tolerance or semantic gate is changed.

## What the source supports

`fit_sources` resets states on every update, makes exactly 300 full-batch Adam
updates per learned source, uses the fixed optimizer and global clip1, and retains
the computational graph across delayed feedback. Current loss averages 14 selected
maps/world and next loss averages 15×3 selected maps/world, each across 384
normalized elements; their independently averaged terms receive .5/.5 weights.
Both learned sources share the fit-V-only coordinates, architecture and budget,
with separately registered initialization seeds. The same-object alias is a
duplicate check. Training targets enter loss after prediction, with no mapping,
truth, pixels or H in the prediction API. Recorded counts/initial and terminal
states do not independently prove the optimizer trajectory; the engine correctly
states that limitation.

At decision t, `_observation` supplies V and W(t−1), plus the previously issued
command; `_feedback` supplies W(t) only after scoring. There are decisions0..14
and updates from reveals1..14. W15 is evaluator archive content only. The reviewed
core assembles current from the last revealed snapshot and next from current,
preserving uncertain inherited addresses. H consumes predicted maps for policy
selection; actual actions consume separately executed W. Neither actual H(W)
nor evaluator truth replaces H(A).

The analytic comparator starts uniformly over all six permutations, intersects
survivors using only permitted revealed completions at the registered tolerance,
and retains ambiguity/zero-survivor failure. Its primary is H(posterior mean map);
expectation of H is labeled secondary. Final queries can be calibrated and are
not evidence of resolving a still-unknown mapping. Ordinary/analytic and scalar
maxima parity remain legitimate explanations.

Six donor prefixes are constructed once per phase/encoder and shared. For each
t=12,13,14 and both schedules, the executor performs t transitions but replay
consumes only t−1 feedback updates. The reached addressed state is patched into
the recipient while recipient observations/W stay fixed. Restoration checks
predictions, state and policy exactly. Exempting the patched address from
nonselected-next isolation correctly permits current-to-next inheritance. Donor
policy actions reference shared executed forks; baseline policy branches execute
again and are explicitly duplicate evidence.

Per encoder/phase the declared counts are consistent: 96 worlds,1440 common
transitions,4320 shared forks,1344 current and4320 next selected maps/source,
17280 full-next item records/source,864 shared queried actions,288 executed policy
duplicates/source,576 donor pairs/source,7488 shared donor transitions and6912
feedback events/source. Sources, restorations, aliases and secondary policies do
not multiply independent worlds. Bootstrap indices use the fixed seed and256
whole-world paired draws. Compact actual-state reconstruction starts from V,
replaces only the addressed item from its saved operator family, and checks W0,
all45 forks and15 common transitions exactly; failure retains raw arrays.

Address permutation uses `_per_world_close`, **which is authorized** by the
amended contract: numerical encoder/readout/permutation checks use atol1e−6,
rtol1e−5. A toy test's stronger exact equality does not amend that contract.
Copy, isolation, restoration and emitted discrete policies remain exact.

## Required corrections and integration evidence

**E1 — completed fit sources can disappear from first-failure preservation.**
At lines128–188, `progress` has a single live checkpoint/optimizer. Once ordinary
starts these overwrite A's live entries. A's terminal optimizer is held only in
local `terminal`, and its completed checkpoint is not put in `partial`; an
ordinary failure prevents the final return containing either. Preserve a
source-indexed map of both active and completed weights/optimizer states in
caller-owned partial state, refreshing after successful steps and before source
handoff. Preserve existing first logs/counts and the original exception; no retry.
Do not describe the hooks as retaining all completed fitted state until this is
fixed. Apply the same principle to completed shared controls/donor-prefix records
before later risky evaluation work.

**E2 — analytic addressed-state isolation is incomplete.**
Lines547–551 check nonpatched `survivors` only; the addressed analytic state also
contains `revealed_symbols`, and core patching copies both. Archive reached donor
revealed-symbol masks alongside survivors (line581), and require exact isolation
for both masks, exact selected-address transfer from the reached donor, and exact
restoration. Retain the complete terminal analytic state or an explicit verified
reference to reconstruct it. The present core implementation appears correct;
the engine's claimed hard-state check does not cover the full state.

**E3 — visibility recipes do not yet bind all actual evaluation calls.**
`_visibility` (lines104–115) omits the candidate-address argument and state tensor
hashes. Primary feedback and `_replay_donor` updates have no source-specific
call-boundary records; shared donor provenance is an executor recipe rather than
evidence of each actual source update. Record candidate input and h/posterior
hashes for predictions, and actual completion/command/history/mask hashes at
primary and donor update boundaries. Bind the manifest to the corresponding
pre/post state and source. Independently reconstruct/enforce the inventory:
45 prediction calls for A (15 primary,15 alias,3 remapping,6 donor,6 restoration),
30 for each other source, and14 primary plus72 donor batch-update calls/source.
Training's expressly declared first-update permission recipe is not a claim of
retained all-update latent states or independently proven fitting trajectory.
Do not infer heldout nonleakage from declared flags alone.

**E4 — advertised copy/observation/assembly hard invariants are absent.**
`software_checks` contains restoration, patch isolation, action/alias/confidence,
permutation, family reconstruction and fitted-weight checks, but no explicit
baseline observation-boundary/assembly checks matching the engine design's
claimed coverage. Add named checks for exact initial V, last-revealed nonselected
copies, next noncandidate inheritance (including a patched current address),
exact allowed histories/timestamps and unchanged source-local nonpatched decoder
outputs. The independent verifier must reconstruct these from retained inputs
and actual call records; production booleans alone are insufficient. Bind the
compaction failure check into the runner's named hard-gate schema before it can
issue a completed receipt. Existing core validators and successful-looking
assembly code are useful static evidence, not a replacement for this integration.

**E5 — the registered diagnostic publication is only partly assembled.**
The bootstrap input (lines591–599) covers selected map/field errors and policy
accuracy, but omits inherited/full-state errors and cell/coarse disagreements
from the declared primary comparisons. Retain all existing raw ledgers and add
whole-world reductions/paired differences for those primary quantities using
the same saved256 draws, with red/blue attribution. Donor records retain maps,
policy fields/maxima and change flags, but do not tabulate current/next H-field
changes, scalar changes or factual disagreement with fixed W across all pairs.
Produce these declared causal summaries from retained maps/frozen H and shared
actual references, with no changed-only subset or new support cutoff. Explicit
derived reconstruction is acceptable; duplicate full tensors are unnecessary.
Record coincident donor addresses/duplicate status. Raw data can support these
calculations, but the current returned summaries do not establish completion.

## Scope after correction

This is a useful learned content/control bridge: addressed recurrence learns
command consequences, estimates concealed current content, forecasts alternatives
and causally supplies operation choice. It advances beyond an unchanged raw-W
filter demonstration. Its task policy still uses scalar sharpness maxima and an
equally informed ordinary model has the same architecture, so command changes
alone do not qualify representation manner or attention-specific consciousness.
The fixed dev literal-content assessment must examine H(A), H(W), truth and
off-family errors separately. Wrong estimates and null donors remain evidence.

After corrections, tiny fixtures and complete runner/verifier source review
remain necessary. Immutable native/input/fitted/dev bindings, source closure,
phase sealing and named gate enforcement are external runner obligations; this
engine cannot release a split. Passing bookkeeping does not qualify presentation
or release Reporter work. Accurate conditional source reports resembling the
registered consciousness-report structure remain the user goal, with the earlier
negative manner endpoint preserved and the full goal unachieved.
