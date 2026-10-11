# Attention-content model v1: experiment engine source draft

2026-10-11. Source-only implementation against the approved amended
[contract](../../../docs/ATTENTION_CONTENT_MODEL_V1_DESIGN.md).
The engine is [attention_content_model_v1_experiment.py](../../../scripts/attention_content_model_v1_experiment.py),
SHA-256 `5c50f7d7107c4e5cf6571311abc3963a9ffefeb6eca66218cba58389f041e7ef`.
It was syntax-compiled, never imported or executed in this subtask. No world,
renderer, checkpoint/archive load, fit, model forward, Reporter or Git operation
was performed. Tiny engine fixtures have not yet run. Independent complete-source
review and committed phase registration remain required before actual execution.
The parent reports accepted corrected native full validation at 206,070/0; this
subtask has not loaded that evidence. Core's separate thirteen passing toy tests
do not establish a fitted A or qualified semantics.

## Public API and execution boundary

`fit_sources(encoder_seed,evaluator_episode,log=None,partial=None)` accepts only
the already released fit episode. It constructs A/ordinary with the registered
seeds and fit-only normalizer, resets all world h on every update, and performs
exactly 300 full-batch Adam updates per source with the fixed settings and clip1.
The complete fifteen-decision unroll uses separately averaged current selected-map
loss at t=1..14 and next selected-map loss at all fifteen decisions×three symbols,
in normalized coordinates, weighted .5/.5. Targets enter losses after prediction;
revealed W(t) enters feedback only after that decision's losses. No H, truth cell,
mapping label, bandwidth or evaluator dictionary is a source input.

Return keys are `models`, `normalizer`, `initial_checkpoints`, `checkpoints`,
`optimizer_terminal_states`, `training_log`, `fit_visibility`,
`completed_updates_per_source=300`, and `visibility_reconstruction`. The phase
runner omits models from serialization and stores normalizer by dataclass fields.
Initial and terminal state dicts are preserved per learned source. There are 600
scalar training records, including the first and final pre-update losses, gradient
norm and update indices. Each source's first-update fifteen permission manifests
and the fixed 300-reset recipe cover the repeated input/feedback schedule.
Transient training predictions are not all duplicated in an archive; the entire
fit trajectory requires deterministic optimizer replay and is not claimed
independently proven by terminal checkpoints. This was explicitly agreed with
the parent before execution and does not weaken training counts or source gates.

`evaluate_sources(models,episode,native_head,native_standardizer,extractor,split,
log=None,partial=None)` accepts immutable A/ordinary plus the released episode and
frozen native apparatus; it adds the empirical analytic mapping source. It never
fits, reinitializes fitted sources, loads artifacts or releases a phase. Required
extractor controls run itemwise. All predicted maps pass frozen H itemwise;
completed actual actions use the existing NativeController. Source ordinary use
and budget are the same as A. A's same-object alias and each source's scalar-maxima
policy are explicit duplicate checks. The analytic primary is H(posterior mean
map); posterior expectation of H is a separately attributed derived alternative.

After fitting, the phase runner evaluates these same fit worlds once, before
the immutable fit receipt. Those fit diagnostics are fitting-split evidence,
never independent validation. Dev/heldout are separate fixed calls using the same
fitted bytes after their registered releases. Every phase has the fixed world,
command, opportunity and donor schedule; evaluation adds no new rendered worlds.

Both functions update optional caller-owned `partial` state before risky operations.
Fit records live weights/optimizer, update counts, losses and current observation;
backward/optimizer failures also refresh partial terminal state/gradients. Evaluation
records current source state, latest prediction, accumulated records/checks/calls
and donor state before patching. The runner must archive this first failure without
resume or retry. These are preservation hooks, not permission to fit again.

`compact_episode(episode)` retains original delivered inputs, truth, mappings,
commands, queries, nuisances and world indices plus actual operator-family maps.
It checks exact reconstruction of initial W, each of 45 full forks and fifteen
common transitions. Healthy full W/fork copies are replaced by explicit references:
start at V; replace the issued address with its mapped member of the saved family;
choose the common command's fork. Failed compaction retains the original full
actual arrays and boolean failure evidence. The runner archives evidence before
enforcing hard gates. Raw released episode inputs may be retained on first failure.

## Archive keys and tensor shapes

The root runner owns `fit|dev|heldout/states-{encoder_seed}.pt.gz` and immutable
receipt hashes. Healthy bundles contain the compact `episode`, `fit` payload in
fit, and `evaluation` payload for every phase. No file is written by the engine.
PyTorch tensors and compact metadata share one compressed bundle per encoder/phase.

Each `evaluation` contains `tensors`, `metadata`, `software_checks`:

* `tensors.shared`: V/pixels/truth/mappings/commands/query roles/world indices;
  `actual_operator_family` [96,4,3,24,4,4]; completed H(W) logits
  [96,16,4,3,4,4]/fields [96,16,4,2,4,4]; family fields [96,4,3,2,4,4];
  actual query cells [96,3,3]; actual W channel means/std/norms; frozen category/
  orientation controls; `bootstrap_indices` [256,96]; and `actual_controls` with
  sham/endogenous/exogenous/restored/remapped emitted logits/fields and action cells.
* `tensors.sources[s]`, s=A/ordinary/analytic: `selected_current_maps`
  [96,14,24,4,4], `selected_next_maps` [96,15,3,24,4,4], matching logits/fields;
  normalized selected/full map errors; full field errors; literal role-specific
  cell/coarse agreement and disagreement with H(W)/truth; expected columns,
  entropy/max/margin/ties; continuous three-operator map distances and channel
  statistics. Source full states are reconstructed from the last permitted actual
  snapshot and these selected maps, including inherited current uncertainty.
* Learned `pre_reveal_hidden` [96,15,4,48]. Analytic
  `pre_reveal_survivors` [96,15,4,6] and `pre_reveal_revealed_symbols` [96,15,4,3],
  with survivor counts/entropy/revealed-symbol counts. `terminal_state` follows
  reveal W14. W15 is archived by the evaluator; there is no scored decision15 or
  fictitious core post-scoring update at15.
* Per-source policy symbols/cells/correctness/tie counts [96,3], role maxima
  [96,3,3] and actual role fields [96,3,2,4,4]. Selected commands execute a real
  evaluator branch again and retain the shared fork reference; wrong outcomes
  are measurements. Red/blue accuracy is assessed separately and pooled.
* `sources[s].donors.current|prospective`: selected current maps
  [96,3,24,4,4], selected next maps [96,3,3,24,4,4], policy symbols/maxima/fields,
  shared-action references, source change flags, reached donor hidden/posterior at
  the patched address, and restored selected maps/policy fields. Restoration and
  isolation are exact hard checks; source change/action effects are measurements.
* `analytic_secondary`: expectation-of-H fields [96,15,3,2,4,4] and its three
  query symbols. These derive from shared family fields and posterior weights;
  they are not extra worlds, hard gates or oracle assignments.

`metadata.source_call_visibility[s]` contains per-call kind, decision/reveal
timestamps, exact update indices and input hashes for primary, alias, address
remapping, donor and restoration calls. It identifies current completion and
fork visibility separately. `donor_provenance` has six shared prefixes, executor
indices 0..t−1, permitted update indices 0..t−2, each completion/address/symbol hash,
pre/post-reveal masks and evaluator-only donor mapping indices. Reached source
states use only permitted donor completions; donor W(t) remains concealed.
Actual controls retain hashes and emitted arrays. Metadata bootstrap summaries
use saved paired whole-world draws and fixed conditional interpretation. There
is no semantic/presentation qualification or report generation.

## Ledgers and hard gates

Per encoder/phase: 96 shared worlds; 1,440 common transitions; 4,320 shared actual
forks; 1,344 current estimates, 4,320 next selected estimates and 17,280 full-next
item records per source; 864 shared query operator/role actions; 288 duplicate
executed baseline policy branches per source; two×288 donor patch/restoration
pairs per source; 7,488 shared donor-prefix transitions and 6,912 feedback updates
per source. Donor prefixes are constructed once per world/time/schedule, shared
across all sources; coincident schedules are not deduplicated. Policy donor outputs
reference already executed shared forks, adding no actual executor transition.
All prior ambiguity and final calibrated mapping outcomes stay in the ledger.

`software_checks` contains only nonempty boolean tensors for copy/restoration,
driver parity/actions, observed-source boundaries, source alias/confidence identity,
patch isolation, address equivariance, family reconstruction and unchanged fitted
weights. Prediction accuracy, posterior ambiguity, content/map changes and causal
null effects never enter hard checks. The runner saves evaluation/partial evidence
before checking them. No performance, manner or new post-fit threshold is added.
Receipts and source closure must bind this engine and independent verifier before
release; the engine's split argument alone cannot certify that phase sealing held.

Status: source draft ready for independent static integration review, with static
compilation only. Existing native gates and the full unachieved scientific goal
remain unchanged. No automatic execution, qualification or publication is implied.
