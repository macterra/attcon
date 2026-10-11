# Attention-content model v1: core implementation before toy execution

2026-10-11. Bounded software only. Implements the independently approved
[exact design](../../../docs/ATTENTION_CONTENT_MODEL_V1_DESIGN.md), preserved at
`6ad1d09` as supplied by the parent. No collector, runner, checkpoint loader,
renderer invocation, training loop, native study execution or Reporter is added.
The parent reports corrected native full reconstruction passed 206,070 checks
with zero failures and was committed at `c84d3da`; this subtask has not loaded
its numerical evidence. Core executable review and registration remain required.

The source is [attention_content_model_v1.py](../../../scripts/attention_content_model_v1.py).
It implements the shared CPU float32 GRUCell(771,48) and residual
435→96→384 decoder, 197,328 parameters; separate ordinary initialization with
identical architecture and an exact same-object alias; fit-only population
normalization; pre-reveal current/next assembly; addressed-only recurrent feedback;
an ambiguity-preserving analytic mapping posterior; source-state snapshot/patch/
restoration; and frozen native H applied itemwise to raw estimated maps before
symbol selection. No bandwidth truth or scored current W is a prediction argument.

The public API is:

* `FitNormalizer.fit(fit_v, split='fit')`, `.normalize`, `.denormalize`; split is
  a required keyword and caller/collector must enforce actual fit provenance.
* `CommandBatch(address, symbol)` with CPU long [B] tensors.
* `ObservationBatch(v, last_revealed_w, decision_time, last_revealed_at,
  previous_command, observed)` with maps [B,4,24,4,4], bool mask [B,4], and one
  common decision timestamp per batch. Initial time requires exact W0=V;
  subsequent calls require feedback only through t−1.
* `SourceState(hidden, revealed_through, update_command_indices, source_key)`;
  hidden [B,4,48], exact update indices 0..revealed_through−1.
* `RevealedFeedback(completed_w, command, completed_at, command_index,
  scored_decision_at)`; feedback has the separately revealed completion and
  command, enforces the next one-time reveal and its explicit post-scoring tag.
* `LearnedContentModel(normalizer, init_seed, source_key='A')` with
  `.initial_state(batch_size)`, `.decode(v_item,hidden,symbols)`, `.update(state,v,
  feedback)` and `.predict(observation,state,candidate_address)`/`forward`.
  `PredictionBatch.current` is [B,4,24,4,4], `.next` [B,3,4,24,4,4], with explicit
  decision/source attribution. Output maps are already denormalized raw units.
* `build_learned_sources(normalizer,encoder_seed)` returns A, ordinary and alias;
  `ExactAlias.predict` delegates to the identical A instance without extra weights.
* `AnalyticMappingSource` provides the same initial/update/predict boundary;
  `AnalyticState.summary` gives posterior counts/entropy and revealed-symbol counts.
  `operator_family` and `.decode` support the declared analytic alternatives.
* `snapshot_state`, `patch_addressed_state(recipient,donor,address)` and
  `restore_state`; patches require matching source/time/update indices and clone
  recipient storage. The collector must supply independently reached donor states
  and preserve their exact pre-reveal command/update provenance. Restoration of
  estimates and policy outputs is checked through a repeated forward call.
* `NativePredictionReadout(head,native_standardizer).consume(raw_maps)` returns
  itemwise local logits and role fields. `.choose(prediction,address,role)` returns
  symbol indices, three maxima and candidate fields. `choose_from_maxima` is the
  identical confidence-only duplicate rule with first-symbol exact tie breaking.

The schema can reject target-as-argument, wrong timestamps, duplicate feedback,
unseen addresses and invalid/nonfinite tensors. It cannot certify that an external
caller told the truth about a post-scoring tag or fit split. The later collector,
phase runner and independent verifier must establish these from per-call manifests,
input permissions and sealed receipts. Direct decoder calls are numerical building
blocks, not evidence of observation provenance or literal-content qualification.
The frozen H readout supplies technical H(A), not source semantics or manner.

Static compilation of the source and tests passed before any test execution.
The proposed first command is `python3 tests/test_attention_content_model_v1.py`;
all tests use tiny handcrafted tensors and freshly initialized toy weights.
One test checks gradient propagation through a permitted feedback update using
backward only; there is no optimizer, fit loop or parameter update. No existing
renderer test is invoked. Preserve first_run.log unchanged as its pre-execution
note, including its original proposed logging location. The parent has clarified
that actual first output instead belongs in a separate, exclusively created
`first_test_run.log`; this supersedes the note's unexecuted append plan. Record
the first command's complete combined output and exit status there unchanged
before considering any repair. Source and tests are ready for root/static
independent review before that command. Hold toy execution until review/root
release; no actual first-test log exists before execution.

Status: implementation written and statically compiled; first toy execution not
yet performed. No production archive/checkpoint/results loads, reserved worlds,
fits, native inference, Reporter or Git operations occurred. Only the owned core,
test and implementation-record files were changed. Actual A execution remains
gated on the reported native validation's accepted receipt, executable review and
phase registration.
