# Attention-content model evaluator collector: first static source review

2026-10-11. `/root/spatial_content_manner_critical_review` reviewed
`scripts/attention_content_model_v1_data.py` and its five handcrafted test fixtures
against the amended A design. Source and AST reads only: no import, tests,
reserved schedule/render call, archive/checkpoint access, model inference, fit,
Reporter, Git mutation or external communication occurred in this review.

**Verdict: the intended evaluator schedule and construction match the design,
but two malformed-input rejection defects require correction before collector
runtime approval.** These are pre-execution draft contract defects, not actual
A study failures. Preserve this source/tests/first review before correction.
The coordinator reports five initial toy tests passing; this reviewer did not
execute or independently verify that log. Those positive-path fixtures do not
cover the two rejected-input cases below.

| Reviewed file | SHA256 |
|---|---|
| Collector | `24f661edf07296e09ba41b17104a78e1df200b38814a693552f2453f272721b3` |
| Handcrafted tests | `fe424b9eabc8fea3a070c99ad026284cd7c140d9a4cb66d586d7c61081718122` |
| Amended design | `b64062f3c07aa1e29b8fd5d3f50bcc04b2188edfe2a8d6b26cafb4a9d04ff2af` |

## Schedule and actual-state construction

`schedule` creates six lexicographic permutations of (1,2,4), sixteen copies
each per address, independently shuffled with the exact address/split/encoder
seed namespace. `repeat_interleave(16)` followed by a permutation retains every
balanced marginal. Commands use their separately seeded per-world/address symbol
permutations, not mapping values. The first twelve commands visit addresses in
row-major order through three rounds; final addresses/symbols/roles use the
registered `(world+j) mod 4/3/2` definitions. Roles are absent for early decisions.
The collector rendering namespace and 96-world count match the design, with exact
registered encoder/split names checked before collection. These findings are
static inspection, not execution of a reserved schedule.

`execute_command` clones completed W and replaces only the selected address from
`block_average(V[address], bandwidth)`. It does not average a previously degraded
W. `actual_episode` constructs all three symbol forks from the same saved V and
current completed snapshot, then copies the matching fork onto the independently
fixed common spine. Other addresses remain copied and initial V remains copied
and unmodified. Shared actual histories/forks are evaluator records, not a
different history per predictor. `operator_family` likewise constructs all
bandwidth alternatives from V, with its finite CPU float32 map validation.

`donor_prefix` advances only the selected mapping to the next lexicographic
permutation, copies V and executes t commands at t=12/13/14. It returns command
indices 0..t−1 and permitted feedback-command indices 0..t−2, correctly producing
t executor completions but t−1 visible feedback updates. Other mappings remain
copied. It returns concealed W(t) as evaluator data; actual recurrent replay,
visibility masks, map hashes, source-local patching and restoration are not yet
implemented here. It does not itself enforce that a model consumes only the
listed feedback.

## Required source corrections

**D1 — reject invalid common-spine symbols before fork construction/indexing.**
`actual_episode` checks command array shapes, but constructs every fork using
new valid symbols 0,1,2. It then indexes those forks directly using
`command_symbols[:,time]`. Consequently a supplied common symbol −1 is not
examined by `execute_command`'s symbol-range check and silently selects the final
fork through negative indexing. That is an unregistered command accepted as a
valid transition. Validate the actual common address/symbol tensors' integer
dtype, CPU device and complete 0..3/0..2 ranges before constructing any episode.
Add handcrafted rejection cases for negative/out-of-range common symbols and
malformed command tensors without calling registered scheduling/rendering.

**D2 — validate evaluator mapping tensors before integer conversion.**
`execute_command` checks mapping shape, but not integer dtype, CPU device or that
each address's three entries is exactly a permutation of {1,2,4}. Its `int(...)`
conversion can truncate a fractional entry such as 1.9 to valid bandwidth 1;
duplicate valid entries can also pass the primitive's bandwidth validation. This
silently alters the declared unknown-permutation contract. Reject noninteger or
non-CPU mappings and any row without exactly those three distinct bandwidths
before execution/conversion. Add purely handcrafted fractional/duplicate/invalid
mapping rejection fixtures. Invalid donor indices must similarly be rejected
where donor schedule indices are consumed, rather than relying on Python/Tensor
indexing to implement the six-permutation contract.

The exact registered `schedule` produces valid tensors, so neither issue is
evidence that a production world or source output was wrong. They are necessary
boundary checks for malformed or miswired evaluator inputs. Correcting them
must preserve all valid schedule/operator/donor behavior and the first toy pass;
new rejection tests should have a separately retained first log.

## Permission and sealing limits

`collect_worlds` calls `_indices`, then rejects `released_split != split` before
rendering or extraction. The sealed-split fixture checks this order while mocking
the renderer. The guard is a caller-supplied string comparison, not verification
of immutable registration, native acceptance, fit/dev receipts or input hashes.
The implementation correctly documents the upstream phase runner's duty to
validate release; this helper must not be presented as enforcing that entire
contract by itself. Direct `schedule`, executor and donor helper calls are also
not intrinsically sealed. Full runner call-site review must exclude dev/heldout
construction and inspection before their registered phase release.

The module is evaluator-only and has no A/ordinary model call. Its returned
dictionary intentionally includes pixels, seeds, mappings, physical cells and
all concealed targets. That dictionary cannot become a source inference input.
The later runner must select only permitted V, command/history snapshots and
delayed feedback, archive exact per-call masks, and verify that mappings/truth/
future outcomes never enter learned source calls. No immutable permission wrapper
or delayed observation source-call enforcement is supplied by this collector.

The five inspected fixtures cover saved-V recovery, common-spine/fork identity
and nonselected copying, donor index counts, early sealed rendering rejection,
and operator-family equivalence on handcrafted maps. They deliberately do not
execute the registered mapping/layout schedule or render native worlds. They do
not qualify a fit, learned A, checkpoint extraction, native reconstruction
dependency, full runtime collector/runner behavior or semantic content/manner.

After source correction, a separate static pass can approve these evaluator
helpers and their bounded toy fixtures. Actual A execution still requires full
native corrected revalidation, exact trained-module/collector/runner/verifier
review, registration and phase receipts. No new user-permission requirement,
human endorsement, private-experience proof or task-necessity condition is added.
The original negative presentation and later source/report gates remain intact.

This is separate AI task allocation within the same model family with inherited
project context, not independent-family numerical evidence or empirical
replication. No scientific goal is marked achieved.
