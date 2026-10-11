# Runner boundary corrections: seven-fixture approval

2026-10-11. Independent task allocation within the same AI family, with inherited
project context; no human or independent-family endorsement. Static source/text
inspection only. No import, compilation, fixture execution, torch/model call,
world generation, archive/checkpoint load, fit, Reporter or Git operation.

Reviewed [runner](../../../../scripts/run_attention_content_model_v1.py) SHA-256
`b7ac221b6a31a19072d0fc9bd9c9cb9c4834c832f68ae83f4c9bf70625f52ae9`
and [seven boundary fixtures](../../../../tests/test_run_attention_content_model_v1.py)
SHA-256 `ce8df3aa11be393919638f0bef96cadeb00d8dfc666bac3f8ddbca6f48be1240`.

**Approve only one first execution of these seven software fixtures after the
coordinator preserves/pushes these exact sources and this review.** No actual
phase, engine fixture, production registration, load, forward or fit is approved.
Preserve the first test log and failure before any correction or separately
labeled repetition. This review does not assert any test result.

The test module sets `dont_write_bytecode` and imports the runner by exact local
path. The runner's module-level imports are standard library only, and its main
entrypoint does not run under the fixture module name. Each test replaces ROOT
with a temporary directory. Archive-looking paths are fake inventory names;
no checkpoint/archive is created or opened. Commit bindings and subprocess calls
are mocked where exercised, so the fixtures perform no real Git operation.

The seven tests exercise:

1. Empty phase artifact inventory rejected before binding.
2. Receipt with another input registration rejected.
3. Required three encoder archives plus first log all passed to binding.
4. Different committed bytes rejected using a mocked subprocess result.
5. Accepted native review commit pin attempted before parsing invalid temporary
   review bytes.
6. Missing registered source closure rejected before review-file access.
7. Original mocked torch-import failure retained in stdlib JSON, independently
   recorded secondary mocked tensor-retention failure, no completed receipt/log,
   original registration hash, and exclusive claim preventing another execution.

The last test mocks registration, ancestry, bindings and native acceptance, then
raises on the first `import torch`. The engine/model/collector imports and
released generation are consequently unreachable. `save_tensor_archive` is also
mocked, so the secondary-retention path never imports torch or serializes data.
The second call fails at the existing exclusive claim before imports. This is an
intentional boundary simulation, not evidence that production native acceptance,
registration or model invariants passed.

## Static correction status and limits

The runner now pins the accepted corrected native review commit and checks its
`actual_results_bytes/<path>` records against current results/completion/archives,
then checks completion's result/archive bindings and recorded verifier provenance.
This addresses the first draft's R1 source omission prospectively. No production
review JSON or archive was read to certify that the live evidence satisfies it.

R2 now requires the exact phase archive/log inventory, matching registration,
registration→fit→dev ancestry, dev's fit-receipt link and heldout assessment's
fitted/dev bindings. R3 adds a fixed fit-return schema, source/checkpoint and
initialization inventories, parameter count, fit-only normalizer, terminal Adam
step/settings,600 ordered finite log records and the fifteen-call visibility
recipe. These inspect recorded fit facts, not an independently proven trajectory.

R4 now enumerates exact named boolean hard-check shapes, prediction-kind counts,
14 primary plus72 donor update calls/source,27 decoder probes/source, named
denominators and exact compaction checks. An arbitrary green boolean no longer
satisfies this schema. The seven fixtures do not exercise these torch-dependent
validators or establish their numerical/source integration. Complete input/hash
key enforcement and independent reconstruction remain necessary.

The coordinator additionally identified an unresolved heldout-release defect:
the current dev assessment can assert `hard_invariants_pass` without binding an
actual independent reconstructed-dev report. A separately reviewed correction
must bind the committed report path/hash, passed status, exact verifier, dev
receipt/fitted artifacts and command/registration/fit lineage, and retain declared
source semantic dispositions. An unqualified semantic disposition may permit the
fixed technical heldout run; it must not become reporting approval. None of the
seven fixtures exercises this heldout boundary. This known production limitation
does not change their exact-source bounded approval.

R5 includes engine, independent verifier and permission schema in the required
registration closure and rechecks exact complete-review bindings before release.
The verifier is still integrating; a filename in closure is not a completed
independent executable review. R6 writes the original traceback and original
registration hash first, then attempts tensor retention separately. Active seed
and returned fit are kept before downstream validation. The last fixture can
test the import/secondary-retention/exclusive-claim case; it cannot certify
mid-training or partial engine retention. Failure of the stdlib failure-record
write itself remains outside this deliberately bounded fixture scope.

The revised engine source was observed at SHA-256
`09a5cbd3e6c7f3c37c65d2bdde35e1db40d65d35904c832d24d7c1533d887878`,
but its software design still bound the original first SHA and no final
correction record was available at this inspection. This review therefore does
not close E1–E5 or approve that evolving engine. Preserve the original rejections
and review its finalized correction record separately.

Actual execution still requires complete corrected engine/runner/verifier review,
toy validation appropriate to each component, exact committed registration and
native/input/fit/dev release evidence. No predictive, literal-content, manner or
reporting qualification follows from these boundary fixtures. The user's faithful
conditional consciousness-report goal and earlier negative semantic endpoint
remain unchanged.
