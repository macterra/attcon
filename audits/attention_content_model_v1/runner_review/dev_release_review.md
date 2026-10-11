# Dev-release fixtures: bounded static approval

2026-10-11. Separate same-AI-family review task with inherited context; no human
or independent-family endorsement. Source/text/hash inspection only. No imports,
compilation, fixtures, torch/model calls, production evidence loads, worlds,
fits, Reporter or Git operations.

Exact reviewed bytes:

| File | SHA-256 |
|---|---|
| [Runner](../../../scripts/run_attention_content_model_v1.py) | `bd4c41b26af823ea75d86dc3e442fedf4c1e18b334da9938e705eb7499d3bf59` |
| [Five fixtures](../../../tests/test_attention_content_model_v1_dev_release.py) | `98dd35365c9de69d6ab095f901a306b3bea1a0ab5a42a3c18e983b8ce8f985f5` |
| [Permission schema](../../../docs/ATTENTION_CONTENT_MODEL_V1_PERMISSION_SCHEMA.md) | `8ea6ca991eb672e38a81c74d594a80bd3678eec955bb17c8ca25384a3bdbde68` |
| [Fixture design](../runner_software/dev_release_design.md) | `9960586a1a817512bff9e98ad24d2a7a7c9915d0384b59c566c6acabe3012b67` |
| [Preexecution record](../runner_software/dev_release_preexecution_metadata.json) | `eeca428f4bbb276e83eba493f0a020915d92064c6701a0bae20e6e889f69e5a1` |

**Approve one first execution of only these five fixtures after preserving/pushing
the exact sources, preexecution records and this review.** The declared command
is `.venv/bin/python tests/test_attention_content_model_v1_dev_release.py`.
Preserve the first exclusive log/result, including failures, before any repair or
separately labeled repetition. No actual phase/engine execution, production
registration or scientific qualification is approved. No fixture result is
asserted here.

The test imports the runner under a fixture module name after disabling bytecode
writes. Its module-level imports are standard library only. ROOT is replaced
with a temporary directory; every report, receipt, registration and acceptance
read/write is synthetic temporary JSON. `bound_files` and `ancestor` are mocked
throughout helper calls, so there is no real Git or production binding operation.
No phase entrypoint, torch-dependent validator, native acceptance, loader,
collector, fit, evaluator or Reporter is invoked.

The five tests cover negative semantic dispositions accepted with a bound passing
synthetic reconstruction; missing reconstruction-report reference rejected;
failed report rejected despite an assessment's green assertion; changed verifier
rejected; and changed fit commit rejected. The positive test also checks both
binding calls and the fit→dev-artifact→acceptance ancestry calls. Its one-check
report and fake hashes are control-flow inputs, not independent reconstruction
evidence or a permitted actual check inventory.

The new helper prospectively closes the previous bare-assessment source omission:
it binds assessment and report at the acceptance commit, checks dev phase/status,
all declared checks, absence of errors/exception, dev receipt, fitted artifacts
and registered verifier hash, then checks registration/fit command provenance and
artifact commit ancestry. It requires all three sources' explicit semantic
dispositions and unreleased reporting. An unqualified disposition can therefore
permit the fixed technical heldout run when hard invariants pass. This preserves
the approved negative endpoint and adds no predictive/manner threshold.

These five fixtures do not establish that real commits, native inputs, archives,
report inventories or independent calculations satisfy those requirements. The
helper checks the inventory declared by the report; the finalized registered
verifier and its complete review must establish which actual checks are required
and emitted. Binary `unqualified` must retain the detailed limited/unresolved/
failed assessment in the independent scientific record. A true presentation flag
in this helper is not itself a reviewed presentation qualification or reporting
release. Full engine/verifier integration, E1r preservation correction, complete
source review and committed phase registration remain pending separately.
