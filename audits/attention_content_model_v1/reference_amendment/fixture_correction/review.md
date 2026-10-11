# First amendment fixture failure: schema-only correction

2026-10-11. Separate task allocation within the same model team, with inherited
hypothesis context; no independent model family or human endorsement.

The first nine-fixture execution remains failed: eight passed, one errored, exit 1.
Its exact source was published at cb754b6a2d0d87192b69c8ff369c4042b9303759; first log,
receipt and review records were preserved/pushed at
`a8e17e307d8c6c564b1a3f7b0cca6e17791bf520`. The original
[first log](../fixtures/first_run.log), [receipt](../fixtures/first_execution.json),
[first fixture review](../fixture_review/review.md), [design](../design.md) and
[source preparation record](../source_preparation.json) are unchanged.

The sole error occurred in test_exact_creation_validation_and_exclusive_immutable_amendment
at test line 171, helper create_amendment line 222. The fixture patched immutable_context
to return only {'handcrafted_context_only': True}. The helper's collision guard correctly
accesses immutable['original_source_hashes']; the incomplete mock raises KeyError before
any exclusive amendment file is created. The actual immutable_context implementation
always constructs original_source_hashes from validated registration source_hashes and
native_input_hashes from validated registration input_hashes. Inspection found no helper
or production-program defect responsible for this failure. These are container-schema
keys; this diagnosis performs no actual original-context validation or archive loading.

The only source change adds 'original_source_hashes': {} to that one handcrafted
fixture mock. Inspection of the full mocked creator/validator path found no other
required immutable-context key access. The dictionary is explicitly synthetic and
empty. No real source/input bytes or hash claims are introduced,
and no test intent, assertion, helper, verifier, runner, tolerance or gate is changed.
The fixture still mocks all authentic bindings/ancestry and cannot approve study
provenance. Its actual helper-source byte identity check remains active.

Original fixture SHA-256:
`66f69f37927d7f27c91a5086cc03b1ccbe9b2fc80c2f7f363ff75aa4ce30b149`.
Corrected fixture SHA-256:
`513919a22fec931cf095b82beabdbe95ce68b12b4885f32e86b8cb0fac96b3e2`.
All three actual correction programs retain their published hashes. The original
nine-fixture result is not relabelled as a pass. This record is a source-only follow-up;
only AST/syntax compilation was performed after the mock correction. No fixture import,
execution, helper creation/validation, verifier, model, fit, data/world generation,
optimizer or Reporter call occurred here.

Proposed next step: independent scoped review and publication of these exact corrected
fixture bytes, followed only by explicit authorization for one separately labelled
first execution of the corrected nine-fixture suite. Preserve its complete stdout/stderr
and terminal outcome at fresh exclusive paths; failure remains failure, with no silent
retry or repair. No actual amendment creation, corrected fit reconstruction, dev or
heldout release follows automatically from those bounded tests. Any exact amendment
approval must bind the corrected fixture hash alongside the unchanged other closure
files. Controller is not Reporter; software checks cannot establish qualia or
presentation, and the scientific goal remains unachieved.
