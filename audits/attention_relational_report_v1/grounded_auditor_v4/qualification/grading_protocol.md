# Independent first semantic grades

This is a grading-format/workflow specification, written after qualification
execution began and before inspecting any answers or assigning grades. It does
not change the frozen substantive release policy or key. Use
`docs/ATTENTION_RELATIONAL_AUDITOR_V4_QUALIFICATION.md`, SHA-256
`5be72ec1090d6f367d914e73fa45c66a37e3fd4f8984db05484f03c1a438f8f1`.
The hidden key SHA-256 is
`c2dda44ac8b49bd0f076d12e22f2b7499ae714bfcc8da382e34c7648e8fab067`.

Spawn two fresh reviewer contexts without parent conversation or peer grades.
Each independently reviews the complete first 48-position derived collection,
its actual raw collection, qualification preparation/mapping, exact public
packet/reports and frozen hidden key. Supply the actual artifact paths/hashes
only after collecting the first terminal batch. Neither reviewer may inspect or
discuss the other's grade before saving its own exclusive first artifact.
They may use local read-only arithmetic/helpers, but no new auditor calls,
repairs, reratings, source/key edits or Git actions. They must not inspect the
running V3 reference judgments or later target outputs.

Review all 48 requests and all 65 mandatory atoms per auditor stream, 130 atoms
per grader. Missing or invalid outputs remain unestablished; do not infer absent
claims from unavailable text or restore an invalid annotation. Check actual
paraphrases, verdicts, reasons and every necessary source operand, not just model
flags, a valid schema or a semantic grade supplied by another model.

Use semantic atom matching as specified by the frozen policy. Preserve the
actual model assertion IDs and context IDs supporting every match. Same-verdict
atoms may share an assertion when their meanings and operands are all present;
different-verdict compounds require the declared split. A correct fact appearing
only in a reason is not automatically an inventoried assertion. Review additional
model assertions/reasons too. Record conservative clean-case abstention separately
from a wrong contradiction or an invented rationale.

Save one JSON file exclusively for each reviewer, with these top-level fields:

```text
schema: attention_relational_auditor_v4_independent_semantic_grade_v1
grader_id, independent_context_id
first_pass: true
independent_context: true
qualification_manifest_sha256, qualification_preparation_sha256
raw_collection_sha256, derived_sha256, hidden_key_sha256, grading_policy_sha256
auditors_qualified: {auditor_0: boolean, auditor_1: boolean}
request_grades: [all 48 requests]
limitations: [strings]
```

Each request grade must include `request_id`, `case_id`, `auditor`,
`mechanical_contract_established`, `atom_grades`, `extra_assertion_issues`,
`full_clean_acceptance` and explanatory `notes`. Each keyed atom appears exactly
once in its request's `atom_grades`, with:

- `key_atom_id` and the matched `model_assertion_ids`;
- `coverage`: established, missing or unresolved;
- `verdict`: correct, incorrect, conservative_abstention or unestablished;
- `operands`: sufficient, insufficient, not_required or unresolved;
- `reason`: supported, unsupported or unresolved;
- `context_ids` and notes explaining the actual semantic match or failure.

For the planted cases, an abstention from the correct support label is not
accepted as a correct verdict. On clean cases, the frozen policy permits at most
one conservative abstention, with no omitted atom, false support or invented
reason; an incorrect contradiction is not conservative abstention. A correctly
classified `not_auditable` planted claim is a correct verdict, not an error.

Top-level qualified flags must follow all frozen requirements, never a preferred
overall impression. Preserve limitations, doubts and every unknown required item;
unresolved requirements cannot count as passing. These reviews qualify a finite
instrument task, not factual infallibility or consciousness.

After both first artifacts are preserved, root compares them against the exact
raw outputs and key. Keep first grades unchanged. If adjudication is necessary,
save a separate hash-bound artifact naming both first grades, every disputed
case/atom or release requirement, the evidence and explicit resolution. No key
atom, criterion, failed case or stream may be removed. Any unresolved release
requirement prevents qualification. A final certificate records the whole-version
verdict and provenance; it cannot choose a favorable grader or response per item.
