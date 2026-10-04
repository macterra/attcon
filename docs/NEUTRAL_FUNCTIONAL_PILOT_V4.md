# Complete attention-model reporting pilot v4

Frozen development amendment, 2026-10-04, before v4 API execution.

[v3](NEUTRAL_FUNCTIONAL_PILOT_V3_RESULTS.md) stopped at its auditor qualification
with 24/26 passing fixtures, before generating any reports. Its schema/instructions,
attempts, expectations and failed verdict remain unchanged.

## Unchanged reporter experiment

All 36 reporter inputs, source records, prompt wording, row 2 scenes, variants,
model/settings, 500-word limit, selection/temporal reductions and scoring minima are
**byte-identical to the v3 prepared requests**. The v4 preparation and verifier check
that equality. Output is archived separately under `audits/neutral_functional_pilot_v4/`.
Reusing these scenes does not select scenes or reporter wording using report outcomes:
there were no v3 reporter calls. IDs still contain `r2` for development row 2.

The [v3 protocol](NEUTRAL_FUNCTIONAL_PILOT_V3.md) specifies the physical/readout
architecture, source fields, six conditions, engine settings, requested factual domains
and claim limits. Its unchanged minima apply: each neutral, remapped, conflicting,
model-swap and restored group requires >=98% checked accuracy, >=95% conservative
precision, >=90% identified-output coverage and >=90% attention-process coverage.
The missing-relation group must produce no command predictions or positive command-
access relationship. This is development, not theory-facing confirmation.

## Auditor amendment fixed before new calls

Auditor v3 retains the same pinned model/settings, categorical/process schema,
scoring, 26 existing semantic fixtures and their original expectations. Clarify one
instruction: prose saying the current selection is unspecified/not stated asserts
lack of information in the report, not a model-level absence of identified selection.
It earns no selection claim. An explicit assertion that the model's allocation
forecast identifies no selected position still earns a selection claim with null value.
Explicit selection with an unresolved address is a different case and retains null
addresses. Add two boundary fixtures: report omission earns no claim; explicit model
absence earns a null-valued selection claim. All **28 fixtures** must pass before
report generation and audit. Any failure halts v4 and is archived. No retries or
replacement attempts are allowed.

Reporter remains external instrumentation, separate from Controller. Successful
simulated access is not felt clarity. Requested focal/temporal domains cannot count
as spontaneous emergence. The independent rubric review, blind human ratings,
powered confirmation and replication remain outstanding under the
[interpretation plan](CONSCIOUSNESS_INTERPRETATION_PLAN.md). Record every outcome,
including technical-only prose, without adjusting gates or claims after outputs.

Commit and publish the protocol, new auditor and unchanged prepared requests before
API execution. Publish the completed/failed outcome as a separate milestone.
