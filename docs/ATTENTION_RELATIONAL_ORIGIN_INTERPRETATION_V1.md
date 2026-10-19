# What would a report about the attention model establish?

This source-trace note was written on 2026-10-19 while the transport-v2 narrator
batch was running, after its earliest narratives had been inspected and before
any prose-auditor or reference-rater outputs existed. It is a descriptive
interpretation aid, not a new registered endpoint, success threshold, or change
to the frozen reporting protocol. The registered primary summaries remain intact.

The goal is evidence about the attention-control model as a proposed source of
qualia. Accurate reporting and consciousness-report resemblance must therefore
be connected to the contents of that model. A report can accurately reproduce
attention estimates while its consciousness-like description concerns other
supplied records. Whole-report accuracy and resemblance alone do not resolve
that attribution.

## Where the supplied facts come from

The reporting panel named `A_only` contains a hybrid record: it starts from W11
and replaces the previous node with the source model's current estimate. Each
conditional candidate inherits that record and replaces the query node with the
source model's conditional estimate. The same format is used for attention A,
ordinary prediction, and analytic prediction. `A_only` therefore does not mean
that every supplied fact originates in attention A. Joined panels additionally
contain native W records.

There are 25,440 required peak-set fact slots across the fixed 552 requests:

| Origin | Attention A | Ordinary | Analytic | Native W | Sensor | Total |
|---|---:|---:|---:|---:|---:|---:|
| Current source estimate | 336 | 336 | 336 | 0 | 0 | 1,008 |
| Conditional source estimate | 1,008 | 1,008 | 1,008 | 0 | 0 | 3,024 |
| Inherited current source estimate | 756 | 756 | 756 | 0 | 0 | 2,268 |
| Copied W11, including inheritance | 3,276 | 3,276 | 3,276 | 0 | 0 | 9,828 |
| Native W, including inheritance | 0 | 0 | 0 | 9,216 | 0 | 9,216 |
| Fixed sensor record | 0 | 0 | 0 | 0 | 96 | 96 |
| Total | 5,376 | 5,376 | 5,376 | 9,216 | 96 | 25,440 |

Attention-derived estimates account for 2,100 slots, including repeated inherited
estimates. In its source panel, 39.06% of slots are attention-derived and 60.94%
are copied W11. All twelve cases use previous node n3; the query nodes cover
n0–n3 for each encoder seed. In three cases the query overwrites n3 in candidate
records; the other nine inherit the current estimate across three commands.
These are reporting opportunities, not independent predictions or worlds.

The [inventory script](../scripts/inventory_attention_relational_origins_v1.py)
reads only the hash-bound preserved export, recursively unwraps inherited
origins, and records per-request and aggregate counts. It never reads generated
reports, changes prompts, or executes a model. This accounting was independently
checked against exporter source and the preserved JSON by a separate agent.
The [first inventory](../audits/attention_relational_report_v1/origin_inventory/first_inventory.json)
completed successfully and preserves every per-request denominator; its
[execution receipt](../audits/attention_relational_report_v1/origin_inventory/first_execution.json)
records the source commit and input/output hashes.

## What the control connection establishes

`NativePredictionReadout.choose` uses source candidate predictions, through the
frozen native readout H, to select a command. The experiment executes that command.
This is a genuine connection between attention-model predictions and control.
The native Controller subsequently targets peaks in **actual executed W** through
H. It does not consume the reporter's text or reporting packets. Direct use of
the attention model's current estimate by that native Controller is not
demonstrated by this reporting export. Reporter, prediction readout, and native
Controller must remain distinct in the interpretation.

Source: [export construction](../scripts/attention_relational_report_v1.py),
[prediction readout](../scripts/attention_content_model_v1.py),
[experiment execution](../scripts/attention_content_model_v1_experiment.py), and
[native Controller](../scripts/spatial_native_task_v1.py).

## How to interpret the reporting results

Keep the registered factual and source-blind reference summaries. Then inspect
whether the reference evidence concerns attention-derived estimates, copied
prior records, native W, a meaningful relation between sources, or an unresolved
mixture. The frozen source-auditor bindings and rater quotations can support this
attribution; neither whole-text COD/CA labels nor correct proposition lists do
so automatically. Supported relational comparisons between A and W can be
relevant; requiring every statement to concern A alone would discard the theory's
relational content.

The separate [descriptive attribution helper](../scripts/attention_relational_origin_attribution_v1.py)
recomputes the complete registered primary summary before linking evidence spans
to supported auditor claims. It retains every matching claim and exact quote
occurrence, marks repeated quotes and record-scope bindings as unresolved, and
preserves all operands of mixed A–W relations. Absence of containment means that
attribution is unresolved, not that consciousness-related structure is absent.
Its implementation author had no access to generated outputs; the coordinator
had previously inspected an early baseline narrative, as disclosed above.

The first independently reviewed software execution passes all **22 synthetic
fixtures** ([receipt](../audits/attention_relational_report_v1/origin_attribution/first_fixture_execution.json)).
Actual attribution awaits complete factual audits, blind ratings, and the
registered primary summary. It adds no endpoint gate or template-attribution
comparison and cannot establish experience from quote containment.

Selective intervention results must likewise identify which attention-derived
content changed and which report evidence followed it. Preserve null changes,
restorations, incomplete outputs, mixed claims, and unmatched evidence. Ordinary
or analytic parity limits attention specificity without automatically invalidating
an accurately reported attention-model relation. This note claims no report
fidelity, reference correspondence, attention specificity, or phenomenal status.
