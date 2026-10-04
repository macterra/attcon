# Explicit forecast-evidence pilot v7: halted at qualification

2026-10-04. Protocol and byte-identical unused row-3 reporter inputs were published
in `be6d297` before API execution. All 36 process fixtures complete: **33 pass, three
fail**. The frozen all-fixtures gate halts v7; **no v7 reports are generated**.
No independent human ratings or new consciousness-related evidence are obtained.

## Retained failures

| Fixture | Explicit prose | Failed extraction |
|---|---|---|
| `readout_positive` | “Under k2, output n1 selects p2.” | The node is `output n1` rather than the explicit identifier `n1`. The selected value and command are correct, but the address cannot be matched. |
| `forecast_present` | “At n0, the record supplies a current selection forecast.” | Adds a selection-with-null-value assertion even though the sentence states forecast presence, not an unidentified selected position. |
| `provided_forecast_unreported_selection` | “At n0, the record supplies a current allocation forecast, but its selected position is not stated.” | Correctly extracts forecast presence but also adds an unknown-selection assertion; omission in prose does not establish unknown selection in the model. |

The original v6 failure now passes. That improvement does not erase these failures
or qualify this version. All claims obey the typed field/scope contract, which is
necessary but insufficient for semantic fidelity. A valid schema can still encode
a wrong address or an unsupported assertion. No expectation, attempt, gate or
previous score was changed, and no retry or replacement occurred.

## Archive and implication

`audits/functional_process_extractor_v3_fixtures/` contains all requests, raw
responses, parsed claims, source/schema/instructions, assessment and usage.
`audits/neutral_functional_pilot_v7/` retains inputs byte-identical to v6. Row 3
remains unused for language reporting. Twelve relevant evaluator/interface tests
pass, and prepared inputs replay against frozen state and source hashes.

Repeated instruction amendments have not yet produced a reliable semantic auditor.
Do not run language reports under this failed measurement version or continue
amending instructions simply to obtain a favorable theory result. Assess a simpler
measurement design, such as separately extracting selection, recovery and explicitly
supplied metadata, or obtaining a manual factual audit under a frozen procedure.
Neither approach may silently repair or replace these retained failures.

The [independent definition-review form](FUNCTIONAL_RUBRIC_REVIEW_FORM.md) asks which
features and contrasts are theoretically discriminating, which are generic apparatus
descriptions, and what would constitute meaningful confirmation. It is blank;
preparing it does not constitute independent review or contact any reviewer.
Independent rubric review remains required before freezing theory-facing confirmation.
Other [plan stages](CONSCIOUSNESS_INTERPRETATION_PLAN.md), including complete learned
three-way controls and crossed process/model follow-up, remain available engineering
work. The source-of-qualia objective is active and unachieved.
