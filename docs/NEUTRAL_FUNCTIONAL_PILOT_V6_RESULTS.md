# Typed process-audit pilot v6: halted at qualification

2026-10-04. The protocol, typed schema, fresh row-3 inputs and evaluator tests were
committed/published in `b53ddfa` before API calls. All 32 process fixtures complete:
**31 pass, one fails**. The all-fixtures gate halts v6. **No v6 reports were generated**.
Row 3 remains unused for language reporting. No new theory-facing evidence is obtained.

The inherited `access_rank_not_focus` fixture says: “At n0 p3 is currently most
recoverable; the currently selected position is not stated.” Expected: no process
assertion. The auditor emits current `selection_forecast_presence=absent` at n0.
Unstated selection does not establish that the supplied record lacks an allocation
forecast: selection may simply be omitted from the prose or unidentified within an
existing forecast. Its recoverability rank also does not establish selection.
This is a presence-semantics failure, not the old malformed-address defect; every
parsed fixture obeys the new typed address/scope contract.

All original expectations, attempted responses and the failed verdict remain
unchanged. The fixture passed in the narrower previous auditor; that result does
not erase this failure after introducing presence metadata. No retry, replacement
or reporter call occurred. Twelve relevant evaluator/interface tests pass. Prepared
inputs replay against archived state and frozen hashes.

Archive: `audits/functional_process_extractor_v2_fixtures/` contains requests, raw
responses, parsed claims, source/schema/instructions, assessment and usage.
`audits/neutral_functional_pilot_v6/` retains prepared row-3 report inputs.

The next protocol should require an explicit statement about supplied allocation/
selection forecast information for presence claims; an unstated selected position
alone must not count. Add report-versus-record and unidentified-selection boundaries.
If an amended version reuses row 3, retain reporter inputs byte-identically and
disclose that v6 generated no reports. Do not alter earlier assessments.
Independent rubric review and human ratings remain outstanding under the
[interpretation plan](CONSCIOUSNESS_INTERPRETATION_PLAN.md).
