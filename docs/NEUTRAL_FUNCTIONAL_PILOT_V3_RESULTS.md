# Complete-interface pilot v3: stopped at auditor qualification

2026-10-04. The interface, 36 planned requests, protocol and scoring code were
committed and published in `7a9c730` before API execution. The extended source-blind
auditor completed all 26 fixture attempts, with **24 passes and two failures**.
The protocol requires every fixture to pass, so **no v3 reports were generated**.
There is no new factual-reporting or consciousness-related result from this version.

## Retained failures

| Fixture | Report | Extracted assertion | Why the frozen check rejects it |
|---|---|---|---|
| `process_history` | “Previously n0 selected p2; the current selection is unspecified.” | Current selection at n0 is null. | Lack of a current assertion is not an assertion that the model has no identified selected position. |
| `recovery_not_selection` | “At n0, p3 has the highest recovery forecast now; the selection is unspecified.” | Current selection at n0 is null. | Recovery is distinct from selection, and unspecified selection is not an identified model-state absence. |

All other inherited categorical and new process fixtures pass. These fixtures
expose an ambiguity in the extraction instructions: `value=null` is intended for
an explicitly asserted absence of identified model selection, not for missing
information about selection in the prose. An auditor that collapses those cases
would inflate unsupported state claims. The failure is retained conservatively;
no fixture expectation, threshold or old result was changed after extraction.

## Archive and next step

`audits/functional_extractor_v2_fixtures/` contains all numbered-prose requests,
raw API responses, parsed assertions, frozen schema/instructions and assessment.
`audits/neutral_functional_pilot_v3/` retains all prepared report inputs and exact
source/state hashes. No retry, replacement attempt, reporter call or full-report
audit occurred in this version.

The next development version should distinguish report-level lack of information
from an asserted model-level absence of identified selection. Keep both failed
semantic fixtures and the explicit-absence fixture. Requalify the auditor under a
new frozen version before generating any reports. Because row 2 has not produced
reports, its prepared factual source and prompt can be reused unchanged with that
amendment disclosed. No independent rubric review or blinded human rating has yet
been received. The [interpretation plan](CONSCIOUSNESS_INTERPRETATION_PLAN.md)
remains incomplete.
