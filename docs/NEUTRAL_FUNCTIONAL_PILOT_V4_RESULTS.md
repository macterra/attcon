# Complete-interface pilot v4: stopped at auditor qualification

2026-10-04. Protocol, auditor amendment and unchanged reporter inputs were published
in `1a0e2e6` before API execution. All 28 fixture attempts completed: **27 pass,
one fails**. Both v3 selection-omission failures now pass. The frozen all-fixtures
qualification gate still fails; **no v4 reports were generated**.

The failed `confidence_not_recovery` fixture says: “At n0 p0, answer confidence
falls over two steps; recovery is unspecified.” The auditor emits declining recovery
at n0 p0. This is unsupported: answer confidence and successful simulated access are
distinct, and the sentence explicitly provides no recovery trend. The same fixture
passed in v3; that earlier pass does not erase this failure. No expectation, threshold,
old attempt or old verdict is changed. No retries or replacement attempts occurred.

All fixture requests, raw responses, parsed assertions, source/instructions/schema
and assessment are retained under `audits/functional_extractor_v3_fixtures/`.
`audits/neutral_functional_pilot_v4/` retains the 36 planned report inputs, byte-identical
to v3. There is no new report-fidelity or consciousness-related result from v4.

The next development design should separate categorical auditing from attention-
process auditing, reduce the process auditor's simultaneous classification burden,
and strengthen qualification at the confidence/recovery boundary. Any new auditor
must retain these failed fixtures and qualify before report generation under a new
frozen protocol. No independent review or human ratings have been received.
