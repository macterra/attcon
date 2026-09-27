# Bound-prose confirmation v5: bounded reasoning with sufficient audit output

Frozen before v5 calls, 2026-09-27. Version 4 generated all 240 reports, but two
blind-audit responses exhausted their 8192-token allowance. At termination there
were 49 completed, two incomplete, eight interrupted reservations, and 181
unattempted audit requests. All records remain archived, including the stop reason.
The completion criterion was already irreversibly failed under the no-retry rule;
remaining calls were stopped rather than spending more on a run that could not pass.
An incomplete audit is not evidence of an inaccurate report or a negative character
judgment. Version 4 cannot establish the registered combined success.

## Fresh configuration

Keep the reporter, full-state interface, models, ten conditions, rubric, and all
acceptance thresholds unchanged. Use process seeds 410040000+visual seed and scene
seeds 420040000+visual seed, eight episodes per pair: 240 new reports. Do not reuse
or silently complete version 4's audit as a fresh confirmation.

Extractor v9 uses the same fixed `gpt-5.4-2026-03-05` snapshot, schema, instructions,
and scoring as v8, with **low reasoning**, maximum **16384 output tokens**, and
240-second request timeout. Validate all 30 fixtures again before the new run.
These changes address resource exhaustion; they do not relax a factual or
report-character criterion. Retain every failed or incomplete attempt.

Generation retains 8192 output tokens, medium reasoning, timeout 120 seconds.
Both stages retain concurrency eight and no retries. At most 240 generation and
270 extraction attempts (including fixtures). Registered output ceilings are
$3.93216 for generation and $66.3552 for extraction; actual use is expected to be
substantially lower and is recorded. Unit prices and model documentation are
linked in [v4](BOUND_PROSE_CONFIRMATION_V4.md). Archive the exact code, requests,
responses, source tensors, token usage, and verdict. Apply every original gate.
