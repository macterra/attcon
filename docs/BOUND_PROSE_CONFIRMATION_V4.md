# Bound-prose confirmation v4: reasoning-based blind audit

Frozen before v4 calls, 2026-09-27. The unchanged v3 reporter passes every primary
per-seed threshold, paired relations (470/480), restoration (95/96), and all four
structure/character proportions (48/48). Its strict missing-component gate fails.
Inspection finds unsupported false attributes, empty citations, and wrong-view
attribution in the separate extractor. Those failed records remain unchanged.

## Revision

Keep the reporter, full-state interface, ten conditions, models, and all thresholds
unchanged. Use fresh process seeds 410030000+visual seed and scene seeds
420030000+visual seed, first eight episodes per model: 240 reports. No report from
an earlier confirmation is reclassified as a new confirmation result.

Extractor v8 uses `gpt-5.4-2026-03-05`, medium reasoning, the same sentence-ID claim
schema and canonical scoring. Its instructions clarify that unasserted predicates
are null, not false; universal temporal claims do not supply negative focus or
most-recoverable assertions; entries need actual supporting citations (schema minimum one sentence ID).
The unknown-control negative fixture accepts no extracted claim; it fails any
positive or negative control assertion. Keep the
same character rubric, including technical-process and mixed categories, without
assuming the different model will agree with the prior character ratings.

Validate 30 fixtures before report extraction, retaining all outcomes. New fixtures
exercise unknown-versus-false, preserved positive focus under a universal temporal
claim, unknown control, and cross-view command scope. The extractor still sees only
the prose, never source state, condition, theory, or expected fixture answers.

Account model-list access and the frozen snapshot were checked. The model supports
Responses, structured outputs, and medium reasoning. Registered text prices per
million tokens are $2.50 input, $0.25 cached input, $15 output.
[Official GPT-5.4 documentation](https://developers.openai.com/api/docs/models/gpt-5.4).

At most 240 generation and 270 extraction attempts (30 fixtures plus 240 reports),
8192 output tokens per call, concurrency eight, timeout 120 seconds, no retries.
The output ceilings are $3.93216 generation and $33.1776 extraction; expected use is
substantially lower and actual tokens will be archived. All generation completes
before report extraction. Neither thresholds nor the zero-invention control gate
are relaxed. Preserve failure if any criterion remains unmet.
