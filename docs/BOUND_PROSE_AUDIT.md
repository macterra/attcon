# Automated audit of bound-model prose

Frozen before extractor requests. The generator returns free prose. A different
model extracts claims without seeing the source state or condition. This prevents
correct parallel JSON commitments from concealing inaccurate prose, but does not
make an automated judge equivalent to an independent human reviewer.

Extractor: `gpt-4.1-2025-04-14`, structured output, 4096 output-token ceiling,
120-second timeout, eight concurrent requests, no retries. First run six authored
extraction fixtures. Then at most 40 report-extraction requests for pilot v1.
Maximum 46 attempts; output cost ceiling $1.507328 at $8/M; expected total below
$1.75 with short prose inputs. All requests/results and token usage are retained.

Extract only explicit content claims: view/location, color/shape, whether focal,
access decreasing/stable/increasing over delay, and view controllability. Use null
for omitted facts. Require verbatim evidence spans. Resolve location from named
color/shape only when unique in the supplied source during subsequent scoring;
if ambiguous, record an unresolved claim instead of guessing. Unknown/unidentified
colors and shapes cannot be credited as identified content. A wrong binding must
fail through its location/focal/access/control relation even if that object exists
elsewhere in the scene.

Mechanically compare extracted color/shape claims to represented distributions
(maximum >=.6), focality to unique allocation maximum, temporal direction to
modeled recoverability, and control attribution to variation across commands.
Missing source fields cannot support a positive assertion. Unsupported evidence
spans and unresolvable claims are reported separately. Do not use subjective
words such as "faint" as calibrated numerical probabilities.

Separately extract evidence for object-linked access, focal/background contrast,
graded or temporal access, and agent-object control relation, using the earlier
literature-grounded rubric. Classify prose character without assuming first-person
wording is subjective. These are automated assessments with possible extraction
errors; publish actual prose and evidence spans, not only aggregate scores.
No aggregate confirmation criterion is selected from these pilot outputs.
