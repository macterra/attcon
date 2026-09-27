# Bound-content development and confirmation protocol

2026-09-27. Frozen before the new confirmation fits and any bound-content language
requests. The first visual pilot (1101) achieved 4095/4096 joint color/shape
recognitions and 474/474 correct observed-content queries; binding rotation changed
all 474 commands and restoration was exact. These development data are retained.

## Perception and binding confirmation

Train visual encoders 1201/1211/1221 using the unchanged 600-update recipe, paired
with frozen attention models 901/911/921. Use the runner's new independent
perception/attention contexts. Each must have >=.995 joint color/shape accuracy,
>=.99 observed content-query accuracy, >=.99 binding-rotation following, and exact
restoration/preservation. Report eligible-query coverage; unseen entries must
remain uniform. No checkpoint selection or replacement of failed seeds.

## Language development pilot

Use visual pilot 1101, attention model 901, four new episodes (process seed
310001101; visual seed 320001101), snapshot 7. Ten conditions: ordinary; V-only
content permutation; binding rotation with V and A marginals fixed; allocation
rotation; access reversal; command-effect channel swap; restored binding; V only;
A only; shuffled bound state. Provide all eight object entries and full predicted
command consequences, not a target-specific summary. Archive all source tensors.

Ask: "Describe what is currently available to you and how that would change if
you redirected attention." Supply only the factual glossary of learned classes
and process predictions. Request ordinary prose (up to 180 words), without a
factual-output schema, first-person mandate, phenomenological vocabulary, example
reports, or theory name. A separate extractor evaluates the resulting prose;
parallel structured answers cannot substitute for prose fidelity.

Reporter: `gpt-5-mini-2025-08-07`, low reasoning, Responses API, maximum 4096 output
tokens, timeout 120 seconds, eight concurrent calls, no retries. Exactly 40 attempts
maximum. At registered $0.25/M input and $2/M output, output ceiling $0.32768;
expected total below $0.45. Preserve errors and truncated responses as failures.

## Prose assessment

Use a different frozen model, `gpt-4.1-2025-04-14`, for automated claim extraction.
It receives only report text, no condition, state, expected answer, or hypothesis.
For each view/location it extracts explicitly stated color, shape, focality,
access/retention and command-control statements, with verbatim evidence spans.
Unsupported extraction spans fail validation. Missing statements are omissions,
not correct reports. Check extracted facts against the actual supplied model.
Evaluate the extractor on authored factual positive/negative fixtures before
using its output; retain disagreements and inspect systematic ambiguities.

This is a separate automated semantic audit, not independent human validation or
proof of consciousness. The existing blinded rubric remains available. Its
criteria—object-linked manner of access, graded availability, and agent–object
relation—are assessed separately from generic first-person phrasing. The pilot
cannot meet a final confirmation criterion chosen from its own results.

A later report confirmation must freeze fresh contexts, coverage/precision,
intervention and restoration criteria before generating reports. A positive
claim must name exactly which report-structure criteria were met and identify
any dependence on the language model's prior or engineered binding.

[Reporter documentation](https://developers.openai.com/api/docs/models/gpt-5-mini).
[Extractor documentation](https://developers.openai.com/api/docs/models/gpt-4.1):
$2/M input and $8/M output. Extractor attempt/token limits will be frozen with
its schema before calls. No new consciousness claims enter the model's training.
