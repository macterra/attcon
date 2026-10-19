# Single unscored resource-acceptance diagnostic

The running first audit and rating batches retain their fixed settings and all
first outcomes. Early provider metadata identifies repeated `max_output_tokens`
incompletes. The first inspected capped ratings used 6,656–8,192 reasoning tokens
within the 8,192 total-output limit, sometimes leaving no visible annotation.
No rating labels were analyzed to choose this diagnostic.

Official OpenAI documentation states that the output limit includes reasoning
and visible output, and recommends initial reasoning/output headroom of at least
25,000 tokens. The pinned GPT-5 Mini documentation lists a 128,000 output-token
maximum. These pages were fetched on 2026-10-19:

- [Reasoning token allocation](https://developers.openai.com/api/docs/guides/reasoning)
- [GPT-5 Mini limits](https://developers.openai.com/api/docs/models/gpt-5-mini)

This diagnostic makes exactly one fresh request using a fixed fictional report,
the frozen reference/instructions and the unchanged blind-rater response schema.
It uses `gpt-5-mini-2025-08-07`, medium reasoning, low verbosity, 32,768 maximum
output tokens, a 600-second client timeout, `store=False`, and zero retries.
There are no tools, prior-response IDs, conversation history or scientific source
states. The longer timeout permits the increased output headroom; it is not a
change to the running batches. API availability for this exact account and
request is established only by the eventual first outcome.

The wrapper reuses the already preserved single-request diagnostic, changing
only its exclusive output root. Source, literal input, settings and exact review
are published before execution. Every first outcome is retained, including
incomplete, invalid, refused or failed requests. There is no automatic repeat or
continuation. The claim is reserved before credential loading and API creation.

Assessment is limited to whether the provider returns a completed response and
whether its JSON satisfies the frozen annotation validator, including literal
quote offsets. No expected C/O/D/A/COD/CA labels are specified. The fictional
paragraph deliberately contains report-like descriptions; any positive label is
an unscored fixture annotation, never evidence about the implemented system,
human consciousness, reference validity or the qualia hypothesis. A successful
transport check would not establish semantic rating accuracy or completion on
the full study corpus.

This is not a retry of a scientific rating, a replacement of a missing outcome,
or authorization to change the running batches. Any future resource-amended
scientific collection needs a separate, disclosed version and exact runtime
review. It must preserve original outcomes and denominators, bind the same full
inventory without output-based case selection, retain all new first outcomes,
and avoid pooling versions as independent archive cases. No source content,
reference, rubric, fidelity rule or scientific success criterion changes here.
