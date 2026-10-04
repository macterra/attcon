# Functional reporting review packet v1

Prepared 2026-10-04 for **author review first**, as requested. No substantive review
or annotation has been received yet. Author comments will be recorded as author
review, separate from independent validation. This packet collects no new reporter
outputs and does not change any existing experimental verdict.

## Start with definitions

Read the [draft rubric](FUNCTIONAL_REPORT_RUBRIC.md) and complete the
[definition-review form](FUNCTIONAL_RUBRIC_REVIEW_FORM.md). The
[standalone definition bundle](assets/functional_review_v1/definition_review.zip)
contains those documents, reviewer metadata and the blank later rating form,
without experimental outputs or factual answer keys. External primary-study links
are retained; local cross-references appear as plain labels in the standalone copy.

The central review question is which independently justified report features would
make accurate attention-model reporting consciousness-related evidence, and which
controls would distinguish that correspondence from ordinary causal description.
In particular, explain why an own-access/external-device contrast should matter
to the source-of-qualia theory. All three current fixture conditions contain an
attention-control model; changing wiring is not removing the proposed substrate.
The functional contrast therefore tests a proposed specificity prediction whose
theoretical relevance needs justification, not a necessary consequence already
derived from the theory. Definitions can be revised or judged insufficient before
confirmation is frozen.

Record prior exposure and relationship to the project. Replying here with comments
is acceptable; retain the original wording and identify the rubric version. No
approval choice is preselected, and an author review must not be counted as an
unexposed independent review.

## Then assess factual-audit mechanics

The [synthetic factual-audit bundle](assets/functional_review_v1/synthetic_factual_audit.zip)
contains the [source-aware procedure](FUNCTIONAL_MANUAL_FACTUAL_AUDIT.md), a glossary,
28 synthetic reports with explicit source records, blank claim/coverage forms and
reviewer metadata. Cases cover:

- Correct and incorrect node/position/command statements.
- Unknown modeled selection, missing fields and prose omission.
- Readout allocation inventions and source-presence metadata.
- Observed versus anticipated events and forecasts versus executed commands.
- Attribute identity, recovery, answer confidence and felt clarity.
- Quoting versus endorsing an untrusted note, unresolved scope, extra assertions
  and coverage omission without false-claim invention.

These are author-written examples, not machine data. Proposed answers are stored
separately as **unverified author expectations**; they need independent checking.
Initial reviewers should not consult them. Procedural withholding is not secure
blinding; record any exposure. No human annotation or qualification is supplied by
the packet or by tests that check its formatting.

## Finally check whole prose

The [whole-prose development bundle](assets/functional_review_v1/whole_prose_development_audit.zip)
contains all 36 unchanged v5 reports and their exact reporter-visible records and
instructions. Twenty-four reports from two model pairs are calibration material;
12 from the third pair are reserved for a later whole-prose check. Each six-variant
family shares one underlying episode, so the split contains four calibration and
two later-check episodes, rather than 36 independent trials.

This is a post-hoc split of already inspected development data. It holds out one
pair from reviewer calibration, not fresh confirmation from the research team.
The original v5 failure remains unchanged. Do not infer qualification from matching
short synthetic examples alone. Quote and audit every additional factual assertion
in the prose, not only the prompted checklist. Preserve first-pass judgments before
adjudication. Independent character raters need separate source-blind materials and
a reviewed rubric; these source-aware audit bundles do not serve that purpose.

## Returning and checking annotations

Use the blank [claims CSV](FUNCTIONAL_FACTUAL_AUDIT_CLAIMS.csv) and
[coverage CSV](FUNCTIONAL_FACTUAL_AUDIT_COVERAGE.csv). Record `role` in the included
metadata CSV as `author`, `independent` or `other`; this session's first review is
`author`. Claim spans are zero-based and end-exclusive Unicode character positions
in the unchanged report. `source_paths` is a JSON array of source pointers, such
as `["/predicted_current/0/selected_position"]`; `$report` refers to report metadata.
Do not invent an unavailable source field as evidence of a negative statement.

```bash
.venv/bin/python scripts/build_functional_review_packet.py --verify
PYTHONPATH=scripts .venv/bin/python -m unittest tests.test_manual_fact_annotations
.venv/bin/python scripts/check_manual_fact_annotations.py --claims /path/to/claims.csv --metadata /path/to/reviewer_metadata.csv
```

The annotation checker verifies IDs, exact quotations/spans, source paths and role
records. It retains ambiguity and never sets `audit_qualified` or
`independence_verified` to true. It cannot judge semantic correctness, inventory
completeness or reviewer independence. Numerical agreement, acceptable errors,
coverage/precision gates and adjudication rules still need review and registration
before new prose collection or confirmation.

The [materials manifest](https://github.com/macterra/attcon/blob/main/audits/manual_factual_review_v1/manifest.json)
records archive/content and dependency hashes. Replay verifies all three ZIPs
byte-for-byte, original report/source preservation, zero-filled annotation forms,
and exclusion of answer keys/linkage from distributed bundles. The public repository
retains proposed gold and experiment linkage separately for reproducibility.
No reviewer contact, independent review, human rating or new language experiment
has been performed. The [goal requirements](CONSCIOUSNESS_REQUIREMENTS_AUDIT.md)
remain incomplete.
