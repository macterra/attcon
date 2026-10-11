# Malformed prose-verdict correction: author source record

2026-10-11. A narrow pre-output software correction after the parent preserved
original pipeline3ac47e02 / fixturec726e964 at94c61a6. No first fixture execution,
scientific archive, report/API output or model call was observed by this author.
This rationale is author review, not independent approval or runtime release.

The critic found that validate_prose_audit marked a list/dictionary verdict
invalid but then passed that unhashable value into Counter, raising instead of
retaining the failed claim. The [corrected pipeline](../../../../scripts/attention_relational_reporting_pipeline_v1.py)
normalizes only permitted string verdicts into their existing count labels;
every other verdict and malformed non-object claim counts as invalid. Original
claim bytes remain in the input/first response. The result stays invalid and
cannot establish fidelity. Existing supported/contradicted/not_auditable behavior
and all scientific criteria, schedule, source/reference/prompt files are unchanged.

[Fixture18](../../../../tests/test_attention_relational_reporting_pipeline_v1.py)
checks list and dictionary verdicts return invalid/fidelityfalse, count one invalid
claim, preserve the malformed original and do not crash. The original17 test
methods are unchanged, verified by AST-function SHA equality in
[original_17_method_hashes.json](original_17_method_hashes.json). This is static
source comparison, not a fixture execution. Original pipeline_software metadata,
preparation log and design record remain unchanged and historical.

Syntax compilation of corrected source and fixture file completed exit0. No
pytest, model, archive/export/report/API execution or Git operation occurred.
The parent independently confirmed original pipeline_software/design.md links
already resolve correctly with three parent components; they were left unchanged.
This nested correction record uses four parent components correctly.

Disposition: corrected source and18-fixture source ready for preservation,
independent review and first-execution release. No runtime/result qualification.
