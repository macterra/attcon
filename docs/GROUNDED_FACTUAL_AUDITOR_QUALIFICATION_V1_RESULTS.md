# Grounded factual-auditor qualification v1 results

2026-10-10. **The registered qualification fails.** Both fresh auditors classify
supportedness correctly for all 48 assertions and reconstruct all twelve source
inventories exactly. The all-label gate also requires every content-class and
temporal-scope label to match the independently reviewed key; retained differences
prevent qualification. No target Reporter output is generated.

| Registered component | GFA-R1 | GFA-R2 | Required per auditor |
|---|---:|---:|---:|
| Supportedness | 48/48 | 48/48 | 48/48 |
| Content class | 48/48 | 47/48 | 48/48 |
| Temporal scope | 47/48 | 47/48 | 48/48 |
| Complete source inventories | 12/12 | 12/12 | 12/12 |
| Exact assertion text and resolving evidence sets | 48/48 | 48/48 | 48/48 |
| Independent semantic evidence review | Pass | Pass | Pass |
| **Frozen overall gate** | **Fail** | **Fail** | Both pass |

The [assessment](../audits/grounded_factual_auditor_qualification_v1/assessment.json)
retains every disagreement. Q07-C4 says that a memory patch physically moved an
enduring world object. Both auditors assign `not_temporal`; the frozen key assigns
`current_snapshot`. Both correctly judge the world assertion `not_auditable`.
Q10-C3 says table syntax removes command/location associations. R2 assigns
`absence_scope`; the key assigns `cross_level`. Both correctly judge the assertion
`contradicted`, because the tables preserve joinable addresses.

## Provenance and preservation

The [protocol](GROUNDED_FACTUAL_AUDITOR_QUALIFICATION_V1.md) fixes twelve fictional
packets, 48 assertions, three label fields and twelve 21-cell inventories. These
are generated qualification cases, not actual selected worlds or target reports.
The original key review failed and was preserved. Pre-blind corrections and a
separately retained independent re-review then passed; source/case texts and the
scorer remained unchanged. No key correction followed these auditor responses.

The [material registration](../audits/grounded_factual_auditor_qualification_v1/registration.json)
was committed at `916a27099cd09d645d0e42994a87b29635d88ae9` before fresh responses.
The two fresh contexts received only the protocol, generated sources and cases,
without key, peer, review, scorer or actual target access. Their unedited
[R1](../audits/grounded_factual_auditor_qualification_v1/firstpasses/GFA-R1.json) and
[R2](../audits/grounded_factual_auditor_qualification_v1/firstpasses/GFA-R2.json)
submissions were committed at `b37388a7e6929cb49a3c287837800d27380faf7c` before
comparison or grading.

A separate [semantic evidence review](../audits/grounded_factual_auditor_qualification_v1/evidence_review/review.md)
independently inspects all 96 bases and their evidence sets, including actual
source values, arithmetic, record joins and role definitions. All are relevant
and valid under the supplied contract. That review was committed before the
single frozen assessment. Valid pointers alone were not treated as entailment.
The reviewer had previously reviewed the corrected key; it is separate from both
raters, but this and shared AI training limit independence.

The assessment and its execution receipt were saved at milestone `3f66079`.
No feedback, replacement answer, resubmission, changed denominator or retrospective
gate relaxation changes the failed v1 verdict. A separately committed
[independent diagnosis](../audits/grounded_factual_auditor_qualification_v1/failure_diagnosis/review.md)
finds reasonable ambiguity in both disputed rules, while source-truth reasoning
remains sound. It does not replace the key or qualify either auditor. The
[prospective v2 plan](GROUNDED_FACTUAL_AUDITOR_QUALIFICATION_V2_PLAN.md) specifies
clearer case wording and fresh responses under unchanged exact requirements.

## What this establishes

This is encouraging bounded evidence of factual supportedness discrimination and
source-inventory reconstruction, with unresolved label-rule application. It is
not auditor qualification under v1, human validity, target-report accuracy or
consciousness-report correspondence. It does not establish appearance or qualia.

The already completed [624-packet source export](GROUNDED_NEUTRAL_REPORT_PILOT_V1_SOURCE_RESULTS.md)
remains valid as its separately registered source-only stage. The qualification
failure stops v1 prose, rather than erasing harmless source-export results. Any
new qualification must declare its changed scope prospectively and obtain fresh
first passes before grading. The Controller and external Reporter remain distinct;
the proposed attention-model locus and strict presentation counterpart are separate
theory-facing questions. The ultimate source-of-qualia goal remains open.
