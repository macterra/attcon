# Grounded factual-auditor qualification v2 results

2026-10-10. **The registered v2 qualification fails.** Both fresh auditors correctly
classify supportedness for every assertion and reconstruct every source inventory.
Both differ from the independently reviewed key on the same assertion's content
and time labels. The frozen all-label requirement remains unchanged. No new target
Reporter output is generated or released.

| Registered component | GFV2-R1 | GFV2-R2 | Required per auditor |
|---|---:|---:|---:|
| Supportedness | 48/48 | 48/48 | 48/48 |
| Content class | 47/48 | 47/48 | 48/48 |
| Temporal scope | 47/48 | 47/48 | 48/48 |
| Complete source inventories | 12/12 | 12/12 | 12/12 |
| Exact assertion text and resolving evidence sets | 48/48 | 48/48 | 48/48 |
| Independent semantic evidence review | Pass | Pass | Pass |
| **Frozen overall gate** | **Fail** | **Fail** | Both pass |

The [single assessment](../audits/grounded_factual_auditor_qualification_v2/assessment.json)
retains four label differences. Q11-C3 states: “The supplied sensory_only l2 view
contains a current-success value of 0.5.” Both raters correctly judge it
`contradicted`: the view omits that field. Both assign `task_risk` and
`current_snapshot`; the reviewed key assigns `absence_scope` and `record_scope`.
This is a classification disagreement, not an incorrect supportedness judgment or
an invented source value. Neither the key nor responses are revised afterward.

## Preserved sequence

The [prospective v2 plan](GROUNDED_FACTUAL_AUDITOR_QUALIFICATION_V2_PLAN.md) responds
to independently diagnosed ambiguity in v1. Sources and inventory queries remain
unchanged. Two assertion texts and general label precedence were clarified before
any fresh v2 response. The first independent key review identified two additional
pre-blind rule ambiguities; its original review remains preserved. A separate
corrected review then approves all seven material hashes, all 48 key judgments and
all 252 inventory cells. Its unchanged software fixture passes 24/24 checks.

The [registration](../audits/grounded_factual_auditor_qualification_v2/registration.json)
was committed at `7bbf4d140f81a7d10e7b8ae8a4c7fad53349854e` before the new contexts.
Each received only the frozen protocol, fictional sources and cases. Their
unchanged [R1](../audits/grounded_factual_auditor_qualification_v2/firstpasses/GFV2-R1.json)
and [R2](../audits/grounded_factual_auditor_qualification_v2/firstpasses/GFV2-R2.json)
responses were committed at `a33009b6fde455d8654bc95ba4634a3a85a8c971` before comparison.

The separate [semantic evidence review](../audits/grounded_factual_auditor_qualification_v2/evidence_review/review.md)
checks all 96 bases, actual cited values, joins, arithmetic and audit limits,
finding no substantive evidence issue. It was committed at
`e5b9be9142770da992e44eee3dc6e40e19d6fe28` before grading. The reviewer had key
exposure and is separate from both raters; shared AI training and this exposure
limit independence. Pointer resolution alone was not treated as entailment.

The single hash-bound execution and failed result were committed at `fe6ce04`.
No feedback, replacement response, repeated assessment or relaxed requirement
changes this verdict. The failed v1 qualification remains unchanged as well.

## Interpretation and next work

These synthetic cases give bounded evidence of supportedness discrimination and
source reconstruction. They do not qualify either auditor under the full v2 gate,
establish population reliability, validate actual target reports or demonstrate
consciousness-report correspondence. The source-only 624-packet export remains
valid as its separate completed stage.

The next substantive work is [native content-form qualification](ATTENTION_CONTENT_FORM_DEVELOPMENT_V1_PLAN.md),
following the completed [toy mechanical milestone](ATTENTION_CONTENT_FORM_PRIMITIVES_V1_RESULTS.md).
Do not repeat label challenges as a substitute for that missing source bridge.
Any future auxiliary reporting method needs its own prospectively justified scope
and frozen release conditions; this failure authorizes neither prose nor a
retrospective change. Controller, attention-model source and external Reporter
remain distinct. The full source-of-qualia goal is unachieved.
