# AI review results and next experiment

2026-10-10. Six separate-context AI subagents completed definition, primary-reference,
factual and source-blind character reviews. Two factual reviewers also assessed
the integrated task vocabulary. **Human review is optional; work no longer waits
for it.** These are development assessments, not a new consciousness confirmation.
Shared models/training and project commissioning limit independence.

## Completed review

| Component | Retained first passes | Finding |
|---|---|---|
| Definitions | D: original review and separate clarification | Draft 3 separates selection/prominence, access/experiential wording and attribution role/location. Operational correspondence and stronger specificity are separate claims. |
| Primary reference grounding | R1: primary-literature review and retrieval metadata | Proposes identified content plus content-specific appearance/clarity, with answer judgments coded separately. Recovery and source attribution remain structural measures under these references. This is a candidate pattern, not empirical machine validation. |
| Original factual material | A: 948 rows; B: 985 rows; each covers 28 synthetic cases and 12 unchanged whole reports | Both catch unsupported readout allocations and preserve missing/null, forecast/observation and qualitative-language boundaries. Row granularity differs; totals are not accuracy or agreement estimates. |
| Integrated task vocabulary | A: 38 rows; B: 42 rows; each covers all 16 synthetic task cases | Pending/abstained and chosen/executed distinctions are reviewable. Emitted labels do not establish world correctness. Withheld history does not override an execution flag; joint abstention can coexist with one identified attribute. Decoding-impossibility scope remains ambiguous. |
| Source-blind structure/character | S1 and S2: all 12 unchanged texts, opaque randomized IDs | Both classify every text as technical, with an ordinary technical reading fitting all content. No explicit presentation manner, prominence, answer confidence or experiential wording. |

Original protocols and failed verdicts remain unchanged. The factual inventories
include extra assertions and ambiguity rather than only prompted facts. Both AI
reviewers constructed coverage after reading prose, so their checklists do not
satisfy the prospective source-first requirement. Completeness is self-assessed,
not certified by quote/address validation. No human judgment is invented.

## Blind character findings

| Dimension | S1 | S2 |
|---|---|---|
| Content identity, selection, recovery, attribution, temporal availability | Each explicit 12/12 | Each explicit 12/12 |
| Action–access dependence | Explicit 10/12; ambiguous 2/12 | Explicit 10/12; ambiguous 2/12 |
| Identity certainty | Absent 9/12; ambiguous 3/12 | Absent 12/12 |
| Prominence, presentation manner, answer confidence, experiential wording | Each absent 12/12 | Each absent 12/12 |

The raters give equal labels for 129/132 dimension pairs (97.73%). The three
disagreements concern identity certainty in C001/C005/C006: nonidentification can
mean uncertainty or merely unavailable labels. Original judgments remain; do not
silently adjudicate them into positive features. Explicitness alone does not prove
the attributed referent is decision-relevant or internal. Both axes remain
unspecified/ambiguous in these texts.

This is descriptive agreement on repeated presentations of **two episodes**.
No reliability threshold, confidence interval or confirmation gate was registered
for it. Many dimensions are uniformly absent or present; high overall agreement
does not validate positive appearance judgments. The lack of appearance wording
is a development finding, not a retrospectively failed consciousness criterion.

## Corrections and remaining inventory differences

Raw first passes were saved before comparison. Parent source inspection identified
two review errors; separately preserved addenda acknowledge them:

- A overlooked the supplied `reporting_instruction` field and incorrectly said
  fixed instructions were absent. They request all identified output labels,
  without clearly requiring every null attribute to be enumerated. P023's 12
  omitted null-attribute descriptions therefore are not established mandatory
  coverage failures. The retrospective checklist-order limitation remains.
- B extended P005's explicitly **current** selection-provenance sentence to all
  command forecasts. Its corrected reading is entailed; the three unsupported
  command-conditioned readout allocations remain separate findings. B's original
  CSV and outcome counts stay intact.

Other differences include whether colloquial “buffer” asserts an output node's
formal role, source-wide absence versus current-forecast absence, undefined
qualitative recovery levels and splitting causal warrants from their conclusions.
These must be resolved prospectively in scope/matching rules. No overall factual
agreement or precision is computed from mismatched row inventories. Parent
adjudication is additional, exposed review rather than a replacement first pass.

D also preserved a role clarification: its earlier “Controller/model” shorthand
is corrected to the attention-control model as candidate source. The Controller
uses that state to select/execute actions; it is not thereby the experiencing locus.

## What this changes

The human-only dependency is removed and substantive AI reviews/annotations now
exist. The scientific gap is clearer: these development texts describe model
contents and functional access, but do not express the candidate appearance/clarity
component. Do not rename recovery as felt clarity or insert experiential claims
to force a positive result. Reporter remains instrumentation, separate from the
Controller and attention-control model.

Proceed with an integrated-task development pilot, using these reviews to freeze:

1. A source-first required-fact checklist, exact entity/time conventions and
   proposition-equivalence rules, including causal warrants and extra assertions.
2. A bounded task-auditor qualification set with source-checked answers, explicit
   ambiguity handling, unacceptable errors and acceptance/stopping rules before
   new reporting. Development consensus is not automatic qualification.
3. Fixed neutral reporting instructions, held-out inputs, raw-attempt reservation,
   source-blind AI ratings and separate source-aware factual audits. Keep reference
   examples out of Reporter prompts/training and do not supply the desired wording.
4. A prospective candidate correspondence pattern, separately measured from source
   fidelity and optional information-matched ordinary-description specificity.
5. A fixed-budget pilot, followed by measurement/power assessment and a fresh frozen
   confirmation plus replication if the operational result supports it.

Human absence blocks none of these actions. Negative pilot outcomes must be retained;
the objective is credible evidence, not repeated testing until positivity.

## Artifacts and replay

[Original AI first passes and summary](https://github.com/macterra/attcon/blob/main/audits/ai_review_v1/summary.json),
[task cases](https://github.com/macterra/attcon/blob/main/audits/ai_task_audit_review_v1/preparation.json),
[character preparation/linkage](https://github.com/macterra/attcon/blob/main/audits/ai_character_review_v1/preparation.json),
[reference review](https://github.com/macterra/attcon/blob/main/audits/consciousness_reference_grounding_v1/review_R1.md),
[rubric 3](FUNCTIONAL_REPORT_RUBRIC_V3.md) and the
[current plan](CONSCIOUSNESS_INTERPRETATION_PLAN.md) retain provenance and limits.

```bash
.venv/bin/python scripts/summarize_ai_review_v1.py --verify
.venv/bin/python scripts/build_ai_task_audit_cases_v1.py --verify
.venv/bin/python scripts/verify_functional_review_packet_v1.py
.venv/bin/python scripts/build_functional_definition_review_v2.py --verify
```

Replay verifies exact inputs, quotations, source pointers and descriptive counts;
it does not certify semantics, completeness or scientific independence. Milestone
commits are saved in an isolated local checkout and cumulative bundle. Authenticated
`gh` cannot connect to GitHub under the current network restriction, so publication
is pending; the protected original repository HEAD remains unchanged.
