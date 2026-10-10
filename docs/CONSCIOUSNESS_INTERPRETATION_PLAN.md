# Plan: test report structure against functional organization

Updated 2026-10-10. **Design plan, not a frozen confirmation protocol. Development
experiments have been run; no theory-facing confirmation has been completed.** This is the current next-stage plan;
earlier protocols and verdicts remain unchanged.

## Goal and current evidence

Test the proposed connection between an attention-control model and independently
specified features of consciousness reports. The required evidence combines
accurate internal-state reporting, independently assessed report structure, and
causal dependence on the system's functional organization rather than supplied
labels. The experiment need not establish that consciousness is necessary for
task performance or that a same-state decoder cannot succeed.

The [bound-content confirmation](BOUND_CONTENT_RESULTS.md) establishes accurate,
intervention-following reports. The [specificity study](SPECIFICITY_RESULTS.md)
shows that the original structural features also occur in external-device reports.
The [self-access replication](SELF_LABEL_RESULTS.md) finds self-coupled access in
26/48 reports versus 3/48 independent-access controls, but nearly equal stated
dependence for self-labelled and camera-labelled tables (41/48 versus 39/48).
The remaining gap is whether the report features track a functional distinction
beyond the supplied attribution. These results do not establish subjective experience.

## Roles and scope

- **Attention-control model:** represents allocation, access, and command effects.
- **Controller:** selects and executes actions using system information.
- **Reporter:** external diagnostic instrumentation describing internal states.
  A learned readout can remain external; it need not become the Controller.
- **Evaluators:** score factual accuracy and independently specified report features.

A native learned reporter is one candidate method, not the only possible method
and not evidence of consciousness by itself. The recurrent consistency diagnostics
remain related mechanism evidence. Building an MSTC Modeler-schema comparator is
not a prerequisite for this reporting experiment or an automatic extension of it.

## 1. Specify the target features before observing new reports

Prepare a rubric grounded in independent descriptions of consciousness reports,
with positive and negative examples not drawn from the new experimental outputs.
Candidate dimensions separate focal/background, recovery forecasts, identity
certainty, presentation manner, answer confidence and action–access relations.
Separate generic factual dependence, attribution to a particular subsystem, and
phenomenological character. A self-only flag cannot by itself establish specificity.
First-person wording, fluency, or use of a consciousness word cannot earn success.

Have independent human reviewers assess the rubric and later score blinded,
randomized reports from every condition. Report agreement, disagreements, and
uncertainty; automated scores are secondary checks, not substitutes for that review.
No human reviews have yet been obtained. Preparing materials is authorized by this
plan; contacting reviewers is a separate action requiring explicit instruction.

Deliverables: [rubric draft 2](FUNCTIONAL_REPORT_RUBRIC_V2.md), [blank definition-review form](FUNCTIONAL_RUBRIC_REVIEW_FORM_V2.md),
blind rating form, scoring instructions, and
prespecified mapping from theoretical predictions to each primary contrast.

The [review packet](FUNCTIONAL_REVIEW_PACKET.md) is now prepared for author review
first, as requested. It separates definition review from factual cases and unchanged
development prose. The received R-C1 review is an exposed AI contributor review;
its [disposition](FUNCTIONAL_RUBRIC_REVIEW_RESPONSE_V2.md) adopts semantic splits and
recasts intervention/revision checks as fidelity controls. It does not establish
independent human validation. Primary theory-facing criteria remain unfrozen, and
human author review, independent review and factual qualification remain pending.

## 2. Make the functional distinction identifiable

Construct paired systems with matched objects, tasks, action opportunities, and
representation/readout resources:

| Condition | What attention commands actually change |
|---|---|
| Own access | Acquisition or retention of content used by the system's decisions |
| External device | An external device's information access; the system's own task-content access is independently maintained |
| Decoupled control | The represented access relation is disconnected from the actual access process |

Verify the wiring using executed interventions and subsequent access outcomes,
not names assigned to tensors. Include crossed model/world interventions: change
the represented relation while holding the physical process fixed, and change the
process while initially holding the model fixed. Immediate reports should follow
the supplied internal state; after new evidence, test whether state and reports
update appropriately. Do not require a faithful reporter to know unobserved wiring.

All three present fixture conditions contain an attention-control model. Wiring
changes do not remove that proposed substrate. The own/external contrast tests a
candidate specificity prediction, whose relevance must be justified in definition
review; it is not already a derived necessary consequence of the source-of-qualia
theory. If review finds the contrast nondiscriminating, revise the theory-facing
protocol before freezing it rather than treating engineering separability as support.

**Identifiability is a design gate.** If all reporter-visible states and histories
are identical across two functional conditions, no reporter can infer which wiring
generated them. Provide equally accessible evidence of command, acquisition,
retention, and decision consequences through neutral channels. Match instantaneous
content where feasible while retaining the temporal evidence required to distinguish
the processes. Audit condition IDs, formatting, order, vocabulary, and task outcomes
for unintended shortcuts. Perfectly indistinguishable inputs are an invariance
control, not a test that a reporter is expected to discriminate.

Deliverables: causal wiring specification, matched simulator, intervention tests,
and a documented inventory of what evidence each reporting condition exposes.

The [readout audit](FUNCTIONAL_READOUT_AUDIT_RESULTS.md) finds that the bridge
preserves dominant category identity and changes confidence/thresholded availability.
The earlier export contained no executed category-decision trace, motivating an
explicit Controller decision rule and declared information boundary. This adds no
phenomenal criterion and does not require consciousness for task success.

The subsequent [explicit Controller assay](FUNCTIONAL_DECISIONS_RESULTS.md) now
implements that rule and archives executed category-task decisions. Modeled effects
guide physical acquisition in the own-buffer condition, effect swapping alters
commands/outcomes, and restoration is exact. External/disconnected task-buffer
outcomes are physically invariant despite residual model-derived answer changes.
The bridge/cutoff remain engineered; the new traces supply functional use rather
than independent consciousness-report characterization. They are not yet integrated
into new qualified language reporting.

The [executed-task input export](FUNCTIONAL_TASK_INTERFACE_RESULTS.md) now connects
these traces to 5,184 neutral source records and a frozen
[field inventory](FUNCTIONAL_TASK_SOURCE_INVENTORY.md). Current predictions, actual
histories, pending commands and emitted answers/abstentions remain distinct; remapping
and restoration replay exactly. This completes that source-integration component,
with no new language reports or semantic-auditor qualification. Review the task
vocabulary as part of factual-audit qualification before new prose collection.

## 3. Establish a factual reporting interface without supplying the interpretation

Begin with neutral identifiers and fixed reporting instructions shared across
conditions. If a learned readout is necessary, train it on factual state-reporting
tasks and action/outcome histories, without consciousness targets or self/attention
labels that supply the tested attribution. Audit both inputs and training targets
for indirect leakage. General language pretraining remains an explicit limitation.

Use held-out seeds, content combinations, and command mappings. Counterbalance
neutral identifiers between training and evaluation; hold out remappings. Compare
factual fidelity and feature expression with the established labelled interface.
Under deliberate conflicting descriptions, keep source-precedence instructions
fixed and evaluate whether outputs track grounded evidence or the distractor label.
An ambiguous-input failure must not be interpreted as absence of consciousness.

Deliverables: frozen readout/interface, training-data manifest, held-out splits,
factual-fidelity checks, and label/identifier leakage controls.

## 4. Freeze and run the causal confirmation

Before confirmation, register sample size and power analysis, independent training
seeds, primary contrasts, minimally meaningful effects, fidelity minima, label
robustness margins, uncertainty estimates, multiplicity handling, and stopping rules.
Use separate development data to assess feasibility; do not choose gates using
confirmation reports. Reuse applicable factual dimensions from the prior reporting
protocol, explicitly documenting any necessary change before data collection.

The primary theory-facing endpoint remains unfrozen. Candidate causal-grounding
comparisons ask whether independently scored attribution:

1. Tracks the decision-content versus other-buffer functional distinction with
   neutral labels, conditional on adequate fidelity and retained generic dependence.
2. Survives identifier remapping and follows grounded evidence under conflicting labels.

These can be ordinary causal inference even when wiring must be inferred. Independent
review must specify what additional report correspondence and alternative control
make them relevant to the source-of-qualia theory. Attribution alone cannot replace
that requirement. Holistic character remains descriptive, while presentation manner
and other proposed correspondence features still need independent justification.

Required fidelity/causal-source controls include model-only interventions followed
by exact restoration, state revision after process changes/evidence, and withholding
command-relation information while retaining ordinary contents. Preserve factual
accuracy for unaffected contents. These checks locate reporting sources and assess
grounding; they do not independently discriminate consciousness from faithful readout.

Include matched perturbations and missing-information controls. Selectivity matters:
destroying all readable content and observing weaker reports is not evidence for
the proposed mechanism. Ordinary object facts, access predictions, and generic
dependence statements should remain separately scored.

The unit of independent evidence is the episode/model, not each correlated report
or extracted assertion. Archive all attempts and failures, exact source states,
requests, checkpoints, code versions, and human/automated ratings. Do not silently
retry or replace failures. Revisions require a new version and fresh confirmation.

## 5. Interpretation, replication, and stopping

| Outcome | Warranted conclusion and next action |
|---|---|
| Factual fidelity inadequate | Interface experiment is uninterpretable; improve it on development data and register a fresh test. |
| Fidelity passes; features follow labels or are equally generic | Accurate description remains supported; the stronger interpretation is unsupported by this assay. Preserve the result. |
| Functional contrast and selective interventions pass, but human characterization fails | Causal reporting of access is supported; consciousness-report correspondence remains unresolved. |
| Fidelity, independent characterization, functional contrast, and label robustness pass | Evidence supports the proposed connection between attention-control organization and consciousness-report structure within the tested system. |

A positive result warrants replication with fresh trained systems and another
task or architecture before a broader claim. It does not establish subjective
experience or uniquely identify its source. Alternative accounts based on ordinary
causal inference and learned description must be discussed, not ruled out by naming
the represented state a self-model.

Do not continue cycles until a positive result appears. Finish each registered test,
publish its actual outcome, and distinguish a completed experiment from an achieved
theory-facing objective. Freeze the experimental protocol only after the rubric,
identifiable comparison, reporting interface, and feasible power analysis exist.

The [requirements audit](CONSCIOUSNESS_REQUIREMENTS_AUDIT.md) maps every stage to
current evidence and outstanding work. Component checks do not establish completion.

## Execution order and documentation

1. Prepare independent rubric and causal wiring specification.
2. Implement and validate the matched systems and neutral factual interface.
3. Run development checks; establish identifiability, fidelity, and leakage controls.
4. Freeze the powered protocol, then collect confirmation reports and blind ratings.
5. Analyze all registered contrasts and replicate any positive result.
6. Update the preprint with retained follow-up results and the resulting claim limits.

Current completion: a [draft rubric](FUNCTIONAL_REPORT_RUBRIC.md), rating form,
[validated causal-wiring fixture](FUNCTIONAL_WIRING_RESULTS.md), and
[learned-model integration](FUNCTIONAL_MODEL_RESULTS.md) are available. All neutral
pilot failures remain retained: v1 has attribute errors, v2 fails coverage, and v3/v4
halt at auditor qualification without reports. The
[complete-interface pilot v5](NEUTRAL_FUNCTIONAL_PILOT_V5_RESULTS.md) supplies current
allocation, unattended recovery and command selection; its qualifications pass and
36 reports plus 72 audits complete. Restoration fails the frozen factual gate,
principally from malformed extracted selection addresses, with separate verified
reporter contradictions and predominantly technical character. The [v6 protocol](NEUTRAL_FUNCTIONAL_PILOT_V6.md) now enforces typed process
addresses and forecast-presence metadata, with prepared inputs from unused row 3.
[Qualification](NEUTRAL_FUNCTIONAL_PILOT_V6_RESULTS.md) fails 1/32 fixtures and
halts before reporting. The [v7 amendment](NEUTRAL_FUNCTIONAL_PILOT_V7.md) adds
explicit forecast-supply evidence rules, retaining identical unused row-3 reporter
inputs, but [qualification fails 3/36](NEUTRAL_FUNCTIONAL_PILOT_V7_RESULTS.md). No
v6/v7 reports are generated; row 3 remains unused. The
[fixed-budget three-way controls study](FUNCTIONAL_CONTROLS_RESULTS.md) now completes
three fresh models: all nine static cases pass and 26/27 matched revision routes
pass. Seed 2021's buffer 1→neither route fails control-mask accuracy (504/512,
below 99%); the overall engineering verdict remains fail. Common physical targets
and unchanged controls address the old unmatched comparison. The
[three-way visual/interface export](FUNCTIONAL_INTERFACE_RESULTS.md) now supplies
664 neutral inputs including static controls, both revision contexts and retained
mistaken-model diagnostics. All input/tensor invariants hold; no new prose is
generated. A [source-aware manual factual-audit draft](FUNCTIONAL_MANUAL_FACTUAL_AUDIT.md)
separates quoted claims, judgments and coverage, with blank forms and no completed
qualification. That procedure needs independent review before new reporting.
The new [three-part review packet](FUNCTIONAL_REVIEW_PACKET.md) supplies 28 synthetic
cases with unverified author-proposed answers and all 36 unchanged v5 reports,
split by model pair for calibration/later whole-prose review. No human annotations
have arrived and no earlier verdict is rescored. User coordination chooses author
review first; independent review remains outstanding.
The [original review form](FUNCTIONAL_RUBRIC_REVIEW_FORM.md) now contains R-C1's
completed exposed AI review. Draft 2 has a separate blank form and definition bundle;
the original review and v1 packet are retained unchanged. No completed human review
exists. Independent rubric review and human ratings remain
pending; none of the six stages is complete in full. The updated preprint retains
the September 27 study and incorporates later findings and limitations.

The [fixed-model disconnection diagnostic](CONTROL_DISCONNECTION_DIAGNOSTIC_RESULTS.md)
now completes the retained-error investigation: eight contradictory probes correct
all eight failed seed-2021 cases, versus seven with random commands and none with
agreement commands. Original histories already contain disagreements; the original
failure remains. Exactly identical-input continuation worlds demonstrate the
separate identifiability limit. No new reporting occurs. Further theory-facing
collection waits on substantive definition review and factual-audit qualification.
