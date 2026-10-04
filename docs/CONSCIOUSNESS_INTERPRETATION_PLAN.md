# Plan: test report structure against functional organization

Updated 2026-10-04. **Design plan, not a frozen confirmation protocol. Development
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
Candidate dimensions are focal/background distinction, uncertainty about content
availability, and explicit dependence of the system's access on attention actions.
Separate generic factual dependence, attribution to a particular subsystem, and
phenomenological character. A self-only flag cannot by itself establish specificity.
First-person wording, fluency, or use of a consciousness word cannot earn success.

Have independent human reviewers assess the rubric and later score blinded,
randomized reports from every condition. Report agreement, disagreements, and
uncertainty; automated scores are secondary checks, not substitutes for that review.
No human reviews have yet been obtained. Preparing materials is authorized by this
plan; contacting reviewers is a separate action requiring explicit instruction.

Deliverables: versioned rubric, blind rating form, scoring instructions, and
prespecified mapping from theoretical predictions to each primary contrast.

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

Primary comparisons should test whether independently scored report features:

1. Track the own-access versus external-device functional distinction with neutral
   labels and fixed instructions, conditional on adequate factual reporting.
2. Survive identifier remapping and follow grounded evidence under conflicting labels.
3. Change under selective interventions in the represented access relation and
   recover on restoration, while generic content-reporting accuracy is preserved.
4. Follow state revision after process changes and new evidence, rather than merely
   repeating a fixed story about the system.

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

## Execution order and documentation

1. Prepare independent rubric and causal wiring specification.
2. Implement and validate the matched systems and neutral factual interface.
3. Run development checks; establish identifiability, fidelity, and leakage controls.
4. Freeze the powered protocol, then collect confirmation reports and blind ratings.
5. Analyze all registered contrasts and replicate any positive result.
6. Update the preprint with retained follow-up results and the resulting claim limits.

Current completion: a [draft rubric](FUNCTIONAL_REPORT_RUBRIC.md), rating form,
and [validated causal-wiring fixture](FUNCTIONAL_WIRING_RESULTS.md) are available.
The [learned-model integration](FUNCTIONAL_MODEL_RESULTS.md) passes engineering
checks. Two neutral prose pilots are retained; [v2](NEUTRAL_FUNCTIONAL_PILOT_V2_RESULTS.md)
improves checked attribute accuracy but fails coverage and omits model fields needed
for focal/background and unattended-availability reports. The [v3 protocol](NEUTRAL_FUNCTIONAL_PILOT_V3.md) now supplies current allocation,
unattended recovery and command-selection forecasts, with prepared inputs and
pre-data process-claim checks. Its [auditor qualification](NEUTRAL_FUNCTIONAL_PILOT_V3_RESULTS.md) fails 2/26 fixtures,
so no v3 reports are generated. The [v4 amendment](NEUTRAL_FUNCTIONAL_PILOT_V4.md) preserves its reporter inputs
and requalifies an auditor with explicit missing-information semantics. Its [qualification](NEUTRAL_FUNCTIONAL_PILOT_V4_RESULTS.md) fails 1/28 fixtures,
so no v4 reports are generated. The [v5 protocol](NEUTRAL_FUNCTIONAL_PILOT_V5.md) now separates process and
category audits, keeping reporter inputs unchanged. Both qualifications pass and all 36 report attempts complete without retries.
Separate source-blind audits and independent rubric review remain pending;
none of the six stages is complete in full. The updated preprint retains the September 27 study and incorporates the later
specificity, self-label and neutral-pilot limitations.
