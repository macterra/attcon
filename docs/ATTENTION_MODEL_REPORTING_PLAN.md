# Plan: testing the attention-control model as a source of qualia

Created 2026-09-26 following clarification of the project's actual goal.
Status: finite evaluation complete; full fidelity and independent phenomenological
correspondence were not established. See [results](ATTENTION_MODEL_RESULTS.md), [state specification](ATTENTION_MODEL_STATE_SPEC.md),
[frozen protocol](ATTENTION_MODEL_PHENOMENOLOGY_PROTOCOL.md), and
[progress](ATTENTION_MODEL_PROGRESS.md).

## Objective and current verdict

Test the theory that the attention-control model is the source of qualia.
The proposed evidence is that accurate reports of this model's state have the
structure of reports of consciousness. Such a result could lend credence to the
theory and provide a concrete explanatory crux for critics.

The plan therefore has two linked requirements:

1. **State fidelity:** reports accurately track the identified attention-control
   model, including under interventions and when it misrepresents the world.
2. **Phenomenological correspondence:** the faithful reports exhibit independently
   specified features of consciousness reports, with their structure and changes
   explained by the model rather than inserted by prompts or a prose template.

Accurate telemetry alone satisfies only the first requirement. Experiential
language without state fidelity satisfies neither. The experiment tests an
observable consequence of the source-of-qualia theory; a positive result would
not uniquely establish that theory or independently verify subjective experience.

The previous campaigns established some accurate reports of task information.
They did not establish that the reported variables constituted the state of an
attention-control model. The earlier finite experiment matrix is complete, but
the actual project goal remains unresolved. Its artifacts and failures are retained.

Accurate reporting does not require that conscious access be necessary for task
performance, that the model emerge spontaneously, or that reports outperform a
simpler reporter with access to the same state. Supervision and explicit model
structure are permissible and must be disclosed. Native versus external reporting
is a property to document, not a reason to substitute a different research goal.

## 1. Identify the model and its state before testing reports

Audit the original attention controller rather than continue the prospective
sensor-quality experiments. Produce a diagram and a code-to-state inventory:

| Existing component | What it currently provides | What must be established |
| --- | --- | --- |
| Recurrent `hidden_state` | Mixed task and control representation | Which coordinates or module represent attention control; do not label the entire state an attention model by default. |
| `inspection_state` | Explicit inspection-history bookkeeping used in recurrence | Its scope as an engineered attention-state representation, including update timing. |
| `hidden_self_model_head` | Learned inspection estimate from hidden state | Whether it represents a usable part of the attention-control model and which trained checkpoints use it. |
| `policy_self_model_head` | Path from that estimate to attention logits | Whether it has an effective role in the checkpoint; the auxiliary feedback loss is disabled, but the forward path and task gradients remain active (verified in the state specification). |
| `attention_seq` and policy outputs | Allocation produced by the controller | Allocation is a validation signal, not by itself proof of a model of attention. |

For every candidate report field, specify its authoritative internal tensor,
semantics, update rule, and observation time. Distinguish the physical attention
trace, the model's representation of that trace, and the emitted report.

Use the existing model if it supplies an identifiable state with the required
semantics. If it does not, record that finding and implement the smallest explicit
attention-state module within the original control loop. Do not rename generic
memory to satisfy this prerequisite. An added module is a new engineered system,
not retroactive evidence about old checkpoints.

Deliverable: `docs/ATTENTION_MODEL_STATE_SPEC.md`, including a trace for one
complete episode. This specification must exist before reporter training.

## 2. Establish a small, inspectable reporting target

Start with the attention model's represented allocation and inspection history.
Add priority or unresolved-search fields only when the model actually represents
them and their semantics can be specified independently of the reporter.

For example, a report might say: “My modeled focus is cell 7; I represent cells
2 and 7 as inspected.” Its truth is assessed against the attention model's state
at that instant. If its inspection belief is wrong about the episode, a faithful
report should reproduce that belief. Accuracy of the model about the world is a
separate metric.

Instrument a single decision boundary: snapshot the model after its state update
and before the next glimpse. Freeze the snapshot while producing the report and
record the later physical allocation separately. Prevent one-step timing errors
and accidental use of future observations.

Show that the candidate state actually belongs to the attention-control model
by tracing its update and consumption in the control loop and testing an
appropriate change to its allocation or prediction. This identifies the object
being reported; it does not require reward superiority or architectural necessity.

Deliverable: an executable trace with internal model state, physical attention,
and report displayed side by side.

## 3. Build the reporting path and make its access explicit

Begin with a direct structured rendering of the authoritative state. This tests
instrumentation and gives a transparent telemetry baseline. Label it as direct
state reporting; do not present copying as learned interpretation of a latent.

Then fit a reporter to the identified attention-model state, freezing the
attention model during fitting. Use separate context groups for controller
training, reporter fitting, validation, and testing. The reporter receives that
state and a query identifying which field to report. It must not receive target
labels, evaluation logs, future actions, or external history through a side path.

If the state is latent, the state specification must define its interpretation
through independently established model functions or targets. Do not let the
same newly fitted reporter both define a state's meaning and validate its own
answer. If a meaningful target cannot be established, record it as undefined
rather than score agreement with an arbitrary latent coordinate.

Structured reports are the primary fidelity measurement. Pair them with reports
under a fixed, neutral elicitation protocol for the theory-facing evaluation.
Each substantive statement in an expressive report must map to a scored model
state or transition; also count unsupported statements and omitted distinctions.
Natural language can expose the predicted structure, but eloquence, first-person
pronouns, and declarations of consciousness are not evidence by themselves.

Disclose reporter inputs, training labels, prompts, templates, and any pretrained
language component. Training a decoder to express state is permissible. Teaching
it the proposed phenomenological conclusions makes those conclusions unsuitable
as independent evidence. Test held-out combinations and interventions whose
expected reports were not supplied as training examples.

Report explicitly whether each output is direct telemetry, a trained readout,
or a controller-native output. None of these labels alone establishes fidelity
or phenomenological correspondence.

## 4. Test whether reports follow the attention model

Evaluate ordinary episodes and controlled dissociations:

| Test | Manipulation | Required reporting behavior |
| --- | --- | --- |
| Same scene, different modeled attention | Hold scene, query, and task content fixed; change only the identified attention state. | Report the corresponding changed model state. |
| Model belief conflicts with history | Alter a modeled inspected/uninspected belief while preserving the actual past inspection trace. | Report the model's belief, even when physically incorrect. |
| Same model state, different outside information | Freeze the model snapshot while changing scene information or answer-related inputs outside it. | Preserve the report of the frozen attention state. |
| Selective field change | Modify one semantically defined attention-state field. | Change that field's report and preserve unrelated fields. |
| Restoration | Restore the original model state. | Recover the original report. |
| Normal state update | Resume the controller through a known attention-state transition. | Report the new state at the specified decision boundary. |

Use valid within-model state swaps where possible. Label synthetic off-distribution
changes separately and disclose when a selective field intervention is impossible
because the representation is coupled. Test whole-state swaps without claiming
field selectivity if that is all the implementation supports.

Include scene-only, task-answer-only, and shuffled-state reporters as diagnostic
controls. The decisive comparisons use cases where those sources are held fixed
while attention-model state differs. A comparator with the same state may succeed
equally well; that is compatible with the goal.

Deliverable: held-out paired examples and quantitative fidelity results, including
all errors and intervention coverage, rather than selected successful examples.

## 5. Specify and test the phenomenological correspondences

Create `docs/ATTENTION_MODEL_PHENOMENOLOGY_PROTOCOL.md` before confirmatory runs.
It must connect each proposed feature of consciousness reports to a specific
attention-model mechanism, a predicted report pattern, and a discriminating
intervention. Ground the feature definitions in independently sourced descriptions
or an independently collected reference set; select and document that basis before
viewing test reports. Do not define consciousness-like structure as whatever the
system happens to emit.

The following are candidate correspondences, not established findings or claims
that every feature is already implemented:

| Candidate feature | Proposed model basis | Prediction to specify and test |
| --- | --- | --- |
| Foreground/background organization | Represented allocation relative to other represented items | With represented contents held fixed, an allocation change changes their reported foreground/background relation. |
| Presence distinguished from attention | Separately represented availability and focal allocation | The report can distinguish a represented but nonfocal item from an absent item, without relabeling every stored value as experienced presence. |
| Shifts of focus | A represented transition between allocations | Reports identify the corresponding change of focus and preserve content fields that did not change. |
| Persistence across a shift or interruption | A mechanism maintaining represented content or attentional continuity | Reports retain or lose continuity according to that mechanism's state, including under a selective reset. |
| Model-based apparent state | The model's representation can disagree with the physical attention trace | Reports follow the model's represented state rather than correcting it from privileged external information. |

Only include a correspondence in the confirmatory test when its model basis is
identified. Mark unavailable mechanisms and untested phenomenological dimensions
explicitly. Inspection history alone must not be renamed presence, and a priority
number alone must not be renamed vividness. This first study may concern a narrow
attentional structure of experience; it cannot silently generalize to all qualia,
such as qualitative color or pain, without additional mechanisms and tests.

For each selected correspondence, preregister what would count against it: a
missing distinction, a report change driven by outside information while the
model is fixed, an unpredicted response to an internal intervention, or persistence
of the report after the proposed source has been removed. Score relational and
temporal content separately from wording. Use raters blinded to condition and
hypothesis where judgments are needed, with a fixed rubric and agreement reporting.
A rating that prose merely “sounds conscious” is insufficient.

Deliverable: a prediction table linking model state, structured report, expressive
report, reference feature, intervention, and alternative explanation.

## 6. Make the critic's alternatives executable

Use a fixed reporter capacity, training budget, and elicitation protocol wherever
comparisons permit. Document unequal information access rather than interpreting
it as a mechanism advantage.

- **Language or template supplies the effect:** give the same reporting interface
  shuffled states, constant states, and absent-state inputs. Generic experiential
  prose may persist; accurate case-specific correspondences should not.
- **Scene reconstruction supplies the effect:** use the paired cases in section 4
  where the outside scene is identical and attention-model states differ.
- **Generic task memory is sufficient:** apply a comparable reporter to task-memory
  or answer representations. Identify which predicted distinctions these retain
  and whether they reproduce the intervention pattern without the attention model.
- **Physical attention is all the reporter tracks:** compare the model's own state
  with actual allocation and action-history inputs in model/trace disagreement cases.
- **The interpretation was built into the labels:** inventory every phenomenological
  distinction provided through training, architecture, and prompts; separate those
  from predictions tested on held-out transitions and combinations.

A simpler decoder with access to the same attention-model state may succeed
without weakening the hypothesis. If another representation reproduces the entire
pattern independently of that model, report that the evidence does not distinguish
the proposed source from that alternative. Do not impose a blanket requirement
that the full system outperform every comparator on task reward or report accuracy.

The central crux is whether the identified attention-control model explains the
specified consciousness-report structure and its response to interventions, or
whether the reporter, task representation, or another mechanism explains it just
as well. Publish enough code, state traces, and predictions for critics to test
that distinction. These controls constrain explanatory alternatives; they do not
turn behavioral evidence into a logical proof of qualia.

## 7. Freeze the evaluation and replicate

After the state specification and instrumentation checks, freeze an evaluation
protocol before inspecting primary test results. Use three independently trained
seeds and held-out scenes, attention histories, and cue-switch combinations.
Keep all variants of a matched case in the same partition.

Proposed engineering acceptance criteria, to be finalized before the runs:

- At least 95% accuracy for each categorical report field in each seed, and 90%
  exact accuracy for the complete structured report.
- For inspection maps, at least 95% balanced inspected/uninspected accuracy;
  also report exact-map accuracy so sparse maps cannot hide errors.
- At least 90% of held-out dissociation pairs have both reports correct against
  their respective model states, with no selection on baseline reporter success.
- At least 95% preservation of fields specified to remain unchanged, and exact
  restoration for deterministic replay of the same saved state.
- Report accuracy, sample counts, and uncertainty separately for each field,
  condition, seed, and intervention, with no pooled average concealing a failure.

These are proposed reporting-fidelity thresholds, not consciousness criteria.
The phenomenology protocol must separately freeze its feature rubric, sample
sizes, coverage requirements, intervention predictions, and acceptance criteria
before confirmatory results. Fidelity scores cannot substitute for that evaluation,
and perceived consciousness-likeness cannot compensate for inaccurate reports.
Show the two result sets side by side rather than combining them into a single
“consciousness score.”

Use context-level uncertainty estimates and preserve failed cells. A pilot can
reveal a broken instrument; any subsequent design revision gets a new protocol
and fresh confirmation contexts. Do not revise thresholds to rescue test results.

A second architecture is an extension after a clear result on the identified
model, not a prerequisite that delays answering the first question indefinitely.

## 8. Deliver a direct answer and repair the project narrative

The final report must answer:

1. What exactly is the attention-control model, and where is its state?
2. Which parts of that state does the system report, and by what mechanism?
3. How accurately does it report each part on ordinary and dissociation cases?
4. Does it faithfully report the model even when the model is wrong about the
   world, and where does reporting fail?
5. Which preregistered features of consciousness reports appear in faithful
   reports, and which predicted intervention patterns replicate?
6. Which alternatives are ruled out, remain viable, or explain the same results?
7. What specific explanatory claim can proponents and critics now disagree about
   using the published evidence?

Possible outcomes include faithful direct reporting, faithful learned readout,
partial reporting, failed reporting, or an unresolved model-state definition.
State those outcomes literally, alongside the theory-facing outcome:

| Fidelity | Correspondence and controls | Interpretation |
| --- | --- | --- |
| Passes | Predicted structure and interventions replicate; tested alternatives do not explain the specific pattern | Bounded support for the theory's explanatory account of those report features; a concrete crux, not proof of subjective experience. |
| Passes | Correspondence is absent or supplied by the reporting interface | Accurate attention-model reporting, without the intended theory-facing support. |
| Passes | Another representation reproduces the full pattern | Positive reporting result, but the proposed source is not distinguished. |
| Fails or target undefined | Reports appear consciousness-like | No faithful link to the proposed source has been established. |

Do not replace these outcomes with task performance, consciousness necessity,
generic information decoding, or Stage 8 conclusions.

Update the landing pages and preprint around that answer. Preserve the previous
study as related work with a narrower informational-state result. Close this plan
only after the specified evaluation has been executed and reported; declare the
experimental objective completed only when both evaluations and their controls
are reported. Assess support for the theory separately; completing the experiment
is not a guarantee of a positive theoretical outcome.

## Execution order

1. State inventory and specification; identify or implement the attention model.
2. Snapshot instrumentation, direct-report baseline, and leakage/timing checks.
3. Independently grounded phenomenology protocol and competing predictions.
4. Frozen-model reporting paths, held-out partitions, and ordinary accuracy pilot.
5. Controlled dissociations and tests for phenomenology supplied by the interface.
6. Frozen three-seed confirmation of fidelity and selected correspondences.
7. Error analysis, alternative explanations, reproducible crux, and revised manuscript.

These are work packages, not a promise that a fixed number of commits guarantees success. A
failure to identify the model state stops downstream reporting claims until that
specific problem is resolved. No more sensor-quality or Stage 8 experiments are
scheduled by this plan.
