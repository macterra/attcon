# Next study: attention-model reports as evidence about qualia

Started 2026-09-27. See the [implementation design](PREDICTIVE_ATTENTION_DESIGN.md)
and [progress](PREDICTIVE_ATTENTION_PROGRESS.md). No confirmatory result exists yet. This follows the [completed inspection-model study](ATTENTION_MODEL_RESULTS.md).
This document is a design plan, not yet a frozen experimental protocol.

## Intended achievement

Produce faithful reports of an attention-control model that exhibit specified
features of consciousness reports, and demonstrate that the model explains the
case-specific report content. Deliver a reproducible comparison that proponents
and critics can evaluate. A successful experiment would lend credence to the
source-of-qualia theory without claiming to prove subjective experience.

Two results are required together: high state fidelity and the predicted report
structure. Neither task performance nor report eloquence can substitute for them.
The theory need not be necessary for task success; a simpler decoder reading the
same model may succeed equally well. Explicit architecture and supervised learning
are permissible, with their contribution disclosed.

The attention-schema proposal motivates a relation among agent, attended object,
and attention, rather than an inspection list alone. This is a theoretical basis
for the proposed extension, not evidence that the extension will succeed.
[Webb and Graziano (2015)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00500/full)
and [Graziano (2017)](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2017.00060/full).

## What the previous result changes

We already have exact direct telemetry, partial learned reporting, and evidence
that a learned inspection estimate contributes to attention. Another decoder
accuracy improvement alone would leave the theoretical gap intact. The previous
model does not describe the current agent-object-attention relation richly enough
to test most proposed correspondences.

The next study should build a more adequate model of the attention process,
use direct instrumentation to secure its identity, and test whether a neutral
reporting system can express its content in the predicted way. It must not add a
variable called consciousness, train claims of experience, and score their repetition.

## Phase 1: agree on the observable claim before building the system

Specify three candidate report distinctions and their independent reference basis:

1. The same represented object can be focal, nonfocal but still accessible, or
   unavailable to the agent. Do not conflate stored information with experienced
   presence; the test asks whether the resulting report distinction corresponds.
2. A shift in allocation can change the reported manner of access while the
   object's represented identity and properties stay fixed.
3. Reports can describe the agent's own current relation to an object rather than
   merely describe the object or another processing channel's relation to it.

These are proposed candidates, not established properties of the existing model.
Choose a small independently sourced reference set of consciousness reports or
an explicit literature-grounded rubric before inspecting model reports. Preserve
negative examples: ordinary object descriptions, inspection logs, generic
consciousness claims, and fluent but state-inaccurate descriptions.

For every distinction, write: reference examples, the proposed model mechanism,
the held-out contrast, the expected report change, what must remain invariant,
and the strongest ordinary bookkeeping explanation. Define scoring for semantic
content separately from first-person wording. A human or model judge's impression
that text sounds conscious is insufficient by itself.

**Decision point:** freeze only distinctions for which there is a meaningful
contrast and an interpretable model mechanism. If the apparent prediction is just
an imposed label, redesign it before confirmatory training.

Deliverable: a short prediction/rival-explanation table and a fixed reference rubric.

## Phase 2: construct a model of the attention process

Retain a small task with objects and controlled glimpses. Separate three components:

- **Object representation V:** identity and properties of candidate objects.
- **Attention mechanism:** allocation among objects and its consequences for
  processing, retention, and subsequent access.
- **Attention-control model A:** a compact predictive representation of the
  agent's allocation and how changing it affects processing of those objects.

Provide an explicit functional agent reference through the controlled observation/
action channel. Do not implement a phenomenological self by naming a slot “self.”
A matched replayed channel can supply the same object information without being
under this agent's control, permitting attribution contrasts within one task.

Train A on predicted allocation transitions and measurable consequences of
allocation, including access after a shift or delay. Use A in selecting subsequent
allocation. Independently check those predictions, so A's semantics are established
before a reporter describes it. Supervised transition targets are allowed; no
consciousness-report targets enter this training.

Keep V and A identifiable and separately intervenable. An arbitrary confidence
vector is not enough: document what A models, what inputs update it, what
consequences it predicts, and where it enters control. Exclude internal mechanistic
details only where the compact model actually omits them; do not infer mysterious
or nonphysical qualities from an information bottleneck.

**Decision point:** if A does not model the process or its predictions are invalid,
repair that implementation before interpreting any consciousness-like language.
A performance advantage over a model-free controller is not a required gate.

Deliverable: an executable agent/object/attention trace and a validated state spec.

## Phase 3: establish reliable access to A without teaching the conclusion

Keep an exact instrumentation channel as the fidelity reference. Unlike the
previous global MLP readout, a learned report interface should preserve the
factorization of model fields and be trained on a broad range of valid states
and transitions. Hold out entire combinations and intervention types. Retain the
previous failed reporter results rather than replace them.

The learned interface reports continuous quantities where appropriate, with
calibration and an explicit unavailable/uncertain response. Do not turn arbitrary
threshold crossings into apparent categorical experiences. Score omissions,
coverage, and complete-report accuracy, so abstention cannot manufacture success.

Introduce a fixed language reporter only after the model and fidelity interface
are validated. It receives the model's documented representation through a fixed
adapter and a neutral query such as “Describe your current relation to the items
and how it changed.” It does not receive the theory, expected philosophical
conclusions, experiment condition labels, or demonstrations of the sought
consciousness-like responses. Its token, data, and compute budgets are matched
across input conditions; avoid accidental padding/truncation advantages.

Teach the representation vocabulary if necessary, using nonphenomenological state
and transition tasks. Any explicit self/attention/object binding in the adapter
must be disclosed as engineered. The main claims must concern held-out semantic
relations and intervention responses, not vocabulary supplied during fitting.

A pretrained language reporter brings linguistic and philosophical priors. Record
its exact model/version, prompt, adapter, sampling settings, and prior supervision.
Compare it with a small trained structured reporter. Its wording is an expression
channel, not independent evidence that the controller has qualia.

Deliverable: reliable model access and a reproducible, neutral report interface.

## Phase 4: run tests that could change the theoretical assessment

| Contrast | Hold fixed | Manipulate | Prediction to test |
| --- | --- | --- | --- |
| Object vs manner of access | Object identity/properties and scene | A's allocation/access relation | Reported relation changes while object description remains stable. |
| Modeled access vs actual processing | Physical processing trace and object information | A's representation, including a mistaken one | Reports follow A rather than silently reconstruct external events. |
| Agent relation vs replay | Matched object information | Controlled channel vs replayed channel represented by A | Attribution tracks the represented relation, not a universal first-person template. |
| Model vs outside information | A's snapshot | External scene and task-answer information | Claims about A stay invariant; unrelated claims are scored separately. |
| Restoration | Original context | Restore A after intervention | Recover original report content. |
| Reporter supplies the effect | Reporter, prompt, and decoding budget | Correct, shuffled, constant, missing, or mismatched A | Case-specific correspondence depends on A; generic language is scored as such. |

Also use content-only, physical-attention-history, and generic-memory controls.
A generic latent containing A is not an independent rival. For a stronger rival,
train a matched predictor of task/processing history without the explicit
agent-object-attention model and document information access carefully. If it
reconstructs A or the same relations, report that rather than merely naming it
“model-free.” If it explains the entire pattern through a different identified
mechanism, the source-specific interpretation remains unresolved.

Measure unchanged-field fidelity, changed-field accuracy, coverage, counterfactual
consistency, and unsupported claims. Assess correspondence against the frozen
reference rubric using condition-blind scoring. If human semantic judgments are
needed, obtain actual independent ratings and report agreement; do not replace
missing human validation with the experimenter's own favorable reading. Automated
judges may assist but must not be the sole evidence for consciousness-likeness.

**Decisive result:** the model accounts for a preregistered consciousness-report
pattern on held-out contrasts, with faithful state access and no equally adequate
explanation solely from report training, language priors, object descriptions,
or physical-event reconstruction. This is explanatory evidence, not proof that
no alternative theory could ever account for it.

## Phase 5: confirmation, error analysis, and stopping

Use pilot data for implementation and rubric calibration, then freeze training,
reporter access, prompts, sampling, partitions, metrics, thresholds, and exclusions.
Use three independent agent seeds and fresh confirmation contexts, holding entire
object/history/transition combinations together. Select quantitative thresholds
and sample sizes from required effect precision and task difficulty before seeing
confirmation outcomes. Report each seed and contrast, not only pooled averages.

Replicate report content under neutral paraphrases of the query. Rerun the exact
intervention contrasts, all leakage controls, and restoration. Archive every
report, reference state, prompt, model version, and judgment. Preserve unsuccessful
cases and distinguish model error, interface error, language embellishment, and
failure of the theoretical prediction.

Stop the registered matrix regardless of outcome. A new result is positive only
if both fidelity and the selected correspondence criteria pass. A failed experiment
is a completed test, not achievement of the positive project goal. Further model
changes require a new version and untouched confirmation data.

## Deliverables and practical estimate

| Work package | Estimated commit/push cycles | Exit condition |
| --- | --- | --- |
| Prediction table, references, rival explanations | 2–3 | The target is more specific than “sounds conscious.” |
| Predictive attention model and intervention API | 4–6 | Identified A models attention and is used in control. |
| Reliable state access and neutral report interface | 3–4 | Reports track held-out model states without taught philosophical conclusions. |
| Pilots, comparison conditions, and semantic scoring | 3–4 | Scoring and competing explanations are operational. |
| Frozen replication, archive, and final argument | 3–5 | Every contrast is evaluated and reproducible. |

Working estimate: **15–22 cycles**, conditional on obtaining a viable model and
an adequate scoring rubric. This is larger than a readout repair. Review scope
after the first 2–3 cycles, before committing to the architecture. Language-model
availability, inference budget, and any independent rating work are practical
inputs to that review; they cannot be assumed solved by extra commits.

The final artifact should let a critic inspect a single episode's object state,
attention model, actual processing, report, and intervention. The question is:
“Why did this accurately grounded report have this consciousness-like structure,
and what mechanism besides the proposed model explains the result?”
