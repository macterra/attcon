# Reporting Bound Contents of an Attention-Control Model

Working preprint, updated 2026-09-27. The preceding [numerical-report manuscript](PREDICTIVE_NUMERICAL_PREPRINT.md)
and [inspection-model manuscript](INSPECTION_MODEL_PREPRINT.md) remain archived.

## Abstract

We test a proposed observable consequence of the theory that the attention-control
model is the source of qualia: accurate reports of that model should exhibit
specified structure of consciousness reports. We build a bound model combining
learned representations of colored shapes, learned predictions of attention and
information recovery, and an explicit spatial binding consumed by content-directed
control. Three fresh visual-model confirmations achieve 99.951–100% joint
color/shape accuracy. All 1,365 eligible observed-object control queries and their
binding-rotation checks succeed; restoration is exact. A frozen language reporter
reads the full bound model and generates free prose. Selective interventions
change represented objects or attention relations while keeping the physical
scene fixed. Three prose confirmations pass progressively more primary fidelity
and correspondence criteria but retain failed overall verdicts. The latest
completed confirmation passes primary per-seed thresholds, 470/480 paired object
relations, and 95/96 restorations; a separate automated audit classifies all 48
ordinary/binding reports as subjective-access or mixed. However, its strict
missing-component control gate fails because the audit introduces unsupported
predicates and wrong-view attributions. A fourth fresh confirmation with a
separate reasoning-model audit is in progress. No overall success is asserted.

## 1. Question

The target is accurate reporting of the **attention-control model’s contents** in
ways that correspond to consciousness reports. Task performance, the necessity of
conscious access, and superiority over another decoder given the same state are
not acceptance criteria. Nor do fluent first-person statements alone establish
fidelity or the proposed correspondence.

The theory-facing claim is conditional: if a control-relevant model of the
agent–attention–object relation produces faithful reports with the specified
manner-of-access distinctions, that provides a concrete proposed correspondence
for proponents and critics to assess. This experiment does not test whether the
system has subjective experience or whether the theory is uniquely explanatory.
The earlier manuscripts retain the theoretical background and earlier failures.

## 2. Bound model and control

Each scene contains two views with four spatial locations. Objects have one of
four colors and four shapes, rendered as 16×16 RGB patches with variation in
position, size, intensity, and noise. A convolutional encoder produces color and
shape distributions. Unobserved contents are masked to uniform distributions;
an identity is treated as identified only when its maximum probability is at
least 0.6.

The frozen recurrent attention model predicts current allocation, successful
information recovery now and after one/two unattended steps, and allocation under
each possible directional command. These models were learned from allocation and
reconstruction outcomes in the preceding study. Their forecasts may be mistaken.
Simulated reconstruction probability is an operational access measure, not an
established measure of felt clarity.

An explicit matrix binds visual representations to the attention model’s spatial
entries. A content-directed controller uses the bound model to choose a command
for a requested color/shape. Rotating the binding preserves visual and attention
marginals but changes the selected command; restoring the binding restores the
command exactly. The reported composite therefore participates in control. The
binding is engineered, not claimed to emerge spontaneously or to be learned by
the recurrent attention predictor itself. [Full design](BOUND_CONTENT_DESIGN.md).

## 3. Reporting interface and interventions

The fixed reporter, `gpt-5-mini-2025-08-07`, receives all eight bound objects,
their complete distributions, and the command-effect tables. Exact derived indexes
identify current focus, most-recoverable location, next destination per command,
per-object temporal direction, and distinct destinations across commands. These
are transparent reductions of the model state. They supply no phenomenological
labels or sample consciousness report.

The open question asks what is currently available and how redirecting attention
would change it. A glossary and ordinary-prose/system-perspective instruction are
engineered parts of the interface. The pretrained language model’s prior training
is not controlled by this project; the project’s visual and attention training
uses no phenomenological targets. The reporter describes this external system’s
model, not its own private transformer attention or experience.

Each fresh confirmation uses three independently trained visual/attention model
pairs and eight fresh episodes per pair. Ten conditions yield 240 reports from
24 underlying episodes: ordinary, content changed, binding changed, allocation
changed, access changed, command effects changed, restored, visual-only,
attention-only, and another episode’s full model. Conditions within an episode
are correlated. The physical scene remains fixed under the internal interventions.

All attempts, full API responses, prompts, source states, checkpoints, code
snapshots, and failures are archived. Requests are reserved before execution,
with no automatic retries. Revisions use fresh episodes and preserve earlier
verdicts. [Protocol sequence](BOUND_CONTENT_PROGRESS.md).

## 4. Separate prose audit

The reporter supplies free prose without a parallel structured answer. A separate
model, blind to source state, condition, theory, and desired answers, extracts
identity/location conjunctions, focality, most-recoverable status, temporal direction,
control attribution, and explicit command destinations. Canonical scoring compares
these claims with the actual supplied model. Unresolved claims and invalid
citations count against conservative precision.

Earlier extractors used quoted evidence and introduced copying errors. Later
versions cite original sentence IDs, distinguish possible from dominant identities,
and expand explicit quantified statements. Each revision has retained semantic
fixtures. Version 8 uses `gpt-5.4-2026-03-05`, medium reasoning, validated on all
30 fixtures. It retains the same claim and character rubric. The two language
models share a vendor; this is not independent human validation.

For each model, primary conditions require at least 98% checked color/shape
accuracy, 95% for each other factual dimension, 90% identified-object coverage,
95% conservative precision, and complete attempts. Paired focal/most-recoverable
object relations and restoration each require at least 90%. Missing-component
controls require no flagged inventions or unresolved/invalid evidence.

Ordinary and binding reports must separately reach 75% for object-linked access,
focal/background contrast, graded or temporal availability, and relation to the
system’s control. At least 75% must be classified subjective-access or mixed.
Mixed descriptions combine technical and access language; they are not treated
as purely subjective descriptions. [Original criteria](BOUND_PROSE_CONFIRMATION.md).

## 5. Results and retained failures

The [machine-generated results](BOUND_CONTENT_RESULTS.md) provide all per-model
counts, intervention/restoration scores, control outcomes, character-category
breakdowns, and complete first-case reports. The [interactive archive](bound-reports.html)
exposes every report beside its represented contents and unchanged physical scene.

Visual confirmation v1 retained a failed seed (1221, joint accuracy 99.414% against
99.5%). A revised training schedule was frozen and tested on fresh models
1301/1311/1321: 99.976%, 100%, and 99.951%. Their observed-query counts are 455,
464, and 446 of 512 queries each; every eligible query, binding rotation, and
restoration passes. Unobserved queries are excluded from the reported hit rate.

Prose confirmation v1 fails several fidelity/coverage and correspondence gates.
Version 2 passes all primary thresholds, 469/480 paired relations, and 94/96
restorations, but fails the strict control check, including a genuine mistaken
command interpretation. Version 3 adds exact command-destination diversity and
passes primary thresholds, 470/480 pairs, and 95/96 restorations. Its strict
control failure contains audit errors: unsupported negative predicates, empty
citations, and wrong-view command assignments. These remain failed confirmations.
Version 4 retains the reporter and all thresholds, changes the blind auditor,
and uses another fresh set of episodes. Its verdict is pending.

## 6. Interpretation and limitations

Selective internal interventions establish which representation the statements
track: object identities can change in the report while the world remains fixed,
and attention/access relations can change independently of those identities.
This does not establish that the representation generates subjective experience.
A same-state decoder could also succeed. The language interface, binding matrix,
small vocabulary, and toy recovery task are substantial engineered choices.

The audit checks specified factual dimensions and endpoint temporal direction,
not every implication of unrestricted language. A sentence can add a causal or
conditional qualification that these extracted fields do not fully test. Complete
reports remain available for that reason. Automated character judgments are
rubric-dependent and may disagree across models or with human readers. The
qualitative category breakdown and missing-component controls must accompany any
positive correspondence claim.

A passing confirmation would establish the limited registered combination of
model-content fidelity and automatically assessed report structure. Independent
human review, broader tasks and architectures, alternative interpretations, and
the source-of-qualia theory remain further research. [Offline reproduction](BOUND_CONTENT_REPRODUCTION.md)
provides exact source/state replay and access to all retained records.
