# Reporting Bound Contents of an Attention-Control Model

Working preprint, updated 2026-10-04. The preceding [numerical-report manuscript](PREDICTIVE_NUMERICAL_PREPRINT.md)
and [inspection-model manuscript](INSPECTION_MODEL_PREPRINT.md) remain archived.

## Abstract

We test a proposed observable consequence of the theory that the attention-control
model is the source of qualia: accurate reports of that model should exhibit
specified structure of consciousness reports. We build a bound model combining
learned representations of colored shapes, learned predictions of attention and
information recovery, and an explicit spatial binding consumed by content-directed
control. Three fresh visual-model confirmations achieve 99.951–100% joint
color/shape accuracy; all 1,365 eligible observed-object control queries and their
binding-rotation checks succeed, with exact restoration. A frozen language reporter
reads the full bound model and generates free prose. After retained failed and
incomplete runs, fresh confirmation v5 meets every registered criterion across
three model pairs, 24 underlying episodes, and 240 reports. Checked color/shape
accuracy is 99.11–100%, identified-object coverage 99.06–100%, paired object
relations 461/480, and restoration 93/96. All 48 missing-component controls pass
the strict check. A separate, source-blind reasoning model finds all four specified
report-structure features in 48/48 ordinary/binding reports and classifies nine
as subjective-access, 36 as mixed, and three as technical-process. Mixed labels
also occur in 17/24 visual-only controls, so the character label alone is
nonspecific. The result establishes the registered limited combination of
model-content fidelity, counterfactual following, and automated report structure
within this engineered system. It does not establish subjective experience or
prove the source-of-qualia theory. Subsequent matched-label controls show that the
original structural features also occur in external-device descriptions. Explicit
self-access reporting replicates (26/48 versus 3/48 independent-access controls),
but the same table produces similar dependence descriptions under a camera label
(41/48 versus 39/48). Self-attribution therefore follows supplied labels in that
interface. The broader theory-facing interpretation remains unresolved.

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
fixtures. Versions 8–10 use `gpt-5.4-2026-03-05`. After medium-reasoning calls exhausted
their token allowance, version 10 uses low reasoning and 16384 output tokens,
validated on 31 fixtures. It retains the same claim and character rubric. The two language
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
and uses another fresh set of episodes. Its audit stopped after two token-limit
failures, so it cannot establish combined success. Version 5 uses another fresh
set, low audit reasoning, and 16384 output tokens. All 240 reports and 240 audits
complete, and every registered gate passes.

In that final confirmation, checked focality is 98.24–100%, most-recoverable and
temporal-direction attribution are 100%, control attribution is 95.39–99.75%,
and explicit command destinations are 100%. Conservative precision, including
unresolved or invalid evidence, is 99.18–99.96%. Paired object relations reach
461/480 (96.04%); restoration reaches 93/96 (96.88%); all 48 missing-component
cases pass. These rates are across extracted assertions in the predefined
conditions, not independent episodes or guarantees about every sentence.

All four specified correspondence features are evidenced in each of 48 ordinary/
binding reports. Nine are classified subjective-access, 36 mixed, and three
technical-process. The broad category is not specific: 17/24 visual-only reports
also receive a mixed label. However, those controls show no graded/temporal-access
or agency-relation evidence, one focal/background contrast, and two object-linked
access instances. The registered conjunction of fidelity and structure therefore
provides the more specific result; a character label alone is insufficient.

The first registered episode illustrates state following. Before intervention,
the report identifies the attended left object in view A as a blue square. With
only the binding changed, it identifies the attended left object as a red cross.
The attention allocation and physical scene remain fixed. The full paired reports
and every other case are available in the results and viewer.

### 5.1 Subsequent specificity and self-access studies

The [specificity study](SPECIFICITY_RESULTS.md) presents equivalent state structure
under neutral fields and as an external spotlight/camera. The originally specified
features persist; mixed character is common in both system and camera descriptions
(23/24 versus 21/24). The v5 structural success is therefore not specific evidence
for an attention-control model as the source of qualia.

In the [self-coupled access study](SELF_COUPLED_ACCESS_RESULTS.md), content certainty
depends on modeled recovery, but inspected reports never state that dependence
(0/24 in both coupled and matched decoupled conditions). Adding an explicit learned
own-content-by-command table produces self-coupled descriptions in 7/24 reports
versus 1/24 in the independent-content control, meeting the frozen threshold in
the first interpretable run, v3. Earlier uninterpretable runs and extraction
problems are retained. [Table study](SELF_ACCESS_TABLE_RESULTS.md).

A [larger replication](SELF_LABEL_RESULTS.md) produces self-coupled descriptions in
26/48 coupled-table reports versus 3/48 independent-table controls (registered
one-sided paired exact p = 2.8 × 10⁻⁶). However, identical tables labelled as an
external camera yield similar stated content dependence: 41/48 for self-labelled
and 39/48 for camera-labelled tables. Self-attribution changes with the label.
Subjective-access character occurs in 11/48 self-labelled and 0/48 camera-labelled
reports; that contrast is descriptive, not a registered primary result. Apparent
report character must therefore be evaluated alongside label and framing controls.

These follow-ups retain accurate state description and demonstrate reporting of
an explicitly represented dependence. They do not establish that the system's
functional organization, rather than its verbal description, explains the
consciousness-related report features. These trials share scenes within conditions;
the quoted counts are paired reports, not independent samples for every condition.

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

The pretrained reporter is external instrumentation and is distinct from the
action-selecting Controller. Its full readout access is not a claim about the
Controller's information access. The related MSTC comparison inquiry remains
separate: recurrent diagnostics establish causal predictive memory but do not
identify a comparison medium for corrective Modeler updates. Neither a separate
reporter nor a control-relevant binding establishes that architecture.

The passing confirmation establishes the limited registered combination of
model-content fidelity and automatically assessed report structure. Independent
human review, broader tasks and architectures, alternative interpretations, and
the source-of-qualia theory remain further research. [Offline reproduction](BOUND_CONTENT_REPRODUCTION.md)
provides exact source/state replay and access to all retained records.

The [current plan](CONSCIOUSNESS_INTERPRETATION_PLAN.md) addresses the remaining
interpretive gap with identifiable functional wiring, neutral factual reporting,
selective model/process interventions, and independent human assessment. A
[causal-wiring fixture](FUNCTIONAL_WIRING_RESULTS.md) and
[learned-model integration](FUNCTIONAL_MODEL_RESULTS.md) are development milestones;
they provide no new phenomenological confirmation. The
[neutral prose pilot](NEUTRAL_FUNCTIONAL_PILOT_RESULTS.md) completed 36 development
reports; verified attribute errors leave the new interface's fidelity unestablished.
It is not a powered confirmation. Independent rubric review and human ratings
have not been obtained. A learned reporter is one possible method, not a guaranteed
route to resolving subjective experience.

A [named-distribution pilot v2](NEUTRAL_FUNCTIONAL_PILOT_V2_RESULTS.md) completes
36 reports and 36 source-blind audits after 14 semantic fixtures pass. Checked
attribute accuracy reaches 100% in the primary presentation groups, but coverage
of identified output attributes is only 71.11–84.44% in three groups, below the
90% feasibility minimum. A model-intervention report overgeneralizes across buffers.
Automated character judgments are 30 technical and six mixed. The neutral interface
also needs current allocation and unattended recovery forecasts to test the proposed
focal/background and changing-availability dimensions. This development result
does not establish the broader theory-facing interpretation.

A [complete-interface development protocol v3](NEUTRAL_FUNCTIONAL_PILOT_V3.md)
adds current model allocation, unattended recovery and command-selection forecasts.
Its [source-blind auditor qualification](NEUTRAL_FUNCTIONAL_PILOT_V3_RESULTS.md)
passes 24/26 semantic fixtures but confuses unspecified selection with an asserted
absence of identified model selection in two fixtures. The frozen gate halts
report generation; this version produces no new language-report evidence. A
[v4 auditor amendment](NEUTRAL_FUNCTIONAL_PILOT_V4_RESULTS.md) fixes those failures
but fails a confidence-versus-recovery fixture (27/28 pass) and also halts before
report generation. These retained failures constrain the use of machine judges.

A [separately audited complete-interface pilot v5](NEUTRAL_FUNCTIONAL_PILOT_V5_RESULTS.md)
passes categorical/process qualifications (14/14 and 20/20) and completes 36 reports
plus 72 source-blind audits. Coverage improves, but the restored group fails the
frozen conservative-precision (89.46%) and process-coverage (72.22%) minima. Forty
selected positions are duplicated into the wrong extraction field, including thirty
in restoration; their values match the state, but the frozen contract rejects their
addresses. Other failed fields include six verified categorical contradictions,
eight unsupported positive readout-selection claims, and seventeen null-valued
readout-selection assertions conflated with missing forecast information. All scores
and actual outputs remain retained. Machine character ratings are 28 technical,
seven object descriptions, one mixed and none experiential. This development result
requires evaluator repair, further fidelity work and independent review; it provides
no new consciousness-related confirmation. Reporter and Controller remain separate.

A [typed process-audit qualification v6](NEUTRAL_FUNCTIONAL_PILOT_V6_RESULTS.md)
passes 31/32 fixtures and eliminates malformed addresses in those responses, but
confuses an unstated selected position with absence of a supplied forecast. The
all-fixtures gate halts before generating any row-3 reports. This is another retained
measurement limitation, not a new consciousness-related result.

An [explicit forecast-evidence amendment v7](NEUTRAL_FUNCTIONAL_PILOT_V7_RESULTS.md)
passes 33/36 fixtures but misaddresses one explicit readout claim and adds unsupported
unknown-selection assertions in two presence/omission fixtures. It also halts before
reporting. Repeated instruction amendments have not yet established semantic audit
reliability. Typed schemas alone do not resolve the problem. A blank independent
rubric-definition review form is prepared; no completed human review exists.

A [frozen learned three-way engineering study](FUNCTIONAL_CONTROLS_RESULTS.md)
trains three fresh forecast models with independent automatic schedules and no
condition labels as inputs. All nine model/condition cases pass their static
allocation, access, command-effect and control-mask minima. All nine process pairs
per model compare feedback and no-feedback predictions against the same physical
next-step target, including unchanged controls; 26/27 routes pass. Seed 2021's
buffer 1→neither route reaches 504/512 correct control masks (98.4375%), below
the registered 99% minimum. Its average effect/recovery metrics pass, but the
overall engineering verdict fails. Every attempt and raw tensor is retained;
checkpoints, scores and traces replay exactly. This closes the missing learned
third-condition and unmatched-comparison gaps while exposing an adaptation
limitation. It generates no language reports and supplies no new consciousness
confirmation. Faithfully reporting a wrong internal model remains distinct from
misreporting that model or accurately describing the physical process.

A [three-way visual/neutral interface export](FUNCTIONAL_INTERFACE_RESULTS.md)
connects those models to existing visual encoders and the explicit categorical
uncertainty bridge. Joint visual accuracy on 512 fresh scenes per pair is
99.9023–100%. It archives 664 inputs across all three controls, presentation and
model interventions, feedback/no-feedback revision contexts and targeted retained
mistakes. Observed and anticipated histories remain separate; no condition or owner
label enters the payload. Fourteen relevant tests and exact payload/tensor replay
pass. No prose is generated, and the parent engineering failure remains unchanged.
The proposed [manual factual-audit procedure](FUNCTIONAL_MANUAL_FACTUAL_AUDIT.md)
separates quoted propositions, source-aware judgments and required coverage; it
remains an unqualified development draft with no human annotations. Independent
rubric review and theory-facing confirmation are still outstanding.

A [review-materials packet](FUNCTIONAL_REVIEW_PACKET.md) separates rubric-definition
review, 28 synthetic source-aware factual cases, and all 36 unchanged v5 development
reports. Proposed synthetic answers remain unverified author expectations. One
model pair is reserved from reviewer calibration for a later whole-prose check;
the reports were already inspected by the research team, so this is not fresh
confirmation. Author review is coordinated but no substantive review or annotation
has been received. All current functional conditions contain an attention-control
model: an own/external wiring contrast needs an independently justified theoretical
prediction, rather than being taken as a direct test of substrate presence.

## 7. Reproducibility

All 185 tests pass. Offline replay verifies all seven visual-model runs, all
archived report source states and prompts, complete API responses, and the
explicitly stopped audit. Checksums and software versions are recorded in
`audits/bound_content/verification.json`; token accounting and the full test log
are adjacent. The previous numerical and inspection-model verifiers also pass.
The [reproduction instructions](BOUND_CONTENT_REPRODUCTION.md) rebuild the public
results and report viewer without API calls.

The 185-test count describes the original reporting-study release. Subsequent
study-specific checks and retained fixture failures are recorded in their own
protocols and result pages. The functional-model verifier exactly recreates its
physical and represented tensors and all 24 neutral example records. No combined
updated test-suite count is claimed here.
The new three-way study has eleven relevant passing tests and exact checkpoint
replay of every saved static/revision tensor, common target, metric and gate.
Its failed engineering verdict remains unchanged by successful verification.
