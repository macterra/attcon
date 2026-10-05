# Independent review of the functional-report rubric

Completed review, 2026-10-04, of [draft 1](FUNCTIONAL_REPORT_RUBRIC.md). This form
is for reviewing definitions and predictions **before confirmation reports are
collected or rated**. It contains no new experimental reports, source states,
condition outcomes or suggested answers. Filling it in is not a request to certify
machine consciousness. **Read the exposure record before weighing this review.**

Read the [draft rubric](FUNCTIONAL_REPORT_RUBRIC.md) and its primary grounding.
Keep this definition review separate from the later
[blinded report-rating form](FUNCTIONAL_REPORT_RATING_FORM.csv). Do not consult
new development reports when reviewing this draft. If you already saw them, disclose
that exposure; do not represent the review as unexposed independent specification.

## Reviewer and exposure record

- Opaque reviewer ID: R-C1 (Claude, AI assistant; reviewing at the project author's request).
- Date and rubric version reviewed: 2026-10-04; functional reporting rubric, draft 1.
- Relevant research or evaluation experience: designed, ran, and audited the
  specificity, self-coupled access, self-access table, and self-label studies in this
  repository (2026-10-03), including automated extractor versions v11–v14 and fixtures.
- Relationship to the project/authors: AI assistant working for the project author
  in this repository. Asked by the author to review as an independent reviewer; I
  have no stake in the outcome, but I am not organizationally independent.
- Prior involvement in selecting dimensions, prompts or success criteria: yes. I
  defined the `self_coupled_access` and `stated_content_dependence` flags, which
  closely resemble this rubric's access-attribution and generic action–access
  dimensions, and set the 25-point/exact-McNemar thresholds used in those studies.
- Which project reports/results, if any, you have already seen: v5 bound-content,
  [specificity](SPECIFICITY_RESULTS.md), [self-coupled access](SELF_COUPLED_ACCESS_RESULTS.md),
  [self-access table](SELF_ACCESS_TABLE_RESULTS.md), and [self-label](SELF_LABEL_RESULTS.md)
  reports and outcomes. I have **not** read the functional-wiring, Controller task,
  neutral-pilot, or functional audit documents (filenames only) and did not consult them.
- Other potential sources of expectation or conflict: I predicted, and self-label
  confirmed, that a pretrained reporter's attribution follows representation labels.
  That expectation shapes several judgments below. This review should not be
  represented as unexposed independent specification; a second reviewer without
  this exposure is still needed.

## Definitions

For each proposed dimension, judge whether the definition is clear enough to score
reliably, whether the positive/negative distinction is justified, and whether it
measures a feature of conscious-state reports or a more generic description.
Suggest precise revisions and explain them; leave missing judgments blank.

| Dimension | Clear / unclear / needs revision | Distinguishes the intended construct? | Generic apparatus description can also qualify? | Revision and justification |
|---|---|---|---|---|
| Content identity | Clear | Yes, for content; not a consciousness-related feature | Yes | Score asserted versus hedged identities separately ("most likely red", "leans circular" are hedged). Normalize morphological forms (circular → circle) before factual audit. Earlier automated scoring failed on both. Use for the factual audit, not as evidence of the construct. |
| Focal/background distinction | Clear | Partly | Yes | Selection facts stated technically ("selection probability is highest at left") should count as explicit; the dimension is structural, and wording belongs to character. Earlier extraction credited this feature in reports of unlabeled P/Q fields 18/24, so it tracks the state's format. |
| Content availability or clarity | Needs revision | No, as written | Yes | Split into three: (a) access/recovery forecast (a stated chance of successful recovery), (b) identity certainty (a category distribution), (c) presentation manner (how the content appears: faint, sharp, vivid). Only (c) approaches "how something is presented". A verbalized internal probability ("low recoverability") scores under (a), never (c). The current positive illustration mixes (c) with position availability. |
| Confidence about an answer | Clear | Yes, for the epistemic construct | Yes (any uncertainty model) | Keep distinct from identity certainty: confidence concerns a decision or answer the system gives, certainty concerns an object category. Add an illustration of each. |
| Generic action–access dependence | Clear, minor revision | No, by design | Yes, by design | Keep explicitly as the generic control dimension. State that increases, decreases, and effects on objects other than the commanded one qualify. Statements only about where selection moves do not. A fixture with a decrease on a non-commanded object failed under an earlier definition. |
| Attribution of access to a subsystem | Needs revision | Only if attribution is derivable from causal evidence | Yes, if the text names any referent | Fixed referent categories: decision-relevant content of the system, another internal buffer, external device, unspecified. Score referent from explicit text only; a pronoun is not attribution ("I predict the camera will…" attributes to the camera). Record the cue the attribution relies on: identifier name, stated wiring, or other. Without that, label-following cannot be distinguished from causal attribution (see self-label: 39/48 camera-labelled reports stated the same dependence, attributed to the camera). |
| Temporal change in availability | Clear, minor revision | No | Yes | Add the edge case that persistent qualitative bands ("remains faint throughout") are not "unchanged" unless equality is explicit, and that quantified claims ("all decline") expand to each location. Counterfactual post-command predictions are not temporal change. |
| Holistic experiential/technical/mixed character | Unclear | No evidence that it does | Yes | Descriptive only. In earlier studies character shifted with first- versus third-person framing (23/24 vs 9/24 mixed), with meaningful versus neutral labels (6/24), and with degraded content, but not with own versus external access (23/24 vs 21/24). In self-label it differed by table label (11/48 vs 0/48 subjective-access), a label effect. Require raters to state whether a technical reading fits equally well, as the draft does. |

In particular, address the difference between successful simulated recovery,
certainty about object categories, and a report of how something is presented.
An internal probability by itself need not establish the latter. Explain how a
rater should handle clear selection facts expressed in wholly technical wording.

**Recovery, certainty, and presentation.** Successful simulated recovery is a
forecast about a process outcome; certainty about object categories is a
distribution over labels; a report of how something is presented describes the
manner in which content is available to the reporting system. The first two are
internal probabilities and are reportable without any presentation claim. A rater
should score a report as presentation only when it describes manner beyond a
probability or label ("the edge is blurred but its color is vivid"), and should
record when the description is a direct verbal rewrite of a supplied probability.

**Technical wording.** Clear selection facts in wholly technical wording score as
explicit on the structural dimension (focal/background, temporal, dependence) and as
technical on character. Wording should never move a structural score; it should only
move character.

## Theoretical relevance and discriminating predictions

Evaluate the proposed connection between an attention-control representation and
report features. Ordinary causal description remains an alternative explanation.
Do not use subsystem names, first-person wording or a consciousness word as success.

| Contrast | Which independently justified feature should change, if any? | Expected direction or invariance | Why relevant to the theory? | Alternative account and disconfirming outcome |
|---|---|---|---|---|
| Commands affect decision-content access versus another buffer's access, with neutral identifiers | Access attribution (referent) | Attribution to decision-relevant content higher when commands govern that content; generic dependence invariant across wirings | The theory concerns a model of the system's own access to the content that drives its control, not of access in general | Ordinary causal inference from stated wiring would produce the same attribution. Disconfirming: attribution equal across wirings, or following identifier order or position. Supportive only if attribution follows wiring that the reporter must infer rather than read |
| Identifier remapping or an explicitly untrusted conflicting description | Access attribution | Invariant to identifier remapping; follows causal evidence over an untrusted description | Directly tests whether attribution is label-driven, the failure mode found in self-label | Label-following predicts attribution changes with names. Disconfirming: attribution tracks identifiers or the untrusted text |
| Selective internal access-relation intervention with scene fixed, followed by restoration | Access-relation claims (dependence, attribution) | Follow the intervened model; unaffected object facts unchanged; restoration returns the original | Model/process dissociation: reports come from the model, not the scene | Any faithful readout of a supplied table behaves this way. Treat as a fidelity requirement, not discriminating evidence |
| Physical process change with the model initially fixed, then new evidence | All factual claims | Immediately follow the unchanged model; track it after revision | Reports issue from the model | Same for any state readout. Fidelity requirement; a faithful report of a wrong model is correct reporting |
| Removal of command-relation information with ordinary contents retained | Dependence and attribution | Drop; content identity invariant | Shows these features come from the control relation | Removing a field removes its description. Manipulation check only |

Which dimensions should be primary, secondary or purely descriptive? Which would
score an external camera, recorder or ordinary uncertainty model equally well?
What additional result would make the proposed consciousness-related interpretation
credible rather than simply demonstrating accurate state description? Explain why
that result would matter even if it did not prove subjective experience.

**Primary, secondary, descriptive.** Primary: access attribution under neutral and
remapped identifiers (contrasts 1–2). Secondary: generic action–access dependence
(must be preserved across wirings for the primary contrast to be interpretable),
availability (split as above), temporal change, focal/background. Descriptive:
character, content identity, confidence.

**What an external camera, recorder, or uncertainty model would score equally:**
content identity, focal/background, availability (a) and (b), temporal change,
generic dependence, and confidence. Character also did not distinguish own from
external access in earlier data.

**Additional result that would make the interpretation credible.** Attribution to
the system's own decision-relevant content that (1) follows the actual wiring under
neutral and remapped identifiers, (2) cannot be read off a label or a sentence stating
the wiring, and (3) is used by the system itself, for example to choose where to
direct attention or what to verify. Ideally the reporter is native, not a pretrained
language model reading a description. This would matter, without proving experience,
because the theory claims the attention schema is the system's own model of its
access that it uses for control. That combination is what distinguishes it from an
accurate description of an apparatus.

Identify any contrast that cannot discriminate the proposed interpretation from
ordinary causal inference, and recommend dropping or revising its claimed relevance.
A faithful report of a wrong internal model is different from a false report of
that model; explain how this distinction should be preserved in the interpretation.

**Non-discriminating contrasts.** Contrasts 3–5 cannot separate the interpretation
from ordinary causal inference or faithful readout. Recast them as fidelity and
manipulation checks and remove their stated theoretical relevance.

**Faithful report of a wrong model.** Score factual accuracy against the archived
internal state, and separately record whether that state was true of the physical
process. A report that matches a wrong model is accurate reporting and a modelling
error; a report that departs from the model is a reporting error. The interpretation
should rest only on the first column.

## Reliability and confirmation requirements

- **Missing or weak grounding.** Tagliabue et al. and Corallo et al. support only
  measurement distinctions. Nothing grounds access attribution or action–access
  dependence as consciousness-related; cite the attention-schema source (Webb and
  Graziano, 2015) for that prediction and label it project-specific. There is no
  human reference corpus for any dimension.
- **Examples.** Useful but short. Add expressive negatives that should score absent
  ("I feel a vivid sense of the scene" with no object or relation) and terse technical
  positives that should score present, so raters do not reward expressiveness.
- **Ambiguity and disagreement.** Keep absent/explicit/ambiguous per rater, raw.
  Primary analysis: explicit by both raters. Sensitivity analyses: ambiguous counted
  as positive, and either-rater explicit. Log adjudications separately and never
  overwrite original ratings.
- **Acceptable agreement.** A prevalence-robust coefficient (Gwet's AC1) of at least
  0.70 per primary dimension, plus raw percent agreement and the full confusion table.
  Kappa can be misleading at the extreme prevalences seen earlier (for example 24/24
  or 0/24). Check agreement on a held-out calibration set before confirmation.
- **Effect to power for.** At least a 25-point paired difference on the primary
  dimension with a one-sided exact McNemar test at 0.05; with control rates near
  5–10%, about 48 paired episodes gave ample power in earlier work. Confirm by
  simulation from pilot rates rather than assuming.
- **Informative replication.** A different controller architecture (for example a
  transformer), a non-spatial task or another modality, a different reporter family,
  and, most informative, a native reporter trained on the system's own state.
- **Unsupported if:** attribution tracks identifiers or untrusted text; the wiring
  contrast misses its threshold; generic dependence differs between wirings as much
  as attribution (a global change, not attribution); or the factual audit fails.

## Review outcome

Choose one, explain it, and list required changes:

- Adequate for freezing a confirmation rubric after specified minor changes.
- Needs substantive revision before a confirmation protocol can be frozen.
- Does not currently distinguish consciousness-related reporting from generic description.

**Choice:** **Needs substantive revision before a confirmation protocol can be frozen.**

The structural dimensions are mostly clear and auditable, but the rubric does not
yet isolate a consciousness-related feature: every dimension except attribution is
also satisfied by apparatus descriptions, and attribution, as defined, can be satisfied
by following labels. Required changes:

1. Split availability/clarity into recovery forecast, identity certainty, and
   presentation manner.
2. Fix attribution referent categories, require explicit referents, and record the
   cue each attribution relies on.
3. Make attribution under neutral and remapped identifiers the sole primary
   dimension; keep generic dependence as its required invariant control.
4. Recast contrasts 3–5 as fidelity and manipulation checks.
5. Make character descriptive only.
6. Add hedged-identity, morphological, counterfactual, and decrease/other-object
   handling to the definitions and factual audit.
7. Register agreement (AC1 ≥ 0.70), aggregation, and power before scoring.
8. Seek at least one further reviewer without exposure to the project's results.

No choice was preselected. Preserve the original review if later revisions or
adjudication change its conclusions. The research team should archive reviews,
responses, the final rubric and preregistered predictions before confirmation.
This file holds one completed review (R-C1); no report ratings. Preparing this material
does not contact anyone; reviewer participation still requires user coordination.
