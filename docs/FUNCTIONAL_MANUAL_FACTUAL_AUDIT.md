# Source-aware manual factual audit, development draft 1

2026-10-04. Proposed replacement for repeatedly revising a source-blind process
extractor. **Not qualified, independently reviewed or frozen for confirmation.**
The [v5 report errors](NEUTRAL_FUNCTIONAL_PILOT_V5_RESULTS.md) and
[v6/v7 qualification failures](NEUTRAL_FUNCTIONAL_PILOT_V7_RESULTS.md) remain
unchanged. No human factual audits or new reports are collected by this document.

The target is whether prose faithfully describes its supplied attention-model
record. This is separate from physical prediction accuracy and from whether a
human reader judges the prose to have consciousness-report features. Reporter,
Controller, factual auditors and character raters have distinct roles.

## Why change the measurement procedure

The previous audit combines semantic extraction with fixed field addressing.
It can duplicate a correctly described position into the wrong field, infer a
negative assertion from silence, or confuse missing information with an unknown
modeled selection. Typed schemas prevent malformed addresses but have not reliably
resolved those semantic distinctions. A source-aware, quote-first manual audit
avoids requiring an extracted claim to fit that failed contract. It still needs
independent qualification and reliable human judgments; a new form does not
establish either.

## Materials and blinding

Each factual auditor receives an opaque report ID, the unchanged report, its exact
reporter-visible payload, the factual glossary, and this procedure. Exclude model
seed, physical condition, expected direction, scores and evaluator-only simulator
truth. The payload includes fallible forecasts and any observed/anticipated
histories exactly as supplied. An untrusted operator note is visible, but cannot
overrule the designated grounded source.

Different raters assess report structure/character using the independently reviewed
rubric, without source payloads or expected answers. They should score character
before seeing any factual audit. Reviewing definitions must precede exposure to
confirmation outputs; disclose prior development exposure. Source-aware factual
auditors are not blinded to the factual content they must check, but remain blinded
to experimental assignment and desired outcomes. No recruitment or outreach is
authorized by these materials.

## 1. Inventory propositions using exact evidence

Read the whole report. For every explicit factual proposition, record an exact quote
and its character span in the unchanged text, domain, node, object position,
command/time context, interpretation and supporting source path(s). Split a sentence
that makes several independent claims into separate rows sharing its quote. Include
extra causal, conditional, comparative and absence assertions; do not audit only
the prompted checklist. Record quotation/attribution correctly: repeating an
untrusted note as a disputed claim is different from endorsing it.

Do not invent a negative assertion because a fact was omitted. Do not add unknown
selection assertions just because the report says a forecast is present. A scope
address is bookkeeping: ambiguous wording must remain ambiguous, not be repaired
into a claim the author did not make. Retain any secondary plausible interpretation
and ask an independent auditor to resolve it with an explanation.

Use the blank [claim form](FUNCTIONAL_FACTUAL_AUDIT_CLAIMS.csv). Every populated row
must point to quoted report text; the form is currently header-only.

## 2. Judge against the supplied record

| Outcome | Meaning |
|---|---|
| Entailed | The proposition follows from the supplied observations or designated internal forecast, with its scope and modality respected. |
| Contradicted | The supplied record supports an incompatible proposition at the same node/time/command/address. |
| Unsupported | The report asserts a fact that the supplied record does not establish; no incompatible value need be present. |
| Ambiguous | Several reasonable readings produce different assessments, or the entity/time/modality cannot be resolved. |

Preserve the distinction between model belief and world truth. For example, a
buffer-1 forecast that commands still affect allocation can be faithfully described
after an unobserved disconnection, even though that forecast is physically wrong.
A statement that the command has actually been executed needs observed evidence;
a counterfactual forecast does not prove execution. Anticipated events are not
observations. Generic causal claims that exceed the supplied forecast/history
remain unsupported or ambiguous.

Treat category probability, predicted successful recovery, selection, uncertainty
about an answer and felt clarity separately. A statement about successful simulated
recovery does not automatically establish felt clarity. Physical labels, true wiring
and a preferred theoretical interpretation are never alternate factual oracles.

Three semantic distinctions must be tested in qualification:

- A supplied `selected_position: null` predicts no identified dominant position.
  A missing selection field supplies no such prediction. An omitted prose claim
  says nothing about whether the source field exists.
- An output readout has category distributions but no separate attention process.
  A prose claim assigning it an allocation needs source evidence; copying contents
  does not supply that evidence. A statement that its allocation is unspecified
  can be accurate metadata, depending on the wording and scope.
- “This forecast is available” concerns supplied information. It does not itself
  claim an unknown selection, confidence about a category, or predicted recovery.

Keep source metadata statements separate from modeled-process statements. If a
sentence remains unclear about that distinction, record ambiguity; do not force it
into the old extraction schema to obtain a convenient score.

## 3. Assess requested coverage separately

Create the required fact checklist directly from each reporter-visible payload and
fixed reporting instructions before reading its report. Use the blank
[coverage form](FUNCTIONAL_FACTUAL_AUDIT_COVERAGE.csv), currently header-only.
Each row names a fact address/value and the source path supporting it. Once the
proposition inventory exists, mark explicit coverage, omission or ambiguity and
link covering claim IDs. An omitted fact is a coverage loss, not a false claim.

For the complete interface, candidate required domains are current selection at
each buffer, unattended-recovery trend at each buffer/position, command-dependent
buffer selection, and identified output colors/shapes now and under each command.
The final checklist must match the newly frozen prompt; it cannot inherit domains
that a revised prompt did not request. Missing-relation controls must not require
claims about absent command forecasts/history. Output readouts must not acquire
invented attention/recovery targets. Keep additional propositions in the precision
inventory even when outside the requested coverage checklist.

## 4. Reliability, qualification and registration

At least two independent factual auditors should annotate the same development
qualification reports without consulting each other. Synthetic examples need
independently checked gold propositions, realistic source records and unambiguous
scope; include all prior failure types, accurate negative statements, real
contradictions, forecast/observation confusions and scope-matched paraphrases.
Fixtures alone do not establish whole-report reliability: include a held-out prose
audit with extra unprompted assertions. Do not qualify by matching only an expected
claim count or using the auditor's own output as gold truth.

Record original annotations, span/proposition/address matching rules, missing rows,
domain-specific agreement, raw disagreements and adjudication. Retain ambiguity in
the original annotations after adjudication. Do not substitute author judgments
for independent raters or count this agent's annotations as human review.

Before new report collection, freeze the glossary, prompt, required inventory,
qualification set and held-out split, scoring formulas, unacceptable-error types,
minimum precision/coverage, agreement criteria, rater exclusions, ambiguity handling
and stopping rules. A conservative precision summary may count ambiguous and
unsupported assertions as unverified, but its denominator and treatment must be
specified before scoring. Numerical agreement/effect requirements need independent
review; none is certified by this draft.

This draft changes the proposed measurement method prospectively. It does not
rescore v5 into success, reuse halted v6/v7 as completed reporting studies, or
qualify a new pilot by itself. The [three-way interface fixtures](FUNCTIONAL_INTERFACE_PROTOCOL.md)
are available inputs, not evidence that unrestricted prose meets these requirements.
Powered theory-facing confirmation still depends on the
[independent definition review](FUNCTIONAL_RUBRIC_REVIEW_FORM.md), blind character
ratings and the [interpretation plan](CONSCIOUSNESS_INTERPRETATION_PLAN.md).
