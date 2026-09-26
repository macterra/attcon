# Registered attention-model reporting and correspondence protocol

Version 1, 2026-09-26. Frozen before new report training and test analysis.

## Hypothesis and independently chosen reference

The user's hypothesis is that the attention-control model is the source of qualia.
The observable test concerns faithful reports of its state and consciousness-report
structure. Webb and Graziano (2015) propose that an internal model of attention
can ground subjective-awareness reports and allow model/process dissociations.
Their account also includes self/object relationships beyond inspection history.
[Primary paper](https://grazianolab.princeton.edu/sites/g/files/toruqf3411/files/graziano/files/webb_graziano_2015_reprint.pdf).

Lamme (2003) argues for distinguishing visual attention from awareness; this is
an independent reason not to equate an attention signal with awareness by label.
[Author's institutional record](https://dare.uva.nl/id/af484fea-9993-4995-bd30-05844513be12).

These references motivate the questions, not numerical human benchmarks. The
operational predictions below are our interpretations, registered here. No human
corpus, human phenomenology judgment, or population match is claimed. Reference
selection precedes test reports, and the source interpretation is not an empirical
validation of our engineering definitions.

## Scope and predictions

The [state specification](ATTENTION_MODEL_STATE_SPEC.md) identifies the learned
inspection model m used by the original attention controller. The new experiment
uses existing controller seeds 107, 207, 307, with no agent retraining.

| Candidate correspondence | Testable part in this model | What is not established |
| --- | --- | --- |
| Foreground/background organization | Rank one model-promoted location above the others; report changed preference after model interventions. | This allocation bias is not actual focus or phenomenal foreground. |
| Shifts | Correctly describe whether the model's preferred location changes between snapshots. | Experiential shifts are not labeled or independently measured. |
| Persistence | Track inspection belief retained across a normal update, and its removal under a belief flip. | Inspected history is not subjective presence or phenomenal continuity. |
| Model/process dissociation | Report m when it disagrees with actual inspection history; under intervention follow m, not physical trace. | Dissociation alone does not prove subjective experience. |
| Presence distinct from attention | No adequate state field identified. | Untested/unsupported; do not relabel memory or add a presence flag. |
| Qualitative character and self-attribution | No adequate mechanism identified. | Untested/unsupported; no generated claims of pain, color experience, or consciousness. |

The tested rows are narrow structural analogues. Full phenomenological
correspondence remains unsupported if these analogues are supplied entirely by
our state definitions/report renderer or if the required broader structure is
absent. Do not claim the source-of-qualia theory confirmed merely because all
operational scores pass.

## Data and reporter training

Generate 512 fit, 128 validation, and 256 test scenes per controller using fixed
RNG seeds 4100+s, 5100+s, 6100+s. Scenes are unique by full scene/cue-schedule
fingerprint across partitions. Each scene yields six pre-action snapshots. Half
have a cue switch at step 3 to the next type cyclically. Fit/validation switch
pairs are 0->1 and 2->3; test pairs are 1->2 and 3->0, with balanced unswitched
cues. Hold every snapshot and intervention of a scene in its partition. These
are fresh random scenes, not a proof that none ever occurred in original online
controller training (its episode list was not retained).

Fit four equal-capacity reporters from: m; recurrent task-answer logits from the
preceding action (zero at step zero); visible scene plus current cue; physical
inspection map plus preceding fixation. Pad to 128 dimensions. Two independent
MLP heads, 128->64 tanh->25 each: BCE inspected map, CE preferred cell. Fit-only
normalization, Adam .003, 1000 full-batch steps; select steps 250/500/750/1000 by
validation minimum of balanced inspected accuracy and preference accuracy, then
exact-report accuracy; earliest tie. Three controller seeds have deterministic
report-training seed 7100+s. Reporters are external supervised readers, not native
agent reports. All loss terms concern explicitly defined model fields, not
consciousness words. Unweighted BCE; retain sparse-class failures.

An additional generic-memory reporter receives h with the same architecture and
budget. Since m is computed from h, its success is an information-source upper
bound, not independent evidence against an attention model. A direct oracle
renderer reports exact m/W fields. No language API or unblinded human judge is used.

## Primary evaluation and controls

Report each seed separately, at each timestep and cue regime as diagnostics, as
well as whole-test metrics. Primary fidelity gates in every seed: preference
accuracy >=.95; inspected balanced accuracy >=.95; exact full report >=.90.
Also report positive/negative recall and exact inspection-map accuracy. Aggregate
scores do not conceal subgroup errors: no subgroup is promoted independently.

At step 3 evaluate all 256 scenes, without selecting baseline-correct cases:

- donor state via cyclic shift of test contexts;
- flip m[q] to 1-m[q], q=context index modulo 25, others unchanged;
- erase m to 0.5; label as synthetic, not absence of consciousness;
- restore original m, requiring exact deterministic restoration;
- keep m fixed and rotate external scene/answer/physical-trace inputs for the
  respective diagnostic reporters; the state reporter must remain unchanged.

Paired baseline+changed exact reports >=.90 for donor, flip, and erase; unchanged
belief fields >=.95 under flip; exact restoration; report every case and coverage.
Donor specificity uses a fixed cyclic donor pairing; learned reporter loss or
selection never sees test interventions. Causal identity: mean absolute allocation
probability change >1e-6 under donor override in each seed; record hard-choice
switches without requiring them. Actual pre-intervention trace is held fixed.

Also evaluate state reporter with shuffled m, zero inputs, and constant m=.5
against original states (information-loss controls) and against the supplied state
where meaningful (fidelity, not original truth). They may emit fluent templates;
case-specific accuracy should decline when relevant information is removed.

Operational correspondence gates: >=.95 preference-shift agreement on successive
pairs; >=.95 per-cell belief-persistence agreement; >=.95 accuracy specifically on
naturally occurring model/physical-trace disagreements, with counts and null if
none. Report erasure-induced preference/inspection changes. These are secondary
structural scores, not a consciousness scale. State identity and fidelity gates
remain separate from correspondence interpretation.

Uncertainty: 1000 scene-bootstrap draws for primary accuracy and paired exact
report scores, deterministic seed 8100+s; intervals conditional on each trained
system. Save all per-scene predictions, fields, interventions, errors, and selected
examples (first scene and first error by deterministic order, never a best case).

## Expressive reports and alternative explanations

A fixed neutral renderer says: “The attention model favors cell X. It represents
cells [...] as inspected.” Pair rendering adds whether preference shifted and
which beliefs persisted. It does not say “I am conscious” or “I experience.”
A separate labeled illustrative paraphrase may use first person, but contributes
zero independent evidence: its style and self-attribution are authored.

Parse renderer output back to fields and verify lossless round-trip. This is a
format check, not a phenomenology metric. Fluent identical syntax under shuffled
or constant states is an explicit language/template alternative. Scene/answer/
physical-trace reporters test reconstruction shortcuts. h reports test whether
readout of the generating latent is equally sufficient. If success is exhausted
by supervised field decoding and template relations, the theory-facing verdict
is **underdetermined**, even if attention-model report fidelity passes.

## Completion and failure handling

Run all three seeds and all registered conditions. No test-driven tuning or
replacement seeds. Save a replayable checkpoint archive, fixed settings, source
hashes, split fingerprints, complete scores, and examples. Deliver an error
analysis and a readable crux regardless of outcome. A failed fidelity gate is a
failed gate; a missing phenomenological mechanism remains unsupported. Do not
schedule indefinite new architectures until a positive result appears.
