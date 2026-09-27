# Reporting a Learned Attention-Control Model: Fidelity, Dissociation, and the Qualia Hypothesis

Completed evaluation, 2026-09-26. This manuscript supersedes the earlier
informational-state reporting paper. The [earlier study](INFORMATIONAL_STATE_RESULTS.md)
is preserved as related work; its data and experimental records are unchanged.

## Abstract

We examine the hypothesis that an attention-control model is the source of qualia
through a proposed observable consequence: accurate reports of that model's state
should exhibit relevant structure of consciousness reports. We distinguish state
fidelity from phenomenological correspondence. In three previously trained
recurrent controllers, we identify a learned inspection-history model whose output
contributes directly to attention selection. Fifteen matched reporters are fitted
on separate contexts and evaluated on ordinary states and controlled interventions.
State-based reporters achieve 95.78–96.79% balanced inspection-belief accuracy and
93.68–96.35% model-preference accuracy, but only 82.36–88.80% exact complete-report
accuracy, below the registered 90% criterion. On cells where the model disagrees
with physical inspection history, reports follow the model with 95.73–97.16%
accuracy. Donor-state swaps change the next attended cell in 16.41–21.88% of cases,
confirming the model's causal role. However, reporting generalization under
isolated belief flips is poor. Direct state rendering is exact by construction;
the trained reports and authored expressive interface do not independently
establish consciousness-like structure. The study identifies and partially reports
an actual attention-model state, improving the specificity of the proposed test,
but leaves the source-of-qualia hypothesis underdetermined.

## 1. Question and explanatory target

The theory being tested is that the attention-control model is the source of
qualia. The proposed evidence has two components: reports must faithfully track
that specific model, and their contents and changes must exhibit independently
specified features of consciousness reports. Generic task-memory decoding does
not answer this question. Nor does merely producing experiential language.

Webb and Graziano's attention schema account proposes that a simplified model of
attention can ground subjective-awareness reports and explain dissociations
between attention and its modeled state. Their proposal includes relationships
among self, attention, and represented objects. [Webb and Graziano, 2015](https://grazianolab.princeton.edu/sites/g/files/toruqf3411/files/graziano/files/webb_graziano_2015_reprint.pdf).
Lamme argues for separating attention from awareness, reinforcing the need to
avoid treating an attention measurement as awareness by definition.
[Lamme, 2003](https://dare.uva.nl/id/af484fea-9993-4995-bd30-05844513be12).

We use these sources to motivate a limited engineering test, not as a human
phenomenology dataset. The operational correspondences are our interpretations,
registered before the new report fits. A positive fidelity result alone cannot
settle the theoretical claim. Conversely, fidelity does not require superiority
over a simpler decoder with the same state, spontaneous emergence, or conscious
access to be necessary for task performance.

## 2. Identifying the model

The original Attcon benchmark is a cue-guided selective-search task on a `5x5`
grid. A recurrent controller allocates attention, reads a single cell through a
straight-through glimpse, receives task information, and updates its state.
Earlier work tested attention control and reporting of general task information.
The latter produced accurate sensor-quality and verified-content readouts, but
did not identify those variables as an attention-control model's state.

The current study returns to the original controller and identifies an existing
learned inspection model:

```text
hidden recurrent state h
    -> hidden_self_model_head -> sigmoid -> inspection model m
    -> policy_self_model_head(m) -> allocation contribution b

attention logits = policy_head(h) + b
```

The 25 values in m estimate which cells have been inspected. They were trained
with inspection supervision. The output b is consumed in actual attention
selection. This is a narrow learned model of attention history used for control;
it is not the entire recurrent state or a complete model of experience.

An inventory correction matters: a zero auxiliary policy-feedback loss weight
does not disable this forward path. Saved weights are nonzero, and the causal
audit confirms an effect on attention. The previous interpretation of that
configuration as disabled feedback was incorrect.

The tested outputs are m's thresholded inspection beliefs and the cell maximizing
b. We call the latter the **model-preferred cell**, not the actual focus: the
other policy contribution can determine a different final allocation. The
[state specification](ATTENTION_MODEL_STATE_SPEC.md) fixes this distinction.

## 3. Methods

### 3.1 Controllers and partitions

We use existing content-memory-v3 GRU controllers trained with seeds 107, 207,
and 307. Each has 32 recurrent units and the original grid task. The controllers
are frozen; the experiment trains new reporters, not new agents. Their previous
task results are known. Original checkpoint hashes and exact configurations are
recorded in the archived results.

Each controller receives 512 new reporter-fitting scenes, 128 validation scenes,
and 256 test scenes, each producing six pre-action snapshots. Half the scenes
switch cue at step 3. Fitting and validation use directed cue switches 0→1 and
2→3; testing uses 1→2 and 3→0. Unswitched cues are balanced. All snapshots and
interventions for a scene stay in the same partition. Full scene/schedule
fingerprints verify disjoint new partitions. The original online controller
training did not retain every scene, so accidental overlap with those historical
training scenes cannot be categorically excluded.

Inference uses `target=None`: no privileged correctness label enters the
controller's feedback in the new evaluation. Snapshots are taken after the
inspection estimate is computed and before the current glimpse. Physical
inspection history contains only preceding fixations. A read-only hook records
h for a diagnostic comparator without altering the original computation.

### 3.2 Report fields and fitting

The authoritative inspected belief for cell i is `m[i] >= 0.5`. The preferred
cell is `argmax(policy_self_model_head(m))`, with the normal lowest-index tie
rule. Their combination is the complete structured report: all 25 belief labels
plus one preferred location. Truth is the model's represented state, even if it
is inaccurate about physical inspection history.

Five equal-capacity reporter families receive m, previous task-answer logits,
visible scene/current cue, physical inspection history/preceding fixation, or h.
Each input is padded to 128 dimensions. Two 128→64→25 tanh MLP heads predict
beliefs and preferred location using unweighted binary cross-entropy and
cross-entropy. Normalization uses fitting data only. Adam at 0.003 runs for
1,000 full-batch steps. Validation selects among steps 250, 500, 750, and 1,000,
first maximizing the minimum of balanced belief and preference accuracy, then
exact-report accuracy, with earliest ties. Controller parameters remain frozen.

The registered primary criteria are at least 95% balanced inspected-belief
accuracy, 95% preferred-cell accuracy, and 90% exact complete-report accuracy in
every seed. Exact-map accuracy and separate positive/negative recalls expose
class imbalance. These are engineering fidelity thresholds, not consciousness
criteria. Direct rendering of m and b provides an exact instrumentation baseline.

### 3.3 Interventions and controls

At step 3, all 256 held-out scenes receive three interventions through the existing
model override hook: a donor state from a fixed cyclic pairing of test contexts,
a flip of one predetermined belief value from m[i] to 1−m[i], and neutralization
to m=0.5. The recipient scene, cue, hidden state at that point, and physical past
remain fixed. The model's changed contribution then enters attention normally.

We score both baseline and changed reports without selecting initially correct
cases. The paired exact-report criterion is 90%; preservation of unrelated
belief fields under a flip must reach 95%. Restoring the original computation
must restore reports and attention exactly. Donor states are observed elsewhere,
but may conflict with the recipient h. Flips and neutralization are synthetic.
The threshold convention classifies neutral 0.5 as inspected, which makes this
an especially artificial boundary case, not a state of “no awareness.”

Additional tests shuffle, zero, or hold model inputs constant; rotate external
reporter inputs while m stays fixed; and compare scene, answer, physical-history,
and hidden-state reporters. Because h generates m, that comparison does not
represent an independent system without an attention model. We record attention
probability changes and hard fixation changes, without requiring reward benefit.

### 3.4 Correspondence and expressive reports

The [registered correspondence protocol](ATTENTION_MODEL_PHENOMENOLOGY_PROTOCOL.md)
selects limited structural tests: model-preference shifts, persistence of
inspection beliefs, and reporting model/physical-history disagreements. The
model lacks independently identified fields for presence distinct from attention,
qualitative character, self-attribution, or phenomenal continuity. Those are
explicitly unsupported, rather than supplied through new report labels.

A fixed neutral renderer expresses the predicted fields and transitions. Its
round-trip parser tests formatting fidelity. Neither fluent syntax nor first-
person paraphrase counts as independent evidence. Preference ranking is not
automatically phenomenal foreground; inspection-history persistence is not
experienced continuity. No human ratings or human report-distribution match is
claimed. The structural scores must therefore be interpreted alongside the
possibility that supervised fields and authored language explain the whole effect.

### 3.5 Uncertainty, records, and correction

Scene-level bootstrap intervals use 1,000 draws and are conditional on each
trained system. Every seed and condition is retained, with timestep and cue-regime
breakdowns and compressed per-case records. A post-evaluation error audit is
labeled descriptive and changes no training decisions or gates.

Seed 207's exact shift agreement, 1216/1280=0.95, was initially represented as
float32 0.949999988 and incorrectly failed the inclusive threshold. We corrected
correspondence fractions to float64 and reevaluated saved models without fitting.
Only that secondary gate changes; the original JSON evaluations are retained in
the [numerical correction record](ATTENTION_MODEL_NUMERICAL_CORRECTION.md).

## 4. Results

### 4.1 Ordinary fidelity and model/world disagreement

| Metric | Seed 107 | Seed 207 | Seed 307 |
| --- | --- | --- | --- |
| Preferred-cell accuracy | 96.35% | 96.03% | 93.68% |
| Balanced belief accuracy | 96.22% | 96.79% | 95.78% |
| Exact belief-map accuracy | 90.89% | 92.38% | 88.15% |
| Exact complete report | 87.30% | 88.80% | 82.36% |
| Accuracy on model/physical-history disagreement cells | 96.09% | 97.16% | 95.73% |

No seed passes all primary criteria. Direct telemetry is exact by construction.
The trained readout is often faithful, including where physical history would
suggest a different answer. In seed 107, scene 0, step 1, for example, both model
and learned report say no cell has been inspected although the physical trace
contains cell 17. At step 4 of the same scene, the model classifies cell 17 as
inspected but the reporter omits it. The former is faithful reporting of an
incorrect model belief; the latter is a reporting failure.

Positive beliefs occupy only 3.2–5.6% of cells. Among ordinary belief-report
errors, 80.5–85.2% occur within 0.1 of the classification threshold. This provides
a possible readout-error explanation, not grounds to relax the criterion.

### 4.2 Causal identity and intervention fidelity

| Intervention | Paired exact-report accuracy, seed range | Next attended cell changes, seed range |
| --- | --- | --- |
| Donor model | 64.84–79.69% | 16.41–21.88% |
| One belief flipped | 16.41–25.00% | 8.98–13.67% |
| Neutral model | 0% | 65.23–85.55% |

No intervention condition passes the paired-report gate in any seed. Unchanged
belief-report fields are preserved only 86.83–90.74% under a single flip, also
below criterion. Thus ordinary accuracy overstates robustness to model changes.

Isolation and exact restoration checks pass in all seeds. Donor overrides change
attention probabilities and some physical fixations while the prior trace and
current hidden state remain fixed. The identified model therefore participates
causally in control. That conclusion is separate from robust reporting fidelity.

### 4.3 Alternative inputs and structural scores

Complete-report accuracy ranges are 77.73–80.86% from h, 45.38–48.24% from physical
history, 39.65–45.57% from prior answer logits, and 5.73–12.17% from the visible
scene/current cue. These comparisons support information specific to the internal
model over reconstruction from those external summaries. They do not establish
that the model is uniquely capable of grounding qualia.

Shuffled model inputs reduce complete accuracy against original states to
20.64–25.78%, constant neutral inputs to zero, and zero inputs to 9.18–45.05%.
The renderer remains grammatically fluent throughout, demonstrating why wording
alone is not evidence of a state-grounded consciousness report.

Preference-shift agreement is 92.11–95.23%, passing the 95% threshold in two seeds.
Inspection-belief persistence agreement is 99.32–99.67%, passing in all three.
However, an always-no-persistence output already reaches 95.36–97.42% because
persistence is sparse. The descriptive error audit gives only 78.87–86.74%
positive shift recall and 92.25–94.67% positive persistence recall. Model/physical
disagreement fidelity passes in all three,
with 2,465–3,255 disagreement cells per seed. No seed-independent complete package
of primary fidelity and the structural criteria is established.

## 5. Interpretation and the critic's crux

The study identifies a learned attention-history model, verifies its causal role,
and shows partial reporting of its state even when the represented history is
wrong. This is more directly relevant to the proposed mechanism than the previous
reports of sensor quality or task content. It is a concrete, inspectable result.

The theory-facing bridge remains unresolved. The tested model is a supervised
inspection estimator; the reporter is a supervised decoder; the expressive
relations are authored. A critic can accept every result and explain it without
a claim about qualia. The current data do not discriminate that explanation from
the broader source-of-qualia theory. This does not refute that theory; the model
and report mechanism cover only a narrow part of its proposed explanatory scope.

A further test would need a specified consciousness-report distinction or
intervention pattern that the inspection-estimator-plus-decoder account does not
already explain. Improving complete-report fidelity is useful but cannot by
itself supply that missing theoretical prediction. Likewise, adding experiential
words or renaming inspection history as presence would not resolve the crux.

## 6. Limitations and reproducibility

These are three old controllers from one small architecture and one synthetic
task. Reports are external, supervised, and restricted to identified fields.
The binary threshold discards graded belief strength. Synthetic interventions
can move off the fitting distribution; natural model error and reporting error
must be distinguished. Bootstrap intervals do not establish population-level
robustness. No independent human phenomenology evaluation was performed.

The [reproduction guide](ATTENTION_MODEL_REPRODUCTION.md) provides archive
restoration, complete replay, and reporter refitting from frozen controllers:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/summarize_attention_model.py --verify --replay
```

All three controllers and 15 reporters are archived with source, configuration,
data, and artifact hashes. Every final metric and case record replays exactly.
The full repository suite passed 160 tests. No paid model API is required.
The [results](ATTENTION_MODEL_RESULTS.md) link the machine-readable summary,
complete cases, and descriptive error audit. Historical studies and Stage 8
artifacts remain unchanged.

## 7. Conclusion

An actual learned model used in attention control is partially reportable and
causally affects allocation. Reports can follow its mistaken beliefs rather than
physical history. Yet complete learned-report fidelity fails the registered bar,
intervention generalization is weak, and consciousness-like structure independent
of supplied semantics and language is not established. The finite evaluation is
complete; the hypothesis that this mechanism is the source of qualia remains
underdetermined.
