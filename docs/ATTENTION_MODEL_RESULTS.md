# Attention-control model reporting: completed evaluation

**We identified and tested an actual learned component used in attention control.**
Reports recover much of its state, including when it disagrees with physical
inspection history. However, the trained reporters fail the registered complete-
report and intervention-generalization criteria. Consciousness-like report
structure beyond the supplied definitions and wording is not established.

The finite study is complete. The source-of-qualia theory remains unresolved;
completion of the experiments is not a claim that the theory has been demonstrated.

## What was actually reported

In the original recurrent controller, `hidden_self_model_head` estimates which
of the 25 cells have been inspected. Its sigmoid output **m** is the tested
attention-model state. `policy_self_model_head(m)` contributes directly to the
next attention logits. Reports describe m's inspected/uninspected beliefs and
which cell that contribution most favors. This is a learned, supervised model
of inspection history with a causal role in attention control.

The model-favored cell is **not necessarily the cell actually attended**. Other
controller inputs also affect allocation. No remembered sensor-quality variable
or generic task-answer value is substituted for this model.

The earlier interpretation of a zero feedback-training weight was wrong: it
turns off an auxiliary loss, not the forward path. Saved feedback weights are
nonzero and the intervention experiment confirms their effect. See the
[state specification](ATTENTION_MODEL_STATE_SPEC.md).

## Evaluation coverage

- Three existing, independently trained GRU controllers: seeds 107, 207, 307.
- Five matched reporters per controller: model state, prior task-answer logits,
  visible scene/current cue, actual inspection history, and recurrent hidden state.
- Per controller: 512 reporter-fit scenes, 128 validation scenes, and 256 held-out
  scenes with six snapshots each; fit and test cue-switch pairs differ.
- All 256 test scenes receive donor-state, isolated-belief-flip, and neutral-state
  interventions at step 3; no selection for baseline report correctness.
- Restoration, external-information invariance, information-loss controls,
  operational correspondence scores, full case records, and error analysis.

Controllers were frozen throughout. These are new reporters and test contexts on
old agents, not new agent training. The old online training episode lists were
not retained, so we do not claim a proof of no accidental overlap with their
original training scenes. New reporting partitions are fingerprinted and disjoint.

## Reporting accuracy

| Metric | Seed 107 | Seed 207 | Seed 307 | Registered threshold |
| --- | --- | --- | --- | --- |
| Model-preferred cell | 96.35% | 96.03% | 93.68% | 95% |
| Balanced inspected-belief accuracy | 96.22% | 96.79% | 95.78% | 95% |
| Exact 25-cell belief map | 90.89% | 92.38% | 88.15% | Diagnostic |
| Entire report correct | 87.30% | 88.80% | 82.36% | 90% |
| Belief report on model/physical-history disagreement cells | 96.09% | 97.16% | 95.73% | 95% secondary criterion |

No seed passes all primary fidelity criteria. A direct renderer of m and its
policy contribution is 100% accurate by construction. That demonstrates an exact
telemetry route; it is not learned interpretation or independent phenomenological
evidence. The measured limitations concern the fitted readout's generalization.

The whole-report criterion matters: a small error rate over individual cells can
still produce many incorrect complete reports. Only 3.2–5.6% of model-belief cells
are positive. We therefore report balanced accuracy and exact maps, not only high
raw cell accuracy. In the descriptive error audit, 80.5–85.2% of belief errors
occur within 0.1 of the fixed 0.5 classification boundary. This diagnosis did not
change fitting, thresholds, or test results.

## A concrete model-versus-world report

In seed 107, the first ordinary case with an exact learned report and a model/
physical-history disagreement is scene 0, step 1:

> The attention model favors cell 17. It represents cells [] as inspected.

The physical trace had already inspected cell 17. The report correctly follows
the model's thresholded belief, even though that belief is wrong about the trace.
This example was selected by a deterministic first-eligible rule in the
post-evaluation error audit, not as the strongest-looking example. All cases,
including failures, are retained.

In contrast, at scene 0, step 4, the model represents cell 17 as inspected but
the learned report says none. This is a reporting error, distinct from a model
that inaccurately represents physical history.

### One complete episode trace

Cells are indexed 0–24. Each row is a snapshot before the next physical fixation.

| Step | Physically inspected so far | Model believes inspected | Learned report believes inspected | Model-preferred cell | Next physical fixation |
| --- | --- | --- | --- | --- | --- |
| 0 | [] | [] | [] | 21 | 17 |
| 1 | [17] | [] | [] | 17 | 17 |
| 2 | [17] | [17] | [17] | 17 | 17 |
| 3 | [17] | [17] | [17] | 17 | 17 |
| 4 | [17] | [17] | [] | 12 | 22 |
| 5 | [17, 22] | [] | [] | 12 | 22 |

## Interventions establish a causal role, but expose reporting failures

| Intervention | Both original and changed reports entirely correct, seed range | Next physical choice changes, seed range |
| --- | --- | --- |
| Donor attention-model state | 64.84–79.69% | 16.41–21.88% |
| One belief flipped | 16.41–25.00% | 8.98–13.67% |
| Neutral m = 0.5 | 0% | 65.23–85.55% |

The paired-report requirement was 90%; no seed passes it in any intervention.
Under an isolated flip, preservation of unrelated belief-report fields is only
86.83–90.74%, below 95%. This is substantial readout fragility.

The prior physical trace and recurrent state at the intervention point remain
identical, and restoring the original unmodified computation restores reports
and attention exactly. Holding m fixed while changing outside reporter inputs
preserves state-only reports. Donor interventions change attention probabilities
in every seed, confirming that this identified model contributes to control.
These findings do not require a reward advantage or necessity for performance.

Flips and neutralization are synthetic states, and donor swaps can conflict with
the recipient's hidden state. Their results test off-distribution report fidelity.
At m = 0.5 the registered `>= 0.5` convention labels every cell inspected; this
is a numerical intervention, not a claim of absent or maximal awareness.

## Controls and the theory-facing result

| Reporter input | Entire-report accuracy, seed range |
| --- | --- |
| Identified attention-model state | 82.36–88.80% |
| Generating recurrent hidden state | 77.73–80.86% |
| Physical inspection history and preceding fixation | 45.38–48.24% |
| Prior task-answer logits | 39.65–45.57% |
| Visible scene and current cue | 5.73–12.17% |

Shuffling model states reduces accuracy against original states to 20.64–25.78%.
Constant neutral states reduce it to zero. Zero states reach 9.18–45.05%, showing
that class imbalance can make a fixed output partly successful. These controls
support information dependence, not the uniqueness of the model as a source of
qualia. The hidden state generates m, so that comparison is not an independent
alternative without an attention model.

The preregistered narrow structural scores are:

- Preference-shift agreement: 92.11–95.23%; two of three seeds meet 95%.
- Inspection-belief persistence agreement: 99.32–99.67%; all seeds meet 95%.
- Reporting model/physical-history disagreement cells: 95.73–97.16%; all seeds
  meet 95%, over 2,465–3,255 such cells per seed.

Persistence is sparse: an always-no-persistence response already scores
95.36–97.42%. In the post-evaluation audit, positive shift recall is only 78.87–86.74%, and
positive persistence recall is 92.25–94.67%. An all-uninspected report on
model/physical disagreement cells would reach only 46.42–63.04%, below the
measured model-report fidelity. These diagnostics expose the different effects
of class imbalance; they do not change the registered gates.

Shift and persistence concern model fields, not independently
measured experiential changes. The neutral prose and its temporal relations are
authored; their syntax remains equally fluent under shuffled or constant inputs.
No claim of an independently produced consciousness report follows from that.

The identified module lacks a tested representation of presence distinct from
attention, qualitative character, self-attribution, or phenomenal continuity.
The source-of-qualia hypothesis is therefore **underdetermined by this study**.
This is neither a positive demonstration nor a refutation of the broad theory.

## The crux critics can inspect

The narrower empirical result is that a learned internal model, used in attention
control, can be reported separately from actual attention history, and changing
it affects both reports and allocation. This is closer to the proposed mechanism
than the earlier sensor-quality reports.

The unresolved bridge is whether that mechanism explains the distinctive structure
of consciousness reports beyond supervised state decoding and supplied language.
A critic can accept every measured result and explain it as reporting an imperfect
inspection estimator. To advance the theory, a subsequent experiment must specify
an additional report distinction or intervention pattern that this alternative
does not already explain. Calling inspection memory “presence” would not supply it.

## Evidence and reproduction

The [registered protocol](ATTENTION_MODEL_PHENOMENOLOGY_PROTOCOL.md),
[numerical correction](ATTENTION_MODEL_NUMERICAL_CORRECTION.md), and
[reproduction guide](ATTENTION_MODEL_REPRODUCTION.md) preserve the procedure.
The float32 correction changes only one exact-95% secondary gate; all initial
JSON evaluations remain available. It does not change the failure of full fidelity.

The [machine-readable summary](https://github.com/macterra/attcon/blob/main/audits/attention_model/summary.json)
links checksums for all three result files, full compressed case records, source,
and checkpoints. The [error audit](https://github.com/macterra/attcon/blob/main/audits/attention_model/errors.json)
is explicitly descriptive and post-evaluation. The archive contains all three
frozen controllers and all 15 fitted reporters. Every final metric and case file
was replayed exactly; restoration into an empty temporary location passed. The
full repository suite passed 160 tests. Refitting all 15 reporters in an isolated
checkout also reproduced the committed checkpoints and results exactly.

The earlier [informational-state study](INFORMATIONAL_STATE_RESULTS.md) remains
related work. Its completed matrix and the historical Stage 8 verdict are unchanged.
