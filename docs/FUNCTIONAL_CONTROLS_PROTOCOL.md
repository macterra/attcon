# Learned three-way and matched-revision controls v1

Frozen engineering-development protocol, 2026-10-04, before model training/evaluation.
The objective remains credible evidence for the attention-control model as source of
qualia. This experiment completes necessary engineering comparisons; it collects no
language reports or consciousness judgments. Reporter and Controller remain separate.

## Gap addressed

The [previous learned integration](FUNCTIONAL_MODEL_RESULTS.md) evaluates two ownership
conditions only. Its process-change “before” error averages four alternatives while
“after” concerns one executed command; those values do not establish a matched
improvement contrast. The [requirements audit](CONSCIOUSNESS_REQUIREMENTS_AUDIT.md)
identifies both gaps. Existing archives, source and verdicts remain unchanged.

## Simulator and architecture

Use two four-position buffers, each with an independent cyclic automatic scan phase
and direction. Commands k0–k3 select p0–p3 at one controlled buffer, or control neither.
Condition 0 controls buffer 0, condition 1 controls buffer 1, condition -1 controls
neither. A downstream task readout can use buffer 0: that defines own-access versus
other-buffer access in this fixture, not a demonstrated phenomenal boundary. The
no-control condition is not implemented by forcing equal channel allocations.

The three physical conditions share commands, automatic schedules and sensor qualities
for each paired evaluation episode. Current recovery starts at zero and evolves as
`q_next = .75*q*(1-allocation) + quality*allocation`. Unattended delays multiply by
.75 and .75². All four future command allocations are computed from the actual
physical process for evaluation/training targets. They never become model inputs.

Use the existing `PredictiveAttention` architecture: 20 observed values (allocation 8,
acquisition 8, command one-hot 4), GRU hidden 64, allocation/access/command-effect heads.
Condition, control-owner labels, raw object identities and evaluator outcomes are
excluded from forward/policy input. The predictive loss remains the existing allocation
and command-effect cross entropy plus 10× recovery MSE, with command-effect burn-in
of three observations. No experience, consciousness or reporting targets are trained.
The numerical control-owner label is used only for physical simulation and scoring.

## Fixed training and independent evaluation

Three fresh seeds: **2011, 2021, 2031**. Train each from initialization for **1600**
updates, batch **192**, sequence length **16**. Each shuffled batch contains 64 episodes
per physical condition. Training data seed is `model_seed*100000 + update` (zero based).
Adam learning rate .003, gradient norm clipped to 1, CPU, two Torch threads. Retain
only the fixed final checkpoint for assessment; do not select best checkpoints, tune
on evaluation, add updates, replace seeds or retry failed attempts.

Evaluation: **512 paired episodes** per model, 12 initial observations, data seed
`730000000 + model_seed`. Assess all three conditions. Static accuracy/recovery metrics
use observations 8–12, and prospective counterfactual recovery uses all four commands
at the next step with known acquisition quality **0.8**. Reuse of a scene across
conditions creates correlated comparisons, not independent replications.

Infer a two-channel control mask from command-conditioned allocation TV spread:
controlled if spread >0.75 (ideal one-hot controlled spread 1.5, independent spread 0).
Unlike an argmax over channels, the mask can express neither channel controlled.
All three classes must be correctly distinguished; no condition label is supplied.

## Selective model intervention and restoration

At each static final state, swap the decoded command-effect channels while holding
current allocation, access, recurrent hidden state and physical process fixed.
Compute prospective recovery from the altered forecasts and the same hidden state.
Restore the original effect tensor and require exact return of predictions.
Swap the inferred control mask exactly. In no-control cases the operation also swaps
automatic destination forecasts; it need not create command dependence. Report change
magnitude descriptively, not as a qualia or general consistency-comparator test.

## Matched process-change comparison

Test all **nine old→new condition pairs**: six changes and three unchanged controls.
Preserve the initial physical recovery/automatic phase and prior modeled state at the
unobserved switch. A previously active modeled relation with physical commands now
controlling neither buffer supplies a represented/physical disconnection case.

Use the same 16 post-switch commands across every pair, seed `740000000 + model_seed`,
with quality 0.8. Run two contexts from the identical prior model state:

1. **No-feedback route:** advance recurrent state using its own predicted next
   allocations and acquisitions under the known commands; receive no new physical data.
2. **Feedback route:** advance recurrent state using actual allocations/acquisitions
   generated by the changed or unchanged physical process.

At windows **1, 2, 4, 8, 16**, compute both contexts' next-step forecasts for all four
commands. Compare them to **one common physical target** constructed from the actual
current recovery, automatic phase and new wiring at that same window. Both predictions
refer to the same episode, next-step time, commands, channels, positions and quality.
Archive one target and both predictions, allowing exact per-episode replay. Do not
compare an earlier four-command aggregate with a later single-command estimate.

This is a matched open-loop versus observed-feedback comparison, not a literal snapshot
before/after at different times. Feedback can improve ordinary filtering in unchanged
worlds too. Include unchanged controls and a switch-specific command-effect gain over
that control. Do not interpret generic access-error improvement alone as adaptation of
the represented command relation, or as evidence of an experiencing comparator.

## Prespecified component gates

Every model and static condition:

- Current allocation, command-effect and exact three-way control-mask accuracy >=99%.
- Recovery MAE over all three delays <=0.04; next-step counterfactual recovery MAE <=0.04.
- In the no-control condition, mean command-effect TV spread <=0.10 and mean
  command-conditioned recovery spread around its command mean <=0.02.
- Exact preservation of current allocation/access/hidden state, exact control-mask
  swap, and exact restored prospective recovery.

Every condition-pair route at the primary 16-observation window:

- Feedback command-effect accuracy >=99% and new control-mask accuracy >=99%.
- Feedback prospective and current recovery MAE <=0.04.
- No-feedback route retains its prior control-mask classification >=99%.
- For unchanged controls, no-feedback command-effect accuracy also >=99%.
- For changed processes, >=95% of paired episodes have lower prospective recovery
  MAE with feedback; feedback-minus-no-feedback effect-accuracy gain exceeds the
  matching unchanged-condition gain by at least **0.30**.

Initial model allocation/access/effects/hidden state and initial physical recovery/
automatic phase must remain byte-identical across new-world choices for each old
history. Global component success requires every listed gate in all three trained
models. Intermediate windows are descriptive; no gate is moved to the best window.
No hypothesis-test significance, human-report effect or independent-rater claim is
made by these engineering minima. Retain every failure and diagnose it separately.

## Provenance, milestones and next-stage limits

Commit/publish source, protocol and preparation manifest before training. Reserve
one attempt per model; archive final checkpoint, training-loss trace, every static
forecast/target and matched-revision window in lossless tensor archives. Record source,
checkpoint and trace hashes and software versions. Commit/publish each completed
model and the aggregate assessment at their milestones. Replay every scored tensor
and metric without retraining or API calls. Preserve interrupted attempts without
silently rerunning them under the same version.

The [v6/v7 semantic qualifications](NEUTRAL_FUNCTIONAL_PILOT_V7_RESULTS.md) remain
failed, and their row-3 language sample remains unused. This engineering study does
not qualify a semantic language auditor, validate consciousness-report features,
collect independent rubric review, or complete theory-facing confirmation. Independent
[definition review](FUNCTIONAL_RUBRIC_REVIEW_FORM.md), a reliable factual assessment,
blinded human ratings, powered causal tests and replication remain required by the
[interpretation plan](CONSCIOUSNESS_INTERPRETATION_PLAN.md). Existing ordinary causal-
description explanations remain viable. Do not infer consciousness from these gates.
