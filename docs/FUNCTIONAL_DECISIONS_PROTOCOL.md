# Explicit Controller category decisions v1

2026-10-04. **Post-hoc engineering assay**, motivated by the
[readout audit](FUNCTIONAL_READOUT_AUDIT_RESULTS.md). Publish this design, source,
tests and manifest before executing task branches. This is a newly implemented
deterministic Controller, not a claim that the earlier interface already ran it.
Reporter and Controller remain distinct; no reporter is called here.

## Question and information boundary

Does an explicit inspect-then-answer Controller use the learned attention model
to select physical commands and emit category responses? Measure command selection,
executed acquisition, modeled readout and actual answer/abstention separately.
This establishes an engineering component of the interpretation plan; it does
not supply human review, reporting qualification or source-of-qualia evidence
by itself. Task necessity and category-identity changes are not success criteria.

Use fixed attention models 2011/2021/2031, paired existing visual distributions
and all 512 archived static episodes in each of the three physical conditions
(commands control buffer 0, buffer 1, or neither). Each episode is branched
independently for each of four queried buffer-0 positions. Visual distributions
and scenes remain fixed during a branch. The attention estimator receives only
command/allocation/acquisition observations; it never receives scene labels,
physical owner flags or evaluator scores. The Controller receives the task query,
visual distributions and modeled forecasts. It never receives reporter prose,
physical recovery, automatic phase/direction or a condition label.

The simulator receives the Controller's chosen command and physical state;
its actual allocation/acquisition observation goes back to the learned estimator.
Scene labels and physical recovery are evaluator-only. No physical value replaces
a supplied modeled forecast. All conditions still contain an attention-control
model; functional separability is not substrate-presence evidence.

## Declared Controller rule and task

1. Construct all-command prospective categorical distributions for buffer 0
   using the existing bridge `b = q*v + (1-q)/4` and learned prospective recovery.
2. At the queried position, score each command by the minimum of its dominant
   color probability and dominant shape probability. Choose the largest score;
   exact ties choose the smallest command index. Planning uses unrounded values.
3. Execute one actual simulator step at acquisition quality 0.8 and update the
   attention model from that observed allocation, acquisition and command.
4. Construct the queried current distribution from the updated model's buffer-0
   recovery and the fixed visual representation. Apply the interface's Python
   five-decimal rounding. Emit both dominant attribute labels only if both
   probabilities are at least 0.6; otherwise emit a single abstention `(-1,-1)`.
5. Evaluate emitted labels against scene truth. An abstention is retained as an
   abstention, not a correct label. Report correct and incorrect answer counts,
   answer fraction and accuracy conditional on answering separately.

The cutoff is an engineered response rule inherited from interface identification,
not a validated phenomenal boundary. The Controller could recover dominant
categories at much lower positive recovery with a different decoder. Retain that
raw-argmax comparison; do not claim the chosen abstention rule is forced by physics.
The visual representation is already supplied, not reacquired from newly noisy
sensory patches. The bridge represents uncertainty explicitly but its calibration
against actual category mistakes is not established by this task.

## Matched policies and interventions

| Policy | Chosen physical command |
|---|---|
| Model-guided | Apply the rule to the unmodified model forecast. |
| Random-command comparator | Uniform command for each episode/query, seed `790000000 + model_seed`; reuse the identical random commands across all physical conditions. |
| Swapped-effect-guided | Swap the two modeled effect channels, preserving current allocation/access/recurrent state and physical wiring; recompute prospective recovery, then apply the same rule. |
| Restored-guided | Restore original modeled effects before planning; require exact equality of full command/world/model/answer traces with model-guided branches. |

Every policy starts from the same archived state. Physical outcomes can change
when the command changes. The effect swap is a transient planning intervention,
not a persistent alteration of the estimator weights or hidden state. Feedback
after actual execution uses the original estimator for every policy.

For each executed branch, separately perturb only the *post-feedback represented
recovery* of buffer 0 by multiplying all three delay forecasts by 0.25, holding
visual content, allocation, effects, hidden state, query, command and physical
outcome fixed. Emit another response using the same rule, retain its raw dominant
labels, restore original recovery and require exact response equality. This is a
transient readout intervention, not new physical acquisition. It deliberately tests
the implemented rule's dependence on represented recovery, without claiming that
this engineered dependence establishes consciousness. Restoration is an invariance
check, not independent replication.

## Fixed measurements, archive and stopping

Retain all 36 model/condition/policy contexts and 73,728 one-step task branches
(three models × three conditions × four policies × 512 episodes × four queries).
These are reused, paired and correlated branches; the count is not an independent
sample size. No training, fresh model selection, language reports, human judgments,
power claim or scientific success gate is introduced. Preserve all older failures.

Report target acquisition, physical queried recovery, and confidence regret against
the evaluator's maximal physical all-command confidence. That oracle is an auxiliary
counterfactual score, not an executed native policy. Also report the discrepancy
between modeled and physical thresholded availability, with the physical comparison
using the same engineered bridge/cutoff. Do not equate this comparison with actual
category-decoding ability. Raw-argmax scene accuracy remains a separate measure.
Retain attenuated responses, raw-label preservation and exact restoration checks.

Archive initial physical and modeled states, histories, all-command forecasts,
queries, planning scores, chosen commands, actual observations/allocations/recovery,
updated model and recurrent states, distributions, answers/abstentions, scene labels,
all metrics and exact source/dependency/archive hashes. Replay every value without
training or API calls. Reserve the attempt before execution and refuse overwrite
or retry; keep exceptions and failed checks. Publish preparation and completed
results as separate milestones.

Substantive author/independent definition review, independently justified report
features and qualified factual auditing remain prerequisites to new theory-facing
reporting. This assay cannot replace those judgments, powered confirmation or
positive replication. No reviewer contact is authorized by this protocol.
