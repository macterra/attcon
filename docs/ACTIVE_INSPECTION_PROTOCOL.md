# Recurrent information-acquisition protocol

Registered before results, 2026-09-26. Six commit/push cycles:

1. Implement and validate a finite-horizon acquisition environment and analytic
   belief-policy comparator; freeze this protocol.
2. Train a pilot at seed 2309: recurrent state, action-score, and confidence
   inspection heads with equal parameter counts and paired initialization.
3. Repeat the three controllers at seeds 2333 and 2351; retain every run.
4. Test history reset and choice-preserving state interventions, restoration, and
   matched random controls on the three state-controller runs.
5. Fit verified-information reporters after controller learning; compare state
   and action-score readouts with identical capacity and shuffled-label controls.
6. Evaluate untouched stress contexts with longer delays and degraded sensors,
   consolidate results, and update the roadmap without upgrading Stage 8.

## Environment and partitions

Six possible values, three initial information conditions: fresh (value with
validity cue), stale (independent old value with invalidation cue), and missing.
Four distractor events identify a context. A query shows cost and remaining
inspection budget, never the hidden answer or a condition label. The first paid
inspection returns a noisy value (75% correct, otherwise uniformly wrong); the
second returns the exact value. The controller may answer at any decision point,
or buy up to two inspections. Correct answers earn 1, wrong answers 0, and each
inspection subtracts its cost. Costs {0.1, 0.25, 0.4} are crossed with every value
and information condition within a context. These are observed environmental
validity cues, not introspective ground truth.

For each seed, randomly partition distinct four-value distractor contexts into
128 controller-train, 32 validation, 64 report-fit, 64 primary test, and 64 stress
contexts. Every answer/condition/cost variant stays together. A noisy sample is
shared across condition/cost variants of the same context/value. Record tensor
fingerprints. At evaluation use one delay blank; train delay alternates between
0 and 2 using an independent RNG. Stress uses five blanks and, separately,
sensor reliability 0.55 at one blank. Stress contexts are unused until cycle 6.

An analytic belief policy knows the task's generative distribution, not each
hidden answer. Fresh histories should answer immediately. Missing/stale histories
have a uniform prior; after one sample their posterior has mass 0.75 on that
sample. It chooses between answering, sampling, and verifying by backward
induction including costs. This is an information-model upper comparator, not a
learned baseline or per-case hindsight oracle. Record never-inspect, inspect-once,
and inspect-twice policies too; these fixed policies answer from the learned
controller at their chosen stopping point.

## Reward-based recurrent control

GRU hidden size 48, 12 input channels: six value indicators, distractor flag,
validity cue, invalidation cue, sample flag, cost, and budget. Query cost/budget
channels are shared across all models. Answer head is linear to six logits,
trained on immediate correctness rewards with cross entropy (equivalent to the
one-hot reward vector over answers). Inspection value is a 128→32 tanh→1 head.
Its input is either recurrent state, six answer logits, or max softmax plus
normalized entropy; all also receive cost and budget, padded to width 128.
Equal parameter counts, initialization, minibatch order, and training schedules.
The answer logits remain available for selecting an answer in every model.

Train all three reached-by-inspection histories using counterfactual replay from
known environment transitions. This is full-information fitted value learning,
not on-policy exploration. Inspection targets are -cost plus the frozen target
network's best next action value; terminal answer rewards train the value decoder.
Target network is refreshed once per epoch. The answer probability estimates an
immediate answer's expected reward; inspection values are unrestricted scalars.
Use Adam at 0.003, 100 epochs, batch 256, gradient clipping at 5. No availability,
report, or optimal-inspection labels train controllers. Select epochs 60,80,100
by mean validation return across all costs (earliest tie). Evaluate by actually
feeding acquired observations back into recurrent state and charging every
inspection; answers terminate the episode. No test-dependent recipe changes.

Primary support requires, at EACH cost and ALL seeds: state-policy mean return
at least 0.02 above each learned action/confidence comparator and each fixed
policy; paired context-bootstrap lower 95% bounds above zero for both learned
comparators; accuracy at least 0.90 on fresh cases and at least 0.90 after two
forced inspections. Record analytic-policy gap, accuracy, counts, cost-conditioned
returns, and per-condition inspection counts. This is a new narrow control gate;
it never replaces the previous reporting gates.

## Interventions and reporting

Cycle 4 uses primary test episodes. Reset initial state to zero, compare intact
and restored state, and transplant a report-fit fresh-minus-stale mean direction
projected into the answer weight matrix's null space. Move fresh recipients to
paired stale donor projections. Random null-space controls match each case's
perturbation norm. Require initial answer-logit residual <=1e-5 and report
initial inspection decision changes; continue the environment to measure actual
returns. Synthetic interventions have no access-label ground truth. Restoring
initial state must restore trajectories and rewards. They establish causal
sensitivity or damage, not beneficial awareness.

Cycle 5 freezes the selected state controllers. Report labels mean *verified
current value*: fresh history and exact second inspection provide a verified
value; stale/missing and one noisy sample without prior fresh information are
unverified. Fresh information remains valid after a noisy sample. Equal-capacity
128→32 tanh→7 state/action-logit reporters, fit-only standardization, Adam 0.01,
200 steps, L2 {0,0.001}, select validation minimum(verified-value,unverified)
accuracy, then balanced accuracy. Fit on all three decision stages equally.
Ten shuffled-label state fits repeat selection. Gate: verified and unverified
accuracy each >=0.90, state balanced gain over action >=0.02, and gain over null
p95 >=0.10 at all seeds. Report all-stage and actually visited-stage metrics.
This is supervised measurement after reward learning, not native verbal reporting.

Cycle 6 freezes every model and threshold. Evaluate the stress contexts under
baseline conditions, five blanks, and reduced sensor reliability; compare all
controllers and the appropriate analytic policy. Apply primary control gates
without refitting and label any failures. Diagnostic stress does not count as
independent system replication. All sources, selections, failures, and gates are
retained. Conscious experience and robust Stage 8 convergence remain untested.
