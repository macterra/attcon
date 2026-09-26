# Final research protocol and completion contract

Registered 2026-09-26 before the final experiment matrix. This completes the
bounded research prototype and evaluation authorized by the user. Completion is
not a guarantee of conscious access, native introspection, or Stage 8 support.
Earlier protocols and thresholds remain immutable. Negative results close an
experiment; they do not justify relaxing its gates.

## Final questions and stopping rule

1. Does future information quality affect acquisition independently of current
   answer confidence? Test variable-quality sequential verification and a
   structurally different two-sensor routing task.
2. Does full state outperform a fair confidence comparator with the SAME quality
   cue? A confidence-only head is an information-restricted diagnostic, not the
   decisive comparator. Both receive the full event stream in their encoder;
   the fair head additionally receives the observed cue directly as a memory aid.
3. Can acquisition and answer/decline behavior be learned from experienced rewards,
   without full-information answer labels or optimal-inspection supervision?
4. Do post-training reports and control share a causal prospective-information
   representation? Test answer-preserving transplants, matched random changes,
   report changes, policy changes, environment reward consequences, and restoration.
5. Which results survive another recurrent architecture, independent contexts,
   longer delays, and sensor misspecification?

Complete every registered cell, retain failures, audit source/data/checkpoint
integrity, deliver reproduction commands and a final evidence map. Stop after
that finite matrix. Promote independent-content convergence only if the fair
control advantage replicates across both architectures and all three seeds in
at least one task; otherwise close that promotion as prerequisite not met and
retain the existing negative/unforced-convergence evidence. No Stage 8 gate is
redefined. A final report must separate completed evaluation from unsupported
scientific hypotheses and untested generalizations.

## Prospective environments

Fresh seeds 2503,2521,2539. Start from the acquisition context generator with
128 train,32 validation,64 reporter-fit,64 test,64 reserved stress contexts.
Cross every initial condition (fresh/stale/missing), value (six), cost
{0.1,0.25,0.4}, and quality {0.55,0.90} within each context. All variants remain
in the same split. Quality is independent of initial answer information. The
initial history includes a quality cue; queries contain cost/budget only.
Use shared random numbers across quality variants and record fingerprints.

- **Serial task:** up to two inspections. First sample has cued reliability;
  second verifies the value exactly. One available inspection action.
- **Routing task:** choose sensor A or B, then answer. Sensor A has cued quality,
  sensor B has the complementary quality (1.45 minus the cue). Both cost the same.
  A route flag distinguishes the acquired sensor. This tests selecting an
  information source, rather than deciding whether to verify a noisy sample.

Four distractor events, a quality-cue event, initial evidence, a validity cue,
delay, query. Fourteen event channels. Values and source-validity cues are normal
environment observations; no labels about internal access are supplied. Primary
inference is about task representations, not subjective availability.

## Fitted control matrix: 36 controllers

Two tasks × GRU48/RNN87 × state/blind-confidence/cue-confidence heads × three
seeds. Same weights at initialization within architecture/seed, same minibatch
order, delay draws, optimizer updates, and head parameter count. RNN87 is near
parameter matched; record actual counts and avoid architectural necessity claims.
Inspection head 128→32 tanh→2; unavailable action masked in serial task. Inputs:
state, or max softmax/normalized entropy; all have cost/budget. Cue comparator
also receives the actually observed quality scalar. Exact cue retention makes
this a strong comparator, not evidence of autonomous memory in that comparator.

Answer logits are trained from the full environmental correctness-reward vector
with CE; inspection values by one-step fitted Bellman targets on replayed
counterfactual branches. GRU/RNN and heads train jointly. Target network copied
once per epoch. Adam .003, batch512,80 epochs, clip gradient norm5. Train delay
randomly 0 or2 with a separate generator. Select epochs40,60,80 by average
validation return; earliest tie. Primary delay1. No test-driven changes.

Primary per-cost gates: fresh-answer and forced-final-observation accuracy each
>=.85; state return exceeds every fixed policy by >=.02; state exceeds the FAIR
cue-confidence policy by >=.02 and its paired context-bootstrap lower95% bound
is positive. Require all costs and seeds for a task/architecture and both
architectures for promotion. Blind-confidence comparison is diagnostic only.
Sensor routing never guarantees 100% accuracy; .85 viability is below its .90
information bound. Record analytic expected/realized policies, calibration,
inspection counts and source choices, and every checkpoint selection.

## Reward-only exploration: 12 controllers

Both tasks × state/cue-confidence GRU48 × three seeds, trained from scratch.
Epsilon-greedy action selection over six answers, permitted inspections, and a
native decline action returning0.30. Epsilon linearly decreases .40→.10 over
600 minibatches of512 sampled training episodes. No answer-label vector, status,
quality-report target, or optimal action is used in a loss. Update chosen-answer
correctness probability by BCE on the experienced binary reward; update chosen
inspection Q from its immediate cost and target-network continuation value.
Target copy every10 updates; Adam .003, clip5. Select updates200,400,600 using
validation return. The learner receives reward/next observations for its chosen
actions only; simulator access to truth is confined to computing rewards.

Decline is a reward-grounded communicative action, not a separately supervised
access report. Record answer coverage, selective accuracy, decline rates,
reward, inspection/source behavior, and comparison with cue-confidence. These
objectives differ from the fitted-control matrix, so do not make a matched-budget
algorithm superiority claim. Repeat all seeds even if the pilot fails. This is
epsilon-greedy exploration in a finite simulator, not general autonomous learning.

## Report/control coupling

Freeze each fitted STATE controller (12 systems). Fit root-state reporters on
report-fit contexts only: joint label (cued quality, verified current value or
unverified). Two qualities × seven value/source labels. State, answer-logit,
and confidence-plus-cue readouts have equal 128→32 tanh→14 capacity, fit-only
normalization, Adam .01,200 steps,L2{0,.001}; validation selects minimum quality
accuracy and balanced verified/unverified accuracy, then their mean. Five
shuffled-label state fits repeat selection; their empirical p95 is a coarse
null diagnostic, not a significance test. Accuracy gates >=.90 for quality and
both verified/unverified components; fair-readout advantage >=.02 on the mean of
quality and balanced source accuracy. Retain results without claiming that
source verification equals internal access.

Fit high-minus-low quality direction on reporter-fit paired root states; remove
answer-weight row space, normalize, transplant donor projection into high-quality
recipient. Donors share context,value,condition,cost. Test on initially
unverified test histories (stale/missing) and record baseline coverage. Match
random null-space controls per-case norm. Require initial answer logits invariant
within1e-5, restoration of decisions/rewards/reports, and record full changes in
quality reports and inspection actions. Environmental quality remains unchanged
under synthetic perturbations; returns measure policy consequences, not
introspective truth. Coupling alone is not superiority over the fair comparator.

## Stress, final audit, and deliverables

Use reserved contexts under delay1, delay5, and actual sensor reliability reduced
by.15 while the old cue remains displayed (misspecification). Exact verification
in serial remains exact. Evaluate all fitted controllers; no refitting. Analytic
stress policy knows actual reliability; mark its information advantage. Keep
all original gate failures. Context bootstrap2000 draws, pointwise intervals,
conditional on trained systems, not population-level certainty.

Deliver immutable experiment artifacts, a validated manifest with source/data
hashes, a reproduction entry point and smoke run, a final claim/evidence matrix,
full tests, limitations, and an updated repository landing page. Preserve prior
Stage 8 evidence; record that it is unsupported if promotion prerequisites fail.
A future research extension is optional after this completed bounded study, not
an unbounded condition for finishing this user request.
