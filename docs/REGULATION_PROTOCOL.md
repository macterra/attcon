# Delay, confidence, and information-seeking protocol

Registered before new results on 2026-09-26. Six further commit/push cycles:

1. Paired GRU pilot, seed 2101: fixed-delay versus variable-delay task training.
2. Repeat both conditions at seeds 2111 and 2129; retain all results.
3. Variable-delay ungated RNN at the same three seeds, width 115 versus GRU width
   64 (15,876 versus 15,942 parameters). This is near parameter matching, not exact
   equivalence; report the difference and avoid architecture-necessity claims.
4. Compare state, action-score, and entropy reports under identical content
   lesions. Add a fitted access-direction transplant in the choice head's null
   space, with a matched-norm random null-space control. Choice logits must remain
   unchanged within 1e-5; otherwise that contrast is invalid.
5. Learn a one-step inspection policy from environmental correctness rewards on
   the frozen variable-delay GRU, seed 2101. Compare state, action-score, and
   confidence features. An inspection reveals the true value at a fixed cost.
6. Replicate inspection learning at seeds 2111 and 2129 and test its response to
   the choice-preserving state interventions. Consolidate evidence conservatively.

## Fixed agent and reporting design

- Existing paired-history task and disjoint context partitions: 1,024 agent-train,
  64 validation, 128 test, 512 reporter-fit groups selected from the existing pool.
  All variants of a context remain together. Fresh seeds are shared across recipe
  and architecture comparisons. Fingerprint actual data and initial agent weights.
- GRU width 64; RNN width 115. Agent training uses value-choice cross entropy only,
  160 epochs, batch size 256, AdamW at 0.003. No access/report/inspection labels or
  reporter gradients train these recurrent agents.
- Fixed recipe always has zero extra blanks. Variable recipe samples extra delay
  uniformly from `{0,1,3,6}` per minibatch using a separate RNG. Paired conditions
  use identical initialization and minibatch order. They have equal update counts,
  but variable training processes more recurrent steps; record that difference.
- Fit and validation report contexts have one deterministic, balanced assignment
  of delays `{0,1,3,6}` per context, the same for all content/status variants and
  both training recipes. Primary test metrics use this mixed-delay assignment.
  Also report every held-out context at delays 0,1,3,6 and unseen delay 9. No tuning
  on those test slices; delay 9 is an extrapolation diagnostic.
- State, action-score, and observation reporters use the same standardized
  128→64 tanh→7 network (8,711 parameters), accommodating both state widths without
  truncation. L2 candidates `{0,0.001}`, Adam 0.01, 300 steps, fit-only normalization.
  Select by validation min(seen, unavailable) accuracy, paired accuracy, then
  balanced accuracy. Entropy gets the same 101-threshold validation selection as
  the preceding campaign. Never substitute the best test reporter for the primary.
- Twenty primary-state permuted-label fits repeat the full selection procedure.
  Retain the existing 15 reporting/transplant gates without changes. Fit content,
  permuted directions, and query probes only on reporter-fit states. Information
  availability is checked by the input-only symbolic oracle for every delay.

## Confidence and intervention design

- Use the three variable-delay GRUs, mixed-delay held-out contexts, and frozen
  selected reporters. Test strengths 0,0.25,0.5,1 for content-subspace erasure and
  norm-matched random/permuted controls. Compare all reporters on all seen cases
  and a shared cohort where choice and every reporter were initially correct.
- Fit the mean seen-minus-unseen state direction on reporter-fit pairs; remove its
  projection onto the row space of the six-way choice weight matrix and normalize.
  Transplant the scalar projection from each matched unseen history into its seen
  recipient. The matched random direction is also in the choice null space.
  Report state-report changes, random-control differences, logit residuals, and
  action/entropy invariance. Degenerate directions invalidate the contrast.
- These are off-distribution perturbations. Choice invariance can isolate a fitted
  readout's additional sensitivity; it does not establish subjective access or
  native regulation. Report direction norms, cohort sizes, and restoration checks.

## Reward-trained inspection design

- A frozen agent first proposes its argmax value. A separate policy may inspect,
  obtaining the true environmental value, or answer immediately. Net return is
  `1-cost` after inspection, otherwise `1` for a correct answer and `0` for error.
  Costs are fixed at `{0.2,0.4,0.6}`; inspection reveals missing information too.
- Learn probability of immediate correctness from task rewards only. Inputs are
  state, action logits, or two confidence statistics (max softmax and normalized
  entropy), padded to 128. All use standardized 128→32 tanh→1 networks (4,161
  parameters). Adam 0.01, 300 steps, L2 `{0,0.001}`, BCE reward prediction.
  Inspect when predicted correctness is less than `1-cost`.
- Select L2 using average validation return across the three costs. No report,
  seen/unseen, or explicit access labels train the policy. Do not refit the agent.
  A hindsight oracle is an unattainable bound, clearly separated from policies.
- Baselines: always inspect, never inspect, matched action-score/confidence
  policies, and 20 shuffled-reward state policies with identical selection.
  Record return, inspection rates, Brier score, and paired context-bootstrap
  intervals for state-minus-baseline returns. Primary gate: at every cost, the
  state policy gains >=0.02 over each learned comparator, >=0.05 over null p95,
  and beats both fixed policies. Require task viability >=0.85 on mixed-delay seen
  cases and positive lower 95% context bounds for each learned-comparator gain.
  This is a fixed narrow diagnostic gate, not a revision of the reporting gates.
- Test policies at delay 9 without retuning, and under the choice-null-space
  transplant. Confidence/action inputs must stay constant in that contrast.
  Policy changes then demonstrate use of additional state information, not that
  the changes are beneficial: there is no established reward ground truth for a
  synthetic lesion. Report both directions of switching and norm-matched controls.

## Boundaries

This is offline one-step policy learning from rewards on frozen representations,
not a fully learned recurrent attention-control loop. Evaluating planned heads on
the same reserved tests across cycles is a prespecified diagnostic reuse, not new
independent confirmation. Claims require fresh-system replication and remain
bounded; none of these results automatically upgrades Stage 8.
