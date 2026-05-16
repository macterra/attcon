# Next Steps Checklist

This checklist turns the revised roadmap into a working execution order. The goal is to move from bounded benchmark support toward robust, comparator-resistant evidence without blurring the current claim boundaries.

## Priority 1: Tighten Existing Claims

- [x] Run matched-capacity baseline audits for Stage 4B hidden self-model probes.
- [ ] Run matched-capacity baseline audits for Stage 6A and Stage 6B report probes.
- [ ] Run matched-capacity baseline audits for the Stage 7 local opaque-token reporter.
- [ ] Add explicit negative-control runs for feedforward, shuffled-feedback, and high-capacity observation-only systems.
- [ ] Add first-class comparator runs for static/feedforward, matched transformer, large-LM-without-loop, and trivial-regulator systems.
- [ ] Re-test Stage 3 under complete zero target-attention shaping.
- [ ] Document which bounded claims survive the audits and which need downgrading.

## Priority 2: Rebuild Stage 4B for Emergence

- [ ] Define a Stage 4B training condition with no direct self-model objective.
- [ ] Train fresh checkpoints under the task-only or indirectly induced self-modeling objective.
- [ ] Evaluate hidden-state inspected-map and target-inspected probes against previous-observation baselines.
- [ ] Test hidden-state interventions for effects on self-model reports and attention policy.
- [ ] Compare against the supervised Stage 4B feedback checkpoint.
- [ ] Decide whether Stage 4B remains an engineered capability probe or gains consciousness-relevant evidence status.

## Priority 3: Finish Stage 7 Variants

- [ ] Evaluate external API LLM reporting under default, cue-switch, and intervention slices once quota is available.
- [ ] Add a VLM-based Stage 7 path using minimally labeled visual internal-state renderings.
- [ ] Compare VLM reports against scene-only and explicit symbolic-dump baselines.
- [ ] Add token-remapping and held-out combination tests for the local opaque-token reporter.
- [ ] Keep the symbolic dump as an upper-bound baseline, not the main Stage 7 claim.

## Priority 4: Build New Theory Branches

- [ ] Extend the benchmark with independently recombinable attributes, held-out conjunctions, and false-binding lures for Branch C.
- [ ] Add Branch C unity/binding experiments with bound-content probes and intervention tests.
- [ ] Extend the benchmark with query-change and alternative-target conditions for Branch D.
- [ ] Add Branch D counterfactual-access experiments for non-current but query-available contents.
- [ ] Extend the benchmark with stale-access, inferred-content, and wrong-access lures for Branch E.
- [ ] Add Branch E higher-order state-representation experiments that separate first-order content from access, confidence, and report-grounding state.
- [ ] Add separable downstream consumers for Branch F: action, report, uncertainty, reallocation, memory, and language-shaped report paths.
- [ ] Add Branch F broadcast/ignition experiments with coordinated intervention tests.
- [ ] Add perturbational-complexity diagnostics over controller and self-model state.

## Priority 5: Replicate Across Systems

- [ ] Replicate supported claims on a structurally different controller architecture.
- [ ] Replicate supported claims on a second benchmark with different surface task structure.
- [ ] Re-run comparator and negative-control suites on the replicated systems.
- [ ] Check whether any Stage 8-relevant contents show cross-validated causal overlap across branches.

## Stage 8 Gate

Do not claim Stage 8 support until all of the following are true:

- [ ] At least one access/report family has robust support.
- [ ] At least one non-reportability family has robust support.
- [ ] The supported families point to the same internal contents, not merely the same checkpoint.
- [ ] Comparator systems fail in predicted ways.
- [ ] Results replicate across at least one different architecture.
- [ ] Results replicate across at least one different benchmark.
- [ ] The final claim is framed as consciousness-relevant evidence, not proof of consciousness.
