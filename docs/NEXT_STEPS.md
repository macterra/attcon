# Next Steps Checklist

This checklist turns the revised roadmap into a working execution order. The goal is to move from bounded benchmark support toward robust, comparator-resistant evidence without blurring the current claim boundaries.

> **Post-rehab note.** The Priority 1 audit machinery below was originally exercised on a
> non-functional checkpoint (under the old fully-soft recipe the recurrent controller collapsed
> to uniform attention and lost to the static baseline). After the discrete-glimpse fix, the
> audits, negative controls, and comparator suite were re-run on a model that actually does the
> task: dissociation/Stage 3/Stage 4A/Stage 5/Stage 6A/Stage 7 are genuinely supported (Stage 3
> robustly), Stage 6B is bounded/provisional, and all negative controls + comparators fail as
> intended. See `audits/post_rehab_full_eval_tune_prob_035_summary.json` and
> `docs/PRIORITY1_AUDIT_STATUS.md`. The Priority 1 boxes are therefore now genuinely validated,
> not artifacts. Remaining open work below (Priorities 2-5) is unchanged.

## Priority 1: Tighten Existing Claims

- [x] Run matched-capacity baseline audits for Stage 4B hidden self-model probes.
- [x] Run matched-capacity baseline audits for Stage 6A and Stage 6B report probes.
- [x] Run matched-capacity baseline audits for the Stage 7 local opaque-token reporter.
- [x] Add explicit negative-control runs for feedforward, shuffled-feedback, and high-capacity observation-only systems.
- [x] Add first-class comparator runs for static/feedforward, matched transformer, large-LM-without-loop, and trivial-regulator systems.
- [x] Re-test Stage 3 under complete zero target-attention shaping.
- [x] Document which bounded claims survive the audits and which need downgrading.
- [x] Calibrate audit thresholds against an empirical permuted-label noise floor (`noise_floor_metrics`). The Stage 6A controller-vs-observation advantages (`~0.38`, `~0.42`) are ~100x above the permuted-label p95 floor (`~0.004`, `~0.003`), so the claim is significant rather than a probe-capacity artifact.

## Priority 2: Rebuild Stage 4B for Emergence

GitHub issue: [#4](https://github.com/macterra/attcon/issues/4)

First pass implemented in `scripts/stage4b_emergence.py` (result in
`audits/stage4b_emergence_tune_prob_035.json`). Honest finding: a **weak** cell-level
inspection-history self-model is **task-induced, not supervision-induced** (the raw hidden
state beats a previous-observation baseline on the inspection map, BCE advantage `~+0.09`; the
dedicated self-model objective adds only `~+0.005`), but target-level inspection does not
emerge. Bounded evidence against the "supervised self-model required everywhere" global
falsifier, not a strong emergence claim.

- [x] Define a Stage 4B training condition with no direct self-model objective.
- [x] Train fresh checkpoints under the task-only or indirectly induced self-modeling objective.
- [x] Evaluate hidden-state inspected-map and target-inspected probes against previous-observation baselines.
- [ ] Test hidden-state interventions for effects on self-model reports and attention policy. (Causal feedback path is disabled in the base config; interventions remain to be run on a task-only checkpoint.)
- [x] Compare against the supervised Stage 4B feedback checkpoint.
- [x] Decide whether Stage 4B remains an engineered capability probe or gains consciousness-relevant evidence status. (Engineered capability probe; the emergent component is weak.)

## Priority 3: Finish Stage 7 Variants

GitHub issue: [#5](https://github.com/macterra/attcon/issues/5)

- [ ] Evaluate external API LLM reporting under default, cue-switch, and intervention slices once quota is available. (Blocked: quota.)
- [ ] Add a VLM-based Stage 7 path using minimally labeled visual internal-state renderings. (Blocked: vision model.)
- [ ] Compare VLM reports against scene-only and explicit symbolic-dump baselines.
- [~] Add token-remapping and held-out combination tests for the local opaque-token reporter. **Investigated: not meaningful against the current local reporter.** The local decoder reads the scored content fields from attended-content token bases the renderer fills *directly* from the model's attended content (not the learned translator's predictions, nor the opaque latent-bit tokens), so it is a schema-aware structural round-trip: a consistent token remapping is invariant by construction and held-out combinations do not bite directly-encoded fields. The genuine anti-memorization test needs a **latent-only decoder** (forced to recover content from the opaque latent-bit tokens alone) or the external LLM/VLM path. See ROADMAP "Sharper decoder caveat".
- [ ] Build a latent-only decoder so token-remapping / held-out-combination tests become meaningful (the real next step for this priority).
- [ ] Keep the symbolic dump as an upper-bound baseline, not the main Stage 7 claim.

## Priority 4: Build New Theory Branches

GitHub issue: [#6](https://github.com/macterra/attcon/issues/6)

- [ ] Extend the benchmark with independently recombinable attributes, held-out conjunctions, and false-binding lures for Branch C.
- [ ] Add Branch C unity/binding experiments with bound-content probes and intervention tests.
- [ ] Extend the benchmark with query-change and alternative-target conditions for Branch D.
- [ ] Add Branch D counterfactual-access experiments for non-current but query-available contents.
- [ ] Extend the benchmark with stale-access, inferred-content, and wrong-access lures for Branch E.
- [ ] Add Branch E higher-order state-representation experiments that separate first-order content from access, confidence, and report-grounding state.
- [ ] Add separable downstream consumers for Branch F: action, report, uncertainty, reallocation, memory, and language-shaped report paths.
- [ ] Add Branch F broadcast/ignition experiments with coordinated intervention tests.
- [x] Add perturbational-complexity diagnostics over controller and self-model state. (`perturbational_complexity_metrics`; first non-reportability evidence family, bounded support: rich-but-recoverable dynamics that propagate far more than a no-recurrence control and recover unlike a frozen-state control. Robust support still needs multiple seeds and cross-system replication.)

## Priority 5: Replicate Across Systems

GitHub issue: [#7](https://github.com/macterra/attcon/issues/7)

- [ ] Replicate supported claims on a structurally different controller architecture.
- [ ] Replicate supported claims on a second benchmark with different surface task structure.
- [ ] Re-run comparator and negative-control suites on the replicated systems.
- [ ] Check whether any Stage 8-relevant contents show cross-validated causal overlap across branches.

## Stage 8 Gate

GitHub issue: [#8](https://github.com/macterra/attcon/issues/8)

Do not claim Stage 8 support until all of the following are true. **Current status: not met.**
The methodology now produces one of each partition type (a robust access/report family and a
bounded non-reportability family) and comparators fail as intended, but both families are not
yet robust, content-identity is unestablished, and cross-system replication is absent.

- [~] At least one access/report family has robust support. (Stage 3 explicit-attention-modeling is robust; Stage 6A is capacity-audited and now backed by an empirical noise floor; Stage 7's local-reporter content claim is weak — a symbolic round-trip — so the access/report side is strong but its strongest reportability leg needs the latent-only / external path.)
- [ ] At least one non-reportability family has robust support. (Perturbational complexity has **bounded** support; needs multi-seed + cross-system for robust.)
- [ ] The supported families point to the same internal contents, not merely the same checkpoint.
- [x] Comparator systems fail in predicted ways. (All negative controls and comparators fail as intended; `shuffle_feedback` drops accuracy by `0.27`.)
- [ ] Results replicate across at least one different architecture.
- [ ] Results replicate across at least one different benchmark.
- [ ] The final claim is framed as consciousness-relevant evidence, not proof of consciousness.
