# Next Steps Checklist

This checklist turns the revised roadmap into a working execution order. The goal is to move from bounded benchmark support toward robust, comparator-resistant evidence without blurring the current claim boundaries.

> **Post-rehab note.** The Priority 1 audit machinery below was originally exercised on a
> non-functional checkpoint (under the old fully-soft recipe the recurrent controller collapsed
> to uniform attention and lost to the static baseline). After the discrete-glimpse fix, the
> audits, negative controls, and comparator suite were re-run on a model that actually does the
> task: dissociation/Stage 3/Stage 4A/Stage 5/Stage 6A are genuinely supported (Stage 3
> robustly), Stage 7 is bounded (schema-aware round-trip only; the faithful-latent leg is open —
> see [Current Focus](#current-focus-latent-only-stage-7-decoder)), Stage 6B is
> bounded/provisional, and all negative controls + comparators fail as
> intended. See `audits/post_rehab_full_eval_tune_prob_035_summary.json` and
> `docs/PRIORITY1_AUDIT_STATUS.md`. The Priority 1 boxes are therefore now genuinely validated,
> not artifacts. Remaining open work below (Priorities 2-5) is unchanged.

## Current Focus: Latent-Only Stage 7 Decoder

A **latent-only decoder** for Stage 7 is now **implemented** (`run_latent_only_report_mode` in
`src/attcon/nl_report.py`, wired through `nl_report_metrics`, enabled in `configs/tune_prob_035.yaml`,
unit-tested, runs in CI). It was the one unblocked piece of work that targets the weakest leg of the
access/report side rather than opening a new branch. **Honest finding: it does not yet clear the
faithful-access bar on the current checkpoint** — a real negative-to-marginal result, recorded in
`audits/stage7_latent_only_tune_prob_035.json`.

Why it matters. Stage 7's bounded support rests on the local calibrated reporter, but that reporter
is a schema-aware structural round-trip: it reads the scored content fields (current and previous
attended visible type, attended digit, glimpse digit) from attended-content token bases (`x111xx`,
`x112xx`, `x113xx`, `x211xx`, ...) the renderer fills *directly* from the model's attended content.
Because content is handed to the decoder pre-labelled, the anti-memorization falsifiers do not bite
(see ROADMAP "Sharper decoder caveat").

What the latent-only decoder does. It recovers the scored content from an **opaque, quantised view of
the controller/attention/memory state alone** (`_latent_feature_matrix`: coarse per-chunk levels, no
schema field names, content tokens withheld), fit on the held-out translator+calibration pool and
evaluated on held-out, cue-switch, and intervention slices. Because content is *learned* from opaque
internal state rather than read from a schema-known token, held-out and counterfactual slices are
genuine faithfulness tests. The unit test confirms the mechanism (it recovers content when present and
provably ignores the exact-content fields as input).

Finding on the current discrete-attention checkpoint (`audits/stage7_latent_only_tune_prob_035.json`):

- A small, non-robust **current-content** advantage on the 8-example slice (`+0.125`, rising to
  `+0.25` as the opaque interface is widened from 8×4 to 48×8 levels) — but it vanishes on the larger
  16-example slice and on both the cue-switch and intervention slices.
- **Remembered/counterfactual content is never recovered** above observation (`content_supported`
  is `false` for every interface width and slice).
- Reading: the coarse opaque latent interface on this checkpoint carries marginal current-attended
  signal at best and no reliable remembered or counterfactual content. This **bounds** the Stage 7
  faithful-access claim to the schema-aware round-trip; the genuine faithfulness leg stays open.

Remaining sub-steps (the real next work):

- [x] Add a bottleneck diagnostic that compares the shipped quantised latent interface against a
  richer continuous-internal-state-only probe, without exposing directly encoded content tokens.
  Result (`audits/stage7_latent_followup_tune_prob_035.json`): the richer continuous probe still
  does **not** recover joint current, remembered, or content-only fields on the current checkpoint.
  Some individual previous-visible-type signal is present (`~0.25-0.46` depending on slice/interface),
  but attended digits and joint content stay near chance. This points away from quantisation as the
  sole bottleneck and toward the checkpoint/state representation itself lacking separable faithful
  content. A follow-up condition that includes the model's actual current/previous observation
  feedback channel improves remembered-field recovery (`memory_content_joint_accuracy_advantage`
  `~+0.08` to `+0.21` depending on slice), but still does not recover full joint content, so the
  remaining target is a checkpoint that carries sensory feedback into separable controller state
  rather than only leaving it available as immediate recurrent input.
- [ ] Re-run the latent-only decoder on a checkpoint whose remembered-attention state is more
  separably encoded (e.g. a memory-regularised or longer-trained recipe), to test whether faithful
  remembered-content recovery is reachable at all.
- [~] First widened-checkpoint pilot completed (`configs/stage7_longer_wide.yaml`,
  `audits/stage7_latent_followup_longer_wide.json`): a 64-hidden / 24-scene-embedding checkpoint
  trained for 3000 steps reached recurrent validation accuracy `0.395` vs static `0.247`, so it is
  viable but weaker than `tune_prob_035`. Latent-only joint content still does **not** clear the bar:
  the best quantised runs reach only `+0.0417` current/memory joint advantages on some slices and
  `content_only_joint_accuracy_advantage` remains `0.0`; continuous and feedback-channel diagnostics
  also remain unsupported. The pilot improves visible-type field recovery but not attended digits or
  full content binding.
- [ ] If it stays negative, treat the external API LLM / VLM path (LLM path now runnable with
  `gpt-5-mini`; a *powered* support run and the VLM path are still outstanding) as the
  only remaining route to the strong Stage 7 faithfulness claim, and keep the round-trip reporter as
  the (clearly labelled) bounded local result.
- [x] Smoke-test the external API LLM path on the strict latent-only interface
  (`scripts/stage7_external_llm_audit.py`, `audits/stage7_external_llm_tiny_tune_prob_035.json`).
  The path is now live with `gpt-5-mini`, so API/model access is no longer the immediate blocker.
  Tiny result: neither latent-only nor observation-only LLM reporting recovered joint current,
  remembered, or content-only fields on the 2-example smoke slice (`content_supported = false`).
  This is not a powered support test, but it confirms the external route is runnable and currently
  follows the same negative direction as the local latent-only probes.
- [x] Extend the external API LLM smoke audit across Stage 7 slices
  (`audits/stage7_external_llm_multislice_tiny_tune_prob_035.json`). A one-example-per-slice run on
  default, cue-switch, and intervened examples completed with `gpt-5-mini`; every slice remained
  negative for latent-only joint current, remembered, and content-only recovery. This is still only a
  route/plumbing result, but it removes "external path not runnable" as the reason Stage 7 is open.
- [ ] **Solution direction:** stop treating Stage 7 as a decoder problem. The local latent probes,
  richer continuous-state probes, feedback-channel diagnostic, widened checkpoint, and external LLM
  smoke tests all point the same way: the current checkpoint uses glimpse content transiently but
  does not carry current/remembered attended content in a separable report state. The next experiment
  is therefore a **memory-regularized Stage 7 checkpoint**: add a post-glimpse report state and train
  it with a small auxiliary content-memory objective for current attended visible type/digit and
  previous attended visible type/digit/glimpse digit. Keep the claim boundary explicit: if this works,
  it supports "faithful latent reportability under explicit content-memory regularization," not
  spontaneous Stage 7 reportability.
- [~] First memory-regularized checkpoint pilot completed
  (`configs/stage7_content_memory.yaml`, `audits/stage7_latent_followup_content_memory.json`).
  The checkpoint is task-viable but weaker than `tune_prob_035` (recurrent validation accuracy
  `0.325` vs static `0.166`). Using the trained `content_memory_state_seq` as the latent report
  state produces the first clear strict-Stage-7 movement: best quantised runs reach current/memory
  joint advantages of `+0.0417/+0.3333` on default, `+0.125/+0.3333` on cue-switch,
  `+0.0/+0.375` on intervention baseline, and `+0.1667/+0.3333` on intervened examples; previous
  attended digit and previous glimpse digit often reach `0.83-1.0` accuracy. However,
  `content_only_joint_accuracy_advantage` remains `0.0` on every slice, so strict Stage 7 is still
  **not supported**. Reading: content-memory regularization is the right direction, but the auxiliary
  target must also bind visible type, digit, location, and report-control fields strongly enough for
  full joint content recovery.
- [~] Second memory-regularized checkpoint pilot completed
  (`configs/stage7_content_memory_v2.yaml`, `audits/stage7_latent_followup_content_memory_v2.json`).
  This version trains the content report state against the stricter report schema: current/previous
  attended cell, current/previous visible type and digit, current/previous glimpse digit, previous
  cue, inspection counts, and the binary report-control flags. The checkpoint remains task-viable
  (recurrent validation accuracy `0.293` vs static `0.166`) and improves several quantised
  report-state slices: best current/memory joint advantages reach `+0.2083/+0.1667` on default,
  `+0.125/+0.375` on cue-switch, `+0.2083/+0.125` on intervention baseline, and
  `+0.25/+0.2083` on intervened examples. Individual digit readouts are strong
  (`0.625-0.9583` on several slices). However, `content_only_joint_accuracy_advantage` is still
  `0.0` in every slice, so strict Stage 7 remains **not supported**. Reading: wider auxiliary
  supervision makes more content recoverable, but the joint bottleneck is now likely field
  compositionality/calibration under the opaque reporter rather than absence of any content signal.
  Next experiment should either factor the reporter output by field before applying the joint claim,
  or train the report state with an explicit joint structured decoding objective instead of only
  independent heads.
- [ ] Keep the symbolic dump as an upper-bound baseline, not the Stage 7 claim.

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

- [~] Evaluate external API LLM reporting under default, cue-switch, and intervention slices. (Path runnable with `gpt-5-mini` — one-example-per-slice smoke run completed, all slices negative for joint content; a *powered* multi-example support run is still outstanding. See `audits/stage7_external_llm_multislice_tiny_tune_prob_035.json`.)
- [ ] Add a VLM-based Stage 7 path using minimally labeled visual internal-state renderings. (Blocked: vision model.)
- [ ] Compare VLM reports against scene-only and explicit symbolic-dump baselines.
- [~] Add token-remapping and held-out combination tests for the local opaque-token reporter. **Investigated: not meaningful against the current local reporter.** The local decoder reads the scored content fields from attended-content token bases the renderer fills *directly* from the model's attended content (not the learned translator's predictions, nor the opaque latent-bit tokens), so it is a schema-aware structural round-trip: a consistent token remapping is invariant by construction and held-out combinations do not bite directly-encoded fields. The genuine anti-memorization test needs a **latent-only decoder** (forced to recover content from the opaque latent-bit tokens alone) or the external LLM/VLM path. See ROADMAP "Sharper decoder caveat".
- [x] **Implemented — see [Current Focus](#current-focus-latent-only-stage-7-decoder).** Built a latent-only decoder (`run_latent_only_report_mode`) that recovers the scored content from an opaque quantised view of internal state alone, with the directly-encoded content bases withheld, so held-out-combination and counterfactual-tension slices become meaningful. **Honest finding:** it does not yet clear the faithful-access bar on the current checkpoint (marginal, non-robust current-content advantage; no remembered/counterfactual recovery); `content_supported = false`. See `audits/stage7_latent_only_tune_prob_035.json`. Remaining: re-run on a checkpoint with more separable remembered-attention state, or fall back to the external LLM/VLM path.
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

- [~] At least one access/report family has robust support. (Stage 3 explicit-attention-modeling is robust; Stage 6A is capacity-audited and now backed by an empirical noise floor; Stage 7's local-reporter content claim is weak — a symbolic round-trip. The latent-only decoder built to harden it does **not** clear the bar on the current checkpoint, so the access/report side is strong but its strongest reportability leg still needs a checkpoint with more separable remembered-attention state or the external path. See `audits/stage7_latent_only_tune_prob_035.json`.)
- [ ] At least one non-reportability family has robust support. (Perturbational complexity has **bounded** support; needs multi-seed + cross-system for robust.)
- [ ] The supported families point to the same internal contents, not merely the same checkpoint.
- [x] Comparator systems fail in predicted ways. (All negative controls and comparators fail as intended; `shuffle_feedback` drops accuracy by `0.27`.)
- [ ] Results replicate across at least one different architecture.
- [ ] Results replicate across at least one different benchmark.
- [ ] The final claim is framed as consciousness-relevant evidence, not proof of consciousness.
