# Next Steps Checklist

This checklist turns the revised roadmap into a working execution order. The goal is to move from bounded benchmark support toward robust, comparator-resistant evidence without blurring the current claim boundaries.

> **Post-rehab note.** The Priority 1 audit machinery below was originally exercised on a
> non-functional checkpoint (under the old fully-soft recipe the recurrent controller collapsed
> to uniform attention and lost to the static baseline). After the discrete-glimpse fix, the
> audits, negative controls, and comparator suite were re-run on a model that actually does the
> task: dissociation/Stage 3/Stage 4A/Stage 5/Stage 6A are genuinely supported (Stage 3
> robustly), Stage 7 has a bounded schema-aware route plus seed-robust remembered-content recovery
> under explicit content-memory regularization (strict `content_only` remains seed-fragile), Stage 6B has bounded
> positive evidence but fails its full calibrated support gate, and all negative controls + comparators fail as
> intended. See `audits/post_rehab_full_eval_tune_prob_035_summary.json` and
> `docs/PRIORITY1_AUDIT_STATUS.md`. The Priority 1 boxes are therefore genuinely validated,
> not artifacts. The active work has moved to the partial Stage 8 gates.

## Current Focus: Remove Imposed Stage 8 Mechanisms

The executable convergence audit is **not met**: three gates pass and five are partial. The
highest-value next experiments are:

- repeat the temporal-relay transformer over fresh data/model seeds
- remove forced shared state from the integrated-content and temporal-relay assays without losing
  task viability
- induce routing through a naturally coupled task or resource constraint with identical training
  and evaluation scaling, then require transfer against the neutral dual-lane control
- align access/report and non-reportability interventions on independently learned
  representations of the same target content
- rerun comparator and negative-control suites on the replicated architectures and benchmark

The latest evidence is in `audits/stage8_convergence_current.json`,
`audits/stage8_task_induced_routing_correction.json`,
`audits/stage8_temporal_relay_multiseed.json`, and
`audits/stage8_temporal_relay_transformer_pilot.json`.

## Completed Focus: Latent-Only Stage 7 Decoder

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

Experiment record (chronological):

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
- [x] Re-run the latent-only decoder on a checkpoint whose remembered-attention state is more
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
- [x] Test the external API LLM / VLM fallback path. Both powered `gpt-5-mini` variants are
  unsupported on the current v3 interface. The VLM result is especially well controlled: the
  label-free latent heatmap is `0/8` on all three joint-content families in all three slices, while
  the explicit symbolic-image upper bound is `8/8` on every joint metric and the entire report.
  Keep the round-trip reporter as the clearly labelled bounded local result.
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
- [x] **Solution direction:** stop treating Stage 7 as a decoder problem. The local latent probes,
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
- [x] Third memory-regularized checkpoint pilot completed
  (`configs/stage7_content_memory_v3.yaml`, `audits/stage7_latent_followup_content_memory_v3.json`).
  The v2 bottleneck audit showed that visible-type fields were the lowest-accuracy strict-report
  fields even when digit memory was strong, so v3 adds the controller's attended visible-type glimpse
  to the post-glimpse `content_memory_state_seq` adapter. The checkpoint remains task-viable
  (recurrent validation accuracy `0.324` vs static `0.166`) and is the first strict-positive
  Stage 7 pilot under the quantised opaque reporter: best quantised current/memory/content-only
  advantages are `+0.3333/+0.6667/+0.0833` on default, `+0.2917/+0.6667/+0.0417` on cue-switch,
  `+0.3333/+0.7083/+0.0833` on intervention baseline, and
  `+0.2083/+0.4583/+0.0417` on intervened examples. Claim boundary: this supports
  **faithful latent reportability under explicit content-memory regularization with visible-glimpse
  binding**, not spontaneous Stage 7 reportability in the original checkpoint. The enriched
  bottleneck fields show the remaining weak spots are mostly attended digit, inspected count, and
  current-wrong-candidate calibration; the continuous diagnostic remains negative, so the next
  robustness step is replicate v3 across seeds and tighten the quantised reporter selection policy.
- [x] Calibrate the v3 positive against a permuted-label noise floor
  (`scripts/stage7_latent_noise_floor.py`, `audits/stage7_latent_noise_floor_content_memory_v3.json`).
  The pilots' `content_supported` gate is only a directional `>0` test, so a `+1/24` joint advantage
  flips it true with no significance floor — the same gap the Stage 6A claim closed with
  `noise_floor_metrics`. This audit refits the latent decoder 120x per slice/interface with fit-time
  labels permuted (features and probe-init held fixed, so labels are the only variable) and compares
  the observed advantage against the permuted p95. **Result: the v3 positive survives.** On both
  claimed-positive interfaces (32x8, 48x8) across all four slices, the permuted `content_only`
  advantage is a point mass at exactly `0.0` (mean = p95 = max = `0.0` over 120 permutations) — under
  shuffled labels the decoder *never once* beats observation on joint content — while the real
  advantage is `+0.0417`/`+0.0833`; current (`+0.21..+0.33` vs p95 `~0.08-0.125`) and memory
  (`+0.50..+0.67` vs p95 `~0.08`) clear by wide margins. So the v3 content result is **not** a
  probe-capacity/label-fit artifact. Honest boundaries this does *not* remove: the `content_only`
  effect is significant-but-tiny (1-2 of 24 joint examples), it is single-checkpoint and
  single-init-seed, current-content recovery is partly circular by construction (the visible-glimpse
  is fed into the report state it is scored from), and cross-seed/architecture/benchmark replication
  is absent. Significance is established; robustness and non-circularity are not.
- [x] Replace the bare `content_supported = (current>0 and memory>0 and content_only>0)` gate in the
  latent pilots with the permuted-label floor gate (`content_supported_vs_floor`), so future Stage 7
  checkpoints report significance rather than direction. Implemented in
  `run_latent_content_noise_floor`: the observed and null decoders share probe initialization, only
  fit-time labels are permuted, and global RNG state is preserved. The follow-up pilot now runs a
  configurable p95 audit (12 permutations by default) for every slice/interface and assigns its
  primary `content_supported` field from the calibrated gate; the old result remains explicitly
  labeled `content_supported_directional`. Published claims should continue to use a powered
  standalone run (80-200 permutations), because the integrated default is a regression guard rather
  than a high-resolution estimate of the null tail.
- [x] Replicate v3 across 3 fresh training seeds (107/207/307, stride 100 from v3's seed 7;
  `configs/stage7_content_memory_v3_seed*.yaml`, `scripts/stage7_multiseed_v3.sh`,
  `audits/stage7_latent_noise_floor_v3_seed*.json`). All three checkpoints are task-viable
  (recurrent val `0.322`/`0.323`/`0.326` vs static `0.261`/`0.171`/`0.209`). **The replication splits
  the claim cleanly into a robust leg and a fragile leg:**
  - **Robust across all 4 seeds (32/32 slice*interface cells clear the floor): remembered-content
    (`memory`) joint recovery.** Observed advantage per-seed means `+0.41..+0.64` (range `+0.33..+0.79`)
    vs a permuted p95 of `~0.08` — large and non-circular (previous/remembered attended content is
    carried by the recurrent memory state, not fed in via the visible-glimpse). `current` joint also
    clears 4/4 seeds (means `+0.24..+0.33`) but is partly circular by construction, so it carries less
    weight. This is the defensible, seed-robust Stage 7 result: faithful latent reportability of
    *remembered* attended content under content-memory regularization.
  - **Seed-fragile: the strict `content_only` leg (content jointly beyond observation).** It clears
    fully on seeds 7 and 207 (8/8 cells), partially on 307 (4/8), and **not at all on 107 (0/8 —
    advantage exactly `0.0` everywhere)**. It is significant whenever it appears (always beats the
    degenerate point-mass-at-0 null) but its magnitude is a knife-edge `1-2/24`, so on some seeds the
    decoder lands exactly at the observation baseline. So strict `content_only` is **significant but
    not robust** — do not claim it as robust Stage 7 support on the strength of v3 alone.
  Net: the earlier "v3 = first strict-positive Stage 7" headline is downgraded to *significant-but-
  seed-fragile* for the strict leg; what actually replicates is the **remembered-content** leg.
- [x] Keep the symbolic dump as an upper-bound baseline, not the Stage 7 claim. The powered VLM
  audit does this explicitly and requires the symbolic-image control to pass before latent support
  is even eligible.

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

The causal follow-up is now complete. A fitted inspection-map direction moves its decoded report
and produces selected-cell attention effects above a 32-direction matched random-state floor, in
both the task-only and supervised checkpoints. However, across scales `0.25/0.5/1.0/2.0`, raising
the direction's "already inspected" report consistently **increases** attention to that cell at the
intervention and following step. This is the opposite of the avoidance/reallocation effect required
for an inspection-history regulatory self-model. Stage 4B therefore remains unsupported: the state
contains decodable inspection-correlated structure, but the causal test does not establish that the
policy uses it with the claimed regulatory semantics.

- [x] Define a Stage 4B training condition with no direct self-model objective.
- [x] Train fresh checkpoints under the task-only or indirectly induced self-modeling objective.
- [x] Evaluate hidden-state inspected-map and target-inspected probes against previous-observation baselines.
- [x] Test hidden-state interventions for effects on self-model reports and attention policy.
  Completed with matched random-direction controls and a four-scale sweep on both task-only and
  supervised checkpoints. Report and cell-specific attention effects are real, but the attention
  effect has the wrong regulatory sign at every scale; `supported = false`.
- [x] Compare against the supervised Stage 4B feedback checkpoint.
- [x] Decide whether Stage 4B remains an engineered capability probe or gains consciousness-relevant evidence status. (Engineered capability probe; the emergent component is weak.)

## Priority 3: Finish Stage 7 Variants

GitHub issue: [#5](https://github.com/macterra/attcon/issues/5)

- [x] Evaluate external API LLM reporting under default, cue-switch, and intervention slices. The
  72-request `gpt-5-mini` run completed with same-model latent-vs-observation controls and exact
  paired sign tests. It is unsupported on the v3 content-memory interface: latent-only current-
  content joint accuracy was `0/12`, `1/12`, and `0/12` across the three slices versus observation-
  only `3/12`, `5/12`, and `1/12`; remembered and full-content joint accuracy were zero throughout.
  Artifact: `audits/stage7_external_llm_powered_content_memory_v3.json`.
- [x] Add a VLM-based Stage 7 path using minimally labeled visual internal-state renderings.
  Implemented with a fixed eight-level, label-free activation heatmap and GPT-5 mini image input.
- [x] Compare VLM reports against scene-only and explicit symbolic-dump baselines. Powered result:
  latent heatmap `0/8` on current/memory/full content in every slice; symbolic upper bound `8/8`
  throughout; `content_supported = false`.
- [~] Add token-remapping and held-out combination tests for the local opaque-token reporter. **Investigated: not meaningful against the current local reporter.** The local decoder reads the scored content fields from attended-content token bases the renderer fills *directly* from the model's attended content (not the learned translator's predictions, nor the opaque latent-bit tokens), so it is a schema-aware structural round-trip: a consistent token remapping is invariant by construction and held-out combinations do not bite directly-encoded fields. The genuine anti-memorization test needs a **latent-only decoder** (forced to recover content from the opaque latent-bit tokens alone) or the external LLM/VLM path. See ROADMAP "Sharper decoder caveat".
- [x] **Implemented — see [Completed Focus](#completed-focus-latent-only-stage-7-decoder).** Built a latent-only decoder (`run_latent_only_report_mode`) that recovers the scored content from an opaque quantised view of internal state alone, with the directly-encoded content bases withheld. The original checkpoint remains negative-to-marginal; the v3 memory-regularized recipe yields seed-robust remembered-content recovery, while strict `content_only` recovery is seed-fragile. See `audits/stage7_latent_only_tune_prob_035.json` and `audits/stage7_latent_noise_floor_v3_seed*.json`.
- [x] Keep the symbolic dump as an upper-bound baseline, not the main Stage 7 claim.

## Priority 4: Build New Theory Branches

GitHub issue: [#6](https://github.com/macterra/attcon/issues/6)

- [x] Extend the benchmark with independently recombinable attributes, held-out conjunctions, and false-binding lures for Branch C. The standalone scaffold is implemented in `attcon.binding` and audited over 4,096 cases in `audits/branch_c_binding_dataset.json`: 808 deterministic held-out conjunctions, complete train coverage of every individual target feature value, and zero invalid false-binding lures.
- [x] Add Branch C unity/binding experiments with bound-content probes and intervention tests. The explicit shared-selector family reaches `1.00` joint accuracy, lure rejection, and intervention coherence in the original and surface-v2 benchmarks, then repeats across three independent data/model-seed pairs per variant (`audits/branch_c_binding_multiseed.json`). A structurally different cue-token set transformer also passes every frozen gate on both variants (`audits/branch_c_binding_cross_model.json`). Its object-pooled controls are exactly parameter matched (20,073 and 19,487 parameters) and reach only `0.014`/`0.499` and `0.006`/`0.500` joint/lure performance, versus `1.00`/`1.00` integrated. Branch C now clears its predeclared support threshold. Keep the status at strong bounded rather than robust until the transformer family is multi-seed and the finding transfers beyond this synthetic selection-task family.
- [x] Extend the benchmark with query-change and alternative-target conditions for Branch D. `attcon.counterfactual_access` now generates balanced unavailable, merely visible, previously attended, and counterfactually accessible targets. The 4,096-case audit (`audits/branch_d_access_dataset.json`) has 831 held-out query/value pairs, keeps current fixation unchanged across 100% of query switches, and creates observation/cache tension in 100% of counterfactual cases. Scene-only accuracy is `1.00` on merely visible and unavailable cases but `0.00` on previously attended and counterfactual-tension cases; the current-glimpse control stays near value-chance and cannot exploit the switched target. This is benchmark infrastructure, not Branch D evidence.
- [x] Add Branch D counterfactual-access experiments for non-current but query-available contents. The unstructured GRU (`audits/branch_d_access_pilot.json`) is a decisive negative control: `1.00` training accuracy but `0.00` across all 3,228 held-out pairs. The relational GRU passes every gate and repeats over three fresh data/model-seed pairs (`audits/branch_d_access_multiseed.json`). The cross-model audit (`audits/branch_d_access_cross_model.json`) replaces recurrence with a permutation-equivariant event-set transformer and changes the surface schema, item/key/value cardinalities, split ratio, and seeds. Both variants again have `1.00` minimum accuracy, no-cache advantage, cache-erasure drop, and observation-conflict retention; controls are exactly parameter matched. Branch D now has strong bounded support across seeds, architectures, and two surface variants. Both successful models still share explicit relational key addressing, so this is not spontaneous access in the original controller or robust Stage 8 evidence.
- [x] Extend the benchmark with stale-access, inferred-content, and wrong-access lures for Branch E. `attcon.higher_order` generates six-way groups that hold the exact first-order key/value constant across fresh-current, fresh-memory, inferred-content, unavailable, stale-access, and wrong-access states. The 4,092-case audit (`audits/branch_e_higher_order_dataset.json`) has 682 complete groups, 816 held-out content/status conjunctions, zero invariant failures, and exact observation matches between every fresh-current/wrong-access pair. A content-only status oracle is capped at `0.167`; an observation-only oracle at `0.333`. This is benchmark infrastructure, not higher-order evidence.
- [~] Add Branch E higher-order state-representation experiments that separate first-order content from access, confidence, and report-grounding state. The first pilot (`audits/branch_e_higher_order_pilot.json`) withholds the exact six-way status labels from representation learning and trains one shared latent through report, confidence, reinspection, and correction behavior. On 3,267 held-out content/status conjunctions, its frozen status probe reaches `1.00`, versus `0.00` first-order and `0.302` observation-only using identical 4,550-parameter probes. Across 537 fresh-current/wrong-access pairs with identical first-order content and observation, latent swaps raise confidence, disable reinspection/correction, and recover the newly accessible content at `1.00`. Every engineering gate passes. This is engineering support only because three behavior targets directly reward access-sensitive distinctions. Branch E remains theoretically unsupported until comparable structure emerges under objectives without those higher-order rewards.
- [x] Add separable downstream consumers for Branch F: action, report, uncertainty, reallocation, memory, and language-shaped report paths. `attcon.broadcast` defines all six interfaces in cue-strength sweeps that hold content and evidence fixed. The 4,095-case audit (`audits/branch_f_broadcast_dataset.json`) contains 819 complete threshold crossings and 819 held-out content/strength conjunctions. Local action availability is `1.00` below and above threshold, while the five broad consumers are jointly unavailable below ignition and have perfectly aligned onset above it. This is benchmark infrastructure, not broadcast evidence.
- [~] Add Branch F broadcast/ignition experiments with coordinated intervention tests. The first exactly matched pilot passes every engineering gate. The three-run robustness audit (`audits/branch_f_broadcast_multiseed.json`) repeats it with fresh data/model seeds and a fixed reduced budget: every gate passes in every run. Minimum shared joint accuracy is `0.962`, onset accuracy/alignment `1.00`, shared-ablation drop `0.841`, coordination advantage `0.659`, and donor-content follow rate `0.984`; maximum private single-route damage is `0.192`, and local action remains invariant. This makes the imposed-bottleneck result seed-robust engineering support. Because all consumers and the shared bottleneck are directly supervised, spontaneous broadcast remains unsupported.
- [x] Add perturbational-complexity diagnostics over controller and self-model state. (`perturbational_complexity_metrics`; initial bounded result: rich-but-recoverable dynamics propagate farther than a no-recurrence control and recover unlike a frozen-state control. The multi-seed and RNN replication is recorded in the next item.)
- [x] Multi-seed + cross-checkpoint robustness for the perturbational family (`scripts/perturbational_multiseed.py`, `audits/perturbational_multiseed.json`). Under a standardized perturbation config, `supported` (rich-but-recoverable AND integration>feedforward AND recovery>freeze) holds on **100% of 25 perturbation seeds on all 4 checkpoints** — the primary `tune_prob_035` controller plus the 3 independently-trained v3 content-memory controllers (seeds 107/207/307). Recurrent attention-propagation exceeds the feedforward control on every seed (`tune_prob_035` `~0.59` vs `~0.13`; v3 seeds `~0.15` vs `~0.06` — smaller margin on the content-memory recipe but still robust), and recurrent recovery (`~0.39-0.46`) exceeds the frozen-state control (`0.000`) everywhere. So the non-reportability family is now **perturbation-seed robust and cross-training-seed / cross-recipe replicated**. Boundary: all four are the *same* recurrent controller architecture, so this does **not** yet satisfy the Stage 8 cross-architecture requirement (item d) — that still needs a structurally different controller.

## Priority 5: Replicate Across Systems

GitHub issue: [#7](https://github.com/macterra/attcon/issues/7)

- [~] Replicate supported claims on a structurally different controller architecture. **First cross-architecture pass done: an ungated `nn.RNNCell` controller** (`ModelConfig.controller_kind: "rnn"`, `configs/tune_prob_035_rnn.yaml`, `outputs/tune_prob_035_rnn/`) vs the gated GRU. The RNN is task-viable but weaker (recurrent val `0.31` vs GRU `0.44`). Of 7 comparable supported claims, **6 replicate**: Stage 3 explicit-attention-modeling (`robust_supported` on 3 seeds), Stage 6A report probes (and the RNN Stage 6A advantages `0.41`/`0.45` clear their own permuted-label floor at p95 `~0.007`/`~0.001`), dissociation, closed-loop adaptation, cue-dependence, and perturbational complexity (`audits/perturbational_multiseed_with_rnn.json`, 100% of 25 seeds). **One drops: `cue_switch_adaptation`** (supported on GRU, false on the RNN — the weaker recurrence does not reallocate attention on a mid-episode cue switch). So both evidence families' headline claims replicate on a different recurrent architecture; a genuinely non-recurrent-*family* architecture (e.g. LSTM's dual state, or a state-space controller) and re-running the comparator/negative-control suite on the RNN remain.
- [~] Replicate supported claims on a second benchmark with different task structure. The generic temporal-relay GRU is a decisive memorization control (`0.020` held-out joint). The relational GRU clears all ten frozen gates in all three fresh runs (`audits/stage8_temporal_relay_multiseed.json`): every minimum task, directional, null-advantage, and stability metric is `1.00`, with `0.934` minimum order-destroyed advantage. A position-aware relational transformer also clears every gate at `1.00` on its first seed (`0.951` order-destroyed advantage; `audits/stage8_temporal_relay_transformer_pilot.json`). This is seed-robust GRU and single-seed cross-architecture engineering transfer. It remains partial because query matching and shared state are explicit. Next: repeat the transformer across seeds and remove forced sharing.
- [ ] Re-run comparator and negative-control suites on the replicated systems.
- [~] Check whether any Stage 8-relevant contents show cross-validated causal overlap across branches. Whole-state swaps and stricter disjoint-split directions are multi-seed robust inside the explicitly shared assay (`audits/stage8_integrated_content_multiseed.json`, `audits/stage8_integrated_content_directional_multiseed.json`). Removing forced sharing remains negative. Ordinary joint supervision yields `0.00` transfer. The apparent dropout-induced effect was confounded by inverted scaling: at `0.95` dropout the surviving private state was `20×` larger during training than evaluation. Corrected zero-or-normal occlusion keeps both controls task-perfect but produces at most `0.021` transfer; a viability-first curriculum produces `0.001` (`audits/stage8_task_induced_routing_correction.json`). Learned routing is unsupported. Next: use a naturally coupled task or resource constraint with identical train/evaluation scaling, and require emergence against the neutral dual-lane control.

## Stage 8 Gate

GitHub issue: [#8](https://github.com/macterra/attcon/issues/8)

Do not claim Stage 8 support until all of the following are true. **Current status: not met.**
The executable artifact audit (`audits/stage8_convergence_current.json`) currently records three
passing gates, five partial gates, and zero failures. Same-content causal convergence and
different-benchmark transfer now each have seed-robust engineered assays. They remain partial
because the shared bottleneck and relational matching are imposed rather than independently
emerging across qualifying families. Branch E/F engineering results remain excluded from
theoretical-family counts. The next decisive experiments remove forced sharing on both benchmarks,
repeat the temporal-relay transformer across seeds, and rerun matched controls on the replicated
systems.
The methodology now produces one of each partition type (a robust access/report family and a
non-reportability family) and comparators fail as intended. Both families' headline claims now
also replicate on a second (ungated-RNN) controller architecture — so cross-*architecture*
replication is under way (6/7 supported claims transfer; `cue_switch_adaptation` drops). Still
open: content-identity across independently learned families is unestablished, temporal-relay
transfer still imposes relational matching/shared state, and the strict Stage 7 content-only leg
is not seed-robust.

- [~] At least one access/report family has robust support. (Stage 3 explicit-attention-modeling is robust; Stage 6A is capacity-audited and now backed by an empirical noise floor; Stage 7's local-reporter content claim is weak — a symbolic round-trip. The latent-only decoder does **not** clear the bar on the shipped `tune_prob_035` checkpoint (`audits/stage7_latent_only_tune_prob_035.json`). On the memory-regularized v3 recipe, a permuted-label noise floor plus a 3-seed replication (`audits/stage7_latent_noise_floor_v3_seed*.json`) splits the claim: **remembered-content (`memory`) latent recovery clears the floor on all 4 seeds** (means `+0.41..+0.64` vs p95 `~0.08`; non-circular) — a genuinely seed-robust reportability result for *remembered* attended content under content-memory regularization — while the strict `content_only` leg is significant-when-present but **seed-fragile** (full on 2/4 seeds, partial on 1, null on 1) and partly circular for current content. So the access/report side is strong and now has one seed-robust latent reportability leg (remembered content) on an engineered recipe; the strict content-only claim is not yet robust, and the spontaneous/emergent version still needs a checkpoint with separably encoded remembered-attention state or the external path. Stage 3 and Stage 6A (the robust access/report claims) now also **replicate on a different architecture** — an ungated RNN controller, Stage 6A clearing its own noise floor there; benchmark replication remains absent. See gate item (d).)
- [~] At least one non-reportability family has robust support. (Perturbational complexity is now **perturbation-seed robust and cross-training-seed replicated** — `supported` on 100% of 25 seeds across `tune_prob_035` + 3 independently-trained v3 controllers; `audits/perturbational_multiseed.json`. It **also replicates on a structurally different architecture** — the ungated RNN controller (`audits/perturbational_multiseed_with_rnn.json`, 100% of 25 seeds, recurrent propagation `0.316` vs feedforward `0.068`). So this leg is now multi-seed robust *and* cross-architecture and cross-training-seed replicated; the remaining gap is a more distant (non-recurrent-family) architecture and a second benchmark.)
- [ ] The supported families point to the same internal contents, not merely the same checkpoint.
- [x] Comparator systems fail in predicted ways. (All negative controls and comparators fail as intended; `shuffle_feedback` drops accuracy by `0.27`.)
- [~] Results replicate across at least one different architecture. (First cross-architecture pass: an ungated `nn.RNNCell` controller vs the gated GRU replicates 6 of 7 supported claims — Stage 3 robust, Stage 6A noise-floor-clearing, dissociation, closed-loop, cue-dependence, and perturbational; `cue_switch_adaptation` drops on the weaker RNN. See Priority 5 and `audits/cross_architecture_rnn_summary.json` + `audits/perturbational_multiseed_with_rnn.json`. Substantially met for both families' headline claims; a more distant architecture and the comparator/negative-control re-run remain.)
- [~] Results replicate across at least one different benchmark. (The relational temporal-relay audit passes all ten gates across three fresh seeds, with `1.00` minimum task/causal/null metrics and `0.934` minimum order-destroyed advantage. This is structurally different and seed-robust at the engineering level, but explicit relational matching and forced shared state prevent a robust Stage 8 pass. See `audits/stage8_temporal_relay_multiseed.json`.)
- [x] The final claim is framed as consciousness-relevant evidence, not proof of consciousness.
