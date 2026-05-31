# Priority 1 Audit Status (post-rehab)

This note supersedes the earlier Priority 1 disposition. The earlier disposition
described audits run on a **non-functional** checkpoint: under the shipped soft-attention
recipe the recurrent controller collapsed to uniform attention (`target_attention ~= 0.04 = 1/25`)
at near-chance accuracy and lost to the static baseline, so every probe-advantage "support"
was an artifact. See the root-cause and fix in the repo history (discrete glimpse readout).

Tracked artifacts:

- full-run summary: `audits/post_rehab_full_eval_tune_prob_035_summary.json`
- checkpoint path: `outputs/tune_prob_035/experiment.pt` (gitignored; regenerate with train+eval)
- config base: `configs/tune_prob_035.yaml`

## What the rehab changed

- The glimpse now reads the single most-attended cell (`model.hard_attention`, straight-through)
  while the attention policy stays soft. This makes the closed-loop search learnable.
- The base config disables the Stage 4B causal policy-feedback path (it destabilises the base
  task); learned-self-model emergence is studied separately.
- Stage 6B is now probed against controller **state** vs a capacity-matched observation
  baseline (it previously had no state probe and compared a ground-truth / untrained "native"
  score, which was circular).
- Stage 4A trains the native self-state head (`self_model_weight`).

## Current full-run dispositions (discrete-attention checkpoint)

- **Dissociation / closed-loop / cue-dependence**: supported. recurrent acc `0.44` vs static
  `0.17`; `target_inspected_rate` `0.39` vs `0.08`.
- **Stage 3 explicit attention modeling**: **robust**. `predictive_probe` and `intervention_test`
  supported on every seed (`stage3_multi_seed` 1.0/1.0); `stage3_checkpoint_family` verdict
  `robust` across the default and `0.25` reduced-shaping checkpoints.
- **Stage 4A engineered self-state**: supported (native cell accuracy `~0.99`).
- **Stage 4B learned self-model feedback**: not supported in the base config (feedback path off
  by design; emergence is Phase 3).
- **Stage 5 cue-switch**: supported (recurrent switch accuracy `0.25` vs baseline `0.0`).
- **Stage 6A structured reportability**: supported, capacity audit passes. An empirical
  permuted-label noise floor (`noise_floor_metrics`) now backs the strong report signals: the
  real controller-vs-observation accuracy advantages (`~0.38`, `~0.42`) are roughly 100x above
  the permuted-label p95 floor (`~0.004`, `~0.003`), so the claim is significant rather than a
  probe-capacity artifact. This replaces the hardcoded directional thresholds for those signals.
- **Stage 6B uncertainty / allocation error**: bounded / provisional. Positive controller-state
  recall advantage on all four gated signals; the accuracy-guarded capacity audit does not pass
  (`revisit_unresolved`, `allocation_error` have marginally negative accuracy advantage).
- **Stage 7 local opaque-token reporter**: supported, capacity audit passes. External API LLM and
  VLM routes remain open. Sharper caveat: the local decoder reads the scored content fields from
  attended-content token bases the renderer fills *directly* from the model's attended content (not
  the learned translator's predictions, nor the opaque latent-bit tokens), so it is a schema-aware
  structural round-trip. Consequently consistent token-remapping and held-out-combination
  anti-memorization tests do not bite it; the genuine faithfulness test needs a latent-only decoder
  or the external LLM/VLM path. See ROADMAP "Sharper decoder caveat".
- **Negative controls**: all fail as intended. `shuffle_feedback` accuracy drop `0.27`,
  `feedforward_summary` `0.21`, high-capacity observation-only and the matched-transformer /
  trivial-regulator comparators all fail as intended.
- **Reduced shaping**: resilient at weight `0.25` (acc `0.34`); complete zero-shaping collapses to
  `~0.19` (≈ static). Complete zero-shaping is **not** supported; the `min_accuracy = 0.15`
  threshold is too lenient and is flagged for empirical calibration.

## Current claim boundary

These are real, capacity-audited, comparator-resistant results on a single working
checkpoint/seed-family. They are **bounded support**, not robust support for a Stage 8
consciousness claim: cross-architecture and cross-benchmark replication, non-reportability
theory branches, and emergent (non-supervised) self-modeling are still required. Stage 6B
remains provisional and complete zero-shaping resilience is still a known weakness.
