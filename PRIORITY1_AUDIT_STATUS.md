# Priority 1 Audit Status

This note records the current disposition of the Priority 1 audit machinery after the implementation review in `FEEDBACK.md`. It no longer treats smoke-test numbers as bounded-claim support.

Tracked smoke artifact:

- artifact: `audits/priority1_smoke_tune_prob_035.json`
- checkpoint: `outputs/tune_prob_035/experiment.pt`
- config base: `configs/tune_prob_035.yaml`
- audit mode: reduced probe batches and epochs for fast sanity checking
- stale checkpoint migration: explicitly allowed for this smoke artifact
- full evaluation report: no

## What Changed After Review

- [x] Stage 4B audits now record checkpoint migration metadata.
- [x] Stage 4B trained-head support is disabled when a checkpoint is migrated because report/self-model heads were missing.
- [x] Evaluating a stale checkpoint now requires `--allow-stale-checkpoint` or `evaluation.allow_stale_checkpoint: true`.
- [x] Probe-capacity-matched baselines are now labeled as `linear_probe_input_dim_only`, not full baseline-capacity matches.
- [x] Audit booleans now use explicit thresholds instead of raw `advantage > 0` signs.
- [x] The high-capacity observation-only negative control now receives a previous-observation window.
- [x] Feedforward-summary and shuffled-feedback controls now use a relative accuracy-drop threshold against the recurrent reference.
- [x] The matched-transformer comparator defaults to the recurrent training budget.
- [x] The large-LM-without-loop entry is marked as a Stage 7 observation-only proxy, not an independent comparator.

## Smoke Artifact Disposition

The tracked smoke artifact shows the new audit keys and the stricter claim boundaries:

- Stage 4B: `probe_effect_positive: true`, but `passed: false` because the checkpoint required migration and lacks trained hidden self-model/report heads.
- Stage 6A: `passed: true` at smoke scale.
- Stage 6B: `passed: true` at smoke scale.
- Stage 7: memory-content advantage is positive, but `passed: false` because the stricter joint matched-token-budget threshold is not met.
- Negative controls: `failed_as_intended: true` at smoke scale.

These are sanity-check outputs only. They are useful for verifying that the instrumentation runs and exposes the right fields; they are not a replacement for a full regenerated evaluation report.

## Current Claim Boundary

No Priority 1 item should be treated as robust support yet. Before relying on any audit disposition, run a full evaluation on a checkpoint trained after the Stage 4B feedback objective, commit or otherwise preserve the resulting report, and quote specific JSON paths from that report.

Until then:

- Stage 4B remains an instrumented probe path, not a trained-head claim, for migrated checkpoints.
- Stage 7 remains bounded to the narrower current/remembered-content result unless the full joint matched-budget audit passes.
- Negative controls and comparator systems are now better shaped, but still require full-budget runs before they can carry interpretive weight.
