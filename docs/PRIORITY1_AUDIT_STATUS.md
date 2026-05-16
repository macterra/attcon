# Priority 1 Audit Status

This note records the current disposition of the Priority 1 audit machinery after the implementation review in `FEEDBACK.md`. It no longer treats smoke-test numbers as bounded-claim support.

Tracked artifacts:

- smoke artifact: `audits/priority1_smoke_tune_prob_035.json`
- full-run summary: `audits/post_stage4b_full_eval_tune_prob_035_summary.json`
- checkpoint path: `outputs/tune_prob_035/experiment.pt`
- report path: `outputs/tune_prob_035/evaluation_report.json`
- config base: `configs/tune_prob_035.yaml`
- smoke audit mode: reduced probe batches and epochs for fast sanity checking
- full-run stale checkpoint migration: no; the full run was evaluated without `--allow-stale-checkpoint`

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

Stage 6B separates the four gated audit signals from informational signals in the smoke artifact. The `passed` flag is computed only from `current_wrong_candidate`, `wrong_candidate_history`, `revisit_unresolved`, and `allocation_error`.

These are sanity-check outputs only. They are useful for verifying that the instrumentation runs and exposes the right fields; they are not a replacement for a full regenerated evaluation report.

## Full Evaluation Disposition

Issue #1 generated a fresh checkpoint with the current model heads and evaluated it end to end:

- `.venv/bin/python -m attcon.train --config configs/tune_prob_035.yaml`
- `.venv/bin/python -m attcon.eval --config configs/tune_prob_035.yaml --checkpoint outputs/tune_prob_035/experiment.pt`

The committed full-run summary pins the local report hash as `4960a574f2488cfd72b86dabd41171e39cfcb37e927517f06c8f68559acceca5` and records `checkpoint_migration_present: false`.

Current full-run dispositions from `audits/post_stage4b_full_eval_tune_prob_035_summary.json`:

- Stage 4B learned self-modeling: supported in this full run. JSON paths: `learned_self_modeling.supported: true`, `learned_self_modeling.capacity_audit.passed: true`, `learned_self_modeling.capacity_audit.hidden_cell_bce_advantage: 0.031124413013458252`, `learned_self_modeling.capacity_audit.hidden_target_bce_advantage: 0.03128062188625336`, and `learned_self_modeling.policy_feedback_evidence: true`.
- Stage 6A structured reportability: not supported in this full run. Two sub-signals are strongly positive, but the AND-gate fails because `target_found_in_glimpse` is flat. JSON paths: `report_probes.supported: false`, `report_probes.capacity_audit.passed: false`, `report_probes.current_search_type.probe_capacity_matched_controller_accuracy_advantage: 0.5869140625`, `report_probes.current_attended_cell.probe_capacity_matched_controller_accuracy_advantage: 0.3682725429534912`, and `report_probes.target_found_in_glimpse.probe_capacity_matched_controller_accuracy_advantage: 0.0`.
- Stage 6B uncertainty/allocation-error reportability: not supported in this full run. The failure mechanism is probe collapse on rare positive classes rather than baseline competitiveness: only `current_wrong_candidate` has a meaningful positive-recall advantage, while the other gated signals tie at zero. JSON paths: `uncertainty_report_probes.supported: false`, `uncertainty_report_probes.capacity_audit.passed: false`, `uncertainty_report_probes.current_wrong_candidate.probe_capacity_matched_native_positive_recall_advantage: 1.0`, `uncertainty_report_probes.wrong_candidate_history.probe_capacity_matched_native_positive_recall_advantage: 0.0`, `uncertainty_report_probes.revisit_unresolved.probe_capacity_matched_native_positive_recall_advantage: 0.0`, and `uncertainty_report_probes.allocation_error.probe_capacity_matched_native_positive_recall_advantage: 0.0`.
- Stage 7 local opaque-token reporter: supported in this full run, but not comparator-resistant because the shuffled-feedback negative control did not fail as intended. JSON paths: `nl_report.supported: true`, `nl_report.content_supported: true`, `nl_report.capacity_audit.passed: true`, `nl_report.probe_capacity_matched_tokenized_joint_accuracy_advantage: 0.625`, `nl_report.probe_capacity_matched_tokenized_current_content_joint_accuracy_advantage: 1.0`, and `nl_report.probe_capacity_matched_tokenized_memory_content_joint_accuracy_advantage: 1.0`.
- Negative controls: not all failed as intended. JSON paths: `negative_controls.failed_as_intended: false`, `negative_controls.shuffle_feedback.failed_as_intended: false`, `negative_controls.shuffle_feedback.accuracy_drop_vs_recurrent: -0.0023437500000000056`, `negative_controls.feedforward_summary.failed_as_intended: true`, and `negative_controls.high_capacity_observation_only.failed_as_intended: true`.
- Stage 3 remains checkpoint-fragile in this full run. JSON paths: `stage3_checkpoint_family.supported: false`, `stage3_checkpoint_family.verdict: "checkpoint_fragile"`, and `evidence.explicit_attention_modeling.supported: false`.

## Current Claim Boundary

No Priority 1 item should be treated as robust support yet. The full run replaces the stale-checkpoint caveat with a concrete current-head evaluation, but the evidence is still a single checkpoint/config result.

For now:

- Stage 4B has full-run support on the current-head checkpoint, but consciousness-relevant emergence is still not established because this remains the supervised Stage 4B feedback condition.
- Stage 6A and Stage 6B should be downgraded from the smoke-scale positive disposition until their full-run capacity audits pass.
- Stage 7 remains bounded to the local opaque-token reporter; external API LLM and VLM routes remain open.
- Negative controls and comparator systems are better shaped, but the failed shuffled-feedback control prevents treating the current bundle as comparator-resistant evidence.
