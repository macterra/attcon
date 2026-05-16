# Roadmap Feedback

Closing review of ROADMAP.md after the convergence-counting and execution-checklist consistency pass, followed by a separate review of the Priority 1 audit implementation.

The roadmap document has stabilized. The audit implementation that began executing against it has been tightened in response to a first round of review; a small number of open items remain — see "Priority 1 audit implementation review" below.

## Items from prior FEEDBACK.md, all closed

- **Convergence-counting rule generalized** ([ROADMAP.md:826-828](ROADMAP.md#L826-L828)) — now stated once at Stage 8 as a single rule covering all overlap-prone interfaces (6A, 6B, D, E, 7), with Branch E's local rule deferring to it. This was the largest residual gap and is now closed.
- **Stage 4B's role in the convergence count is explicit** ([ROADMAP.md:783-784](ROADMAP.md#L783-L784)) — Stage 4B is named as a bridge condition that "can strengthen a Stage 8 package, but it should not be one of the two minimum required convergence families." This is sharper than I had asked for.
- **Branch E placement on the partition is bidirectional** ([ROADMAP.md:778-780](ROADMAP.md#L778-L780)) — both halves are spelled out: access/report when it tests access-status or report-grounding, non-reportability when it tests higher-order content representation without those readouts. The "when it tests" phrasing already carries the per-result reading.
- **Execution checklist grouped** ([ROADMAP.md:889-934](ROADMAP.md#L889-L934)) — split into "Completed groundwork," "Immediate engineering and audit work," "Branch builds," and "Cross-system replication." Communicates priority and scope without committing to a schedule.

## Open items

None worth flagging. The previous draft of this file listed four "still loose" items; on review, they were all cosmetic or already answered by the existing text:

- The asymmetry between Stage 4B (bridging) and Branch D (full family) is justified by the doc's existing explanation — 4B's bridge status comes from double-counting risk with HOT and reportability evidence, which does not apply to D's distinct counterfactual-availability claim.
- The "Branch E is per-result, not per-branch" reading is already carried by the "when it tests..." conditional phrasing in the partition.
- The at-minimum Stage 8 package list and the convergence-family partition serve different purposes (package conditions vs convergence count) and don't conflict.
- Section-label cosmetics ("Completed groundwork") are trivial.

## Net assessment

The roadmap is structurally complete. Further textual revision yields diminishing returns. The work has fully shifted out of the document and into the experimental program: comparator runs, capacity audits, the four new branches with their required benchmark extensions, and cross-architecture / cross-benchmark replication.

## Priority 1 audit implementation review

Initial review of commits 5044e4d → 615681c (2026-05-16) added Stage 4B/6A/6B/7 capacity audits, negative controls, comparator suite, zero-shaping retest, and `PRIORITY1_AUDIT_STATUS.md`. Follow-up commit be5660f addressed most of the review; sub-sections below mark each item's current state.

### Fully addressed in be5660f

- **"Matched-capacity" naming.** `_capacity_matched_features` renamed to [`_probe_capacity_matched_features`](../src/attcon/eval.py#L886); all output keys renamed `probe_capacity_matched_*`; every audit dict now carries `"scope": "linear_probe_input_dim_only"`; docstring states it's a probe-capacity match, not a baseline-processing-capacity match.
- **High-capacity observation MLP needed temporal context.** [`_collect_temporal_observation_probe_dataset`](../src/attcon/eval.py#L304) builds a previous-N-observation window (default 3) before training the MLP. Window value is preserved in the output ([eval.py:2652](../src/attcon/eval.py#L2652)).
- **Shuffle-feedback / feedforward-summary `< 0.95` ceiling.** Replaced with `accuracy_drop_vs_recurrent >= min_accuracy_drop` (default 0.02) against a fresh recurrent reference ([eval.py:2576](../src/attcon/eval.py#L2576), [eval.py:2611](../src/attcon/eval.py#L2611)). The new criterion is doing real work: the smoke artifact shows `shuffle_feedback.failed_as_intended: false` (accuracy drop = 0.0), correctly exposing that this control isn't actually doing its job.
- **Comparator training budget asymmetry.** `match_recurrent_training: True` by default ([train.py:131-135](../src/attcon/train.py#L131-L135)); the transformer inherits the recurrent's training budget. `training_budget.matched_to_recurrent` recorded in output ([eval.py:2789-2795](../src/attcon/eval.py#L2789-L2795)).
- **`large_lm_without_loop_proxy` double-counts.** Explicitly marked `independent_comparator: False` and `double_counts_stage7_observation_only: True` ([eval.py:2813-2814](../src/attcon/eval.py#L2813-L2814)). Excluded from the comparator-level `failed_as_intended` check.
- **Checkpoint migration silently masks staleness.** Now requires `--allow-stale-checkpoint` CLI flag or `evaluation.allow_stale_checkpoint: true` ([eval.py:119-124](../src/attcon/eval.py#L119-L124), [eval.py:5033-5040](../src/attcon/eval.py#L5033-L5040)). Migration metadata recorded under `_checkpoint_migration`. Stage 4B `passed` gated on `not stale_checkpoint` ([eval.py:1560](../src/attcon/eval.py#L1560)). Smoke artifact correctly shows `migrated: true, passed: false, probe_effect_positive: true`.
- **Disposition unbacked by on-disk results.** `audits/priority1_smoke_tune_prob_035.json` is now committed and referenced by the status doc. The doc no longer presents smoke output as bounded-claim support and adds a "this is sanity-check only" caveat.

### Partially addressed

- **Pass thresholds are unprincipled.** Now configurable with non-zero defaults (e.g., `min_cell_bce_advantage: 0.01`, `min_accuracy_drop: 0.02`) and `nonnegative_directional_effect` / `probe_effect_positive` flags reported alongside the stricter `passed`. Real improvement, but the thresholds are still hardcoded constants rather than empirically calibrated. Suggest a small noise-floor routine that runs each probe against shuffled labels and uses the 95th-percentile spurious advantage as the threshold.

### Still open

- **Full eval against post-ba75ae5 checkpoint.** The status doc explicitly calls for this, but it hasn't been run. The committed `outputs/tune_prob_035/experiment.pt` is still pre-Stage-4B.

### New issues introduced by the tightening pass

Two claims in `PRIORITY1_AUDIT_STATUS.md` contradict the smoke artifact it cites:

| Doc claim | Smoke artifact value |
|---|---|
| Stage 6A: `passed: true` | `stage6a.passed: false` |
| Negative controls: `failed_as_intended: true` | `negative_controls.failed_as_intended: false` |

Both should flip to `false` in the doc (or, for negative controls, "feedforward and high-capacity passed; shuffle-feedback did not"). The negative-control miss is the same finding the tightened threshold is supposed to surface honestly — letting the doc claim the opposite undercuts the point of the tightening.

Subtler third item: in [audits/priority1_smoke_tune_prob_035.json](../audits/priority1_smoke_tune_prob_035.json) the `stage6b.positive_recall_advantages` dict lists `relevant_region_inspected: -0.25` and `unresolved_search: -0.96` next to `passed: true`. Those two signals aren't part of `capacity_audit_signals`, so the pass is correct, but a casual reader will see strongly negative numbers next to a green flag. Either split the dict into `audit_signals` vs `informational` or annotate which entries gate `passed`.

### What was solid from the start

- Zero-shaping commit (854b565) is well-scoped.
- `TrivialUniformRegulator` floor is good to have explicit.
- Stage 7 disposition correctly downgrades the joint-accuracy claim rather than overstating it.
- 36/36 tests pass throughout; the new tests verify audit keys are produced (but not their directional behavior).

### Net assessment

The tightening commit lands the structural feedback cleanly — naming, thresholds, temporal context, stale-checkpoint gating, comparator budget, and proxy bookkeeping all moved in the right direction. Remaining work, in order:

1. Fix the two doc/JSON disagreements (Stage 6A, negative-controls `failed_as_intended`).
2. Run the full evaluation on a post-ba75ae5 checkpoint and commit the resulting report.
3. Calibrate audit thresholds against an empirical permuted-label noise floor.
4. Annotate which Stage 6B signals gate `passed` in the artifact.

## Review history summary

- Round 1: Stage 8 bridge undefended; Stage 4B engineered-objective problem; inconsistent "supported" labeling; absent falsification criteria and negative controls; cumulative-confidence problem.
- Round 2: bounded/robust distinction; Self-Model Vocabulary; Stage 8 philosophical bridge; Stage 4B and Stage 7 caveats; per-stage falsification criteria. Critique: falsifiers were threshold restatements.
- Round 3: alternative-explanation falsifiers added; "no stage meets robust support" stated explicitly; capacity-audit and negative-control gaps acknowledged.
- Round 4: Stage 4A internalization, Stage 8 positive-update limit, Stage 4B existence-proof contingency. Surfaced benchmark/architecture generalization and absent positive operational definition.
- Round 5 (under stipulation that consciousness-evidence is the goal): proposed multi-theory convergence at Stage 8; Branches C, D, perturbational; comparator systems first-class; cross-architecture/cross-benchmark replication; methodology-development reframing; global falsifier.
- Round 6: all of the above implemented. Critique: HOT and GWT had no branches; convergence-on-same-contents needed sharpening; new branches lacked claim thresholds.
- Round 7: Branches E and F added; content-identity criterion; claim thresholds added; global falsifier triaged. Critique: partition implicit, overlap counting policed only at Branch E, Evidence-Family list still sequential, two fatal items overlap.
- Round 8: partition added, Branch E ↔ 4B counting rule added, list regrouped, fatal items differentiated, "endpoint not apex" notes added to 6A and 6B. Critique: counting rule only at one branch, 4B role implicit, Branch E placement conditional, checklist undifferentiated.
- Round 9: convergence-counting rule generalized at Stage 8; Stage 4B role explicit as bridge; Branch E placement bidirectional; checklist grouped. Critique: four items, all cosmetic or already answered.
- Round 10: no substantive open items. The roadmap document has stabilized.
- Round 11 (audit-implementation review, 2026-05-16): roadmap unchanged; review pivots to the Priority 1 audit code. Status doc claims results not present in any committed eval report; "matched-capacity" matches probe-input-dim only; pass thresholds are directional > 0 with no noise floor; high-capacity negative control sees no temporal context; matched-transformer comparator undertrained relative to recurrent; checkpoint migration silently loads stale heads at random init.
- Round 12 (audit-implementation tightening review, 2026-05-16): commit be5660f addresses six of the seven structural items cleanly — probe-capacity naming, temporal-context observation control, relative accuracy-drop threshold for ablations, matched comparator training budget, proxy-bookkeeping flag, and CLI-gated stale-checkpoint migration. Pass thresholds raised from `> 0.0` to configurable non-zero defaults with `nonnegative_directional_effect` reported alongside, but still hardcoded constants rather than empirically calibrated. Smoke artifact `audits/priority1_smoke_tune_prob_035.json` now committed and referenced. Two doc/JSON disagreements introduced by the tightening (Stage 6A `passed`, negative-controls `failed_as_intended`). Full evaluation on a post-ba75ae5 checkpoint still not run.
