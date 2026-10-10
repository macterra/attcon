# Grounded attention outcomes v1: functional qualification passes

2026-10-10. All three primary models pass **56/56 registered engineering gates**. This qualifies a functional acquisition-outcome bridge on the declared synthetic worlds. It does not establish conscious appearance or the attention-model-as-qualia hypothesis. The Controller consumes forecasts; the Reporter remains a separate diagnostic system and was not run in this study.

The [frozen protocol](GROUNDED_ATTENTION_OUTCOMES_V1_PROTOCOL.md), [raw results](../audits/grounded_attention_outcomes_v1/results.json), [completion manifest](../audits/grounded_attention_outcomes_v1/completion.json), [pre-output source review](../audits/grounded_attention_outcomes_v1/design_review/review.md) and [independent interpretation review](../audits/grounded_attention_outcomes_v1/interpretation_review/review.md) preserve the scope and limitations.

## What was measured

For each frozen visual encoder 1301/1311/1321, generate 512 training, 256 calibration and 512 evaluation worlds. Four independent orientation/contrast locations are observed through initially unknown command permutations. Latest-glimpse hidden features and raw orientation logits are stored by delivered location. A learned attention model predicts current decision correctness, fresh-glimpse correctness, command delivery and post-command correctness. Labels, physical contrast and future branch images remain evaluator-only. The sensory adapter’s failed confidence map is excluded; its original FAIL is unchanged.

There are 1,536 evaluation worlds, four pre-command snapshots and eight fresh-noise repetitions of each of four commands: 196,608 evaluation branches. Training and calibration add 73,728 branches. Branches and repeated labels are dependent; whole worlds are the split and bootstrap units. Three checkpoints share an architecture and synthetic family, so this is not broader architectural replication.

## Primary forecasts and real Controller outcomes

| Model | Current Brier | Refresh Brier | Post-command Brier | Known delivery | Unseen-query Controller gain, 95% interval | Original minus corrupted success, 95% interval |
|---|---:|---:|---:|---:|---|---|
| 1301 | 0.1559 | 0.0817 | 0.1490 | 100.00% | 22.28 pp [18.54, 25.84] | 28.30 pp [22.99, 33.39] |
| 1311 | 0.1568 | 0.0819 | 0.1512 | 100.00% | 22.18 pp [18.30, 26.07] | 28.76 pp [23.59, 34.31] |
| 1321 | 0.1601 | 0.0811 | 0.1507 | 100.00% | 21.29 pp [18.05, 24.41] | 29.01 pp [24.26, 33.21] |

Each pooled binary Brier improves on its calibration-only constant. Multiclass delivery Brier is .3009/.3005/.3021 versus constants .7533/.7519/.7524. Delivery NLL also improves. Current/refresh seen-unseen and post-command query-seen × command-known calibration gates pass. All sixteen delivery probability cells within each known/unknown status pass, with maximum cell ECE .0574/.0664/.0606. Zero-history top-1 is .2446/.2344/.2651; this is a null/label-leak integrity check, not nontrivial generalization.

The preselected unseen-query sets contain 459/465/452 worlds. The all-world gains are 20.21/20.34/19.03 percentage points; all 512 worlds per model remain in that analysis. Queries are selected without truth or forecast/outcome inspection. Commands are selected from pre-command forecasts and scored against real retained branches. Ties and zero-contrast worlds are retained. Intervals use 256 whole-world bootstrap resamples.

Rotating represented delivery rows while fixing sensory evidence, history, physical mapping and branches changes Controller choices and reduces actual success. Unique-winner rotation and exact restoration pass. Physical remap initially leaves forecasts wrong; four new delivered observations restore 100% mapping top-1 in each model. Intermediate mixed old/new route histories are outside static training. Restoration verifies saved model inputs and predictions, not a re-executed physical world trajectory. Weights are not updated online.

## The calibration pass is bounded

Calibration gates use observable seen/command-known status and point estimates. They do not guarantee population or every sensory-condition calibration. The maximum post-command status-cell ECE is .0273/.0212/.0728; seed1321 also has a .0728 status-cell residual, near the .075 boundary. Actual contrast and orientation curves remain visible evaluator diagnostics. Unobserved future contrast cannot be known to an honest pre-command forecast, but the seen-contrast errors are also a practical limitation; passing coarse status cells does not erase them.

| Model | Seen zero-contrast current overprediction | Seen zero-contrast refresh overprediction |
|---|---:|---:|
| 1301 | 0.1181 | 0.1661 |
| 1311 | 0.1375 | 0.1679 |
| 1321 | 0.1367 | 0.1823 |

Low positive-contrast current forecasts also underpredict success by about .075–.119. No diagnostic curve is relabeled a new qualification gate after outcomes. These errors constrain downstream claims and any proposed appearance mapping.

## Comparison models and what causality does not establish

| Model | Structured gain | Equal-information joint gain | Sensory-only gain | History plus observed-status gain |
|---|---:|---:|---:|---:|
| 1301 | 22.28 pp | 15.03 pp | 0.93 pp | 22.55 pp |
| 1311 | 22.18 pp | 16.32 pp | -2.93 pp | 22.18 pp |
| 1321 | 21.29 pp | 13.18 pp | -2.25 pp | 21.67 pp |

All comparisons use the same worlds and query selection. Structured/ablated networks have 4,338 parameters; the equal-information joint network has 20,584 and a different multihead loss. No superiority gate is imposed and comparator gate verdicts are null, not passes from empty gate sets. These descriptive differences do not isolate parameterization, capacity or architectural necessity.

On an unseen query, structured risk inputs are identical zeros plus observed=0. Current and refresh estimates are therefore constants across those query locations. When refresh exceeds current, maximizing post-command success reduces to maximizing delivery probability to the queried location. The Controller endpoint consequently tests routing/permutation learning; history-only parity is expected from its construction. Richer sensory evidence improves current Brier relative to history-only (.1718/.1710/.1722), but this task does not identify a content-specific appearance role. Routing corruption demonstrates functional consumption, not a selective qualia-source effect.

## Verification and preservation

18 generated-fixture tests passed, plus an end-to-end toy-sensor pipeline check. The independently commissioned design reviewer separately ran 14 fixtures. Registration and source hashes were committed before production inference. A trailing-whitespace-only registration correction is preserved with original snapshots and an AST-equivalence review. One reserved study attempt completed successfully. All 177 raw archives include pixels, noise, decoded observations, evaluator truth, branch labels, fitted models and forecasts. No failed predecessor was rescored and no replacement attempt was selected.

Exact full-budget collection/refit/calibration replay passes with all 177 archive files verified. The [independent reconstruction](../audits/grounded_attention_outcomes_v1/results_review/review.md) also passes: 44,856 checked values across all 177 archives, maximum numerical discrepancy 1.11e-16. See its [numerical receipt](../audits/grounded_attention_outcomes_v1/results_review/numerical_reconstruction.json) and [freeze provenance](../audits/grounded_attention_outcomes_v1/results_review/freeze_provenance.json). Independent audit reconstructs raw worlds/pixels, frozen sensory decoding, memory/targets, calibrated forecasts, metrics, Controller/interventions and gates. Training loss traces are cross-checked against the fitted archive; separate exact replay refits the outcome networks. The independent audit preserves an initial missing-diagnostic implementation and its pre-output correction; frozen production sources were not changed.

## Next step toward the actual goal

Use this as a functional substrate with explicit uncertainty limits. Qualify an independently justified appearance/report endpoint and its stimulus/reference transfer; include uncertainty and criterion alternatives. Then test matched sources (sensory evidence, history/attention, addressed content, bound/joint state) on naturally reached seen-content states and selective represented-state interventions/restoration while sensory inputs remain fixed. Do not make this routing task a test of whether consciousness is required to perform.

Neutral factual reporting of this new model can proceed as a separate fidelity component, with represented forecasts explicitly distinguished from observed events and actual correctness. Such reports are not appearance evidence by definition. Human review is optional; separately commissioned AI reviews and their shared-model limits remain documented. Consciousness-related correspondence, theory-facing power and broader replication remain unestablished.
