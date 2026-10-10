# Isolated synthetic orientation task v1 results

**Overall registered engineering gate: FAIL. The attention-model origin of qualia remains untested.** The [frozen protocol](ISOLATED_ORIENTATION_TASK_V1_PROTOCOL.md) removes paired evaluation inputs, holds out classifier/calibration/evaluation draws, and estimates the probability that a fixed synthetic orientation classifier is correct. That instrument is separate from the attention model, Controller and Reporter; its numbers do not measure clarity or participant confidence.

## Frozen result

Three frozen encoders each receive 256 classifier-development bases, 256 calibration bases and 512 evaluation bases. All 1,536 evaluation bases and 9,216 dependent images are retained. Highest-contrast classifier fitting and six-contrast logistic calibration use distinct splits. No participant ratings, encoder retraining, new reports, exclusions, retries or favorable-source selection occur.

| Visual seed | Hidden highest-contrast accuracy | Conditional base-bootstrap 95% interval | Hidden pooled ECE | Hidden Brier / calibration-constant Brier | Registered result |
|---|---:|---:|---:|---:|---|
| 1301 | 89.84% | 87.01–92.38% | .0074 | .2325 / .2398 | FAIL: accuracy below .90 |
| 1311 | 94.73% | 92.58–96.48% | .0260 | .1956 / .2144 | Pass |
| 1321 | 93.95% | 91.80–95.90% | .0204 | .2198 / .2320 | Pass |

All physical edge controls reach 100% at highest contrast. Primary zero-contrast hidden accuracy is exactly 50% in each model; physical-control null accuracy is 51.17%, 52.73%, 48.83%. Every separately executed duplicate-null source is identical across labels and exactly 50% correct. These are integrity/sanity checks; the primary balanced labels do not make an arbitrary classifier exactly chance in finite samples.

The point threshold was frozen before outputs. Seed 1301's interval crosses .90, but the registered point-accuracy gate still fails. Conditional intervals describe these fixed fits and independently sampled bases; they do not include training/calibration/checkpoint uncertainty or demonstrate population replication.

## Pooled calibration hides conditional failure

Hidden correctness calibration passes the pooled ECE/Brier gates, but this artificial equal-six-contrast mixture conceals substantial contrast-specific errors. These conditional values are retained rather than repaired after evaluation.

| Synthetic contrast level | Hidden accuracy 1301 / 1311 / 1321 | Hidden ECE 1301 / 1311 / 1321 |
|---|---|---|
| Zero | .5000 / .5000 / .5000 | .2004 / .2633 / .2434 |
| 2.2 | .5000 / .5059 / .5000 | .1498 / .2334 / .1975 |
| 3.9 | .5020 / .6035 / .5117 | .2303 / .1504 / .2108 |
| 5.0 | .5625 / .7578 / .6328 | .1831 / .0978 / .1544 |
| 5.5 | .6445 / .8184 / .7227 | .1079 / .1480 / .1597 |
| 6.9 | .8984 / .9473 / .9395 | .3841 / .2441 / .3806 |

The levels label the synthetic conversion, not verified participant stimuli. Scores may encode contrast indirectly even though calibration receives no explicit contrast input. A pooled improvement over the calibration-derived constant does not qualify the instrument under changing sensory/attention conditions. These fits are not promoted as the grounded acquisition bridge.

## Integrity and audit

[Raw results](../audits/isolated_orientation_task_v1/results.json), per-model compressed states, fitted coefficients, reliability bins, bootstrap draws and [hash receipt](../audits/isolated_orientation_task_v1/completion.json) are preserved. Source snapshots and checkpoint hashes were frozen at local milestone `e7e98c3`, before the single reserved run. Five leakage/measurement tests pass.

The frozen script's original verifier fails on tensor lists because its inherited comparator only handles dictionaries and tensors. The [companion verifier](../scripts/verify_isolated_orientation_task_v1.py) adds list/tuple traversal, with the original failure recorded in [verification.json](../audits/isolated_orientation_task_v1/verification.json). Exact reconstruction passes for all inputs, features, fits, convergence checks, probabilities, scores, bootstrap draws and results. The frozen code and scientific outputs are unchanged.

A separate subagent's [result review](../audits/isolated_orientation_task_v1/results_review/review.md) independently reconstructs the study; see its metadata for executable audit source. Independent reconstruction matches every archived value exactly. Supplemental single-image versus mixed-batch encoder checks differ by at most 3.05176e-5 in float32 hidden features; the initial absolute 1e-5 audit tolerance failed and is retained, while disclosed atol 1e-4/rtol 1e-6 checks pass. This numerical check is separate from registered scientific gates. AI reviewers share commissioning/model context; this does not constitute human or institutional independence.

## Decision and remaining goal

This failure does not show that orientation is absent from visual features or that the theory is false. It limits the fixed highest-contrast linear task adapter and its margin-only correctness calibration. The earlier paired-information success remains valid for its different estimand. Neither result demonstrates native human-task transfer, appearance correspondence or attention-model causation.

The next task-development design must train on independent sensory decisions spanning its intended operating conditions, qualify conditional calibration before using it for an acquisition model, and preserve these failed fits as a comparator. Any acquisition model must learn from observed synthetic outcomes rather than receive appearance-rating targets. A separately qualified attention-manipulated appearance reference and selective attention-model intervention/restoration remain necessary for theory-facing interpretation. The full report-correspondence and source-of-qualia objective remains unchanged and unachieved.

Milestones are saved in a writable isolated checkout. Authenticated `gh` cannot reach GitHub, so the latest commits have not been pushed.
