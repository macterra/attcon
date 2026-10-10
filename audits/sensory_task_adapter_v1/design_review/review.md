# Pre-output sensory task adapter v1 design review

Verdict: accept for one reserved engineering-development attempt, subject to recording terminal exceptions as described below. This review does not approve a theory-facing endpoint or identify the correctness instrument with attention, clarity or qualia.

## Scope and independence

An independently tasked subagent reviewed the proposed design, then the actual protocol, script, imported renderer/calibrator/feature extraction/replay traversal and five small fixture tests before this new study's checkpoint inference. The reviewer did not load checkpoints, execute the study, inspect new evaluation outputs, alter code, tune parameters, contact people or retrieve participant data. Existing failed-study information came from the commissioning context and protocol, not a reanalysis of its outcome archive. This is AI review with shared model/training, project context and commissioning; it is not independent human validation, blinded external replication or appearance judgment.

## Findings

The new supervised readout is a declared task-development change after an immutable failed predecessor. It trains on all six artificial operating conditions with new splits, so the change can legitimately address deployment mismatch. It cannot attribute any improvement solely to condition coverage: width, nonlinearity, fitting budget and training-base count also change. Applying frozen predecessor readouts/calibrators to the same new evaluation bases is useful descriptively, without replacing their original verdict.

Code matches the fixed architecture, binary BCE, full-batch Adam settings, 1,500-step budget, isolated initialization, training-only float32 normalization and per-source parameter counts. Fit returns detached weights; prediction uses frozen training statistics and an individual feature vector, without a counterfactual partner or evaluation-batch normalization. All-condition training gives equal image weight. Frozen upstream encoders are evaluated under no-grad; supervised task readouts alone are optimized. Calibration is fit on its separate split using actual synthetic correctness, absolute task logit and the frozen Newton/L2 settings. No contrast metadata, human ratings or attention variables enter a predictor.

The assessment fixes ten equal bins, per-contrast ECE and signed-residual gates, visible true/predicted-direction strata and pooled/conditional constant Brier comparators derived exclusively from calibration outcomes. Whole-base resampling preserves dependence among contrast variants and identical indices support source comparisons. Direct small-fixture enumeration verifies bootstrap residual, ECE and conditional Brier intervals. Intervals condition on fixed trained instruments and three checkpoints; finite-sample ECE and artificial equal-condition sampling remain limitations, not population calibration or theory proof.

Duplicate-null inference checks feature, score and calibrated-probability equality exactly and yields exactly 50% correctness under opposite labels. The primary null remains an independent-image sample with gross-chance tolerance. This control cannot prove every possible nuisance relation absent, but label generation is separate from nuisance generation, and flipped-label fixture regeneration leaves zero-contrast images unchanged.

All three models must pass the predeclared engineering gates, and every source remains archived. The protocol preserves uncertainty under future attention/acquisition histories: passing only permits designing a separately grounded acquisition model. It does not transfer sensory confidence into the proposed experiencing locus or establish consciousness-related correspondence. The Controller, attention model and Reporter remain distinct.

## First-pass issues and disposition

1. Pre-code concerns about pooled calibration hiding condition errors, coarse bins, unspecified optimization and unequal source parameter counts were supplied before outputs. The actual protocol/code adopt conditional gates, 10 bins, explicit optimizer/activation/init, counts, direction diagnostics, conditional baselines and exact null probabilities. These changes are pre-output specification, not evaluation tuning.
2. Nonblocking execution caveat: build exceptions currently propagate after attempt.json is reserved, without an internal exception receipt. The protocol requires retained failed/incomplete attempts. The parent must record exception type/message and terminal execution status externally if a run or replay fails, preserve the reservation/partial artifacts, and never remove or retry it. A reserved file alone is not evidence of a live process or a successful/failed scientific result.
3. Retained limitations: same width/steps/data do not match parameter count/information; synthetic luminance conversion and phase/mask recurrence do not establish native sensory transfer; calibrated correctness is neither participant confidence nor appearance; no attention model or qualia-source hypothesis is tested here. These are correctly disclosed and are not repaired by a passing gate.

## Verification

Executed only `.venv/bin/python -m unittest discover -s tests -p test_sensory_task_adapter.py -v`: 5 tests passed. Inspected source-specific fitting, calibration/evaluation separation, feature dimensions, constant-baseline provenance, source retention, duplicate checks, immutable legacy archive hashing and exact replay traversal statically. No new study outputs existed or were used in this review. Final reviewed file hashes are recorded in metadata.json; the parent should recheck them before preparation/collection.
