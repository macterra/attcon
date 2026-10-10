# Independent AI pre-output design review

Disposition: suitable synthetic development diagnostic after two small alignment corrections below. This is not consciousness-related empirical evidence, nor qualification of human confidence or appearance measurement.

## Scope and exposure

Reviewed the isolated-task script, its protocol and tests, and the prior orientation diagnostic implementation used for rendering constants/features. No frozen checkpoint was loaded by this reviewer; no new study build/run, evaluation predictions, fitted checkpoint instruments or results were seen. Ran only the five supplied unit tests and 200 synthetic calibration-score fixtures. This reviewer is a separately commissioned subagent of the same AI-assisted project, shares model-family and parent-task context, and is neither a human nor institutionally independent reviewer. Earlier paired-study rationale is known. Procedural independence cannot remove shared-model biases.

## Required pre-freeze corrections

1. The protocol says evaluation null images are duplicated and identical scores required, while the implementation only compares one cached score against both labels. This mathematically ensures 50% by relabeling, but does not execute or validate a second inference. Either execute duplicated input/feature predictions and assert equality, or accurately describe the supplemental control as algebraic relabeling of the same cached image score. Keep it distinct from the genuinely independent primary null stream.
2. The fixed Newton loop checks only finite coefficients. A finite result need not be a converged penalized logistic fit. Freeze a final gradient tolerance and fail a nonconverged attempt, preferably archiving gradient norm and objective; alternatively explicitly limit the claim to a fixed-iteration approximation. None of the reviewer fixtures showed instability: 200 standardized heavy-tailed, separated or inversely separated score mixtures had maximum final gradient 5.2909066017292616e-17. This fixture result is not a global convergence proof.

## Design and implementation alignment

Balanced labels are shuffled using a separate generator; nuisance draws do not depend on label values. Independent split streams avoid sharing paired counterfactuals. Six within-base contrasts share geometry/orientation but independently drawn noise; whole-base bootstrap correctly preserves those dependencies. Primary null images retain geometry and noise without orientation texture. No guarantee of population chance follows from one observed finite primary null sample; the broad chance gate is an engineering sanity check.

Classifier fitting uses highest-contrast development rows only, with development mean/scale and intercept frozen. Predictions consume one feature row and fixed coefficients, with no evaluation centering, label/pair/contrast information or another image. Logistic calibration uses only calibration scores/correctness and calibration standardization; evaluation correctness does not fit it. Its slope is unrestricted and probabilities are not clipped to .5, consistent with correctness of a fixed classifier rather than a maximum-class posterior. Absolute scores may encode contrast indirectly, as the protocol acknowledges.

ECE bin boundaries, zero-score horizontal tie rule, ridge/penalty constants, split counts, RNG offsets, bootstrap settings and engineering gates are specified before checkpoint inference. Brier compares to a calibration-derived constant, not a constant fitted to evaluation outcomes. All five feature families are retained without selecting the favorable one. Highest-contrast physical control and hidden representation gates are both included. Conditional source/calibration assessments remain visible even if pooled gates pass.

Pooled ECE and Brier characterize the fixed equal-six-contrast artificial mixture; they can conceal errors at particular contrasts and do not guarantee calibration under later attention/acquisition interventions. Conditional bootstrap intervals condition on fixed fits and checkpoints, exclude training/calibration uncertainty and do not justify population generalization from three previously developed model instances. No participant appearance, human confidence, phenomenal clarity or attention-state source is tested here.

Per-checkpoint compressed state archives retain the whole trace and are checked separately during exact replay. Source snapshots/checkpoint hashes are frozen at preparation; one reserved attempt forbids retry. The initial reservation remains reserved on an exception, which is an incomplete failed attempt rather than a successful completed result; retain the exception/failure evidence rather than restarting it. Registration and commit sequencing must be verified by the parent before execution.

## Verification

`python -m unittest tests.test_isolated_orientation_task`: all five tests passed. They cover label-invariant null rendering, lower-contrast exclusion from classifier fitting, batch-independent scoring, nonmutating calibration prediction, a known Brier/ECE fixture, physical edge feasibility, invariant-feature ties and whole-base resampling. Calibration evaluation-label isolation is established primarily by function signature/code inspection, rather than the existing nonmutation test alone. The tests are leakage/measurement guards, not evidence for consciousness.
