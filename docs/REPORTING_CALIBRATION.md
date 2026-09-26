# Availability readout diagnostic

Protocol registered 2026-09-26, before running the new seeds. This follows the
failed availability gates in the reporting-first coverage experiments.

## Question

Does ordinary readout fitting/calibration explain weak reports of unavailability,
or does that limitation persist after controlling the measurement procedure?
Success would support calibrated frozen-state reporting in this task, not
spontaneous self-report, a higher-order state, or Stage 8 convergence.

## Fixed experiment

- Fresh data/model seeds: 1801, 1811, 1823. Retain every run.
- Agent and task unchanged: generic 64-unit GRU, value-choice training only,
  1,024 training contexts, 160 epochs. Disjoint reporter-fit/validation/test context
  counts remain 128/64/128. Verify actual input isolation as well as context IDs.
- Freeze the agent before fitting any reporter. No report labels, access flags,
  confidence labels, or report gradients train the agent.
- Primary reporter: linear seven-class readout of state. Compare raw features
  with per-feature standardization fitted **only on reporter-fit data**; use a
  standard-deviation floor of 0.01. Fit each with L2 coefficient in
  `{0, 0.0001, 0.001, 0.01}` (penalty `0.5 * coefficient * sum(weight**2)`),
  500 Adam steps at 0.03. These are eight fixed candidates.
- Select within each feature family using validation scores only: maximize the
  smaller of seen-content accuracy and unseen-unavailable accuracy, then paired
  accuracy, then balanced accuracy. Exact ties retain the first candidate.
- Action-score comparator: identical eight-candidate selection, zero-padded to
  64 features for identical readout parameter counts. It is a function of state,
  and success here does not establish privileged access beyond the choice head.
- Entropy comparator: use action argmax for content, but emit unavailable when
  normalized action entropy exceeds a threshold. Select from 101 fixed thresholds
  in `[0, 1]` on validation using the same scoring rule. No test-set tuning.
- Retain uncalibrated 250-step state/action reporters, observation-only reporting,
  and untrained-state reporting as diagnostics on the same fresh seeds.
- Twenty permuted-label primary-state nulls repeat the entire eight-candidate
  fitting and validation-selection procedure. Permute reporter-fit labels; keep
  validation labels genuine, giving the null the same selection opportunity.
- Fit content and permuted-content subspaces exclusively on reporter-fit states.
  Reuse the original rank-five donor transplant, random and permuted controls,
  and controls with per-example perturbation norms matched to the true transplant.
  Evaluate all selected reporters under the same interventions; the primary
  state reporter alone determines the main gates.
- Keep every existing reporting/intervention threshold, including 85% seen and
  unavailable accuracy, 75% paired accuracy and baseline eligibility, 70% joint
  donor following, and 90% report-access stability. Keep the observation/null
  advantages and unrelated query-identity gates. No retrospective threshold changes.
- Record candidate validation scores, selected settings, test and causal scores,
  null distributions, normalization statistics in checkpoints, dataset fingerprints,
  and source hashes. Do not refit on validation or test after selection.

## Interpretation and next decision

Compare calibrated and original reporters within each seed. Improvements show a
measurement limitation; they do not alter agent memory or demonstrate regulatory
use of an access representation. Persisting failures bound this readout family,
not every possible decoder. Stronger action-score/entropy comparators weaken any
argument that a privileged full-state reporter is needed.

All-seed passage would justify selective access-erasure and stale-memory tests,
followed by a second architecture. If gates remain unmet, identify the failed
criterion before deciding whether to change the task, representation, or reporter.
No result here changes the Stage 8 audit automatically.
