# Six-cycle reporting experiment protocol

Registered before the first run on 2026-09-26. The user requested six additional
commit/push cycles. Each cycle records a result and a reproducible research step.
This protocol stays immutable; outcomes are recorded in REPORTING_CAMPAIGN.md.

1. Nonlinear reporter pilot, fresh seed 1901.
2. Replicate the pilot at seeds 1907 and 1913; summarize all three, including failures.
3. Prespecified reporter-data comparison: expand fitting from 128 to 512 context
   groups on the same three frozen agents. Validation and test groups stay fixed.
   This planned repeated evaluation is a diagnostic comparison, not a fresh
   confirmation sample. Do not tune subsequent settings on its test results.
4. Cross-architecture replication with an ungated RNN, seeds 1949, 1951, 1973,
   using 512 reporter-fitting groups. Keep hidden width and task/training recipe
   fixed; report parameter counts. This is replication, not a parameter-matched
   architecture superiority comparison.
5. Delay stress: insert 0, 1, 3, and 6 extra blank events immediately before the
   final query in held-out histories. Evaluate the frozen 512-group GRU and RNN
   pipelines, with no refitting or selection. Test state and action-score reports;
   the fixed-width learned history observer cannot accept longer streams. The
   symbolic history oracle remains an information-availability upper bound.
6. Selective content-erasure diagnostic on the 512-group GRU pipelines: remove
   0%, 25%, 50%, and 100% of displacement from the fitting-set seen-state mean in
   the content subspace. Compare random and permuted-label subspaces with per-case
   perturbation norms matched to the content erasure. Report choice/report errors,
   unavailable responses, query stability, and exact restoration controls, both
   unconditionally and on a fixed baseline-correct cohort. Do not label an erased
   state objectively inaccessible merely because a particular decoder fails.

## Fixed task, splits, and reporters

- Existing paired-history task: eight keys, six values, four record positions,
  a blank delay, then a query. Generic recurrent agent, hidden width 64, trained
  only on value choice for 160 epochs on 1,024 context groups. No report gradients
  or explicit access targets enter agent training.
- Generate disjoint partitions in this order: agent training 1,024, validation 64,
  test 128, reporter pool 1,024. The first 128 or 512 pool groups are reporter fit
  data. This fixes validation/test inputs across the prespecified data comparison.
- Reporters receive state, action logits, untrained-agent state, flattened full
  history, or only the final observation. Pad all to width 96 without information
  loss. The six-event history has 90 coordinates; state has 64.
- Identical nonlinear readout for every family: standardized inputs (fit-only
  statistics, standard-deviation floor 0.01), 96→64 tanh→7, 6,663 parameters.
  Two L2 candidates `{0, 0.001}`; Adam at 0.01 for 300 full-batch steps, penalty
  `0.5 * coefficient * sum(weight**2)` on both linear layers. Same initialization
  and tuning budget across families. Labels are six values plus unavailable.
- Select using validation only: maximize min(seen accuracy, unavailable accuracy),
  then paired accuracy, then balanced accuracy; deterministic first-candidate ties.
- A symbolic full-history oracle reads only event inputs and the final query,
  resolving the last matching record. It validates information availability; its
  success is not evidence for the agent. A learned history observer's failure does
  not invalidate the oracle or establish that history is uninformative.
- Twenty primary-state shuffled-label controls repeat the entire two-candidate
  fitting/validation procedure. Other feature families are comparators, not
  separately promoted claims with their own null audits.
- Reuse existing content-transplant assay and all 15 frozen thresholds: seen and
  unavailable reporting >=85%, paired reports and eligibility >=75%, joint donor
  following >=70%, access and query stability >=90%, query baseline >=85%, report
  advantage over observation >=25 points and null p95 >=20 points, and causal
  advantages over raw/magnitude-matched random and permuted controls >=25 points.
- Subspaces and query probes fit only reporter-fit states. A state-report adapter
  pads state to 96; the action reporter uses a frozen copy of the choice head.
- Check actual-input and context isolation, oracle agreement, parameter equality,
  checkpoint provenance, and source/data fingerprints. Save every audit and
  selected reporter locally. Never overwrite earlier experiment artifacts.

## Boundaries

All-seed passage means bounded supervised reportability on this task, not
spontaneous introspection. A strong untrained-state decoder bounds the role of
task learning. A strong history observer bounds compression; it is not an agent
self-report. Action scores may suffice without privileged full-state access.
Neither these results nor the later stress diagnostics automatically upgrade
Stage 8, prove inaccessible content, or establish regulatory self-modeling.
