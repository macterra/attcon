# Six-cycle reporting campaign

Protocol: [REPORTING_SIX_CYCLE_PROTOCOL.md](REPORTING_SIX_CYCLE_PROTOCOL.md).

| Cycle | Work | Status |
| --- | --- | --- |
| 1 | Nonlinear reporter and full-history pilot, seed 1901 | Complete; reporting gates unmet |
| 2 | Fresh-seed replication, seeds 1907 and 1913 | Complete; failures replicate |
| 3 | Fixed-agent reporter-data comparison, 128 vs 512 groups | In progress |
| 4 | Ungated RNN replication, three fresh seeds | Pending |
| 5 | Frozen-pipeline delay stress | Pending |
| 6 | Selective content erasure and restoration diagnostic | Pending |

Results, checks, and interpretations are added after each experiment. Failures
remain failures; the protocol's thresholds will not be relaxed.

## Cycle 1: nonlinear pilot

The agent reaches 97.7% held-out choice accuracy. The state reporter reaches
88.9% seen-content and 70.3% unavailable accuracy; paired accuracy is 62.1%.
Joint action/report donor following is 83.3%, but access stability is 87.5%.
Twelve of fifteen frozen gates pass; unavailability, paired reporting, and access
stability remain below threshold. All five reporter families have 6,663 parameters.

Balanced report accuracies: state 79.6%, action scores 82.6%, untrained state
69.1%, learned full-history observer 46.2%, current observation 50%. The symbolic
history oracle is perfect, so the weak learned history observer reflects this
decoder/training setup, not lack of information in the history. Nonlinearity alone
does not resolve the reporting problem. Replication and the prespecified larger
reporter-fitting set remain necessary.

Artifact: `audits/report_sufficiency_gru_fit128_seed1901.json`. All 76 tests pass,
including new split-invariance, oracle, capacity, frozen-gradient, and variable-
delay architecture checks. The protocol and artifact record provenance and claim
boundaries; Stage 8 is unchanged.

## Cycle 2: replicated nonlinear reporting

The three seeds reach 85.5–90.0% seen-content reporting, but only 60.2–70.3%
unavailable and 52.1–62.1% paired reporting. Joint action/report donor following
replicates at 81.6–84.1%. No seed passes the complete assay. Ten of fifteen gates
pass on every seed; nonlinearity does not resolve availability reporting with
128 fitting contexts.

The learned history observer remains weak (44.6–47.3% balanced accuracy), despite
perfect oracle access. Untrained-state reporters reach 61.7–69.1%; action scores
reach 78.6–82.6%, versus 72.9–79.6% for state. These controls prevent interpreting
the fitted state reporter as privileged introspection.

`scripts/summarize_sufficiency.py` reconstructs predictions from saved checkpoints
and adds 2,000-resample context-cluster intervals. Entire content/status groups
are resampled, preserving paired histories and avoiding pseudoreplication of the
six identical unseen inputs per context. Intervals are descriptive, conditional
on each fitted system; they do not change gates or measure between-seed uncertainty.
Four dedicated tests cover ordering, pairing integrity, determinism, and intervals.

Artifact: `audits/report_sufficiency_gru_fit128_multiseed.json`, with all three
source artifacts linked. The prespecified larger-fitting-set comparison uses the
same frozen agents, validation histories, and test histories.
