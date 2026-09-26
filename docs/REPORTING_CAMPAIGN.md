# Six-cycle reporting campaign

Protocol: [REPORTING_SIX_CYCLE_PROTOCOL.md](REPORTING_SIX_CYCLE_PROTOCOL.md).

| Cycle | Work | Status |
| --- | --- | --- |
| 1 | Nonlinear reporter and full-history pilot, seed 1901 | Complete; reporting gates unmet |
| 2 | Fresh-seed replication, seeds 1907 and 1913 | In progress |
| 3 | Fixed-agent reporter-data comparison, 128 vs 512 groups | Pending |
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
