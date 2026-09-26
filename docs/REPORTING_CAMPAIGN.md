# Six-cycle reporting campaign

Protocol: [REPORTING_SIX_CYCLE_PROTOCOL.md](REPORTING_SIX_CYCLE_PROTOCOL.md).

| Cycle | Work | Status |
| --- | --- | --- |
| 1 | Nonlinear reporter and full-history pilot, seed 1901 | Complete; reporting gates unmet |
| 2 | Fresh-seed replication, seeds 1907 and 1913 | Complete; failures replicate |
| 3 | Fixed-agent reporter-data comparison, 128 vs 512 groups | Complete; more data helps, gates unmet |
| 4 | Ungated RNN replication, three fresh seeds | In progress |
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

## Cycle 3: reporting-data coverage

Increasing reporter fitting from 128 to 512 context groups improves primary
balanced accuracy by 6.4–11.5 percentage points within seed. Paired context-bootstrap
intervals for these differences exclude zero on each seed, conditional on the
fixed fitted agents. `scripts/compare_reporter_data.py` verifies identical agent
weights and identical training, validation, and test fingerprints before comparing.

Seen reporting reaches 91.4–94.8%, unavailable reporting 76.6–78.1%, and paired
reporting 71.7–73.7%. Joint donor following is 82.6–86.1%. Twelve of fifteen gates
pass across all seeds; unavailable and paired reporting fail everywhere, while
access stability passes only one seed. The overall verdict remains unmet.

The learned history observer rises to 85.9–88.8% balanced accuracy, from
44.6–47.3%; the information was present, but this reporter needed more fitting
data. State reaches 84.4–86.1%, action scores 79.6–84.3%, and untrained state
75.7–80.5%. Thus stronger state decoding is now measurable, but externally trained
reporting and reservoir decoding remain important alternative explanations.

Artifacts: `audits/report_sufficiency_gru_fit512_multiseed.json` and
`audits/report_sufficiency_data_comparison.json`, plus the three seed audits. This
is the prespecified repeated-test comparison, not a new independent confirmation
sample. Earlier split-invariance tests and the checkpoint/fingerprint checks protect
its interpretation; no agent was retrained for the larger-reporting-data condition.
