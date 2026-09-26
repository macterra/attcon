# Six-cycle reporting campaign

Protocol: [REPORTING_SIX_CYCLE_PROTOCOL.md](REPORTING_SIX_CYCLE_PROTOCOL.md).

| Cycle | Work | Status |
| --- | --- | --- |
| 1 | Nonlinear reporter and full-history pilot, seed 1901 | Complete; reporting gates unmet |
| 2 | Fresh-seed replication, seeds 1907 and 1913 | Complete; failures replicate |
| 3 | Fixed-agent reporter-data comparison, 128 vs 512 groups | Complete; more data helps, gates unmet |
| 4 | Ungated RNN replication, three fresh seeds | Complete; task viability fails |
| 5 | Frozen-pipeline delay stress | Complete; temporal generalization weak |
| 6 | Selective content erasure and restoration diagnostic | Complete; bounded sensitivity, residual false reports |

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

## Cycle 4: cross-architecture boundary

The ungated RNN reaches 95.3–96.0% choice accuracy on training contexts but only
67.6–72.9% on held-out contexts. All three seeds fail the task-viability gate.
Seen reporting is 57.8–66.8%, unavailable reporting 30.5–33.6%, and paired reporting
20.2–20.4%. Joint donor following is 49.1–55.2%, below the 70% threshold.

This is a failed cross-architecture replication under the fixed recipe. It cannot
support a higher-level claim about reporting independently of the task failure.
The RNN has 5,574 agent parameters versus the GRU's 15,942: width and training
recipe are matched, not parameter count, and seeds differ. The result therefore
does not establish that GRU gating is necessary or isolate architectural superiority.

Full-history observers remain capable (81.6–86.8% balanced accuracy), while
untrained RNN-state readouts reach 64.4–70.5%, exceeding the trained-state
45.7–49.4%. This further bounds claims that task training necessarily creates a
privileged reporting representation. A viable, fairly matched architecture
comparison remains open.

Artifact: `audits/report_sufficiency_rnn_fit512_multiseed.json`, with three source
audits and reconstructed, context-cluster intervals. The RNN checkpoint and
variable-delay interfaces were tested in cycle 1; no thresholds changed.

## Cycle 5: temporal stress

All six frozen pipelines reproduce their original zero-delay scores. Blank events
preserve every historical record and the final query; the symbolic history oracle
remains perfect at every delay. No agent or reporter is refitted.

| Extra blank events | GRU choice accuracy | GRU seen-value reports | GRU paired reports |
| --- | --- | --- | --- |
| 0 | 93.6–97.7% | 91.4–94.8% | 71.7–73.7% |
| 1 | 92.3–96.1% | 87.0–91.4% | 68.1–72.7% |
| 3 | 83.7–87.5% | 69.9–76.7% | 50.8–58.3% |
| 6 | 65.6–71.4% | 44.7–53.0% | 32.6–37.6% |

The already nonviable RNN pipelines fall to 21.9–23.6% choice accuracy and
11.5–15.5% seen reporting at six extra steps. This is evidence of weak temporal
generalization under the fixed recipes, not an architecture superiority claim.
The history retains the information; recurrent representations and their fitted
readouts do not preserve their original performance under this timing shift.

Artifact: `audits/report_delay_stress.json`, covering six systems and four delay
conditions, with conditional paired context-bootstrap intervals. Its four basic
task/report checks are stress diagnostics, not substitutes for the full 15-gate
assay. Tests verify history/query/label invariance and zero-delay identity.

## Cycle 6: selective content erasure

The fixed baseline-correct cohort contains 88.9–93.0% of seen cases: both choice
and report are initially correct. This is a single-case cohort, distinct from the
donor/recipient eligibility criterion in the transplant assay.

At full-strength content-subspace erasure, cohort choice accuracy falls to
14.8–17.8%, and true-value reporting falls to 4.2–5.8%. Unavailable reports rise
to 70.9–78.3%, versus 5.4–10.5% under norm-matched random perturbations and
15.7–21.8% under norm-matched permuted-content perturbations. This is repeatable
direction-specific sensitivity of the fitted reporter to disrupted content.

The reporter still asserts an incorrect value in 17.4–23.3% of the cohort.
Among choice-error trials, 71.6–79.3% receive unavailable reports, 20.0–26.4%
receive incorrect-value reports, and 0.7–2.0% retain a correct report despite the
choice error. Those retained correct reports are not counted as unavailable.
All norm-matching checks pass, and restoring the perturbation recovers the original
choice and report on every case in every condition.

This is an off-distribution lesion diagnostic. It does not prove that all internal
access was removed, that an unavailable report is introspective, or that ordinary
confidence/rejection mechanisms cannot explain the effect. The original reporting
gates still fail. The assay explicitly separates decoder failure, retained correct
reports, incorrect-value reports, and unavailable responses.

Artifact: `audits/report_content_erasure_diagnostic.json`, covering three frozen
systems, four strengths, and three perturbation subspaces. The complete final suite
passes **85 tests**, including selective projection, restoration, per-case norm
matching, and separation of correct reports from choice failure.

## Campaign decision

All six cycles are complete. Six agents were trained (three GRUs and three RNNs);
nine sufficiency runs include the three fixed-GRU data comparisons. Every run and
negative result is retained. `audits/reporting_campaign_status.json` consolidates
the conservative verdict and source artifacts.

The clearest progress is better measurement: report-data coverage improves decoding
on unchanged agents, history controls expose decoder limitations, and targeted
lesions causally change content and unavailable reports. The remaining obstacles
are reliable unavailable reporting, temporal generalization, and task-viable
cross-architecture replication. Full reporting support and Stage 8 remain unmet.

Next work should establish task viability under variable delays and a parameter-
aware architecture comparison, while testing whether state reports add anything
beyond action-score/entropy responses to the same lesions. Keep reporter supervision
outside agent training. Endogenous information seeking and stronger access claims
remain later steps, not conclusions of this campaign.
