# Availability calibration results

> **Historical campaign report.** Results, test counts, and unresolved questions
> below describe this campaign at closeout. The proposed follow-up was subsequently
> executed in the [six-cycle reporting campaign](REPORTING_CAMPAIGN.md).
> See [final results](PROJECT_RESULTS.md) for current reporting evidence and scope.

Executed 2026-09-26 under the fixed
[availability-readout protocol](REPORTING_CALIBRATION.md). All three fresh seeds
are retained in `audits/paired_history_calibration_seed{1801,1811,1823}.json`;
the validated summary is `audits/paired_history_calibration_multiseed.json`.

## Main finding

Validation-selected normalization/regularization improves unavailable reports
within every seed, but trades away some correct reports of seen content. It does
not clear the frozen reporting criteria. This identifies a partial readout-fitting
limitation, not a successful solution or evidence that the remaining information
is absent from state.

| Seed | Task choice accuracy | Unavailable report, original → calibrated | Seen-value report, original → calibrated | Paired reports, original → calibrated |
| --- | --- | --- | --- | --- |
| 1801 | 95.8% | 64.8% → 78.9% | 84.1% → 80.1% | 55.1% → 63.5% |
| 1811 | 93.8% | 64.8% → 78.1% | 78.5% → 72.1% | 50.0% → 55.1% |
| 1823 | 95.4% | 69.5% → 83.6% | 85.2% → 82.0% | 60.4% → 69.0% |

Unavailable accuracy gains are 13.3–14.1 percentage points, while seen-value
accuracy drops 3.1–6.4 points. Balanced accuracy improves 3.5–5.5 points. These are
matched comparisons on the same frozen agents and splits; they describe the
combined fitting/selection procedure, not an isolated causal effect of L2 or
normalization. Reported ranges are seed minima/maxima, not confidence intervals.

## Controls and causal findings

- The primary state reporter's balanced accuracy is 75.1–82.8%, clearing the
  current-observation advantage and selection-matched empirical-null gates on all
  seeds. Each null repeats all eight fitting candidates and validation selection.
- Action-score reporters reach 80.0–82.9% balanced accuracy; entropy reporters
  reach 78.9–85.0%. Thus access to the full hidden state is not shown to be
  necessary for these report scores. The entropy baseline needs only a threshold
  on choice uncertainty, selected without test data.
- Joint action/report donor following under content transplants is 68.5–76.3%:
  the 70% gate passes on two seeds. Effects exceed both norm-matched causal nulls
  by at least 65.5 percentage points; query identity remains 99.7–100% stable.
- Access-report stability is 74.5–80.1%, below its 90% gate on every seed. Baseline
  eligibility is 54.7–67.6%, below its 75% gate. Conditional successes do not rescue
  these failures.
- Raw observation and untrained-state reporters remain diagnostics. The untrained
  reporter did not receive the calibrated fitting sweep, so its comparison does
  not isolate the necessity of task training under equal tuning budgets.

Every seed fails the seen-report, unavailable-report, paired-report, access
stability, and eligibility gates. **Reporting gates remain unmet; Stage 8 is
unchanged.** These are supervised measurement instruments on one task and one GRU
architecture, with no demonstration of spontaneous reporting or regulatory use.

## Follow-up executed after this campaign

The follow-up proposed at closeout was to test representation
sufficiency with capacity-matched nonlinear reporters on state, action scores,
untrained state, and the full observable history. The history observer supplies
an information upper bound; its success is not evidence for the agent. Use fresh
seeds and validation-only model selection, and repeat selection-matched nulls.
Freeze that diagnostic's capacities and thresholds before running it.

Only after reliable reporting should selective access erasure, stale-memory
trials, and endogenous information seeking become the main tests. A stronger
decoder's success alone will remain decodability, not introspection.

## Reproduction and checks

Run each seed (1801, 1811, 1823) with:

```bash
.venv/bin/python scripts/paired_history_calibration.py --seed 1801 --out audits/paired_history_calibration_seed1801.json
```

Summarize without changing thresholds:

```bash
.venv/bin/python scripts/summarize_history_reporting.py audits/paired_history_calibration_seed1801.json audits/paired_history_calibration_seed1811.json audits/paired_history_calibration_seed1823.json --audit-name paired_history_calibration_multiseed --out audits/paired_history_calibration_multiseed.json
```

At campaign closeout, all 71 tests passed, including normalization isolation, validation-only selection,
agent-gradient isolation, action-score adapter equivalence, and entropy behavior.
The updated summary tool reproduces the prior coverage summary byte-for-byte and
rejects inconsistent source fingerprints. New artifacts record source/data hashes,
selected candidates, controls, and checkpoints. No external APIs were used for the
experiments.
