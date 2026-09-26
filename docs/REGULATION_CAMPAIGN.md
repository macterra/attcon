# Delay and information-seeking campaign

Fixed protocol: [REGULATION_PROTOCOL.md](REGULATION_PROTOCOL.md).

| Cycle | Experiment | Status |
| --- | --- | --- |
| 1 | Paired fixed/variable-delay GRU pilot, seed 2101 | Complete |
| 2 | Paired replication, seeds 2111 and 2129 | Complete |
| 3 | Near-parameter-matched RNN comparison | Complete |
| 4 | Confidence and choice-preserving intervention controls | Complete |
| 5 | Reward-trained inspection pilot | Complete |
| 6 | Inspection replication and causal sensitivity | Complete |

All results, including failures, will be retained. Report fitting and policy
learning remain separate from agent training. The earlier reporting campaigns and
their thresholds are not overwritten.

## Cycle 1: paired delay pilot

Seed 2101 improves from 9/15 to 13/15 reporting gates under variable-delay
training. Mixed-delay seen choice accuracy rises from 86.2% to 95.4%; paired
report accuracy rises from 43.4% to 70.6%. At extra delay 9, choice accuracy rises
from 53.3% to 93.5%. Unavailable reporting (74.2%) and paired reporting (70.6%)
remain below their original gates. This is one seed, not replication.

The paired audit verifies identical initialization, updates, data, and reporter
delay assignments. Variable training has additional recurrent compute. Sources:
[paired pilot](../audits/regulation_delay_pilot.json),
[fixed](../audits/regulation_gru_fixed_seed2101.json),
[variable](../audits/regulation_gru_variable_seed2101.json).
Validation: 91 unit tests passed; source and dataset fingerprints are recorded.

## Cycle 2: paired replication

All three variable-delay GRUs improve over their fixed-delay counterparts.
Mixed-delay seen choice is 95.2–97.4% versus 86.2–89.8%; delay-9 choice is
91.4–94.8% versus 53.3–59.5%. Paired report accuracy is 67.7–70.6% versus
43.4–55.9%. The context-bootstrap paired-report improvement interval excludes
zero at every seed. These are pointwise within-seed intervals, not population
uncertainty across independently sampled systems.

Variable training passes 12/15 gates in every seed (individual totals 13,12,12),
versus 8/15 universally for fixed training (individual totals 9,11,9). Unavailable
and paired reporting fail universally; access stability passes only one seed.
The paired checks pass, including equal initial weights and 7,680 optimizer
updates. Variable training processes about 1.42 times the recurrent steps.

Sources: [paired replication](../audits/regulation_delay_replication.json),
[fixed aggregation](../audits/regulation_gru_fixed_multiseed.json),
[variable aggregation](../audits/regulation_gru_variable_multiseed.json).
Aggregation checks validate source/configuration comparability and recompute gates.

## Cycle 3: near-parameter-matched recurrent architecture

The width-115 ungated RNN has 15,876 parameters versus the width-64 GRU's 15,942
(66 fewer, about 0.4%). All three comparisons preserve task contexts, delay
assignments, optimizer updates, recurrent-example steps, and reporter capacity.
Mixed-delay seen choice accuracy is 62.2–70.1%, below the 85% viability gate;
seen reporting is 53.1–62.1%, and paired reporting is 9.9–18.1%. Individual runs
pass 6,4,6 of 15 reporting gates. This recipe provides no cross-architecture
replication. Different optimization requirements remain a live explanation;
near parameter matching does not establish a necessity of gates.

Sources: [architecture comparison](../audits/regulation_architecture_comparison.json),
[RNN replication](../audits/regulation_rnn_variable_multiseed.json).
Validation: comparison checks pass; full suite passes 100 tests, including the
forthcoming intervention and reward-policy implementation checks.

## Cycle 4: matched lesions and choice-preserving controls

All three GRUs pass the choice-invariance check: maximum logit residual is below
3.4e-6 (registered tolerance 1e-5), action/entropy reports remain identical, and
subtracting the perturbation restores every report. The fitted availability
projection changes state reports on 4.0–6.3% of seen cases, versus 0.1–1.7% for
the norm-matched random null-space control. This isolates modest additional
state-readout sensitivity beyond choice logits; synthetic perturbations have no
validated access ground truth.

Full content erasure produces unavailable state reports on 59.1–69.3% of seen
cases, but also incorrect-value reports on 23.8–31.9%. Action-score and entropy
reporters become unavailable more often (93.2–96.0% and 96.5–99.6%). The shared
initially-correct cohorts contain 590–616 cases per seed, and retain substantial
incorrect-value reporting after erasure. Thus erasure sensitivity alone does not
distinguish a reliable access report from confidence-related behavior.

Source: [all strengths, controls, cohorts, and restoration checks](../audits/regulation_report_interventions.json).
Validation: five focused tests cover pairing, rank deficiency, choice invariance,
matched norms, and invalid directions; all pass within the 100-test suite.

## Cycle 5: reward-trained inspection pilot

Seed 2101 learns three equal-capacity policies from immediate-answer correctness
rewards only; report/access labels do not enter training or model selection.
The recurrent agent remains frozen. Validation selects between two L2 settings,
and twenty shuffled-reward fits repeat that selection.

| Inspection cost | State-policy return | Action-score return | Confidence return |
| --- | --- | --- | --- |
| 0.2 | 0.7788 | 0.8352 | 0.8352 |
| 0.4 | 0.6647 | 0.7228 | 0.7117 |
| 0.6 | 0.5941 | 0.6327 | 0.6329 |

The state policy fails the advantage gate at every cost. It improves over never
inspecting, but at cost 0.2 also loses to always inspecting (return 0.8). No
beneficial full-state advantage beyond confidence is demonstrated. This is
external one-step reward learning, not endogenous recurrent regulation.

Source: [inspection pilot, delay-9 stress, null fits, and paired intervals](../audits/inspection_seed2101.json).
Validation: four focused tests cover reward accounting, context resampling,
fit-only normalization/frozen agents, and action/confidence invariance; all pass.

## Cycle 6: inspection replication and causal sensitivity

The state policy loses to both learned comparators at every cost in all three
seeds. State-minus-action return ranges are -0.0564 to -0.0323 (cost 0.2),
-0.0581 to -0.0441 (0.4), and -0.0385 to -0.0214 (0.6). All three agents satisfy
task viability, but none clears any complete inspection-advantage cost gate.
The gate remains unchanged; no costs or seeds were dropped.

Choice-preserving availability transplants switch state-policy decisions in
4.4–12.5% of seen cases across seeds/costs, versus 0.8–4.4% for matched random
controls. Both answer-to-inspect and inspect-to-answer changes are retained.
Action/confidence policies make no decision switches, logit residuals pass, and
restoration returns all decisions to baseline. This demonstrates additional state
sensitivity without showing that synthetic-lesion responses improve reward.

Sources: [inspection replication](../audits/inspection_multiseed.json),
[policy interventions](../audits/inspection_interventions.json),
[machine-readable campaign summary](../audits/regulation_campaign.json).
All earlier artifacts and the Stage 8 audit are retained unchanged.

## What remains

Delay robustness is substantially better. Full reporting support remains unmet,
and useful regulation beyond action confidence remains unsupported. The next
substantive experiment should couple information acquisition to a recurrent
policy in the environment, distinguish fresh, stale, and missing information,
and predeclare matched confidence/action controls. Fit any access-report head
only after task/control learning. Use fresh contexts and seeds for confirmation,
and require a reward benefit as well as report sensitivity before advancing the
regulation claim. These are proposed next steps, not completed results.

## Reproduction

From the repository root, rebuild agents/reporters before dependent assays.
Checkpoints live locally under `outputs/regulation/` and are not versioned; the
JSON results, seeds, source hashes, settings, and dataset hashes are versioned.
The fixed protocol is immutable: [REGULATION_PROTOCOL.md](REGULATION_PROTOCOL.md).

```bash
for seed in 2101 2111 2129; do
  for recipe in fixed variable; do
    .venv/bin/python scripts/train_regulation.py --seed "$seed" --recipe "$recipe" --out "audits/regulation_gru_${recipe}_seed${seed}.json"
  done
  .venv/bin/python scripts/train_regulation.py --seed "$seed" --recipe variable --architecture rnn_matched --out "audits/regulation_rnn_variable_seed${seed}.json"
  .venv/bin/python scripts/train_inspection.py "audits/regulation_gru_variable_seed${seed}.json" --out "audits/inspection_seed${seed}.json"
done
for recipe in fixed variable; do
  .venv/bin/python scripts/summarize_history_reporting.py audits/regulation_gru_${recipe}_seed*.json --out "audits/regulation_gru_${recipe}_multiseed.json" --audit-name "regulation_gru_${recipe}_multiseed"
done
.venv/bin/python scripts/summarize_history_reporting.py audits/regulation_rnn_variable_seed*.json --out audits/regulation_rnn_variable_multiseed.json --audit-name regulation_rnn_variable_multiseed
.venv/bin/python scripts/compare_delay_training.py --fixed audits/regulation_gru_fixed_seed*.json --variable audits/regulation_gru_variable_seed*.json --out audits/regulation_delay_replication.json
.venv/bin/python scripts/compare_regulation_architectures.py --gru audits/regulation_gru_variable_seed*.json --rnn audits/regulation_rnn_variable_seed*.json --out audits/regulation_architecture_comparison.json
.venv/bin/python scripts/audit_regulation_interventions.py audits/regulation_gru_variable_seed*.json --out audits/regulation_report_interventions.json
.venv/bin/python scripts/summarize_inspection.py audits/inspection_seed*.json --out audits/inspection_multiseed.json
.venv/bin/python scripts/audit_inspection_interventions.py audits/inspection_seed*.json --out audits/inspection_interventions.json
.venv/bin/python scripts/summarize_regulation_campaign.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

Validation also rejects duplicate seeds, changed source fingerprints, and
unearned inspection gate passes. Context-bootstrap intervals are pointwise and
conditional on a trained system; they are not population-level uncertainty.

Final validation: all 104 unit tests passed. Compilation, whitespace checks,
source fingerprints, comparison controls, and unchanged Stage 8/protocol checks
passed. No paid model APIs were used.
