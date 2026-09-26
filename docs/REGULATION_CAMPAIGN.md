# Delay and information-seeking campaign

Fixed protocol: [REGULATION_PROTOCOL.md](REGULATION_PROTOCOL.md).

| Cycle | Experiment | Status |
| --- | --- | --- |
| 1 | Paired fixed/variable-delay GRU pilot, seed 2101 | Complete |
| 2 | Paired replication, seeds 2111 and 2129 | Complete |
| 3 | Near-parameter-matched RNN comparison | Complete |
| 4 | Confidence and choice-preserving intervention controls | Pending |
| 5 | Reward-trained inspection pilot | Pending |
| 6 | Inspection replication and causal sensitivity | Pending |

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
