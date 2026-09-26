# Delay and information-seeking campaign

Fixed protocol: [REGULATION_PROTOCOL.md](REGULATION_PROTOCOL.md).

| Cycle | Experiment | Status |
| --- | --- | --- |
| 1 | Paired fixed/variable-delay GRU pilot, seed 2101 | Complete |
| 2 | Paired replication, seeds 2111 and 2129 | Pending |
| 3 | Near-parameter-matched RNN comparison | Pending |
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
