# Recurrent acquisition campaign

Fixed protocol: [ACTIVE_INSPECTION_PROTOCOL.md](ACTIVE_INSPECTION_PROTOCOL.md).

| Cycle | Work | Status |
| --- | --- | --- |
| 1 | Environment, partitions, and analytic policy | Complete |
| 2 | Three-controller pilot, seed 2309 | Complete |
| 3 | Replication, seeds 2333 and 2351 | Complete |
| 4 | Recurrent-history and choice-preserving interventions | Pending |
| 5 | Frozen-controller verified-information reporting | Pending |
| 6 | Untouched stress contexts and consolidation | Pending |

## Cycle 1

The environment supplies a validity/invalidation cue with an initial value, or
no initial information. Each query offers six answers or a paid inspection.
The first inspection is noisy; the second verifies the value. Observations feed
back into recurrent state, and every inspection is charged. An analytic belief
policy provides a comparator based on known sensor probabilities rather than
hindsight. Controller learning will use environmental answer rewards and replayed
transitions, without access/report or optimal-inspection labels.

Eight focused tests validate disjoint contexts, hidden-answer exclusion, shared
sensor samples, analytic stopping, equal controller parameter counts, recurrent
rollout equivalence, cost accounting, and verified-information label semantics.
All three seed partition audits pass. No primary/stress outcome has been used to
select the design. See [environment audit](../audits/acquisition_environment.json).

## Cycle 2

The pilot trains three recurrent controllers with exactly 13,383 parameters,
identical initial weights, and 2,700 updates each. The state head selects epoch
60 using validation return; action and confidence heads select 100 and 80.
All representations learn from answer rewards and inspection continuation values,
without report or optimal-inspection labels.

| Cost | State return | Action-score return | Confidence return | Analytic realized return |
| --- | --- | --- | --- | --- |
| 0.1 | 0.8667 | 0.8674 | 0.8667 | 0.8667 |
| 0.25 | 0.6476 | 0.6602 | 0.6606 | 0.6458 |
| 0.4 | 0.5458 | 0.5458 | 0.5458 | 0.5458 |

The state controller answers every fresh case correctly without inspection. At
cost 0.1 it buys two observations for stale/missing cases and reaches 100%
accuracy; at cost 0.4 it buys one and accepts sensor uncertainty. Forced two-step
verification also reaches 100%. This establishes useful closed-loop behavior in
this finite task, but all full-state advantage gates fail: action-score and
confidence heads match or exceed its return. Differences from the analytic
policy's realized return can reflect finite-sample variation and alternative
stopping at the cost-0.25 indifference point; expected return is recorded too.

Source: [pilot](../audits/acquisition_seed2309.json). Ten environment/training
tests pass, including recurrent trajectory restoration and frozen selected weights.

## Cycle 3

Three-seed replication preserves matched parameter counts, initial weights,
training budgets, and sources. At every cost and seed, the state controller beats
each fixed inspection count by at least 0.02 reward, and fresh-answer and forced
verification accuracy pass 0.90. Returns range from 0.8613–0.8667 at cost 0.1,
0.6476–0.6641 at 0.25, and 0.5413–0.5667 at 0.4.

The required 0.02 gain over either learned comparator passes zero of three seeds
at every cost. One seed has positive context bounds over both learned controls
at cost 0.1, but it still misses the effect-size gate. No cost or seed is removed,
and no threshold is relaxed. This replicates adaptive acquisition within the
task, not an advantage of full-state inspection beyond confidence.

Source: [replication](../audits/acquisition_multiseed.json). Four summary tests
reject duplicate seeds, unequal initialization, and forged gates, and preserve
the negative conclusion. The underlying artifacts retain every checkpoint
selection, loss trace, conditional policy metric, and paired context interval.
