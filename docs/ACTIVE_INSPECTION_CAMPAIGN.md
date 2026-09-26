# Recurrent acquisition campaign

Fixed protocol: [ACTIVE_INSPECTION_PROTOCOL.md](ACTIVE_INSPECTION_PROTOCOL.md).

| Cycle | Work | Status |
| --- | --- | --- |
| 1 | Environment, partitions, and analytic policy | Complete |
| 2 | Three-controller pilot, seed 2309 | Pending |
| 3 | Replication, seeds 2333 and 2351 | Pending |
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
