# Recurrent acquisition campaign

> **Historical campaign report.** Results, test counts, and unresolved questions
> below describe this campaign at closeout. The proposed follow-up was subsequently
> executed in the [final evaluation](PROJECT_RESULTS.md).
> See [final results](PROJECT_RESULTS.md) for current reporting evidence and scope.

Fixed protocol: [ACTIVE_INSPECTION_PROTOCOL.md](ACTIVE_INSPECTION_PROTOCOL.md).

| Cycle | Work | Status |
| --- | --- | --- |
| 1 | Environment, partitions, and analytic policy | Complete |
| 2 | Three-controller pilot, seed 2309 | Complete |
| 3 | Replication, seeds 2333 and 2351 | Complete |
| 4 | Recurrent-history and choice-preserving interventions | Complete |
| 5 | Frozen-controller verified-information reporting | Complete |
| 6 | Untouched stress contexts and consolidation | Complete |

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

## Cycle 4

Resetting initial recurrent state reduces mean return by 0.2002–0.2686 across
seeds. Restoring the state restores every answer, inspection count, visited-stage
mask, and reward. The learned controllers therefore depend causally on history
in this task; reset is a broad lesion and does not isolate an access mechanism.

The fitted fresh-versus-stale direction, projected out of the answer head's row
space, causes **zero initial inspection switches and zero return changes** in
all three seeds. Matched random controls cause 0–1.0% switches and small reward
losses. Maximum initial answer-logit residual is below 3.9e-6; all restoration
checks pass. This provides no evidence that this fitted availability direction
regulates inspection beyond the answer logits. We retain the null result.

Source: [interventions](../audits/acquisition_interventions.json). Paired
condition alignment and answer-null-space tests pass; the 118-test suite passed.
Synthetic perturbations retain environmental reward accounting, but their
responses are not verified introspective access reports.

## Cycle 5

Frozen state controllers support 99.97–100% balanced verified-information
reporting across all decision stages, and 100% on stages actually visited.
State reporters pass both accuracy gates and the shuffled-label advantage gate
in all three seeds. Ten shuffled-label fits per seed repeat validation selection;
their 95th-percentile balanced accuracy is about 50%.

Matched action-logit reporters also reach 99.71–100% balanced accuracy across
all stages (99.61–100% on visited stages). The state reporter fails the required
0.02 advantage over action logits in every seed. Both reporter families have
4,359 parameters; controller fingerprints remain unchanged after fitting.

These labels describe an explicit environmental guarantee: verified current
value versus unverified information. They are easier to establish than actual
internal availability after arbitrary memory loss. High accuracy here neither
resolves the earlier paired-history reporting failures nor demonstrates a native
reporting mechanism or subjective experience.

Sources: [reporting aggregation](../audits/acquisition_reporting_multiseed.json)
and its three linked seed artifacts. Focused tests verify label scoring, visited
stage dimensions, matched capacity, and isolation from controller training.

## Cycle 6

Every seed reserves 64 contexts not used for controller learning, checkpoint
selection, reporter fitting, or earlier primary evaluations. Frozen controllers
are evaluated on these contexts under baseline conditions, five delay blanks,
and an unannounced sensor-reliability reduction from 0.75 to 0.55. All other
variables are paired; there is no refitting.

At cost 0.1, five blanks reduce state-policy return from 0.8593–0.8667 to
0.7826–0.8517. At cost 0.25, sensor degradation yields 0.5816–0.6474 versus
0.6667 for the analytic policy that knows the changed reliability. At cost 0.4,
degraded-sensor state return is 0.4260–0.4469 versus the analytic policy's 0.4667.
This comparison deliberately tests distribution shift: the learned controllers
are not informed of the changed sensor probability. Fresh-answer and forced
verification accuracy still pass 0.90 at every cost/condition/seed, while no
complete full-state advantage gate passes on the reserved contexts.

Sources: [stress audit](../audits/acquisition_stress.json),
[campaign summary](../audits/active_inspection_campaign.json).
The protocol, previous reporting results, and Stage 8 audit remain unchanged.

## Campaign conclusions and subsequent work

Recurrent controllers now learn useful multi-step information acquisition from
answer rewards and replayed transitions. They condition inspection on information
and price, preserve history, and exceed fixed inspection policies across seeds.
This advances beyond the earlier external one-step inspection head. It remains
full-information fitted learning rather than autonomous exploration.

The stronger comparative-advantage hypothesis was unsupported at closeout.
It is separate from the goal of accurate reporting. Accurate verified-source
reporting and adaptive acquisition are both explained by action/confidence
representations in this task. Initial-state reset demonstrates broad memory
dependence; the more specific choice-preserving availability intervention is null.
The earlier paired-history reporting failures and Stage 8 verdict are not changed.

The subsequent [final study](PROJECT_RESULTS.md) varied expected information
quality independently of current answer confidence, with remembered sensor
reliability cues and a comparator receiving the same cues. It tested reward
benefits and causal report/control coupling across fresh seeds. Native
answer/decline and reward-only exploration were also evaluated. Native reports of content and verification status remain unestablished.

## Reproduction

Checkpoints are local, unversioned files under `outputs/acquisition/`; source,
settings, seeds, selections, dataset fingerprints, and JSON results are versioned.
Rebuild controllers before dependent assays. Run from the repository root:

```bash
.venv/bin/python scripts/audit_acquisition_environment.py
for seed in 2309 2333 2351; do
  .venv/bin/python scripts/train_acquisition.py --seed "$seed" --out "audits/acquisition_seed${seed}.json"
  .venv/bin/python scripts/audit_acquisition_reporting.py "audits/acquisition_seed${seed}.json" --out "audits/acquisition_reporting_seed${seed}.json"
done
.venv/bin/python scripts/summarize_acquisition.py audits/acquisition_seed*.json --out audits/acquisition_multiseed.json
.venv/bin/python scripts/audit_acquisition_interventions.py audits/acquisition_seed*.json --out audits/acquisition_interventions.json
.venv/bin/python scripts/summarize_acquisition_reporting.py audits/acquisition_reporting_seed*.json --out audits/acquisition_reporting_multiseed.json
.venv/bin/python scripts/audit_acquisition_stress.py audits/acquisition_seed*.json --out audits/acquisition_stress.json
.venv/bin/python scripts/summarize_acquisition_campaign.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

Context-bootstrap intervals are pointwise and conditional on a trained system.
All registered seeds/costs and all negative comparisons are retained.

Campaign validation: all 124 unit tests passed. Source compilation, whitespace,
finite JSON, source fingerprints, paired controls, and unchanged protocol/prior
Stage 8 artifacts were verified. No paid model APIs were used.
