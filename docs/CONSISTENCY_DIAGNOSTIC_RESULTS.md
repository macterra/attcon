# Existing consistency checking: mechanism diagnostic

2026-09-29. The existing predictor adapts its recurrent state to a changed
controlled channel, but does not learn a new command mapping in this assay.
The reported bound representation does not participate in the training-error or
runtime feedback path. These are exploratory findings, separate from the passing
bound-content reporting confirmation.

## What was tested

Following the [protocol](CONSISTENCY_DIAGNOSTIC_PROTOCOL.md), three existing frozen
checkpoints each received 512 fresh 24-step sequences. Dynamics changed before
step 8. Random executed commands and observations were shared across intact,
reset, and shuffled-history treatments. Thus these are observation-replay probes
with prospective policy scoring, not new closed-loop policy rollouts. Query success
uses the true controlled channel as a supplied task query; it does not measure
autonomous discovery of which channel to query.

## Findings

The following values pool equally sized runs over three checkpoints. Effect
accuracy scores the eight command/channel destinations per episode. Observation
counts refer to samples received since the change, including the current sample.

| Condition and history | After 1 observation | After 4 observations | After 8 observations | After 16 observations |
|---|---:|---:|---:|---:|
| Unchanged, intact | 100% | 100% | 100% | 100% |
| Owner swap, intact | 25.03% | 82.81% | 99.58% | 100% |
| Owner swap, reset | 63.00% | 99.72% | 100% | 100% |
| Owner swap, shuffled | 43.66% | 79.35% | 99.79% | 100% |
| Command offset, intact | 50.00% | 49.92% | 49.89% | 49.87% |

After the owner swap, intact-history prospective controlled-target success rises
from 80.40% after one observation to 99.67% after four and 100% after eight.
Under the two-slot command offset it remains 0% at those times and after sixteen
observations. Roughly 50% overall effect accuracy in that condition comes from the
unchanged replay channel, not successful learning of the new controllable mapping.

Resetting history accelerates recovery after an owner swap but impairs prediction
under unchanged dynamics. This demonstrates a causal role of history and a cost
of stale history. It is compatible with implicit consistency checking, but also
with ordinary recurrent state estimation; the assay does not distinguish those
implementations. A trained checkpoint can infer a different familiar channel role
without acquiring a new command mapping. Changes within an episode and the offset
mapping were absent from training, so the failure is bounded to this stress test.

Overwriting decoded effect forecasts, with the recurrent state and subsequent
observation held fixed, leaves all subsequent forecast fields exactly unchanged
in every tested seed/condition. This agrees with the code: the forward input is
the observation and hidden state, not the decoded forecast or a residual computed
from it. Hidden state may nevertheless encode predictions and implicit comparisons.
Parameter tensors remain exactly unchanged throughout this diagnostic.

## Where comparisons actually occur

- `prediction_loss` compares allocation, all-command effects, and recovery
  forecasts with simulator targets. Optimizer updates refine the predictor;
  recovery supervision uses simulator probabilities, not sampled recovery alone.
- `predictive_closed_loop.rollout` sends executed commands, observed allocation,
  and glimpses into recurrent state. Target success and forecast error are scored
  externally, not returned as explicit error inputs.
- `BoundState.content_policy` matches requested content against represented
  objects and combines that match with command forecasts. It chooses a command;
  it does not compare subsequent bound observations or correct the encoder,
  binding, or attention predictor. Reporting uses frozen bound snapshots.
- The older controller in `models.py` feeds task loss (when labels are supplied)
  or negative confidence back into its next decision. That implementation is
  separate from the predictor used in the current bound-report study.

## Theory-facing interpretation and next discriminating test

There is already feedback regulation and training-time error correction. The
unestablished connection is whether the reported integrated representation serves
as the comparison medium that corrects the Modeler. The external language reporter
is diagnostic instrumentation, not the Controller.

Heile's [MSTC v5](https://arxiv.org/abs/2512.01073v5) assigns consistency checking
and Modeler refinement to the Modeler-schema. These results establish neither
that identification nor its absence inside a recurrent computation.

A next assay should match the current observation while independently manipulating
the history-derived expectation and relevant hidden-state pathways. Decode
expectation and mismatch on held-out data, then test whether selective interventions
alter corrective updates while preserving content and ordinary state estimation.
Decodability alone is insufficient. Only if that existing mechanism fails to supply
the required role should an explicit comparison/update pathway be introduced and
evaluated as a new architecture. Preserve the original reporting goal throughout.

## Reproduction

Run `.venv/bin/python scripts/diagnose_attention_consistency.py` in a fresh checkout
without the diagnostic output. The script refuses to overwrite retained results.
The archived `audits/attention_consistency_v1/results.json` includes per-step
metrics, checkpoint/source hashes, observation hashes, and software version.
No API requests or retraining were used. The protocol has no pass/fail threshold.
Exact replay into a temporary directory reproduced the complete archived JSON.
Eight existing predictive-model tests and two diagnostic simulator tests passed.
