# Causal predictive memory under matched observations

2026-09-29. Existing recurrent memory causally influences subsequent forecasts
under identical sensory evidence. Most of the effect of replacing an eight-step
history is reproduced by replacing the hidden-state components readable by the
command-effect head. This does not isolate a mismatch computation: those components
also affect other state estimates.

## Design

The [frozen exploratory protocol](HIDDEN_EXPECTATION_PROTOCOL.md) crossed two
histories with two continuations for each of 512 paired sequences and three
existing checkpoints. One history has channel 0 controllable, the other channel 1;
commands, replay motion, and sampled glimpse quality are otherwise shared.
Each continuation's observations are identical across all hidden interventions.
The continuation either agrees or disagrees with the history's controlled channel.
There are 1,536 underlying paired sequences, not independent samples for each
intervention and continuation. There was no retraining or API use.

The existing linear effect head has rank 32 in a 64-dimensional hidden state in
each checkpoint. Its row space contains changes that can affect its logits; its
null space contains changes that cannot immediately affect those logits. These
are mathematical subspaces of the trained readout, not identified biological
modules or necessarily disentangled semantic features.

## Results

Next-command effect accuracy, pooled equally across three seeds and both
continuation owners (all eight command/channel predictions per sequence):

| Initial hidden-state treatment | 1 new observation | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| Matching history | 100% | 100% | 100% | 100% |
| Opposite history | 25.02% | 37.77% | 62.52% | 82.96% |
| Replace effect-head row-space component | 27.70% | 40.01% | 66.90% | 84.71% |
| Replace effect-head null-space component | 95.76% | 100% | 100% | 100% |
| Random perturbation, row-replacement norm matched | 96.81% | 99.47% | 99.88% | 99.95% |

Row replacement reproduces the opposite history's effect logits before the next
input, while null replacement preserves the matching history's effect logits;
both identities were checked numerically in every seed/continuation. The much
larger effect of directed row replacement than norm-matched random perturbation
supports a structured causal role for predictive memory rather than sensitivity
to perturbation magnitude alone. These artificial hybrids may be off the natural
hidden-state manifold. The random control is one fixed direction per sequence,
not an exhaustive perturbation distribution.

Collateral effects prevent an expectation-specific interpretation:

| Treatment | Pre-input allocation probability MAE from matching history | Pre-input access probability MAE |
|---|---:|---:|
| Opposite history | 0.37872 | 0.17507 |
| Effect-row replacement | 0.21961 | 0.12501 |
| Effect-null replacement | 0.19241 | 0.13303 |
| Norm-matched random | 0.04301 | 0.19907 |

After one shared observation, current-allocation accuracy is at least 99.65% for
all treatments. Thus current allocation can be tracked accurately while predictions
of command consequences remain influenced by stale history. Null replacement also
changes the next forecast despite preserving the preceding decoded effect logits:
that decoded table is not a sufficient summary of recurrent state.

## What this resolves

The previous diagnostic ruled out direct feedback of the decoded output tensor.
This assay establishes a causal pathway through the hidden state from past control
conditions to future forecasts. The system already has predictive memory and
evidence-responsive state updates. It remains possible that the recurrent update
implements an implicit comparison, but this test does not distinguish that from
other recurrent state-estimation computations.

Neither assay establishes that the visual bound state supplies the medium of a
corrective Modeler update. That state is still downstream of this predictor and
absent from its feedback/training paths. The external reporter is instrumentation
and remains separate from the action-selecting Controller.

The next mechanistic test would need to preserve current content/access while
varying a history-derived expectation, and demonstrate a discrepancy-dependent
corrective update rather than just changes in forecasts. A mechanism-specific
claim requires interventions and matched estimation controls; decoding an error
alone would be insufficient. These findings do not justify declaring an implicit
comparator absent or adding a new architecture solely to obtain a positive result.

## Reproduction

`audits/hidden_expectation_v1/results.json` retains per-seed, per-continuation,
per-treatment values, observation hashes, checkpoint/source hashes, and the Torch
version. Run `.venv/bin/python scripts/diagnose_hidden_expectation.py` with a fresh
output location; existing results are protected against overwrite. Parameters are
verified unchanged. The previous reporting study and its acceptance criteria remain
untouched.
Two diagnostic tests passed, covering the subspace identities and paired simulator
conditions. Exact replay into a temporary directory reproduced the archived JSON.
