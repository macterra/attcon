# attcon

`attcon` is a minimal PyTorch benchmark for testing whether a model is merely computing attention or actually controlling it over time.

The current implementation trains and compares:

- a static cue-conditioned attention baseline
- a recurrent attention controller
- several ablations, including frozen recurrence and a feedforward summary controller

The benchmark is a small cue-guided selective-search task on a `5x5` grid. Each scene contains visible cell types plus hidden target/digit information that only becomes useful through attention.

## Repository Guide

- [SPEC.md](/home/david/dev/attcon/SPEC.md): original conceptual spec and motivation
- [configs/minimal.yaml](/home/david/dev/attcon/configs/minimal.yaml): default experiment config
- [src/attcon/train.py](/home/david/dev/attcon/src/attcon/train.py): training entrypoint
- [src/attcon/eval.py](/home/david/dev/attcon/src/attcon/eval.py): evaluation, ablations, and reporting

The latest local evaluation report is written to `outputs/minimal/evaluation_report.json` after running eval.

## Quickstart

Create or activate a Python environment, install the package, then run training and evaluation:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m attcon.train --config configs/minimal.yaml
.venv/bin/python -m attcon.eval --config configs/minimal.yaml --checkpoint outputs/minimal/experiment.pt
```

## What The Eval Reports

The evaluation report includes:

- task accuracy and target attention
- trajectory divergence and temporal reallocation
- wrong-cue sensitivity metrics
- a predictive probe comparing controller-state vs observation-only prediction of the next attention map
- a causal intervention test that perturbs controller state and measures the next-step attention shift
- a mid-episode cue-switch evaluation that tests whether attention reallocates after priorities change
- reduced-shaping retraining runs that test whether reallocation survives weaker or zero target-attention supervision
- ablations over recurrence and feedback channels
- an `evidence` summary for the three core claims:
  `dissociation`, `closed_loop_adaptation`, `cue_dependence`, plus Stage 3-style
  `explicit_attention_modeling`, `causal_attention_intervention`, and
  `reduced_shaping_resilience` results

## Current Result Shape

On the current default run, the recurrent controller outperforms the static baseline and stronger non-recurrent ablations, while also showing positive temporal reallocation and cue sensitivity.

The current evaluation also adds three positive Stage 3-style signals:

- a predictive probe where controller state predicts the next attention map better than the current observation summary alone
- a causal intervention test where controller-state perturbations shift the next attention map in a systematic, cue-like direction
- reduced-shaping retraining runs showing that useful reallocation weakens without direct target-attention shaping, but does not disappear entirely when that shaping term is removed

It also includes a Stage 5-style cue-switch test. On the current default checkpoint, that test is not yet passed: the controller is stronger on fixed-cue control than on rapid redirection after a mid-episode cue change.

The eval artifacts now also include intervention comparison plots that show baseline versus intervened attention around the intervention step, with both the original and alternate cue targets marked.

The exact numbers depend on the saved checkpoint in `outputs/minimal`, but the intended workflow is:

1. train both baseline and recurrent models
2. run ablations
3. inspect the JSON report, predictive-probe results, intervention results, cue-switch results, reduced-shaping results, and attention plots

## Tests

Run the smoke and regression tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
