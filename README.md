# attcon

`attcon` is a minimal PyTorch benchmark for testing whether a model is merely computing attention or actually controlling it over time.

The current roadmap treats Stages 1 through 3 as the sequential foundation, then splits into parallel lines of work around engineered self-state tracking, stronger learned self-modeling, and flexible reallocation under changed priorities. After structured reportability, the next planned step is a natural-language reporting layer grounded in tokenized internal state rather than a hand-authored symbolic dump.

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
- report probes that test whether controller state can support simple readouts of current regulatory content, including cumulative target-found status and unresolved regions
- reduced-shaping retraining runs that test whether reallocation survives weaker or zero target-attention supervision
- ablations over recurrence and feedback channels
- an `evidence` summary for the core benchmark claims plus the later roadmap stages:
  `dissociation`, `closed_loop_adaptation`, `cue_dependence`,
  `explicit_attention_modeling`, `engineered_self_state_tracking`,
  `learned_self_modeling_of_attention`, `structured_reportability`,
  `structured_reportability_uncertainty_and_allocation_error`,
  `natural_language_reportability`, `causal_attention_intervention`, and
  `reduced_shaping_resilience`

## Current Result Shape

On the current default run, the recurrent controller outperforms the static baseline and stronger non-recurrent ablations, while also showing positive temporal reallocation and cue sensitivity.

The current evaluation also adds several positive later-stage signals:

- a predictive probe where controller state predicts the next attention map better than the current observation summary alone
- a causal intervention test where controller-state perturbations shift the next attention map in a systematic, cue-like direction
- an engineered self-state evaluation where the recurrent controller maintains an explicit inspected-cell state and reports it more accurately than an observation-only probe
- report probes where controller state supports simple readouts of current search type, current attended cell, cumulative target-found status, and unresolved regions
- reduced-shaping retraining runs showing that useful reallocation weakens without direct target-attention shaping

It also includes a Stage 5-style cue-switch test. On the current default checkpoint, that test is now passed after training on a mix of stationary and switched-cue episodes: the recurrent controller redirects attention better than the baseline after a mid-episode cue change.

The earlier cue-switch tuning exposed a real tradeoff, but the current default checkpoint now recovers both signals: `cue_switch_adaptation` and `reduced_shaping_resilience` are both supported in the latest report.

The evaluator now also reports `stage3_multi_seed`, a repeated-seed summary over the predictive-probe and intervention checks together with the reduced-shaping result. That summary is intentionally conservative: it makes it easier to see when Stage 3 evidence is unstable across probe seeds instead of looking strong only on one slice. The same stability numbers are also surfaced in the `evidence.explicit_attention_modeling` block so the headline claim summary does not hide repeated-seed fragility. That block now distinguishes `single_run_supported` from `robust_supported`, and the final `supported` flag follows the stricter robust interpretation.

The current Stage 4A-style result is stronger than the earlier decoder-only report probes. The recurrent model exposes an explicit inspected-cell memory and a native self-state report head, and the evaluation report tracks this as `engineered_self_state_tracking`.

The current Stage 6A-style result is also now positive. The latest report tracks this as `structured_reportability`, with positive advantages for search type, attended cell, cumulative target-found status, and unresolved-region reporting. The stricter Stage 6B-style category, `structured_reportability_uncertainty_and_allocation_error`, now separates `current_wrong_candidate`, `wrong_candidate_history`, `revisit_unresolved`, and `allocation_error`. That gives the repo a finer distinction between active pursuit of a wrong candidate, cumulative wrong-candidate memory, and revisits during unresolved search. The overall Stage 6B bundle is still provisional rather than broadly settled, but it now has bounded positive evidence with a more interpretable internal decomposition.

The Stage 7 natural-language harness now exposes the same Stage 6B variables in its schema and metrics: `relevant_region_inspected`, `unresolved_search`, `current_wrong_candidate`, `wrong_candidate_history`, `revisit_unresolved`, and `allocation_error` are present in the symbolic baseline, the tokenized internal-state interface, and the observation-only control. The Stage 7 example format also now carries cue-history and inspection-history fields, and the evaluator can run dedicated `cue_switch_slice` and `intervention_slice` NL evaluations through the same reporting harness. That does not yet make Stage 7 supported, but it does mean later language-report claims can be checked against the same uncertainty/allocation-error distinctions already tracked in structured form and under stronger perturbation settings.

The eval artifacts now also include intervention comparison plots, switched-cue comparison plots, self-state trajectory plots, self-model trajectory plots, Stage 6B uncertainty-report comparison plots, and Stage 7 visual report panels for default, cue-switch, and intervention slices. Those new Stage 7 panels place scene-only information next to explicit symbolic dumps and minimal tokenized state views, which makes it easier to inspect the reporting interfaces without reading raw JSON arrays.
Stage 3 now also has its own repeated-seed diagnostics plot, which makes predictive and intervention instability visible without reading the raw seed table.

The exact numbers depend on the saved checkpoint in `outputs/minimal`, but the intended workflow is:

1. train both baseline and recurrent models
2. run ablations
3. inspect the JSON report, predictive-probe results, report-probe results, intervention results, cue-switch results, reduced-shaping results, and attention plots

## Tests

Run the smoke and regression tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
