# attcon

`attcon` is a minimal PyTorch benchmark for testing whether a model is merely computing attention or actually controlling it over time.

The current roadmap treats Stages 1 through 3 as the sequential foundation, then splits into parallel lines of work around engineered self-state tracking, stronger learned self-modeling, and flexible reallocation under changed priorities. After structured reportability, the repo now includes a bounded Stage 7 route: a local calibrated reporter reads opaque tokenized internal state and emits faithful natural-language-shaped reports, while external API LLM and VLM variants remain follow-up work.

The current implementation trains and compares:

- a static cue-conditioned attention baseline
- a recurrent attention controller
- several ablations, including frozen recurrence and a feedforward summary controller

The benchmark is a small cue-guided selective-search task on a `5x5` grid. Each scene contains visible cell types plus hidden target/digit information that only becomes useful through attention.

## Benchmark Mechanism (discrete glimpse)

The controller's attention policy is a soft distribution (so divergence and probe
metrics stay graded), but each glimpse **reads the single most-attended cell** through a
straight-through estimator (`model.hard_attention`). This matters: with a fully soft
glimpse the readout averages the digits of every same-type cell, diluting the per-cell
target evidence below what the controller can localise, and the task does not train (the
recurrent controller collapses to uniform attention and loses to the static baseline).
Reading one cell per step — as the original spec intended ("inspect one or two cells per
timestep") — makes the closed-loop search learnable. Because a discrete searcher reads
the target on a single decisive fixation and carries the digit forward, mean
`target_attention` understates success, so the eval also reports `target_inspected_rate`
(did the argmax fixation ever land on the true target).

## Repository Guide

- [SPEC.md](/home/david/dev/attcon/docs/SPEC.md): original conceptual spec and motivation
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
- learned-self-model diagnostics that compare hidden-state-only probes against observation-only baselines and perturb hidden state along native self-model readout directions
- reduced-shaping retraining runs that test whether reallocation survives weaker target-attention supervision
- ablations over recurrence and feedback channels
- an `evidence` summary for the core benchmark claims plus the later roadmap stages:
  `dissociation`, `closed_loop_adaptation`, `cue_dependence`,
  `explicit_attention_modeling`, `engineered_self_state_tracking`,
  `learned_self_modeling_of_attention`, `structured_reportability`,
  `structured_reportability_uncertainty_and_allocation_error`,
  `natural_language_reportability`, `causal_attention_intervention`, and
  `reduced_shaping_resilience`

## Current Result Shape

On the current discrete-attention checkpoint (`configs/tune_prob_035.yaml`, 5000 steps),
the recurrent controller solves the search task and clearly beats the static baseline and
the non-recurrent ablations. Representative numbers from a regenerated full eval
(`audits/post_rehab_full_eval_tune_prob_035_summary.json`):

- recurrent accuracy `0.44` vs static `0.17` (chance `0.10`); `target_inspected_rate` `0.39` vs `0.08`
- all negative controls fail as intended, including `shuffle_feedback` (accuracy drop `0.27`) and `feedforward_summary` (`0.21`); the matched-transformer and trivial-regulator comparators also fail as intended

Honest current status by stage (discrete-attention checkpoint):

- **Stage 2 / 3** (closed-loop control, explicit attention modeling): supported, and Stage 3 is **robust** — the predictive-probe and intervention checks pass on every seed (`stage3_multi_seed` 1.0/1.0) and the `stage3_checkpoint_family` verdict is `robust` across the default and `0.25` reduced-shaping checkpoints.
- **Stage 4A** (engineered self-state tracking): supported; the native self-state head reports the explicit inspected-cell map at `~0.99` cell accuracy.
- **Stage 4B** (learned self-model feedback): **not** part of the base config — the destabilising policy-feedback path is disabled, and learned-self-model *emergence* is studied as its own experiment.
- **Stage 5** (cue-switch reallocation): supported (recurrent switch accuracy `0.25` vs baseline `0.0`).
- **Stage 6A** (structured reportability): supported, capacity audit passes — controller state beats a capacity-matched observation probe on current search type and current attended cell.
- **Stage 6B** (uncertainty / allocation-error reportability): **bounded / provisional**. Controller state beats the capacity-matched observation baseline on positive *recall* for all four gated signals, but the stricter accuracy-guarded capacity audit does not pass (`revisit_unresolved` and `allocation_error` have marginally negative accuracy advantage). 6B is now probed against controller state directly (it previously had no state probe and leaked ground-truth labels).
- **Stage 7** (faithful NL reportability): supported for the local calibrated opaque-token reporter, capacity audit passes; external API LLM and VLM routes remain open. Sharper caveat: the local decoder reads the scored content fields from attended-content token bases the renderer fills directly from the model's attended content (a schema-aware structural round-trip), so consistent token-remapping and held-out-combination anti-memorization tests do not bite it — the genuine faithfulness test needs a latent-only decoder or the (blocked) external LLM/VLM path. See `docs/ROADMAP.md` "Sharper decoder caveat".
- **reduced-shaping resilience**: holds at weight `0.25` (acc `0.34`, clearly above static); complete **zero-shaping collapses to `~0.19`** (≈ static), so complete zero-shaping is *not* supported (a known weakness; the `0.15` accuracy threshold is too lenient and is flagged for calibration).

> Note: the pre-rehab soft-attention checkpoints/reports described a non-functional model
> (recurrent collapsed to uniform attention and lost to the baseline); every "supported"
> label from that era was a probe artifact and has been superseded by the discrete-attention
> rehab.

The eval artifacts also include intervention, switched-cue, self-state, self-model, Stage 6B
uncertainty, and Stage 7 visual-report plots, plus Stage 3 repeated-seed and checkpoint-family
diagnostics.

The exact numbers depend on the saved checkpoint, but the intended workflow is:

1. train both baseline and recurrent models
2. run ablations
3. inspect the JSON report, predictive-probe results, report-probe results, intervention results, cue-switch results, reduced-shaping results, and attention plots

## Tests

Run the smoke and regression tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
