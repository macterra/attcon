# attcon

`attcon` is a minimal PyTorch benchmark for testing whether a model is merely computing attention or actually controlling it over time.

The current roadmap treats Stages 1 through 3 as the sequential foundation, then splits into parallel lines of work around engineered self-state tracking, stronger learned self-modeling, and flexible reallocation under changed priorities. After structured reportability, the repo now includes a bounded Stage 7 route: a local calibrated reporter reads opaque tokenized internal state and emits faithful natural-language-shaped reports. Powered external LLM and VLM variants have both now been evaluated with negative results on the current v3 interface.

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
  The task-only emergence audit finds weak decodable inspection-history structure, but its causal
  direction has policy-inconsistent semantics: increasing "already inspected" increases attention
  to that cell across four intervention scales, so Stage 4B remains unsupported.
- **Stage 5** (cue-switch reallocation): supported (recurrent switch accuracy `0.25` vs baseline `0.0`).
- **Stage 6A** (structured reportability): supported, capacity audit passes — controller state beats a capacity-matched observation probe on current search type and current attended cell. An empirical permuted-label noise floor (`noise_floor_metrics`) confirms significance: the real advantages (`~0.38`, `~0.42`) are ~100x above the permuted-label p95 floor (`~0.004`, `~0.003`).
- **Stage 6B** (uncertainty / allocation-error reportability): **bounded positive evidence; full support false**. Controller state beats the capacity-matched observation baseline on positive *recall* for all four gated signals, and all four recall advantages clear their permuted-label p95 floors. The stricter calibrated capacity audit still fails because `revisit_unresolved` and `allocation_error` have marginally negative accuracy advantages and do not clear their accuracy noise floors. See `audits/stage6b_noise_floor_tune_prob_035.json`.
- **Stage 7** (faithful NL reportability): supported for the local calibrated opaque-token reporter, capacity audit passes. Sharper caveat: the local decoder reads the scored content fields from attended-content token bases the renderer fills directly from the model's attended content (a schema-aware structural round-trip), so consistent token-remapping and held-out-combination anti-memorization tests do not bite it. A **latent-only decoder** (`run_latent_only_report_mode`) that recovers content from an opaque quantised view of internal state alone is implemented but does **not** clear the bar on the original checkpoint. Powered external GPT-5 mini audits are also negative on the v3 state. The text route's latent-only current-content joint accuracy is `0/12`, `1/12`, and `0/12` across default, cue-switch, and intervened slices, with remembered/full content `0/12` throughout. The VLM route uses an eight-level label-free state heatmap and scores `0/8` on current, remembered, and full content in all three slices; its explicit symbolic-image upper bound scores `8/8` on all joint metrics and the entire report in every slice, validating the vision/control path. See `audits/stage7_external_llm_powered_content_memory_v3.json` and `audits/stage7_external_vlm_powered_content_memory_v3.json`.
- **Branch C** (unity / binding): **strong bounded support** across two synthetic surface variants and two model families. The explicit shared-selector family passes all frozen gates in the original and surface-v2 benchmarks, then repeats that result over three independent data/model-seed pairs per variant. A structurally different cue-token set transformer also reaches `1.00` minimum joint accuracy, lure rejection, and intervention coherence across both variants; its pooled controls have exactly equal parameter counts and reach only `0.014`/`0.499` and `0.006`/`0.500` joint/lure performance. This clears the predeclared Branch C threshold. It is not called robust for Stage 8: the transformer replication is single-seed per variant, and both tasks remain in one synthetic selection family. See `audits/branch_c_binding_cross_model.json`.
- **Branch D** (counterfactual access): **strong bounded support across seeds, architectures, and two surface variants**. The unstructured GRU memorizes training conjunctions (`1.00`) and scores `0.00` held out, establishing the negative control. Relational GRU runs pass every frozen gate over three fresh data/model-seed pairs. A permutation-equivariant set transformer then passes the original and changed-vocabulary surface task with exactly parameter-matched no-cache controls; all minimum accuracy, comparator-advantage, and intervention metrics remain `1.00`. The decoder never receives the explicit cache. Boundary: both successful families share explicit relational key addressing, so this is not spontaneous access in the original controller or Stage 8-robust evidence. See `audits/branch_d_access_cross_model.json`.
- **Branch E** (higher-order state): **engineering support only; theoretical support false**. A shared latent trained on report, confidence, reinspection, and correction—but never the exact six-way status labels—supports `1.00` held-out status decoding, versus `0.00` first-order and `0.302` observation-only with identical 4,550-parameter probes. On 537 observation- and content-matched wrong/current pairs, latent swaps switch every access-sensitive behavior correctly at `1.00`. Because confidence, reinspection, and correction directly reward higher-order distinctions, this demonstrates an engineered representational route rather than spontaneous HOT-style emergence. See `audits/branch_e_higher_order_pilot.json`.
- **Branch F** (broadcast / ignition): **seed-robust engineering support only; spontaneous broadcast unsupported**. Shared and private-shortcut models are exactly parameter matched and both are task viable. Across three fresh seed pairs, every frozen gate passes: minimum shared joint accuracy `0.962`, aligned-onset accuracy `1.00`, shared-ablation drop `0.841`, coordination advantage `0.659`, and content-swap follow rate `0.984`; private single-route damage stays below `0.192`. Local action is invariant. Because all consumers and the bottleneck are explicitly engineered, Branch F remains theoretically unsupported. See `audits/branch_f_broadcast_multiseed.json`.
- **Stage 8** (multi-theory convergence): **not met**. The current artifact-level audit records three pass, three partial, and two failed gates. The decisive failures are absence of cross-validated causal overlap for the same internal content across families and absence of a genuinely different benchmark. Surface/cardinality variants do not count as different benchmarks, and Branch E/F engineering results are excluded from theoretical-family counts. See `audits/stage8_convergence_current.json`.
- **Stage 8 integrated-content scaffold**: ready, but not evidence yet. The generator creates 2,048 paired episodes (8,192 cases) in which one target identity and initial feature bundle are held exactly fixed across unavailable, merely-visible, previously-attended, and counterfactually-accessible transitions. Binding and access queries name that same target, current attention remains fixed elsewhere, false-binding lures are absent conjunctions, and a shared-state perturbation point is reserved. All seven dataset gates pass; no trained-model or causal-overlap claim is made. See `audits/stage8_integrated_content_scaffold.json`.
- **Stage 8 integrated-content pilot**: all ten engineering gates pass. Exactly parameter-matched shared-state and split-state models (22,003 parameters each) both reach `1.00` held-out binding/access joint accuracy, while the identity-destroying pooled control reaches `0.011`. Swapping the binding content state makes both binding and accessible access outputs follow the donor in the shared model (`1.00` joint), but access retains the receiver in the split model (joint donor-follow `0.115`), a coordination advantage of `0.885`. This validates a causal-overlap assay on the same content, but the bottleneck is imposed and the result is single-seed, so the Stage 8 same-content gate remains open. See `audits/stage8_integrated_content_pilot.json`.
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
