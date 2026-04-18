# A Minimal Benchmark and Staged Program for Recurrent Attention Control

## Abstract

Many machine learning systems compute attention, but fewer cleanly demonstrate **attention control**: the ability of a distinct controller to regulate future attention on the basis of task demands and the consequences of previous allocations. We present a minimal PyTorch benchmark for that distinction and report the current repository status of the broader staged research program built around it. The task is a cue-guided selective-search problem on a `5x5` grid in which visible cell types are globally available, but task-relevant target identity becomes useful only through attention. On the current default checkpoint, a recurrent attention controller outperforms a static cue-conditioned baseline in held-out accuracy (`0.312` vs. `0.229`), shows strong temporal reallocation (`0.751` vs. `0.000`), and achieves positive target-attention gain (`0.125` vs. `0.000`). Additional evaluations extend the benchmark beyond closed-loop control alone: predictive probes show that controller state predicts the next attention map better than observation alone, causal interventions and reduced-shaping tests are implemented as stricter Stage 3 checks, explicit inspected-state variables support bounded engineered self-state tracking of attention history, cue-switch training yields positive changed-priority reallocation, and structured probes support bounded reportability of search type, attended cell, target-found status, and unresolved regions. The strongest open problem is now Stage 7 reportability of current and remembered attended contents: tokenized-state language reports still do not beat observation-only baselines, and a parallel VLM route may be more natural for the spatial form of the underlying internal state. The current repository therefore supports a meaningful Stage 2 benchmark, later bounded results around engineered self-state tracking and structured reportability, and an implemented but not yet successful Stage 7 harness, while stronger claims remain provisional or open. Recent additions make the later stages more inspectable: Stage 6B now distinguishes active wrong-candidate pursuit from cumulative wrong-candidate history and unresolved revisits, and the evaluator now emits Stage 7 visual report panels that place scene-only, explicit symbolic, and minimal tokenized state views side by side.

## 1. Introduction

The phrase *attention control* is often used loosely. In many architectures, attention is simply a learned weighting mechanism inside a single feedforward computation. That is not yet the same thing as a system that **controls its own attention**.

We use a stricter criterion. A system exhibits attention control only if:

1. it has an object-level attention process that allocates attention over inputs,
2. it has a distinct controller with access to a representation of that allocation or its consequences, and
3. it can modify future allocation on the basis of task demands, performance, or internal state.

The goal of this project is not to solve a large-scale perceptual problem. The goal is to build the smallest credible setting in which the difference between **attending** and **controlling attention** can be measured directly, then extend that benchmark into a staged program about explicit attention modeling, engineered and learned self-state modeling, reportability, and eventually natural-language access to internal state.

## 2. Benchmark Setup

### 2.1 Task

The benchmark uses a cue-guided selective-search task on a `5x5` grid. Each cell has:

- a visible type identity,
- a hidden cue-specific target flag,
- and a hidden digit identity.

For each cue type, exactly one cell of that visible type is designated as the target for that cue. The model must report the digit associated with the target cell for the current cue.

This matters because the scene contains structure that is globally available, but task-relevant evidence only becomes useful after an attention allocation and cue-conditioned interpretation.

### 2.2 Sequence Structure

An episode lasts multiple timesteps. At each step the model:

1. produces an attention distribution over cells,
2. extracts a glimpse,
3. converts that glimpse into a cue-conditioned observation,
4. predicts the target digit,
5. optionally updates future allocation from previous attention outcomes and feedback.

The static baseline uses the same scene and cue information, but it does not carry state across steps. Its attention distribution is fixed within the episode. The recurrent controller instead updates attention from a recurrent summary of previous attention, previous observation, previous loss proxy, previous confidence, and cue.

## 3. Models

### 3.1 Static Baseline

The static baseline is a cue-conditioned attention model without recurrence. It encodes the visible scene and cue into a scene summary, produces one attention distribution over grid cells, extracts a hidden glimpse, maps that glimpse into a cue-conditioned observation, and predicts the target digit.

This baseline answers the question: how far can one get with attention *without* attention control?

### 3.2 Recurrent Attention Controller

The recurrent model augments the same scene encoding and task head with a recurrent controller. Its recurrent summary includes:

- previous attention,
- previous cue-conditioned observation,
- previous detached loss proxy,
- previous detached confidence,
- cue embedding.

That summary is passed through a `GRUCell` and learned summary adapter to produce the next hidden state, which in turn produces the next attention logits. Future allocation is therefore explicitly conditioned on a representation of previous allocation and its task-level consequences.

The current repository extends this controller with additional internal state used in later stages:

- an explicit inspected-cell state,
- a native self-model head over inspected history,
- a cumulative found-state variable,
- a target-found report head.

Those additions matter because the project now evaluates not only whether the controller improves attention regulation, but also whether it maintains bounded internal state about its own attentional history and supports report-like access to that state.

## 4. Training and Evaluation

### 4.1 Optimization

Training uses:

- final-step cross-entropy on digit prediction,
- a small auxiliary loss on intermediate predictions,
- a final-step target-attention loss that rewards placing mass on the true target cell,
- auxiliary self-model and target-found reporting losses for the recurrent controller.

The direct target-attention term makes the benchmark easier to interpret, but the repository now also includes reduced-shaping evaluations to test whether useful reallocation survives when that term is weakened or removed.

### 4.2 Evaluation Axes

The original benchmark emphasized three claims:

- dissociation from static and weaker non-recurrent controls,
- closed-loop adaptation,
- cue dependence.

The current repository now evaluates a broader staged set of claims:

- closed-loop attention control,
- explicit attention modeling via predictive probes, intervention, and reduced-shaping checks,
- engineered self-state tracking of attention history,
- flexible cue-switch reallocation under changed priorities,
- structured reportability of bounded internal content,
- natural-language reportability.

## 5. Current Results

The current default report in `outputs/minimal/evaluation_report.json` supports a richer picture than the original benchmark alone.

### 5.1 Main Comparison

On the current default checkpoint:

- Static baseline accuracy: `0.229`
- Recurrent controller accuracy: `0.312`
- Static baseline target attention: `0.065`
- Recurrent controller target attention: `0.070`

The absolute target-attention gap on this checkpoint is smaller than in some earlier runs, but the recurrent controller still produces meaningfully better final task performance.

### 5.2 Closed-Loop Dynamics

- Static temporal reallocation: `0.000`
- Recurrent temporal reallocation: `0.751`
- Static target-attention gain: `0.000`
- Recurrent target-attention gain: `0.125`

These are the clearest Stage 2 signals in the current run. The recurrent controller does not merely learn a better static map; it changes its attention over time in a task-relevant way.

### 5.3 Explicit Attention Modeling

The repository now includes predictive-probe and intervention tests.

Predictive probe:

- controller-state test cross-entropy: `1.752`
- observation-only test cross-entropy: `2.269`
- controller top-1 advantage: `0.276`

Causal intervention:

- attention-change KL: `0.0181`
- original-target attention drop: `0.00241`
- alternate-target attention gain: `0.0174`

Reduced-shaping condition:

- at `attention_target_weight = 0.0`, accuracy remains `0.184`
- temporal reallocation remains `0.484`
- target-attention gain remains `0.0255`

Together, these results provide positive but still threshold-sensitive evidence for a bounded Stage 3 claim: controller state is not merely generic recurrent memory, but carries structured information about future attention and causally influences later allocation. In the revised roadmap and evaluator, this stage should count as supported only when predictive, intervention, and reduced-shaping thresholds are all met together and the repeated-seed robustness gate also passes. The evaluator now explicitly distinguishes a weaker single-run pass from the stricter robust pass, and it also extends that robustness check across the default checkpoint and reduced-shaping checkpoint family. The current Stage 3 summaries therefore report not only that the stage is still unstable, but also which metric and which seed family form the present bottleneck.

### 5.4 Engineered Self-State Tracking

The recurrent controller now maintains an explicit inspected-cell state and a native report head over that state.

Current default results:

- native inspected-map cell accuracy: `0.990`
- observation-only inspected-map accuracy: `0.973`
- native target-inspected accuracy: `0.988`
- native target-inspected positive recall: `0.721`

This supports a bounded Stage 4A-style claim: the model contains an explicit internal variable about where it has already attended, and that variable supports more faithful reporting than observation-only baselines. The stronger Stage 4B-style claim, that the controller has learned and uses its own attentional self-model without relying mainly on an engineered scaffold, remains open.

### 5.5 Flexible Reallocation Under Changed Priorities

The current default training mixes stationary and switched-cue episodes, and the repository now evaluates mid-episode cue switching directly.

Current default results:

- baseline switch-target gain: `0.0233`
- recurrent switch-target gain: `0.1169`
- baseline switch accuracy: `0.000`
- recurrent switch accuracy: `0.500`

This supports Stage 5 in the current benchmark: the recurrent controller can redirect attention under changed priorities better than the static baseline.

### 5.6 Structured Reportability of Internal Content

The current report probes test whether controller state supports explicit readouts of its own regulatory state.

Current default results:

- search-type accuracy advantage over observation-only: `0.308`
- attended-cell accuracy advantage: `0.287`
- target-found accuracy advantage: `0.0124`
- unresolved-region advantage from native self-model: `0.0172`

This supports a bounded Stage 6A-style claim: the same controller state that guides attention also supports structured reports about current search type, attended cell, target-found state, and unresolved regions. The stronger Stage 6B-style target, reportability of uncertainty and allocation error, is now implemented as a distinct evaluation family. It remains provisional overall, but the evaluator now separates current wrong-candidate pursuit, cumulative wrong-candidate history, revisit-under-unresolved-search, and allocation error. That finer decomposition makes the positive result more interpretable: current wrong-candidate and wrong-candidate-history signals provide bounded positive evidence that some uncertainty-style report variables can beat observation-only baselines on positive-recall style reporting, while revisit-under-unresolved-search and allocation error remain weaker.

### 5.7 Natural-Language Reportability

The repository now includes a Stage 7 natural-language reporting harness using `gpt-5-mini`. It evaluates three reporting conditions:

- symbolic internal-state serialization as a weak baseline,
- tokenized internal-state reporting as the real Stage 7 target,
- observation-only reporting as the weaker external baseline.

A plausible parallel branch now under consideration is a VLM-based Stage 7 route in which internal attention/self-model state is rendered as compact visual panels or overlays. That branch would be useful only if it obeys the same anti-cheating discipline as the text route: explicit labeled visual dumps would count only as baselines, while minimally labeled internal-state renderings would need to beat scene-only baselines on held-out current and remembered attended-content reports.

The current picture is now judged under a stricter, more skeptical setup:

- evaluation examples are restricted to non-initial timesteps so remembered previous-attention content is genuinely required,
- the report schema now asks for both current attended content and previous attended content,
- the same schema now also carries Stage 6B-style variables for relevant-region inspection, unresolved search, current wrong-candidate pursuit, wrong-candidate history, revisit-under-unresolved-search, and allocation error,
- symbolic reporting is strong and can achieve exact structured reports on held-out slices,
- tokenized-state reporting still does not beat observation-only on the full report bundle.

In a recent skeptical Stage 7 slice:

- tokenized search-type accuracy: `1.0`
- tokenized attended-digit accuracy: `1.0`
- tokenized previous-attended-cell accuracy: `0.0`
- tokenized previous-attended-visible-type accuracy: `0.0`
- tokenized previous-attended-digit accuracy: `0.0`
- tokenized previous-glimpse-digit accuracy: `0.0`
- tokenized joint accuracy: `0.0`

Against the same slice:

- observation-only joint accuracy: `0.0`
- observation-only previous-attended-cell accuracy: `0.0`
- observation-only previous-attended-visible-type accuracy: `0.0`
- observation-only previous-attended-digit accuracy: `1.0`
- observation-only previous-glimpse-digit accuracy: `1.0`

So the tokenized internal-state interface is not yet strong enough to support a positive Stage 7 claim. The more skeptical memory-focused probe is useful precisely because it narrows the interpretation: the current tokenized representation still does not support convincing language reports of either current attended content or remembered previous attended content. The harness is nevertheless better aligned with the roadmap than before, because the language-report schema now covers the same finer Stage 6B uncertainty/allocation-error variables used in the structured evaluator, the example format now carries cue-history and inspection-history fields too, the evaluator can run dedicated cue-switch and intervention NL slices through the same reporting interface, and the artifact bundle now includes Stage 7 visual panels that make the three reporting interfaces directly comparable.

## 6. Interpretation

The main result is no longer just that recurrence is generally useful. The more specific repo-level result is that a small recurrent controller, given access to previous attention and its consequences, can support:

- closed-loop attention regulation,
- explicit attention-dynamics probes,
- bounded engineered self-state tracking of attentional history,
- flexible reallocation under changed priorities,
- structured internal report variables.

That is already stronger than the original attention-control benchmark framing.

The stronger claim should still be stated carefully. The current evidence supports:

- recurrence improves attention regulation,
- controller state predicts future allocation and can be probed and perturbed in ways consistent with Stage 3-style modeling, though thresholded support remains provisional,
- the model tracks inspected history explicitly through an engineered self-state scaffold,
- structured internal contents are available for bounded report,
- changed-priority reallocation can be trained successfully.

It does **not yet** support:

- a clean Stage 4B-style claim of learned self-modeling of attention,
- a broad or fully stable Stage 6B-style claim of uncertainty and allocation-error reportability,
- faithful natural-language report from tokenized internal state,
- faithful language access to the current and remembered contents of attention,
- a strong claim that the controller’s internal state is already a sufficient consciousness-like schema in anything but a speculative sense.

### 6.1 Relation to Good Regulator and Modeler Schema Framing

The benchmark still admits a natural interpretation in the language of the Good Regulator Theorem and modeler-schema ideas. On that framing, the most plausible candidate for consciousness-like content is not the raw scene representation or the raw attention mask. It is the controller state that carries forward:

- previous attention allocation,
- previous cue-conditioned observation,
- previous task feedback,
- explicit inspected-state variables,
- and later report-oriented self-model variables.

What the current repository adds is a sharper boundary around that interpretation. Bounded structured reportability is now supported for a limited set of internal variables, and Stage 6B-style uncertainty reporting now has an initial foothold through the wrong-candidate-history signal, but the broader uncertainty/allocation-error bundle and natural-language reportability from tokenized internal state are not yet settled. That distinction is valuable: it prevents the project from overclaiming and keeps the theoretical interpretation tied to empirical tests.

## 7. Limitations

This system is still intentionally minimal.

- The environment is synthetic and low-dimensional.
- Attention is soft rather than hard fixation.
- Some checkpoint-level metrics vary across training recipes.
- Stage 7 natural-language reporting still depends on an external language model and remains unstable enough that small evaluation slices are more reliable than large aggregate runs.
- Tokenized internal-state reporting remains the current bottleneck.
- The sharper memory-focused probe makes the present Stage 7 result more informative, but also harder to pass.

So while the repository now supports much stronger claims than the original benchmark paper draft, it is still best understood as a disciplined toy program rather than a comprehensive model of attentional control or consciousness.

## 8. Immediate Next Work

The next highest-value experiments are now concentrated in Stage 3 thresholding, Stage 6B, and Stage 7:

1. set and test explicit Stage 3 claim thresholds across repeated seeds and checkpoints,
2. strengthen the Stage 6B bundle beyond wrong-candidate history so unresolved search and allocation-error reports also beat observation-only baselines,
3. continue documenting the now-completed split between Stage 6A-style structured reportability and Stage 6B-style uncertainty/allocation-error reportability across writeups and artifacts,
4. improve the tokenized internal-state interface so current and remembered attended semantic content are easier to recover than from observation-only input,
5. separate current-attention content tokens from memory-of-previous-attention tokens more sharply,
6. test natural-language reporting under cue switches and controller interventions,
7. add a VLM branch that reads minimally labeled visual internal-state renderings and compare it against scene-only and explicit-dump baselines.

## 9. Reproducibility

The implementation lives in this repository:

- benchmark/task generation: [src/attcon/data.py](/home/david/dev/attcon/src/attcon/data.py)
- models: [src/attcon/models.py](/home/david/dev/attcon/src/attcon/models.py)
- training: [src/attcon/train.py](/home/david/dev/attcon/src/attcon/train.py)
- evaluation: [src/attcon/eval.py](/home/david/dev/attcon/src/attcon/eval.py)
- Stage 7 NL reporting helpers: [src/attcon/nl_report.py](/home/david/dev/attcon/src/attcon/nl_report.py)
- default config: [configs/minimal.yaml](/home/david/dev/attcon/configs/minimal.yaml)

Default commands:

```bash
.venv/bin/python -m attcon.train --config configs/minimal.yaml
.venv/bin/python -m attcon.eval --config configs/minimal.yaml --checkpoint outputs/minimal/experiment.pt
```

Current evaluation artifacts also include intervention comparison plots, switched-cue comparison plots, self-state diagnostics plots, self-model trajectory plots, Stage 6B uncertainty-report comparison plots, and Stage 7 visual report panels in addition to the JSON report.

## 10. Conclusion

The repository now goes well beyond a minimal Stage 2 benchmark. In the current default setup, a recurrent attention controller outperforms a static baseline, shows strong temporal reallocation, supports predictive, intervention, and reduced-shaping analyses around explicit attention modeling, maintains an explicit engineered state about inspected history, handles cue switching better than the baseline, and supports structured internal report variables. The evaluator also now exposes repeated-seed and checkpoint-family Stage 3 summaries plus dedicated Stage 3 diagnostics plots, switched-cue artifacts, self-state artifacts, self-model artifacts, and Stage 6B uncertainty diagnostics artifacts, which make current weaknesses easier to inspect rather than masking them behind a single headline run. Those Stage 3 robustness numbers are also surfaced directly in the evidence summary, including a distinction between single-run and robust support plus the current bottleneck family and metric, which makes the remaining fragility harder to overlook. Those results are enough to support a bounded attention-control benchmark plus later-stage engineered self-state and structured-reportability results, but not yet enough to collapse the later roadmap stages into a single settled ladder.

The strongest remaining open problem is Stage 7: faithful natural-language access to internal attention state, especially for the current and remembered contents of attention. Symbolic state dumps are easy for a language model to report faithfully. Tokenized internal-state reporting is not yet good enough, and a VLM route may prove more natural for spatial internal-state readout if held to the same baseline controls. That gap is now the clearest frontier in the project, and it is precisely what makes the benchmark useful as a disciplined stepping stone rather than a vague consciousness metaphor.
