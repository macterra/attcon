# A Minimal Benchmark and Staged Program for Recurrent Attention Control

## Abstract

Many machine learning systems compute attention, but fewer cleanly demonstrate **attention control**: the ability of a distinct controller to regulate future attention on the basis of task demands and the consequences of previous allocations. We present a minimal PyTorch benchmark for that distinction and report the current repository status of the broader staged research program built around it. The task is a cue-guided selective-search problem on a `5x5` grid in which visible cell types are globally available, but task-relevant target identity becomes useful only through attention. On the current default checkpoint, a recurrent attention controller outperforms a static cue-conditioned baseline in held-out accuracy (`0.312` vs. `0.229`), shows strong temporal reallocation (`0.751` vs. `0.000`), and achieves positive target-attention gain (`0.125` vs. `0.000`). Additional evaluations now support stronger claims than closed-loop control alone: predictive probes show that controller state predicts the next attention map better than observation alone, causal interventions on controller state shift later attention, explicit inspected-state variables support bounded self-modeling of attention history, cue-switch training yields positive changed-priority reallocation, and structured internal-content probes support reportability of search type, attended cell, target-found status, and unresolved regions. The strongest open problem is now natural-language reportability from tokenized internal state under a stricter memory-focused test: symbolic state dumps are reported faithfully, but tokenized-state reports do not yet reliably describe current or remembered attended contents better than observation-only baselines. The current repository therefore supports Stages 1 through 6 of the roadmap and an implemented but not yet successful Stage 7.

## 1. Introduction

The phrase *attention control* is often used loosely. In many architectures, attention is simply a learned weighting mechanism inside a single feedforward computation. That is not yet the same thing as a system that **controls its own attention**.

We use a stricter criterion. A system exhibits attention control only if:

1. it has an object-level attention process that allocates attention over inputs,
2. it has a distinct controller with access to a representation of that allocation or its consequences, and
3. it can modify future allocation on the basis of task demands, performance, or internal state.

The goal of this project is not to solve a large-scale perceptual problem. The goal is to build the smallest credible setting in which the difference between **attending** and **controlling attention** can be measured directly, then extend that benchmark into a staged program about explicit attention modeling, self-modeling, reportability, and eventually natural-language access to internal state.

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

Those additions matter because the project now evaluates not only whether the controller improves attention regulation, but also whether it maintains bounded internal models of its own attentional history and supports report-like access to that state.

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
- explicit attention modeling via predictive probes and intervention,
- self-modeling of attention history,
- flexible cue-switch reallocation under changed priorities,
- structured reportable internal content,
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

Together, these results support a bounded Stage 3 claim: controller state is not merely generic recurrent memory, but carries structured information about future attention and causally influences later allocation.

### 5.4 Self-Modeling of Attention

The recurrent controller now maintains an explicit inspected-cell state and a native self-model head.

Current default results:

- native inspected-map cell accuracy: `0.990`
- observation-only inspected-map accuracy: `0.973`
- native target-inspected accuracy: `0.988`
- native target-inspected positive recall: `0.721`

This supports a bounded Stage 4 claim: the model contains an explicit internal variable about where it has already attended, and that variable supports more faithful reporting than observation-only baselines.

### 5.5 Flexible Reallocation Under Changed Priorities

The current default training mixes stationary and switched-cue episodes, and the repository now evaluates mid-episode cue switching directly.

Current default results:

- baseline switch-target gain: `0.0233`
- recurrent switch-target gain: `0.1169`
- baseline switch accuracy: `0.000`
- recurrent switch accuracy: `0.500`

This supports Stage 5 in the current benchmark: the recurrent controller can redirect attention under changed priorities better than the static baseline.

### 5.6 Structured Reportable Internal Content

The current report probes test whether controller state supports explicit readouts of its own regulatory state.

Current default results:

- search-type accuracy advantage over observation-only: `0.308`
- attended-cell accuracy advantage: `0.287`
- target-found accuracy advantage: `0.0124`
- unresolved-region advantage from native self-model: `0.0172`

This supports Stage 6 in a bounded sense: the same controller state that guides attention also supports structured reports about current search type, attended cell, target-found state, and unresolved regions.

### 5.7 Natural-Language Reportability

The repository now includes a Stage 7 natural-language reporting harness using `gpt-5-mini`. It evaluates three reporting conditions:

- symbolic internal-state serialization as a weak baseline,
- tokenized internal-state reporting as the real Stage 7 target,
- observation-only reporting as the weaker external baseline.

The current picture is now judged under a stricter, more skeptical setup:

- evaluation examples are restricted to non-initial timesteps so remembered previous-attention content is genuinely required,
- the report schema now asks for both current attended content and previous attended content,
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

So the tokenized internal-state interface is not yet strong enough to support a positive Stage 7 claim. The more skeptical memory-focused probe is useful precisely because it narrows the interpretation: the current tokenized representation still does not support convincing language reports of either current attended content or remembered previous attended content.

## 6. Interpretation

The main result is no longer just that recurrence is generally useful. The more specific repo-level result is that a small recurrent controller, given access to previous attention and its consequences, can support:

- closed-loop attention regulation,
- explicit attention-dynamics probes,
- bounded self-modeling of attentional history,
- flexible reallocation under changed priorities,
- structured internal report variables.

That is already stronger than the original attention-control benchmark framing.

The stronger claim should still be stated carefully. The current evidence supports:

- recurrence improves attention regulation,
- controller state predicts and causally influences future allocation,
- the model tracks inspected history explicitly,
- structured internal contents are available for bounded report,
- changed-priority reallocation can be trained successfully.

It does **not yet** support:

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

What the current repository adds is a sharper boundary around that interpretation. Structured reportability is now supported, but natural-language reportability from tokenized internal state is not yet. That distinction is valuable: it prevents the project from overclaiming and keeps the theoretical interpretation tied to empirical tests.

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

The next highest-value experiments are now concentrated in Stage 7 and beyond:

1. improve the tokenized internal-state interface so current and remembered attended semantic content are easier to recover than from observation-only input,
2. separate current-attention content tokens from memory-of-previous-attention tokens more sharply,
3. test natural-language reporting under cue switches and controller interventions,
4. add uncertainty and allocation-error report targets that distinguish “not yet inspected” from “inspected but failed,”
5. continue measuring robustness over repeated seeds and checkpoints.

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

## 10. Conclusion

The repository now goes well beyond a minimal Stage 2 benchmark. In the current default setup, a recurrent attention controller outperforms a static baseline, shows strong temporal reallocation, supports predictive and intervention evidence for explicit attention modeling, maintains an explicit self-model of inspected history, handles cue switching better than the baseline, and supports structured internal report variables. Those results are enough to support Stages 1 through 6 of the roadmap in a bounded benchmark sense.

The strongest remaining open problem is Stage 7: faithful natural-language access to tokenized internal state, especially for the current and remembered contents of attention. Symbolic state dumps are easy for the language model to report faithfully. Tokenized internal-state reporting is not yet good enough. That gap is now the clearest frontier in the project, and it is precisely what makes the benchmark useful as a disciplined stepping stone rather than a vague consciousness metaphor.
