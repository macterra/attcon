# A Minimal Benchmark for Recurrent Attention Control

## Abstract

Many machine learning systems compute attention, but fewer cleanly demonstrate **attention control**: the ability of a distinct controller to regulate future attention on the basis of task demands and the consequences of previous allocations. We introduce a minimal PyTorch benchmark for this distinction. The task is a cue-guided selective-search problem on a `5x5` grid in which visible cell types are globally available, but task-relevant target identity and digit information become useful only through attention. We compare a static cue-conditioned attention baseline to a recurrent attention controller that updates future attention from previous attention, previous observation, and detached task feedback. On the current benchmark, the recurrent controller outperforms the static baseline in held-out accuracy (`0.367` vs. `0.223`) and target attention (`0.108` vs. `0.062`). Ablations show that removing recurrence or replacing it with a feedforward summary policy substantially reduces performance. Probe metrics further show temporal reallocation, positive target-attention gain, and stronger wrong-cue sensitivity for the recurrent controller. These results support the claim that even a very small recurrent architecture can instantiate a meaningful form of attention control.

## 1. Introduction

The phrase *attention control* is often used loosely. In many architectures, attention is simply a learned weighting mechanism inside a single feedforward computation. That is not yet the same thing as a system that **controls its own attention**.

We use a stricter criterion. A system exhibits attention control only if:

1. it has an object-level attention process that allocates attention over inputs,
2. it has a distinct controller with access to a representation of that allocation or its consequences, and
3. it can modify future allocation on the basis of task demands, performance, or internal state.

The goal of this project is not to solve a large-scale perceptual problem. The goal is to build the smallest credible experimental setting in which the difference between **attending** and **controlling attention** can be measured directly.

## 2. Benchmark Setup

### 2.1 Task

The benchmark uses a cue-guided selective-search task on a `5x5` grid. Each cell has:

- a visible type identity,
- a hidden cue-specific target flag, and
- a hidden digit identity.

For each cue type, exactly one cell of that visible type is designated as the target for that cue. The model must report the digit associated with the target cell for the current cue.

This design matters because the scene contains structure that is globally available, but task-relevant evidence only becomes useful after an attention allocation and cue-conditioned interpretation.

### 2.2 Sequence Structure

An episode lasts multiple timesteps. At each step the model:

1. produces an attention distribution over cells,
2. extracts a glimpse,
3. converts that glimpse into a cue-conditioned observation,
4. predicts the target digit, and
5. optionally uses the previous attention outcome to determine the next allocation.

The static baseline uses the same scene and cue information, but it does not carry state across steps. Its attention distribution is fixed within the episode. The recurrent controller instead updates attention from a recurrent summary of previous attention, previous observation, previous loss proxy, and previous confidence.

## 3. Model

### 3.1 Static Baseline

The static baseline is a cue-conditioned attention model without recurrence. It encodes the visible scene and cue into a scene summary, produces one attention distribution over grid cells, extracts a hidden glimpse, maps that glimpse into a cue-conditioned observation, and predicts the target digit.

This baseline answers an important question: how far can one get with attention *without* attention control?

### 3.2 Recurrent Attention Controller

The recurrent model augments the same scene encoding and task head with a recurrent controller. Its summary state includes:

- previous attention,
- previous cue-conditioned observation,
- previous detached loss proxy,
- previous detached confidence,
- cue embedding.

This summary is passed through a `GRUCell` and a learned summary adapter to produce the next hidden state, which in turn produces the next attention logits. In that sense, future allocation is explicitly conditioned on a representation of previous allocation and its task-level consequences.

Under a Good Regulator interpretation, this recurrent summary plus hidden state is the model's internal **model of attention**. It is the representation through which the controller tracks what it previously attended to, what that attention revealed, and how useful that allocation was for the task. The raw attention mask alone is not yet the model; the model is the internal state that makes the previous allocation available for regulation of the next one.

## 4. Training and Evaluation

### 4.1 Optimization

Training uses:

- final-step cross-entropy on digit prediction,
- a small auxiliary loss on intermediate predictions,
- a final-step target-attention loss that rewards placing mass on the true target cell.

The attention-target loss is important because it gives the controller a direct signal about where useful evidence should end up by the end of the episode.

### 4.2 Evidence Criteria

We evaluate three claims.

#### Dissociation

If the controller matters, the recurrent model should outperform:

- a static attention baseline,
- a frozen-recurrence ablation,
- a feedforward summary ablation that uses the current summary but no recurrent state.

#### Closed-Loop Adaptation

If the controller is genuinely regulating attention, the model should show:

- nonzero temporal reallocation,
- positive target-attention gain across timesteps,
- degradation when recurrent control is removed.

#### Cue Dependence

If attention is task-sensitive, behavior should change under the wrong cue. We therefore measure:

- wrong-cue accuracy,
- wrong-cue target attention,
- cue-accuracy delta,
- cue-target-attention delta.

## 5. Results

On the current saved checkpoint:

### 5.1 Main Comparison

- Static baseline accuracy: `0.223`
- Recurrent controller accuracy: `0.367`
- Static baseline target attention: `0.062`
- Recurrent controller target attention: `0.108`

### 5.2 Stronger Ablations

- Freeze recurrence accuracy: `0.135`
- Feedforward summary accuracy: `0.188`

These ablations are important because they show the result is not merely due to a richer policy head or stepwise access to a summary. The recurrent state itself matters.

### 5.3 Dynamic Attention Metrics

- Recurrent temporal reallocation: `0.741`
- Freeze temporal reallocation: `0.000`
- Recurrent target-attention gain: `0.129`
- Feedforward target-attention gain: `0.005`

The recurrent model does not merely maintain a better fixed attention map. It changes its attention over time in a way that improves concentration on the target.

### 5.4 Cue Sensitivity

- Baseline cue-accuracy delta: `0.000`
- Recurrent cue-accuracy delta: `0.250`
- Baseline cue-target-attention delta: `0.027`
- Recurrent cue-target-attention delta: `0.155`

The recurrent controller is therefore not only more accurate; it is more strongly governed by task cue in the specific sense expected of attention control.

## 6. Interpretation

The main result is not that recurrence is generally useful. The more specific result is that a small recurrent controller, given access to previous attention and its consequences, can produce behavior that is better described as **regulation of attention** than mere computation of attention.

In the language of the Good Regulator Theorem, the controller appears to embody a task-relevant model of its own attentional process. In this implementation, that model is not a separate symbolic object; it is the recurrent internal representation that combines previous attention, previous cue-conditioned observation, and detached feedback into the next allocation policy. What counts as the model of attention here is therefore the controller state that summarizes prior allocation and its consequences well enough to guide future allocation.

## 7. Limitations

This benchmark is intentionally minimal.

- The environment is synthetic and low-dimensional.
- Attention is soft rather than hard fixation.
- The wrong-cue and shuffle-cue probes are useful diagnostics, but not the only possible tests of cue dependence.
- Some ablations remain noisy and may require repeated runs or confidence intervals for publication-quality claims.

The benchmark should therefore be seen as a proof-of-principle environment, not a comprehensive account of attentional control in larger systems.

## 8. Future Work

The next natural extensions are:

1. add explicit task switching within episodes,
2. add explicit error-corrective reallocation tasks,
3. test hard attention variants,
4. measure robustness over multiple seeds,
5. refine qualitative visualizations into paper-ready figures.

## 9. Reproducibility

The implementation lives in this repository:

- benchmark/task generation: [src/attcon/data.py](/home/david/dev/attcon/src/attcon/data.py)
- models: [src/attcon/models.py](/home/david/dev/attcon/src/attcon/models.py)
- training: [src/attcon/train.py](/home/david/dev/attcon/src/attcon/train.py)
- evaluation: [src/attcon/eval.py](/home/david/dev/attcon/src/attcon/eval.py)
- default config: [configs/minimal.yaml](/home/david/dev/attcon/configs/minimal.yaml)
- latest report: [outputs/minimal/evaluation_report.json](/home/david/dev/attcon/outputs/minimal/evaluation_report.json)

Default commands:

```bash
.venv/bin/python -m attcon.train --config configs/minimal.yaml
.venv/bin/python -m attcon.eval --config configs/minimal.yaml --checkpoint outputs/minimal/experiment.pt
```

## 10. Conclusion

We introduced a minimal benchmark designed to separate attention from attention control. In this setting, a recurrent controller that conditions future allocation on previous attention outcomes outperforms both static and stronger non-recurrent alternatives, shows positive temporal reallocation, and is more strongly cue-sensitive. These results support the claim that even a small recurrent system can instantiate a minimal but defensible form of attention control.
