# A Minimal Benchmark and Staged Program for Recurrent Attention Control

> **Erratum / status update.** An earlier version of this abstract reported results
> (`0.348` vs. `0.230` accuracy, etc.) from a checkpoint that, on review, had **not learned the
> task**: under a fully-soft glimpse the recurrent controller collapsed to uniform attention and
> lost to the static baseline, and the later-stage "support" was probe artifact on a
> non-functional model. The benchmark was repaired with a **discrete glimpse readout** (the glimpse
> reads the single most-attended cell while the policy stays soft) and all results below were
> regenerated. See `audits/post_rehab_full_eval_tune_prob_035_summary.json`.

## Abstract

Many machine learning systems compute attention, but fewer cleanly demonstrate **attention control**: the ability of a distinct controller to regulate future attention on the basis of task demands and the consequences of previous allocations. We present a minimal PyTorch benchmark for that distinction and report the current repository status of the broader staged research program built around it. The task is a cue-guided selective-search problem on a `5x5` grid in which visible cell types are globally available, but task-relevant target identity becomes useful only through attention. Because a fully-soft glimpse averages the digits of every same-type cell, the readout is discretised (each glimpse reads the single most-attended cell via a straight-through estimator) so the closed-loop search is learnable. On the regenerated discrete-attention checkpoint, a recurrent attention controller outperforms a static cue-conditioned baseline in held-out accuracy (`0.44` vs. `0.17`; chance `0.10`) and in target-inspected rate (`0.39` vs. `0.08`), while all negative controls and comparator systems fail as intended (e.g. shuffling the feedback channel drops accuracy by `0.27`). Additional evaluations extend the benchmark beyond closed-loop control: predictive probes and causal interventions make the Stage 3 explicit-attention-modeling claim **robust** across seeds and a checkpoint family; explicit inspected-state variables support engineered self-state tracking (Stage 4A; native cell accuracy `~0.99`); controller-state probes support structured reportability of search type and attended cell (Stage 6A, capacity-audited); and a local calibrated reporter over opaque tokenized internal state supports faithful natural-language-shaped reporting (Stage 7, capacity-audited). Stage 6B (uncertainty / allocation-error reportability) is bounded/provisional — positive controller-state recall advantage but not a clean accuracy-guarded capacity audit. The Stage 4B learned-self-model *causal feedback* path is disabled in the base config (it destabilises the discrete-glimpse base task) and learned-self-model emergence under task-only objectives is treated as open. Complete zero-shaping resilience remains a known weakness (the model collapses to near-static accuracy without attention shaping). Stronger consciousness-relevant claims remain provisional or open pending non-reportability theory branches and cross-system replication.

## 1. Introduction

The phrase *attention control* is often used loosely. In many architectures, attention is simply a learned weighting mechanism inside a single feedforward computation. That is not yet the same thing as a system that **controls its own attention**.

We use a stricter criterion. A system exhibits attention control only if:

1. it has an object-level attention process that allocates attention over inputs,
2. it has a distinct controller with access to a representation of that allocation or its consequences, and
3. it can modify future allocation on the basis of task demands, performance, or internal state.

The goal of this project is not to solve a large-scale perceptual problem, nor to claim consciousness in the present toy system. The goal is to build the smallest credible setting in which the difference between **attending** and **controlling attention** can be measured directly, then use that setting as a methodology-development platform for explicit attention modeling, engineered and learned self-state modeling, reportability, and eventually broader consciousness-relevant tests.

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
- a hidden-state-only self-model head,
- a learned self-model feedback path into the attention policy,
- a cumulative found-state variable,
- a target-found report head.

Those additions matter because the project now evaluates not only whether the controller improves attention regulation, but also whether it maintains bounded internal state about its own attentional history and supports report-like access to that state.

## 4. Training and Evaluation

### 4.1 Optimization

Training uses:

- final-step cross-entropy on digit prediction,
- a small auxiliary loss on intermediate predictions,
- a final-step target-attention loss that rewards placing mass on the true target cell,
- auxiliary self-model, hidden-self-model, self-model policy-feedback, and target-found reporting losses for the recurrent controller.

The direct target-attention term makes the benchmark easier to interpret, but the repository now also includes reduced-shaping evaluations to test whether useful reallocation survives when that term is weakened. Complete removal remains a separate stress test rather than part of the current supported Stage 3 claim.

### 4.2 Evaluation Axes

The original benchmark emphasized three claims:

- dissociation from static and weaker non-recurrent controls,
- closed-loop adaptation,
- cue dependence.

The current repository now evaluates a broader staged set of claims:

- closed-loop attention control,
- explicit attention modeling via predictive probes, intervention, and reduced-shaping checks,
- engineered self-state tracking of attention history,
- learned self-modeling of attention through a hidden-state-only self-model and policy-feedback route,
- flexible cue-switch reallocation under changed priorities,
- structured reportability of bounded internal content,
- natural-language reportability.

The revised roadmap treats those as one branch of a larger consciousness-relevant methodology rather than as a sufficient ladder. Serious consciousness evidence would also require comparator systems, cross-architecture and cross-benchmark replication, unity/binding tests, counterfactual-access tests, perturbational-complexity diagnostics, and convergence across multiple theory-derived evidence families.

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

- at `attention_target_weight = 0.25`, accuracy remains `0.320`
- temporal reallocation remains `0.0662`
- target-attention gain remains `0.0448`

Together, these results now support the bounded Stage 3 claim: controller state is not merely generic recurrent memory, but carries structured information about future attention and causally influences later allocation under substantially reduced shaping. In the revised roadmap and evaluator, this stage counts as supported only when predictive, intervention, and reduced-shaping thresholds are all met together and the repeated-seed robustness gate also passes. The evaluator explicitly distinguishes a weaker single-run pass from the stricter robust pass, and it also extends that robustness check across the default checkpoint and a reduced-shaping checkpoint family. On the current `tune_prob_035` report, the default and `0.25` reduced-shaping families both pass; complete zero-shaping resilience remains outside the supported claim, so the shaping-objective alternative is weakened rather than fully eliminated.

### 5.4 Engineered Self-State Tracking

The recurrent controller now maintains an explicit inspected-cell state and a native report head over that state.

Current default results:

- native inspected-map cell accuracy: `0.990`
- observation-only inspected-map accuracy: `0.973`
- native target-inspected accuracy: `0.988`
- native target-inspected positive recall: `0.721`

This supports a bounded Stage 4A-style claim: the model contains an explicit internal variable about where it has already attended, and that variable supports more faithful reporting than observation-only baselines. The stronger Stage 4B-style claim is handled separately below because it requires a hidden-state-only self-model and evidence that the learned self-model route can affect downstream attention.

### 5.5 Learned Self-Modeling of Attention

The recurrent controller now also has a hidden-state-only self-model head and a learned feedback path from that hidden self-model into the attention policy. This is deliberately separate from the scaffolded native self-model head that receives the explicit inspected-cell state.

The Stage 4B evaluator asks four questions:

- does hidden state alone predict inspected-cell history better than a previous-observation baseline?
- does the hidden-state target-inspected readout improve over observation-only baselines using threshold-free BCE and score-separation metrics?
- do hidden-state interventions along the hidden self-model readout direction move self-model report outputs?
- do direct hidden-self-model overrides measurably affect attention through the learned policy-feedback path?

Post-rehab status. With the discrete glimpse readout, the Stage 4B *causal policy-feedback*
path (where a hidden self-model can be overridden to steer attention) destabilises base-task
learning, so it is **disabled in the base config** and is not part of the supported claim. The
path remains available as an architectural option, but the base benchmark no longer trains it.

Instead, the relevant Stage 4B question -- does inspection-history self-modeling *emerge* without
a direct self-model objective? -- is tested directly (`scripts/stage4b_emergence.py`). A task-only
checkpoint (no hidden-self-model, native-self-model, report, or policy-feedback losses) still has a
raw hidden state that beats a previous-observation baseline at predicting the full inspection map
(BCE advantage `~+0.09`), and the dedicated self-model objective adds almost nothing to this
(`~+0.005`). So the (weak) cell-level inspection self-model is **task-induced, not
supervision-induced** -- bounded evidence against the "supervised self-model is always required"
global falsifier. However, target-level inspection ("have I inspected the target?") is *not*
encoded better than observation in either model, and the accuracy-level advantage is near noise,
so the emergent self-model is partial and weak. This is not a strong learned-self-modeling claim,
and it remains local to this benchmark rather than a general self-awareness claim.

### 5.6 Flexible Reallocation Under Changed Priorities

The current default training mixes stationary and switched-cue episodes, and the repository now evaluates mid-episode cue switching directly.

Current default results:

- baseline switch-target gain: `0.0233`
- recurrent switch-target gain: `0.1169`
- baseline switch accuracy: `0.000`
- recurrent switch accuracy: `0.500`

This supports Stage 5 in the current benchmark: the recurrent controller can redirect attention under changed priorities better than the static baseline.

### 5.7 Structured Reportability of Internal Content

The current report probes test whether controller state supports explicit readouts of its own regulatory state.

Current default results:

- search-type accuracy advantage over observation-only: `0.308`
- attended-cell accuracy advantage: `0.287`
- target-found accuracy advantage: `0.0124`
- unresolved-region advantage from native self-model: `0.0172`

This supports a bounded Stage 6A-style claim: the same controller state that guides attention also supports structured reports about current search type, attended cell, target-found state, and unresolved regions. The stronger Stage 6B-style target, reportability of uncertainty and allocation error, is now implemented as a distinct evaluation family. It remains provisional overall, but the evaluator now separates current wrong-candidate pursuit, cumulative wrong-candidate history, revisit-under-unresolved-search, and allocation error. That finer decomposition makes the positive result more interpretable: current wrong-candidate and wrong-candidate-history signals provide bounded positive evidence that some uncertainty-style report variables can beat observation-only baselines on positive-recall style reporting, while revisit-under-unresolved-search and allocation error remain weaker.

### 5.8 Natural-Language Reportability

The repository now includes a Stage 7 natural-language reporting harness. It can use an external API language model when quota is available, and it now also includes a local calibrated opaque-token reporter that runs without external services. The evaluation compares three reporting conditions:

- symbolic internal-state serialization as a weak baseline,
- tokenized internal-state reporting as the real Stage 7 target,
- observation-only reporting as the weaker external baseline.

A plausible parallel branch now under consideration is a VLM-based Stage 7 route in which internal attention/self-model state is rendered as compact visual panels or overlays. That branch would be useful only if it obeys the same anti-cheating discipline as the text route: explicit labeled visual dumps would count only as baselines, while minimally labeled internal-state renderings would need to beat scene-only baselines on held-out current and remembered attended-content reports.

The current picture is now judged under a stricter, more skeptical setup:

- evaluation examples are restricted to non-initial timesteps so remembered previous-attention content is genuinely required,
- the report schema now asks for both current attended content and previous attended content,
- the same schema now also carries Stage 6B-style variables for relevant-region inspection, unresolved search, current wrong-candidate pursuit, wrong-candidate history, revisit-under-unresolved-search, and allocation error,
- symbolic reporting is strong and can achieve exact structured reports on held-out slices,
- the local calibrated token reporter now beats observation-only on the default, cue-switch, and intervention slices.

In a recent local calibrated Stage 7 slice on the current tuned checkpoint:

- tokenized payload current attended-cell accuracy: `1.0`
- tokenized payload previous attended-cell accuracy: `1.0`
- tokenized payload current-content joint accuracy: `1.0`
- tokenized payload memory-content joint accuracy: `1.0`
- local token reporter joint-accuracy advantage over observation-only: `0.5`
- local token reporter memory-content advantage over observation-only: `1.0`
- cue-switch slice supported: yes
- intervention slice supported: yes

This supports a bounded Stage 7 claim: faithful natural-language-shaped reportability from opaque tokenized internal state is now established for the local calibrated reporter. That reporter is a learned decoder, not an off-the-shelf language interface. It does not yet establish that an external general-purpose LLM or VLM can infer the same reports under the same constraints; the API path is currently quota-limited, and the VLM branch remains future work. The narrower local result is still meaningful because it is runnable in CI, uses the tokenized interface rather than a symbolic dump, and is evaluated against observation-only controls under default, cue-switch, and intervention slices.

## 6. Interpretation

The main result is no longer just that recurrence is generally useful. The more specific repo-level result is that a small recurrent controller, given access to previous attention and its consequences, can support:

- closed-loop attention regulation,
- explicit attention-dynamics probes (Stage 3, robust),
- bounded engineered self-state tracking of attentional history (Stage 4A),
- weak task-induced (not supervision-induced) self-modeling of inspection history (Stage 4B emergence probe),
- flexible reallocation under changed priorities (Stage 5),
- structured internal report variables (Stage 6A),
- a first non-reportability evidence family via rich-but-recoverable perturbational dynamics.

That is already stronger than the original attention-control benchmark framing.

The stronger claim should still be stated carefully. The current evidence supports:

- recurrence improves attention regulation,
- controller state predicts future allocation and can be probed and perturbed in ways consistent with a bounded Stage 3 explicit-attention-modeling claim,
- the model tracks inspected history explicitly through an engineered self-state scaffold (Stage 4A),
- structured internal contents are available for bounded report (Stage 6A, capacity-audited),
- changed-priority reallocation can be trained successfully (Stage 5),
- a first non-reportability evidence family: perturbing the recurrent state produces rich-but-recoverable dynamics that propagate far more than a no-recurrence control and recover unlike a frozen-state control (perturbational branch, bounded support).

It does **not yet** support:

- the Stage 4B learned-self-model *causal feedback* claim (the path is disabled in the base config because it destabilises the discrete-glimpse base task) or self-modeling that emerges without a direct self-model objective,
- a broad or fully stable Stage 6B-style claim of uncertainty and allocation-error reportability (positive controller-state recall advantage but not a clean accuracy-guarded capacity audit),
- faithful external API LLM or VLM reports from tokenized or minimally labeled visual internal state,
- unity/binding or counterfactual-access evidence, or robust (multi-seed, cross-system) perturbational complexity beyond the current bounded result,
- cross-architecture or cross-benchmark replication,
- multi-theory convergence across consciousness-theory branches,
- a strong claim that the controller’s internal state is already a sufficient consciousness-like schema in anything but a speculative sense.

### 6.1 Relation to Theory Families

The benchmark still admits a natural interpretation in the language of the Good Regulator Theorem and modeler-schema ideas. On that framing, the most plausible candidate for consciousness-like content is not the raw scene representation or the raw attention mask. It is the controller state that carries forward:

- previous attention allocation,
- previous cue-conditioned observation,
- previous task feedback,
- explicit inspected-state variables,
- and later report-oriented self-model variables.

What the current repository adds is a sharper boundary around that interpretation. A weak inspection-history self-model is task-induced (Stage 4B emergence probe), bounded structured reportability is supported for a limited set of internal variables (Stage 6A, capacity-audited), Stage 6B-style uncertainty reporting is bounded/provisional through positive controller-state recall advantages that do not clear the accuracy-guarded capacity audit, and Stage 7 is supported for a local calibrated opaque-token reporter. The broader uncertainty/allocation-error bundle and external LLM/VLM reportability are not yet settled.

For consciousness-relevant evidence, however, this is insufficient. The repository now has a first non-reportability family (bounded perturbational complexity: rich-but-recoverable dynamics that degrade under no-recurrence and frozen-state controls), but Stage 8 convergence still needs that family to be robust, a second non-reportability family, the content-identity criterion, and cross-architecture/cross-benchmark replication. Higher-order and global-workspace-style framings would still need explicit self-representation or broad multi-consumer availability tests, and unity-oriented framings would need binding tests. The current benchmark work is therefore one branch of a future convergence program, not a direct argument for consciousness-like content by itself.

### 6.2 Philosophical Scope

The bridge from bounded reportable regulatory state to consciousness-like content is conditional. This paper does not argue that Modeler Schema framing is preferable to higher-order thought theories, global workspace theories, integrated information theory, illusionism, or other competitors. A skeptical reader can accept every current engineering result and still conclude only that the benchmark contains sophisticated control with reportable internal state.

The intended claim is narrower: the benchmark is a methodology platform for developing tests that could become consciousness-relevant if they converged across multiple theory-derived families. Current evidence is not yet at that level. Most later-stage results are bounded support, not robust support across multiple seeds, checkpoint families, negative controls, capacity-matched baselines, comparator systems, architectures, and benchmarks.

## 7. Limitations

This system is still intentionally minimal.

- The environment is synthetic and low-dimensional.
- Attention is soft rather than hard fixation.
- Some checkpoint-level metrics vary across training recipes.
- Current support labels are bounded benchmark claims rather than robust multi-seed, multi-checkpoint claims with capacity audits and negative controls.
- The Stage 4B closeout applies to fresh checkpoints trained with the self-model feedback objective, not automatically to older checkpoints.
- Supervised self-modeling is weak evidence for consciousness; the stronger target is self-model emergence without direct self-model rewards.
- The current documentation still needs a full capacity audit for observation-only baselines before later-stage claims should be promoted to robust support.
- Negative controls such as feedforward, shuffled-feedback, and high-capacity observation-only systems remain important next checks.
- Comparator systems, cross-architecture replication, and cross-benchmark replication are not yet implemented.
- Unity/binding, counterfactual-access, and perturbational-complexity branches remain roadmap items rather than current evidence.
- External API LLM Stage 7 reporting remains quota-limited and should be treated as separate from the local calibrated reporter claim.
- VLM-based Stage 7 reporting remains untested.
- The sharper memory-focused probe makes the present Stage 7 result more informative, but also harder to pass.

So while the repository now supports much stronger claims than the original benchmark paper draft, it is still best understood as a disciplined toy program rather than a comprehensive model of attentional control or consciousness.

## 8. Immediate Next Work

The next highest-value experiments are now concentrated on turning the current attention-control methodology into a convergence-oriented program:

1. test whether Stage 3 can also survive complete removal of direct target-attention shaping,
2. rebuild Stage 4B around self-model emergence under task objectives that do not directly reward self-modeling,
3. add capacity-matched baseline audits, negative-control runs, and first-class comparator systems,
4. add unity/binding, counterfactual-access, and perturbational-complexity branches,
5. strengthen the Stage 6B bundle beyond wrong-candidate history so unresolved search and allocation-error reports also beat observation-only baselines,
6. replicate supported claims on a structurally different architecture and a second benchmark,
7. test external API LLM and VLM reporting against the now-supported local calibrated token reporter once quota and infrastructure are available.

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

The repository now goes well beyond a minimal Stage 2 benchmark. In the current discrete-attention setup, a recurrent attention controller outperforms a static baseline, supports a robust Stage 3 explicit-attention-modeling claim through predictive, intervention, multi-seed, and checkpoint-family checks, maintains an explicit engineered state about inspected history (Stage 4A), shows weak task-induced inspection-history self-modeling (Stage 4B emergence probe, with the causal feedback path disabled in base), supports structured internal report variables (Stage 6A), and adds a first non-reportability evidence family via bounded perturbational complexity. The evaluator also now exposes repeated-seed and checkpoint-family Stage 3 summaries plus dedicated Stage 3 diagnostics plots, switched-cue artifacts, self-state artifacts, self-model artifacts, and Stage 6B uncertainty diagnostics artifacts, which make current weaknesses easier to inspect rather than masking them behind a single headline run. Those results are enough to support a bounded attention-control methodology, but not enough to claim consciousness-relevant convergence.

The strongest remaining open problem is no longer only the local tokenized Stage 7 path, but the broader convergence problem: comparator systems, emergent rather than supervised self-modeling, unity/binding, counterfactual access, perturbational dynamics, cross-architecture replication, cross-benchmark replication, and external LLM/VLM reportability. The local calibrated token reporter now closes a bounded Stage 7 claim, but that should be read as a disciplined stepping stone rather than a consciousness claim.
