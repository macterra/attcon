# A Minimal Benchmark and Staged Program for Recurrent Attention Control

> **Erratum / status update.** An earlier version of this abstract reported results
> (`0.348` vs. `0.230` accuracy, etc.) from a checkpoint that, on review, had **not learned the
> task**: under a fully-soft glimpse the recurrent controller collapsed to uniform attention and
> lost to the static baseline, and the later-stage "support" was probe artifact on a
> non-functional model. The benchmark was repaired with a **discrete glimpse readout** (the glimpse
> reads the single most-attended cell while the policy stays soft) and all results below were
> regenerated. See `audits/post_rehab_full_eval_tune_prob_035_summary.json`.

## Abstract

Many machine learning systems compute attention, but fewer cleanly demonstrate **attention control**: the ability of a distinct controller to regulate future attention on the basis of task demands and the consequences of previous allocations. We present a minimal PyTorch benchmark for that distinction and report the current repository status of the broader staged research program built around it. The task is a cue-guided selective-search problem on a `5x5` grid in which visible cell types are globally available, but task-relevant target identity becomes useful only through attention. Because a fully-soft glimpse averages the digits of every same-type cell, the readout is discretised (each glimpse reads the single most-attended cell via a straight-through estimator) so the closed-loop search is learnable. On the regenerated discrete-attention checkpoint, a recurrent attention controller outperforms a static cue-conditioned baseline in held-out accuracy (`0.44` vs. `0.17`; chance `0.10`) and in target-inspected rate (`0.39` vs. `0.08`), while all negative controls and comparator systems fail as intended (e.g. shuffling the feedback channel drops accuracy by `0.27`). Additional evaluations make Stage 3 explicit-attention modeling robust across seeds and a checkpoint family, support engineered self-state tracking (Stage 4A) and capacity-audited structured reportability (Stage 6A), and establish seed-robust remembered-content recovery under explicit Stage 7 content-memory regularization. Branch C binding and Branch D counterfactual-access assays have strong bounded support across seeds and model families; perturbational dynamics replicate across checkpoints and an ungated RNN; and a structurally different temporal-relay assay passes across three relational-GRU seeds and one transformer seed. These remain engineering results where relational matching, shared state, or content-memory objectives are imposed. The executable Stage 8 audit therefore remains **not met** (`3` pass, `5` partial, `0` fail), and the work does not establish consciousness or minimal consciousness-like content.

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

The revised roadmap treats those as one branch of a larger consciousness-relevant methodology rather than as a sufficient ladder. The repository now implements comparator systems, cross-architecture and engineered cross-benchmark assays, unity/binding, counterfactual access, perturbational diagnostics, and higher-order/broadcast pilots. The remaining problem is not branch absence but independent convergence: the successful later assays still build in important relational, shared-state, or supervision assumptions.

## 5. Current Results

The committed consolidation artifact `audits/post_rehab_full_eval_tune_prob_035_summary.json`
summarizes the regenerated `tune_prob_035` evaluation. The executable report can be regenerated at
`outputs/tune_prob_035/evaluation_report.json`.

### 5.1 Main Comparison

On the current tuned discrete-attention checkpoint:

- Static baseline accuracy: `0.167`
- Recurrent controller accuracy: `0.442`
- Static baseline target-inspected rate: `0.078`
- Recurrent controller target-inspected rate: `0.387`

The recurrent controller therefore improves both final task performance and actual target inspection.

### 5.2 Closed-Loop Dynamics

- Static temporal reallocation: `0.000`
- Recurrent temporal reallocation: `0.129`
- Static target-attention gain: `0.000`
- Recurrent target-attention gain: `0.247`

These are the clearest Stage 2 signals in the current run. The recurrent controller does not merely learn a better static map; it changes its attention over time in a task-relevant way.

### 5.3 Explicit Attention Modeling

The repository now includes predictive-probe and intervention tests.

Predictive probe:

- controller-state test cross-entropy: `2.817`
- observation-only test cross-entropy: `3.181`
- controller top-1 advantage: `0.229`

Causal intervention:

- attention-change KL: `1.625`
- original-target attention drop: `0.233`
- alternate-target attention gain: `0.205`

Reduced-shaping condition:

- at `attention_target_weight = 0.25`, accuracy remains `0.337`
- at zero shaping, accuracy falls to `0.187`, approximately the static baseline

Together, these results now support the bounded Stage 3 claim: controller state is not merely generic recurrent memory, but carries structured information about future attention and causally influences later allocation under substantially reduced shaping. In the revised roadmap and evaluator, this stage counts as supported only when predictive, intervention, and reduced-shaping thresholds are all met together and the repeated-seed robustness gate also passes. The evaluator explicitly distinguishes a weaker single-run pass from the stricter robust pass, and it also extends that robustness check across the default checkpoint and a reduced-shaping checkpoint family. On the current `tune_prob_035` report, the default and `0.25` reduced-shaping families both pass; complete zero-shaping resilience remains outside the supported claim, so the shaping-objective alternative is weakened rather than fully eliminated.

### 5.4 Engineered Self-State Tracking

The recurrent controller now maintains an explicit inspected-cell state and a native report head over that state.

Current default results:

- native inspected-map cell accuracy: `0.990`
- observation-only inspected-map accuracy: `0.925`
- native target-inspected accuracy: `0.995`
- native target-inspected positive recall: `0.980`

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

- baseline switch-target gain: `0.0150`
- recurrent switch-target gain: `0.0976`
- baseline switch accuracy: `0.000`
- recurrent switch accuracy: `0.250`

This supports Stage 5 in the current benchmark: the recurrent controller can redirect attention under changed priorities better than the static baseline.

### 5.7 Structured Reportability of Internal Content

The current report probes test whether controller state supports explicit readouts of its own regulatory state.

Current default results:

- search-type accuracy advantage over a capacity-matched observation probe: `0.388`
- attended-cell accuracy advantage: `0.427`
- target-found accuracy advantage: `0.008`
- target-found positive-recall advantage: `0.029`

This supports a bounded Stage 6A-style claim: the same controller state that guides attention also supports structured reports about current search type, attended cell, target-found state, and unresolved regions. An empirical permuted-label noise floor backs the strong signals: the real controller-vs-observation accuracy advantages (`~0.38`, `~0.42`) are roughly 100x above the permuted-label 95th-percentile floor (`~0.004`, `~0.003`), with the permuted null centred at `~0`, so the advantage is significant rather than a probe-capacity artifact. The stronger Stage 6B-style target, reportability of uncertainty and allocation error, is now implemented as a distinct evaluation family. A matched-capacity 12-permutation audit finds that all four gated positive-recall advantages clear their empirical p95 floors. However, revisit-under-unresolved-search and allocation error retain slightly negative accuracy advantages (`-0.0015`, `-0.0020`) and fail their accuracy noise floors. The result is therefore bounded positive evidence, not Stage 6B support.

### 5.8 Natural-Language Reportability

The repository now includes a Stage 7 natural-language reporting harness. It can use an external API language model when quota is available, and it now also includes a local calibrated opaque-token reporter that runs without external services. The evaluation compares three reporting conditions:

- symbolic internal-state serialization as a weak baseline,
- tokenized internal-state reporting as the real Stage 7 target,
- observation-only reporting as the weaker external baseline.

A parallel VLM-based Stage 7 route now renders internal state as a fixed eight-level, label-free heatmap and compares it against observation-only and explicit-symbolic image controls. In a powered 72-request audit, the heatmap condition recovers none of the current, remembered, or full joint content (`0/8` in each family and slice). The explicit symbolic-image upper bound is perfect (`8/8` on all joint metrics and the entire report in every slice), validating the vision path while keeping that labeled dump out of the main claim.

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

This supports a bounded Stage 7 claim: faithful natural-language-shaped reportability from opaque tokenized internal state is now established for the local calibrated reporter. That reporter is a learned decoder, not an off-the-shelf language interface. Powered GPT-5 mini text and vision audits do not extend the result to off-the-shelf interfaces. The text route's latent-only current-content joint accuracy was `0/12`, `1/12`, and `0/12` across default, cue-switch, and intervened slices, with remembered/full content zero throughout. The VLM heatmap route was `0/8` on every joint-content family and slice despite a perfect symbolic-image upper bound.

A sharper caveat further narrows the local claim. On inspection, the local decoder reads the scored content fields (current and previous attended visible type, attended digit, and glimpse digit) from dedicated attended-content token bases that the renderer fills *directly from the model's attended content*, not from the calibration-fit translator's predictions (those occupy separate token bases the decoder ignores for these fields), and not from the opaque latent-bit tokens. The local content report is therefore a schema-aware structural round-trip of directly-encoded attended-content tokens — closer to the symbolic-dump baseline (relabelled with opaque IDs whose schema the decoder is told) than to "learn to attach labels to opaque latent state". Two consequences follow: the local reporter's advantage over the observation-only baseline comes from the tokenized stream *containing* the attended-content tokens rather than from a learned decode of opaque state, and the two named anti-memorization falsifiers — consistent token remapping and held-out cue/content combinations — are not meaningful against it (the former is invariant by construction for a schema-aware decoder; the latter does not bite directly-encoded fields). The genuine faithfulness / anti-memorization test therefore requires a decoder forced to recover content from the opaque latent-bit tokens alone, or the external API LLM / VLM path that is not told the schema. The narrower local result is still useful because it is runnable in CI and confirms the attended-content is recoverable from the token interface, but it should not be read as the stronger learned-decode claim.

The latent-only follow-up now separates two claims. On the original checkpoint it remains negative
to marginal. Under explicit content-memory regularization, however, remembered attended content
clears a permuted-label noise floor on all four training seeds and all tested slices/interfaces
(mean advantages `+0.41..+0.64` versus a p95 floor near `0.08`). The strict `content_only` joint
advantage is seed-fragile: full on two seeds, partial on one, and absent on one. This supports
seed-robust remembered-content reportability under an engineered memory objective, not spontaneous
faithful reporting and not the strict beyond-observation content claim.

### 5.9 Theory Branches and Stage 8 Audit

The wider program now has executable results rather than roadmap-only proposals:

- Branch C binding and Branch D counterfactual access have strong bounded support across seeds,
  surface variants, and GRU/transformer-style model families; their successful models use explicit
  relational selection or addressing.
- Perturbational complexity passes 25 perturbation seeds across four checkpoints and replicates on
  an ungated RNN controller. Six of seven base claims replicate from GRU to RNN; cue-switch
  adaptation is the exception.
- Branch E higher-order state and Branch F broadcast pass their engineering gates, including a
  three-seed Branch F audit, but direct access-sensitive objectives or an imposed broadcast
  bottleneck prevent them from counting as independent theory-family evidence.
- A same-content binding/access assay has seed-robust directional causal overlap against permuted
  and split-state controls, but only with an explicitly shared state. Corrected task-induced routing
  reaches at most `0.021` transfer, so emergent shared routing is unsupported.
- On the structurally different temporal-relay benchmark, a generic GRU memorizes training
  conjunctions and reaches only `0.020` held-out joint accuracy. A relational GRU passes all ten
  gates across three seeds (minimum order-destroyed advantage `0.934`), and a position-aware
  transformer passes all ten on one seed (`0.951` order-destroyed advantage). Relational matching
  and state sharing remain explicit.

Accordingly, `audits/stage8_convergence_current.json` records three pass, five partial, and zero
failed gates, with `stage8_supported = false`.

## 6. Interpretation

The main result is no longer just that recurrence is generally useful. The more specific repo-level result is that a small recurrent controller, given access to previous attention and its consequences, can support:

- closed-loop attention regulation,
- explicit attention-dynamics probes (Stage 3, robust),
- bounded engineered self-state tracking of attentional history (Stage 4A),
- weak task-induced (not supervision-induced) self-modeling of inspection history (Stage 4B emergence probe),
- flexible reallocation under changed priorities (Stage 5),
- structured internal report variables (Stage 6A),
- seed/checkpoint/RNN-replicated perturbational dynamics,
- strong bounded binding and counterfactual-access assays,
- engineered same-content and different-benchmark causal-transfer assays.

That is already stronger than the original attention-control benchmark framing.

The stronger claim should still be stated carefully. The current evidence supports:

- recurrence improves attention regulation,
- controller state predicts future allocation and can be probed and perturbed in ways consistent with a bounded Stage 3 explicit-attention-modeling claim,
- the model tracks inspected history explicitly through an engineered self-state scaffold (Stage 4A),
- structured internal contents are available for bounded report (Stage 6A, capacity-audited),
- changed-priority reallocation can be trained successfully (Stage 5),
- a non-reportability family in which recurrent-state perturbations produce rich-but-recoverable dynamics across checkpoints, perturbation seeds, and an ungated RNN,
- bounded binding and counterfactual-access results that survive seed and model-family checks,
- engineered causal transfer on the same-content and temporal-relay assays.

It does **not yet** support:

- the Stage 4B learned-self-model *causal feedback* claim (the path is disabled in the base config because it destabilises the discrete-glimpse base task); weak task-induced inspection-history decoding exists, but its causal direction has the wrong regulatory sign,
- a broad or fully stable Stage 6B-style claim of uncertainty and allocation-error reportability (positive controller-state recall advantage but not a clean accuracy-guarded capacity audit),
- faithful external API LLM or VLM reports from tokenized or minimally labeled visual internal state,
- spontaneous binding, counterfactual access, higher-order state, or broadcast without the successful assays' engineered relational/supervisory mechanisms,
- independent same-content convergence or benchmark transfer after forced sharing and relational matching are removed,
- Stage 8 multi-theory convergence (`stage8_supported` remains false),
- a strong claim that the controller’s internal state is already a sufficient consciousness-like schema in anything but a speculative sense.

### 6.1 Relation to Theory Families

The benchmark still admits a natural interpretation in the language of the Good Regulator Theorem and modeler-schema ideas. On that framing, the most plausible candidate for consciousness-like content is not the raw scene representation or the raw attention mask. It is the controller state that carries forward:

- previous attention allocation,
- previous cue-conditioned observation,
- previous task feedback,
- explicit inspected-state variables,
- and later report-oriented self-model variables.

What the current repository adds is a sharper boundary around that interpretation. A weak inspection-history self-model is task-induced (Stage 4B emergence probe), bounded structured reportability is supported for a limited set of internal variables (Stage 6A, capacity-audited), Stage 6B-style uncertainty reporting is bounded/provisional through positive controller-state recall advantages that do not clear the accuracy-guarded capacity audit, and Stage 7 is supported for a local calibrated opaque-token reporter. The broader uncertainty/allocation-error bundle and external LLM/VLM reportability are not yet settled.

For consciousness-relevant evidence, however, this is insufficient. The repository now has
multiple implemented theory branches, robust perturbational checks, cross-model replications, an
integrated same-content assay, and a structurally different temporal-relay benchmark. The limiting
issue is independence: the positive binding/access, higher-order, broadcast, and relay results rely
on explicit relational addressing, shared bottlenecks, or access-sensitive supervision. The current
benchmark work is therefore an active convergence program, not a direct argument for
consciousness-like content by itself.

### 6.2 Philosophical Scope

The bridge from bounded reportable regulatory state to consciousness-like content is conditional. This paper does not argue that Modeler Schema framing is preferable to higher-order thought theories, global workspace theories, integrated information theory, illusionism, or other competitors. A skeptical reader can accept every current engineering result and still conclude only that the benchmark contains sophisticated control with reportable internal state.

The intended claim is narrower: the benchmark is a methodology platform for developing tests that could become consciousness-relevant if they converged across multiple theory-derived families. Current evidence is not yet at that level. Most later-stage results are bounded support, not robust support across multiple seeds, checkpoint families, negative controls, capacity-matched baselines, comparator systems, architectures, and benchmarks.

## 7. Limitations

This system is still intentionally minimal.

- The environment is synthetic and low-dimensional.
- The policy is soft, but the glimpse is a straight-through discrete fixation; conclusions may depend on that estimator and shaping recipe.
- Some checkpoint-level metrics vary across training recipes.
- Later-branch support labels are mostly bounded engineering claims; Stage 3 and the perturbational family have stronger seed/checkpoint evidence, but no result closes the Stage 8 audit.
- The Stage 4B closeout applies to fresh checkpoints trained with the self-model feedback objective, not automatically to older checkpoints.
- Supervised self-modeling is weak evidence for consciousness; the stronger target is self-model emergence without direct self-model rewards.
- Capacity-matched baselines and empirical noise floors exist for the main reportability claims, but not every later branch has a complete end-to-end capacity audit.
- The base negative-control and comparator suite passes; equivalent suites still need to be repeated on every replicated system.
- Cross-architecture and temporal-relay replications are implemented but partial under the Stage 8 standard.
- Unity/binding, counterfactual-access, higher-order, broadcast, and perturbational branches are implemented; several remain engineering-only because their target mechanisms are imposed.
- External API LLM Stage 7 reporting is unsupported on the current powered v3 interface and should be treated as separate from the local calibrated reporter claim.
- VLM-based Stage 7 reporting is unsupported on the current powered heatmap interface; its symbolic-image upper-bound control passes perfectly.
- The sharper memory-focused probe makes the present Stage 7 result more informative, but also harder to pass.

So while the repository now supports much stronger claims than the original benchmark paper draft, it is still best understood as a disciplined toy program rather than a comprehensive model of attentional control or consciousness.

## 8. Immediate Next Work

The next highest-value experiments target the five partial Stage 8 gates:

1. remove forced state sharing from the integrated-content and temporal-relay assays while keeping task viability,
2. repeat the temporal-relay transformer across fresh seeds and rerun matched comparator/negative-control suites,
3. test naturally coupled tasks or resource constraints that can induce shared routing without train/evaluation scaling mismatches,
4. align access/report and non-reportability interventions on independently learned representations of the same content,
5. retain the unresolved base limitations: complete zero-shaping, policy-consistent Stage 4B emergence, full Stage 6B capacity support, and strict seed-robust Stage 7 `content_only` recovery.

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

The repository now goes well beyond a minimal Stage 2 benchmark. In the current discrete-attention setup, a recurrent controller outperforms a static baseline; Stage 3 is robust across seeds and a checkpoint family; Stage 4A and Stage 6A pass their bounded audits; remembered-content Stage 7 recovery replicates under explicit memory regularization; and binding, counterfactual-access, perturbational, higher-order, broadcast, integrated-content, and temporal-relay experiments make the convergence criteria executable. These results support a serious methodology-development program, but not consciousness-relevant convergence: the current Stage 8 audit remains `3` pass, `5` partial, `0` fail and explicitly unsupported.

The strongest remaining problem is independence: successful theory-branch assays must survive when relational matching, shared bottlenecks, and access-sensitive objectives are not built in. Multi-seed transformer relay replication, unforced shared routing, same-content alignment across independently learned families, and comparator reruns are the immediate tests. The local and memory-regularized Stage 7 results remain disciplined stepping stones rather than consciousness claims.
