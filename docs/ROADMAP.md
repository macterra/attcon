# Roadmap Toward Consciousness-Relevant Evidence

This document separates the current benchmark result from the larger research goal.

The project already supports a meaningful claim about **closed-loop attention control**. The longer-term goal is more ambitious: to build an experimental methodology that could eventually count as evidence for **minimal consciousness-like processing** in artificial systems.

Because that goal is easy to overstate, this roadmap is intentionally conservative. It is meant to discipline the argument, not accelerate it.

The current `5x5` attention-control benchmark should be treated as a methodology-development system, not as a system that is already large or rich enough to carry a serious consciousness claim. Even if every current benchmark-local stage passed robustly, the honest interpretation would be: this toy setting demonstrates an experimental approach that might scale to consciousness-relevant evidence in larger and more varied systems.

## Positive Operational Target

By `minimal consciousness-like content`, this roadmap means: integrated internal content that is behaviorally accessible for flexible report and action, counterfactually available beyond the currently attended item, causally tied to the system's own regulatory state, and supported by convergent tests derived from multiple consciousness theories across stable contexts.

This is an operational target, not a metaphysical definition. It is meant to pick out something more specific than sophisticated control and less ambitious than full subjective experience. A result becomes consciousness-relevant only if it survives theory-diverse tests, comparator systems, cross-architecture replication, and cross-benchmark replication.

## What This Roadmap Is Not Claiming

This roadmap does not aim to establish:

- human-like consciousness
- subjective experience in a strong philosophical sense
- proof rather than interpretation
- open-ended self-awareness outside this benchmark
- consciousness in the current toy system

The narrower near-term goal is to accumulate bounded evidence for increasingly rich attentional regulation, self-modeling, reportability, unity, counterfactual access, higher-order state representation, broadcast dynamics, and perturbation response signatures. The larger goal requires convergence across those families rather than a single-theory bridge.

## Claim Status Conventions

Each stage should be read using three different labels:

- `implemented`: the experiment or mechanism exists in the repo
- `positive evidence`: current results point in the expected direction
- `supported`: the stage meets a bounded claim threshold strong enough to use as an input to the next experiments

A stage should not count as supported merely because a favorable example exists. The stronger standard is reproducible evidence that survives fair baselines, leakage controls, and alternative explanations.

In practice this roadmap now distinguishes two levels of support:

- `bounded support`: a clearly scoped benchmark claim passes the implemented thresholds, often on the current tuned checkpoint or a specific fresh-training recipe
- `robust support`: the same claim survives multiple seeds, checkpoint families, explicit stress tests, comparator systems, and where relevant cross-architecture and cross-benchmark replication

Some sections also include a `consciousness-evidence status` note. That is not a third support tier. It is an interpretive warning about whether a technically supported benchmark result should count as an input to Stage 8. A result can have bounded engineering support while still having no consciousness-evidence status if it depends on direct supervision, symbolic scaffolding, or benchmark-specific shortcuts.

Only robust support should be treated as strong evidence for Stage 8. No stage in the current repo yet meets that robust-support bar, because the capacity audits and negative-control runs below have not been completed. Current `supported` labels should therefore be read as bounded engineering/evaluation milestones rather than settled philosophical premises.

Several items in this roadmap do not yet meet bounded support. Stage 6B has only provisional positive evidence, the external API LLM and VLM versions of Stage 7 are implemented or planned but not supported, and complete zero-shaping resilience for Stage 3 is not established.

## Self-Model Vocabulary

This roadmap uses `self-model` in a deliberately narrow operational sense. A self-model is a representation whose target is some internal regulatory state of the system, such as prior attention allocation, inspected-history, uncertainty about access, or allocation error.

This is weaker than many philosophical uses of self-modeling. It does not by itself imply phenomenal selfhood, introspective understanding, or a representation that the system conceptualizes as "me." The stages below test increasingly strong versions of this weaker operational notion:

- Stage 4A: an engineered internal variable tracks prior attention
- Stage 4B: a hidden learned representation tracks prior attention and can affect attention policy
- Stages 6A, 6B, and 7: parts of that internal state become behaviorally reportable

When the roadmap says `learned self-modeling`, it means this bounded regulatory sense unless explicitly stated otherwise.

## Evaluation Discipline

The roadmap only works if later stages are protected against easy forms of self-deception.

Across Stages 3 through 7, the following standards should apply:

- each reporter or probe must have a clearly specified input interface
- observation-only baselines must be matched in capacity as closely as possible
- train and evaluation splits must prevent simple memorization of scenes, prompts, or report templates
- benchmark-specific symbolic dumps count only as weak baselines, not as the main claim
- the strongest tests should place observation and internal state under tension so that faithfulness can be distinguished from post-hoc guessing
- robust support claims should survive multiple seeds and should be reported with explicit effect sizes or margins over baseline

The current repo does not yet include a complete capacity audit for every observation-only baseline. That is a known weakness. Before any later-stage claim is promoted from bounded support to robust support, the report should include either parameter-count matching or an explicit argument that the baseline is not capacity-limited in the relevant comparison.

To move a stage from bounded to robust support, the repo should show all of the following for that stage:

- multiple seeds and, where training is involved, multiple checkpoint families
- capacity-matched observation-only and scene-only baselines
- negative-control systems that fail for the intended reason
- enough of the stage's listed alternative-explanation tests to make the intended interpretation stronger than the live alternatives
- replication on at least one structurally different architecture for claims meant to support Stage 8
- replication on at least one benchmark with different surface task structure for claims meant to support Stage 8

For newly proposed branches that are not implemented yet, the first target is bounded support under their own claim thresholds. The bounded-to-robust ladder above applies only after a branch has a concrete implementation, baseline suite, and initial positive result; it should not be read as a requirement to jump directly from proposal to robust Stage 8 evidence.

## Negative Controls

The staged criteria should fail in systems that merely solve the task without using reportable regulatory state. For a consciousness-evidence roadmap, comparator systems are first-class evidence rather than only failure-mode protection. If the criteria cannot distinguish the target system from systems that should not be consciousness candidates, the criteria are not measuring the intended target.

Useful comparator systems include:

- a static or feedforward model with no recurrent access to prior allocation
- a recurrent model whose feedback channels are frozen, shuffled, or zeroed
- a matched-capacity non-recurrent transformer controller
- a large language model with no embodied attention-control loop, tested only through prompt/report behavior
- an intentionally trivial regulator, such as a thermostat-like controller, at the absurd-comparator end
- a high-capacity observation-only decoder that sees the current scene/glimpse but not controller state
- a symbolic dump condition treated only as an upper-bound baseline, not as evidence for the main claim

If any of these controls pass a later-stage claim under the same thresholds, the stage should be downgraded or the claim should be rewritten.

## Global Falsifier

The whole consciousness-evidence program should be downgraded, even if several bounded stages pass, if any of the following occur. These failures are not symmetric:

- fatal for the current criteria: comparator systems that should fail pass the same convergence criteria, showing that the criteria do not discriminate intended candidates from deliberately inappropriate ones
- fatal for the recurrent-control interpretation: a sufficiently large feedforward or observation-only system satisfies the roadmap without recurrent regulatory state, showing that the claimed mechanism is unnecessary
- fatal for any Stage 8 claim in this program: results fail to replicate across structurally different architectures or benchmarks
- degrading but recoverable: the only positive theory branch is Modeler Schema/self-modeling, with no support from access, binding, broadcast, or perturbational families
- degrading but recoverable: supervised self-model objectives are required for every self-modeling result, with no emergence under task objectives that do not directly reward self-modeling

Per-stage falsifiers are not enough. A project aimed at consciousness-relevant evidence needs this global falsifier because otherwise failures can always be absorbed by rescoping individual stages.

## Dependency Structure

Stages 1 through 3 are the attention-control foundation. After that, the roadmap branches into several theory-derived evidence families and then rejoins at convergence:

- Stage 4A asks whether the system contains engineered self-state tracking
- Stage 4B asks whether it learns a model of its own attention as an object of regulation
- Stage 5 asks whether attention control remains flexible when priorities change
- Stage 6A asks whether internal state is behaviorally reportable in structured form
- Stage 6B asks whether uncertainty and allocation error are also reportable
- Branch C asks whether multi-feature contents are bound into unified accessible representations
- Branch D asks whether unattended content remains counterfactually available for report or action on demand
- Branch E asks whether the system has higher-order representations of its own first-order content-bearing states
- Branch F asks whether selected contents become broadly available through broadcast-like or ignition-like dynamics
- a perturbational branch asks whether state perturbations produce rich but recoverable dynamics rather than trivial collapse or rigid reset

Stage 4A is foundational for instrumentation and for the engineered self-state branch, but it is not load-bearing for the Stage 8 convergence argument. For that argument, Stage 4A is a scaffold and diagnostic aid; the learned-regulatory-model burden sits on Stage 4B.

Stage 7 is one reportability endpoint, not the apex of the roadmap. Stage 8 should only be discussed after multiple theory-derived branches are in place and after comparator, cross-architecture, and cross-benchmark tests have been run.

## Stage 1: Attention

Baseline question:

Can the system compute attention over inputs at all?

This stage is satisfied by ordinary attention mechanisms. It is not yet evidence of attention control or consciousness.

What counts:

- attention distributions over inputs
- successful task performance using attended information

What does not count:

- any claim of self-direction
- any claim of self-modeling

## Stage 2: Attention Control

Question:

Can the system regulate future attention using internal state derived from prior allocation and its consequences?

This is the stage the current benchmark is designed to test.

Current evidence in this repo:

- the recurrent controller outperforms a static baseline
- freeze-recurrence and feedforward-summary ablations degrade performance
- temporal reallocation is present in the recurrent model
- cue sensitivity is stronger in the recurrent model

Current assessment:

- implemented: yes
- positive evidence: yes
- supported: yes, for closed-loop attentional regulation within this benchmark

Interpretation:

This is evidence of **closed-loop attentional regulation**. It is stronger than ordinary attention, but it is not yet evidence for an explicit model of attention.

## Stage 3: Explicit Attention Modeling

Question:

Does the controller state explicitly model attentional dynamics, rather than merely functioning as useful recurrent memory?

This is the most important next step after Stage 2.

Current status in this repo:

- a predictive probe is implemented
- on the default run, controller state predicts the next attention map better than a baseline built from current observation alone
- a causal intervention test is implemented
- on the default run, perturbing controller state causes a measurable shift in the next attention map away from the original cue target and toward an alternate cue target
- reduced-shaping retraining runs are implemented
- on the current tuned checkpoint, useful reallocation persists when the direct target-attention supervision term is reduced to `0.25`
- the complete zero-shaping condition remains informative but is no longer part of the supported Stage 3 closeout claim

Current assessment:

- implemented: yes
- positive evidence: yes
- supported: bounded support, with a known weakness, for the default checkpoint plus the `0.25` reduced-shaping checkpoint family under the current thresholds

Required experiments:

1. Predictive probe

Train a readout from controller state at time `t` to predict:

- the next attention map
- target-attention gain at `t+1`
- or the effect of a reallocation

2. Intervention test

Perturb controller state while holding scene and cue fixed.

Best version:

- identify a subspace associated with previous attention or previous observation
- intervene on that subspace only
- measure systematic changes in the next attention allocation

3. Reduced shaping-loss condition

Remove or reduce the direct target-attention supervision term.

Claim threshold:

This stage should count as supported only when all three conditions hold:

- controller-state probes beat fair observation-only baselines by a predeclared margin
- interventions produce directional changes in attention with consistent effect size across seeds
- useful reallocation persists under reduced shaping strongly enough to rule out the claim that the controller is merely optimizing an engineered shaping objective

What remains missing:

- a stronger complete-removal condition where the direct target-attention term is set to zero
- stronger evidence that the signal reflects structured attentional dynamics rather than generic recurrent state

Interpretation:

The current results close the bounded Stage 3 claim. The repo now includes a repeated-seed Stage 3 summary so predictive and intervention support can be checked across multiple evaluation seeds, and the top-level evidence summary distinguishes a weaker single-run pass from the stricter robust pass. It also includes a checkpoint-family Stage 3 summary spanning the default checkpoint and the supported reduced-shaping variant, with explicit bottleneck metrics and worst-seed reporting. On the current `tune_prob_035` evaluation, predictive, intervention, reduced-shaping, multi-seed, and checkpoint-family checks all pass.

Known weakness:

The reduced-shaping closeout currently uses `attention_target_weight = 0.25`, not complete removal of target-attention shaping. This weakens the alternative-explanation argument: a critic can still ask whether one quarter of the original shaping signal is enough to drive the result. The bounded Stage 3 claim is therefore "explicit attention modeling under substantially reduced shaping," not "explicit attention modeling with no engineered attention-shaping signal." Complete zero-shaping resilience remains a separate stress test.

Falsification criterion:

Stage 3 should be downgraded if controller-state probes fail to beat observation-only probes under the predeclared thresholds, if causal state interventions do not change later attention in the predicted direction, or if the effect disappears across the reduced-shaping checkpoint family.

Alternative-explanation falsifier:

Stage 3 should also be downgraded even if those thresholds pass if the same predictive and intervention results can be recovered from generic recurrent-memory surrogates that do not encode attention as an object, such as previous-glimpse summaries plus cue, shuffled attention-history labels, or low-rank history features matched for capacity. That result would support "useful recurrent state correlated with attention," not an explicit attention model.

## Branch A, Stage 4A: Engineered Self-State Tracking

Question:

Does the system contain explicit internal state that tracks its own prior attentional allocation?

This stage is deliberately weaker than learned self-modeling. It captures benchmark-engineered self-monitoring rather than a stronger emergent claim.

Needed capabilities:

- explicit internal variables that track what has been attended
- explicit representation of what remains uninspected
- reportability of that tracked state better than observation-only baselines

Current status in this repo:

- the recurrent controller maintains an explicit inspected-cell state
- that state tracks which cells have already been fixated and which remain uninspected
- the default evaluation includes a native self-state test over the full inspected/uninspected cell map
- on the default run, the native self-state beats an observation-only probe on both full-map reporting and the binary question of whether the target has already been inspected

Current assessment:

- implemented: yes
- positive evidence: yes
- supported: bounded support, as engineered self-state tracking within this benchmark

Interpretation:

This is meaningful progress, but it should not be conflated with a stronger claim that the model has learned its own attentional self-model. The right reading is that the system now contains an explicit internal variable about prior allocation and can behaviorally report it. For the Stage 8 convergence argument, Stage 4A is a scaffold and diagnostic aid rather than a load-bearing learned self-model claim.

## Branch A, Stage 4B: Learned Self-Modeling of Attention

Question:

Does the system model not just the world, but its own attentional state as an object of regulation, without relying only on a hand-authored scaffold?

This is the stronger version of the self-modeling claim.

Needed capabilities:

- self-referential attentional state that is useful for regulation rather than only for decoding
- uncertainty tied to access rather than just output confidence
- error-correction specifically about allocation mistakes
- evidence that the relevant representation is learned and causally used, not merely exposed by design

Good experiments:

- the model must state whether it has already inspected the relevant region, and that answer must depend on internal attention history rather than final task success alone
- hidden-state-only probes should predict inspected-cell state better than previous-observation baselines
- interventions on the learned self-model representation should alter both report and downstream reallocation
- the same representation should generalize to changed task conditions rather than only to the original benchmark regime

Current status in this repo:

- the evaluator now includes `learned_self_modeling`
- the recurrent controller now has a hidden-state-only self-model head, separate from the scaffolded native self-model head
- the policy has a learned self-model feedback path, initialized to zero for checkpoint compatibility, so new training can make hidden self-model content causally available to attention selection
- it trains hidden-state-only probes for the full inspected-cell map and target-inspected variable
- it compares those probes with previous-observation-only baselines
- it perturbs hidden state along hidden-self-model readout directions and measures the effect on native self-model output and target attention
- it can directly override hidden self-model content at a timestep and measure whether the learned policy feedback path changes target attention
- the top-level evidence summary surfaces hidden-state probe advantages and bidirectional intervention gaps under `learned_self_modeling_of_attention`

Current assessment:

- implemented: yes, for the hidden-self-model feedback path (available but disabled in the base config)
- positive evidence: not on the current base checkpoint
- supported: no. The base config disables the Stage 4B causal policy-feedback path because, with the
  discrete glimpse readout, that path destabilises base-task learning. `learned_self_modeling` is
  therefore not supported on the base checkpoint. The architecture can still learn a hidden self-model
  under the dedicated objective, but that is now studied as its own experiment rather than baked into
  the base benchmark.
- consciousness-evidence status: not supported until comparable self-modeling appears without a direct self-model objective and survives comparator tests

Interpretation:

The bounded Stage 4B claim is now closed for the hidden-self-model feedback route: the controller learns a hidden-state-only inspected-cell model, that representation beats a previous-observation baseline on held-out inspected-state prediction, hidden-state interventions move self-model report output, and direct hidden-self-model overrides measurably affect attention through the learned policy feedback path. On a fresh closeout probe with the Stage 4B feedback objective, the learned-self-model metrics passed with positive hidden cell BCE advantage, positive target BCE/separation advantage, a bidirectional self-model intervention gap, and nonzero policy-feedback causal effect. This remains a benchmark-local claim rather than a broad self-awareness claim, and older checkpoints trained before the feedback objective should not be counted as Stage 4B-supported.

Engineered-objective caveat:

This result should not be presented as emergent self-modeling discovered in an otherwise unmodified controller. The Stage 4B objective explicitly trains a hidden self-model and a feedback path through which that model can affect policy. The current claim is therefore narrower: the architecture is capable of learning a hidden regulatory model under direct training pressure, and the learned model can be causally used by attention rather than merely decoded after the fact. That existence proof is non-trivial only to the extent that the alternative-explanation falsifier below passes; otherwise it could still collapse into an auxiliary-head shortcut.

For consciousness-relevant evidence, supervised self-modeling is weak evidence at best. The stronger Stage 4B target should be rebuilt as: does self-modeling emerge under task objectives that do not directly reward self-modeling? If it does not, that is a substantive negative result rather than a hidden success. A supervised self-model feedback route remains useful as instrumentation and as a capability probe, but it should not be treated as direct consciousness evidence.

Falsification criterion:

Stage 4B should be downgraded if the hidden self-model readout does not beat a previous-observation baseline, if hidden-state interventions fail to move self-model outputs, if hidden-self-model overrides do not affect attention through the learned feedback path, or if these effects disappear across fresh seeds/checkpoint families.

Alternative-explanation falsifier:

Stage 4B should also be downgraded if the learned route is shown to be an auxiliary-head shortcut rather than a regulatory self-model: for example, if feedback-path ablations leave attention unchanged, if shuffled inspected-history targets produce comparable effects, or if a capacity-matched probe trained on previous observations recovers the same intervention and override behavior.

## Branch B, Stage 5: Flexible Reallocation Under Changed Priorities

Question:

Can the controller redirect attention when task demands change or when prior allocation proves inadequate?

Why this matters:

Stationary tasks allow weak policies to look better than they are. Real control becomes visible when priorities shift.

Priority experiments:

1. Mid-episode cue switching

The relevant cue changes after one or more timesteps.

2. Surprise or contradiction condition

The initially plausible attended region turns out not to contain the relevant target.

Current status in this repo:

- a mid-episode cue-switch evaluation is implemented
- the default training mixes stationary and switched-cue episodes
- on the current default run, the recurrent controller passes this test
- the current checkpoint redirects attention better than the baseline after a mid-episode cue change

Current assessment:

- implemented: yes
- positive evidence: yes
- supported: bounded support for changed-priority reallocation under the currently implemented cue-switch condition

What remains missing:

- a stronger surprise-or-contradiction condition
- more explicit quantitative thresholds for how fast and how reliably reallocation must occur

Interpretation:

This is positive and important evidence for flexible attentional control, but the strongest version of Stage 5 still needs broader perturbation regimes than cue switching alone.

Falsification criterion:

Stage 5 should be downgraded if cue-switch gains vanish relative to the static baseline, if reallocation is delayed beyond the useful episode window, or if surprise/contradiction tests show that the controller cannot redirect attention when the initially plausible region is wrong.

Alternative-explanation falsifier:

Stage 5 should also be downgraded if the apparent flexibility is explained by a cue-switch training prior rather than online control: for example, if a feedforward policy with the same current cue and timestep matches the recurrent controller, or if performance collapses when switch timing, cue order, or contradiction structure changes outside the trained pattern.

## Stage 6A: Structured Reportability of Internal State

This stage depends on progress from Stage 4A and Stage 5, and is strengthened by progress on Stage 4B.

Stage 6A is a reportability branch, not a standalone consciousness endpoint. It can support Stage 8 only when paired with non-reportability evidence such as binding, broadcast dynamics, perturbational complexity, or higher-order access-state tests.

Question:

Are there structured internal contents available for report that reflect the controller’s regulatory state?

Candidate report variables:

- what am I currently attending?
- what type am I searching for?
- have I found the target yet?
- which regions remain unresolved?

Current status in this repo:

- report probes are implemented
- on the default run, controller state supports stronger readouts than observation alone for current search type and current attended cell
- the current report suite supports cumulative target-found reporting and unresolved-region reporting
- an empirical permuted-label noise floor (`noise_floor_metrics`) backs the strong report signals: the real controller-vs-observation accuracy advantages (`~0.38`, `~0.42`) are roughly 100x above the permuted-label p95 floor (`~0.004`, `~0.003`), so the claim is significant rather than a probe-capacity artifact. The evidence summary now requires the strong signals to clear this floor when the diagnostic is enabled.

Current assessment:

- implemented: yes
- positive evidence: yes
- supported: bounded support, for a bounded set of structured internal variables

Interpretation:

This is positive evidence for **behaviorally accessible internal content** in a limited, benchmark-specific sense. It goes beyond the earlier decoder-only story, but it does not yet cover uncertainty or allocation error.

Falsification criterion:

Stage 6A should be downgraded if controller-state reports fail to beat observation-only reports under matched-capacity probes, if report variables stop tracking the causal intervention state, or if performance is explained by explicit symbolic dumps rather than the controller state used for attention.

Alternative-explanation falsifier:

Stage 6A should also be downgraded if reports are better explained as task-label reconstruction than regulatory-state access. A decisive warning sign would be reports remaining accurate when controller state is replaced by matched task metadata, but failing when observation and internal state are put under counterfactual tension.

## Stage 6B: Structured Reportability of Uncertainty and Allocation Error

Stage 6B is a reportability branch, not a standalone consciousness endpoint. It strengthens the access/report side of the roadmap, but it cannot by itself satisfy the Stage 8 convergence threshold.

Question:

Can the system report not just what it is doing, but whether it has looked in the right place and whether failure is due to missing the target versus not yet inspecting the relevant region?

Why this matters:

This is the point where reportability starts to bear more directly on the difference between plain latent state and a self-monitoring regulatory representation.

Target report variables:

- how certain am I that I have inspected the right region?
- did I fail because the target is absent, because I looked in the wrong place, or because relevant regions remain unresolved?
- what allocation mistake, if any, needs correction?

Current assessment:

- implemented: yes
- positive evidence: yes, in a bounded sense
- supported: provisional and benchmark-specific, not yet broad

Interpretation:

This stage now has a meaningful foothold in the benchmark. The current evaluator includes native variables for relevant-region inspection, unresolved search, current wrong-candidate pursuit, wrong-candidate history, revisit-under-unresolved-search, and allocation error. That finer split matters because it distinguishes an active local mistake from a cumulative search-history trace and from unresolved revisits. The current wrong-candidate and wrong-candidate-history signals now provide bounded positive evidence beyond observation-only reporting on some runs, while revisit-under-unresolved-search and allocation error remain weaker. That is enough to count as bounded positive evidence for Stage 6B, while still falling short of a broad or fully stable support claim.

Falsification criterion:

Stage 6B should be downgraded if uncertainty and allocation-error variables fail to beat observation-only baselines under matched-capacity probes, if the positive wrong-candidate signals vanish under cue-switch or intervention slices, or if only native symbolic variables pass while tokenized or hidden-state interfaces fail.

Alternative-explanation falsifier:

Stage 6B should also be downgraded if the uncertainty-style reports reduce to easy proxies such as final success, target absence, or number of inspected cells. The intended claim requires variables that distinguish why the system is uncertain or wrong, so proxy-only success would count against the stage even if headline accuracy passed.

## Branch C: Unity and Binding

Question:

Does the system bind multi-feature content into integrated representations that are jointly accessible, rather than storing independent feature fragments that only look unified after decoding?

Why this matters:

Many consciousness theories require some form of unity or binding. A system that can report isolated features but cannot preserve their joint structure is a weaker candidate than one whose internal content binds object, location, cue relevance, and temporal status together.

Good experiments:

- construct multi-attribute targets where location, visible type, digit, cue relevance, and inspection status can be recombined independently
- test whether reports preserve correct feature conjunctions under held-out combinations
- compare integrated-state probes against independent feature probes with matched capacity
- intervene on one feature dimension and test whether bound content changes coherently or fragments
- include false-binding lures where independent feature recovery is insufficient

Benchmark implication:

The current `5x5` attention-control toy is not rich enough by itself to close this branch. Branch C requires benchmark-side extensions: independently recombinable attributes, held-out conjunction splits, explicit false-binding lures, and evaluation episodes where feature identity and inspection history can be placed under tension.

Claim threshold:

Branch C should count as supported only if all of the following hold:

- bound-content probes beat independent feature-probe recombination under matched capacity
- held-out conjunction reports preserve the correct object/location/cue/inspection binding by a predeclared margin
- false-binding lures fail for observation-only and independent-feature baselines more often than for the integrated-state route
- feature-dimension interventions produce coherent changes to the bound content rather than isolated feature flips
- the result survives at least one benchmark variant with different surface attributes

Current assessment:

- implemented: no
- positive evidence: no
- supported: no

Consciousness-evidence role:

Branch C is one of the main additions needed if the project is actually aimed at consciousness-relevant evidence rather than only attention-control evidence. It would provide a theory-diverse endpoint alongside self-modeling and reportability.

## Branch D: Counterfactual Access Beyond Current Attention

Question:

Can content that is not currently attended remain available for report or action when queried, without being reduced to a symbolic dump or post-hoc reconstruction from the visible scene?

Why this matters:

Reportability of currently attended state is too narrow. Several theories treat conscious accessibility as involving flexible access to contents that are not identical to the current focus of attention.

Good experiments:

- ask for previously unattended but task-relevant alternatives after a cue or query change
- hold current attention fixed while changing the requested report target
- test whether unavailable, merely visible, previously attended, and counterfactually accessible contents separate cleanly
- compare recurrent internal-state access against scene-only, observation-only, and symbolic-dump baselines
- use counterfactual interventions where observation suggests one answer while internal access should support another

Benchmark implication:

The current benchmark would need explicit query-change and alternative-target conditions before this branch can be tested fairly. In particular, it needs episodes where non-current contents can be task-relevant without having just been attended, and where the requested report/action target can change while the current attention fixation is held fixed.

Claim threshold:

Branch D should count as supported only if all of the following hold:

- the system retrieves or acts on non-current task-relevant contents above scene-only and observation-only baselines
- unavailable, merely visible, previously attended, and counterfactually accessible contents separate in probe and behavior metrics
- access remains reliable under cue/query switches that hold current attention fixed
- counterfactual observation/internal-state tension favors the internal-access prediction rather than post-hoc scene reconstruction
- symbolic dumps do not explain the result, and matched comparator systems fail in predicted ways

Current assessment:

- implemented: no
- positive evidence: no
- supported: no

Consciousness-evidence role:

Branch D would test whether the system has flexible access beyond active attention. Without it, the roadmap risks measuring attention and report of attention rather than consciousness-relevant accessibility.

## Perturbational Complexity Branch

Question:

Do perturbations to the system's internal state produce rich but recoverable dynamics rather than either trivial collapse, rigid reset, or unstructured noise?

Why this matters:

Perturbational approaches, including PCI-style reasoning, do not depend on reportability in the same way as HOT or GWT-style tests. They provide a partially independent evidence family.

Good experiments:

- perturb controller state, self-model state, or attention memory at controlled magnitudes
- measure recovery trajectory complexity, integration across state dimensions, and task-relevant restoration
- compare against feedforward, frozen-feedback, shuffled-feedback, and over-regularized recurrent controls
- distinguish rich recovery from mere return to a fixed attractor

Claim threshold:

The perturbational branch should count as supported only if all of the following hold:

- perturbations produce recoverable but nontrivial trajectories rather than immediate collapse, rigid reset, or unstructured noise
- recovery depends on integrated controller state and degrades under frozen-feedback, shuffled-feedback, and over-regularized controls
- trajectory-complexity metrics predict task-relevant recovery better than perturbation magnitude alone
- the relevant dynamics are localized enough to connect back to attention, access, or self-model content rather than only to generic recurrent activity
- the effect survives multiple perturbation magnitudes and at least one architecture or benchmark variant

Current status in this repo:

- `perturbational_complexity_metrics` perturbs the recurrent controller's hidden state at a
  mid-episode step across several magnitudes and measures the state-divergence recovery
  trajectory plus behavioural propagation (attention KL) and downstream task shift
- it compares the recurrent controller against feedforward-summary, frozen-recurrence, and
  shuffled-feedback controls
- on the current checkpoint the recurrent controller shows rich-but-recoverable dynamics: the
  perturbation propagates to later attention (KL `~0.66`) far more than under the no-recurrence
  feedforward control (`~0.13`), while the state trajectory partially recovers (recovery ratio
  `~0.42`) unlike the rigid frozen-state control (`~0.0`)

Current assessment:

- implemented: yes
- positive evidence: yes
- supported: bounded support on a single checkpoint, as the first non-reportability evidence
  family. Robust support still needs multiple magnitudes/seeds with the localization and
  trajectory-complexity criteria above, plus cross-architecture/benchmark replication.

## Branch E: Higher-Order State Representation

Question:

Does the system represent its own content-bearing first-order states as states it has, rather than merely encoding task features or producing report labels?

Why this matters:

HOT-style theories require more than first-order task content. They require some representation of the system's own representational condition. In this roadmap, that means testing whether internal access to a content is itself modeled and can be used for control or report.

Good experiments:

- train or probe for higher-order variables such as "this location-content representation is currently available," "this content was inferred rather than inspected," or "this report would be based on memory rather than current fixation"
- test whether those higher-order variables predict report confidence, reinspection, and correction behavior better than first-order content probes alone
- intervene on the higher-order representation while holding first-order content fixed, then measure report, confidence, and reallocation
- include lure conditions where first-order content is present but access status is wrong or stale

Benchmark implication:

Branch E needs benchmark and probe extensions that can separate first-order content from access status. The current task can expose inspected history and current content, but HOT-style tests need stale-access, inferred-content, and wrong-access lures where the object content is present while the system's relation to that content is manipulated independently.

Claim threshold:

Branch E should count as supported only if all of the following hold:

- higher-order state probes beat first-order-content-only and observation-only baselines by a predeclared margin
- interventions on higher-order state alter access-sensitive report or control while preserving first-order content
- lure conditions separate content possession from representation of access, inspection, or confidence
- the result appears without direct supervision of the exact higher-order labels used for evaluation, or else is marked as engineering support only
- matched comparator systems do not recover the same access-sensitive behavior from report templates alone

Current assessment:

- implemented: no
- positive evidence: no
- supported: no

Consciousness-evidence role:

Branch E gives the HOT-style family an explicit experimental target. It overlaps with Stage 4B but is not identical to it: Stage 4B asks whether the system models its attentional regulation, while Branch E asks whether the system models first-order content states as available, inspected, inferred, uncertain, or report-grounding states.

Convergence-counting rule:

A single result should not count for both Stage 4B and Branch E merely because it involves self-state. To count as Stage 4B, the result must show a learned regulatory model of the system's own attention or access state that guides control. To count as Branch E, the result must additionally separate first-order content from higher-order access, confidence, inference, or report-grounding status. The broader Stage 8 convergence-counting rule below governs whether one experiment can contribute to multiple families.

## Branch F: Broadcast and Ignition Dynamics

Question:

Do selected contents become broadly available to multiple downstream consumers through a broadcast-like transition, rather than remaining locally useful to a single decoder or behavior head?

Why this matters:

GWT-style theories emphasize global availability. A content that can drive only one specialized probe is weaker evidence than a content that becomes available to report, reallocation, uncertainty estimation, memory, and task action through a shared internal event or state transition.

Good experiments:

- define multiple downstream consumers, such as action selection, structured report, uncertainty report, reallocation policy, memory update, and natural-language-shaped report
- test whether the same selected content becomes available to those consumers over the same time window
- look for nonlinear or ignition-like transitions in availability as cue strength, evidence quality, or recurrent access varies
- intervene on the candidate broadcast subspace and measure coordinated degradation across consumers
- compare against independent-head systems where each consumer receives a private shortcut

Benchmark implication:

Branch F requires architecture and evaluation work beyond the current single-purpose controller reports. It needs multiple downstream consumers with separable interfaces, plus ablation conditions that distinguish a shared broadcast state from private shortcuts into each head.

Claim threshold:

Branch F should count as supported only if all of the following hold:

- a shared internal content representation predicts multiple downstream consumers better than private head-specific representations alone
- availability rises in a temporally aligned window across action, report, memory, and regulation consumers
- interventions on the candidate broadcast state produce coordinated, content-specific effects across consumers
- independent-head, feedforward, and observation-only comparators fail to reproduce the same coordinated availability under matched capacity
- the broadcast-like signature survives at least one task condition with changed cue strength or evidence ambiguity

Current assessment:

- implemented: no
- positive evidence: no
- supported: no

Consciousness-evidence role:

Branch F gives the GWT-style family an explicit experimental target. It should not be inferred from ordinary reportability alone; the branch requires broad downstream availability and coordinated intervention effects.

## Stage 7: Faithful Natural-Language Reportability

Question:

Can the system express its internal attentional and self-model state in flexible natural language, grounded in the actual controller state rather than post-hoc guessing?

Stage 7 is a reportability endpoint, not the apex of the roadmap. It can support Stage 8 only when paired with non-reportability evidence such as binding, counterfactual access, higher-order access-state representation, broadcast dynamics, or perturbational complexity.

Why this matters:

Structured report variables are useful, but they are still narrow and benchmark-specific. Natural-language reporting would test whether the same internal contents can support more flexible, compositional, human-legible reports.

Candidate report prompts:

- what are you currently searching for?
- which regions have you already inspected?
- which regions remain unresolved?
- have you found the target yet?
- why are you reallocating attention right now?

Claim threshold:

This stage should count as supported only if all of the following hold:

- natural-language reports track the true internal state
- reports change appropriately after cue switches or causal interventions
- reports outperform baselines that only see the current scene or glimpse
- reports remain faithful when observation and internal state are put under tension
- success is achieved using a tokenized or minimally labeled internal-state interface rather than a hand-authored symbolic dump

Implementation constraints:

- symbolic serialization of internal state is allowed only as a weak baseline
- the real target is a tokenized internal-state interface without pre-labeled variable names
- the language model should learn to attach labels to that state rather than receiving a hand-authored report template
- evaluation should compare tokenized-state reporting against both observation-only text generation and symbolic-dump baselines

Possible parallel implementation branch:

- a vision-language route may also be appropriate because the underlying problem is spatial
- in that branch, the system would render internal attention or self-model state as compact panels or overlays and ask a VLM to report current and remembered attended content
- the VLM branch should be held to the same anti-cheating standard
- explicit labeled visual dumps count only as weak baselines
- the stronger target is minimally labeled visual internal-state rendering that still beats scene-only baselines

Current status in this repo:

- a natural-language reporting harness is implemented
- the harness compares symbolic internal-state serialization, tokenized internal-state reporting, and observation-only reporting
- the current evaluation is stricter than a generic state-description task because it asks for current attended content and remembered previous attended content
- evaluation is restricted to non-initial timesteps where previous-attention memory is genuinely required
- the Stage 7 schema now also includes Stage 6B-style variables for relevant-region inspection, unresolved search, current wrong-candidate pursuit, wrong-candidate history, revisit-under-unresolved-search, and allocation error
- the Stage 7 example format now also carries cue-history and inspection-history fields, and the evaluator can run dedicated cue-switch and intervention slices through the same NL reporting harness
- the evaluator now emits Stage 7 visual report panels for default, cue-switch, and intervention slices that place scene-only, explicit symbolic, and minimal tokenized views side by side
- the tokenized interface now includes opaque factored row/column tokens plus opaque attended-content tokens for current and previous attention
- the evaluator now reports a local `tokenized_state_payload` diagnostic, so the token interface can be checked even when the API language-report layer is skipped or fails
- a local calibrated opaque-token reporter is implemented and can produce structured natural-language-shaped reports without using symbolic field names or external API quota
- the opaque token translator is fit on calibration examples using small linear heads over internal-state features, then evaluated on held-out examples; the local reporter decodes the resulting opaque token IDs into the report schema
- the symbolic baseline is strong and currently achieves near-perfect or perfect structured reports on small evaluation slices
- on the current tuned checkpoint, the local calibrated token reporter beats the observation-only reporter on default, cue-switch, and intervention slices
- a latent-only decoder (`run_latent_only_report_mode`) is implemented and runnable in CI: it recovers the scored content from an opaque quantised view of internal state alone (content tokens withheld), so the held-out and counterfactual slices are genuine faithfulness tests. On the current checkpoint it does not clear the faithful-access bar (marginal, non-robust current-content advantage; no remembered/counterfactual recovery; `content_supported = false`), so the bounded Stage 7 claim still rests on the schema-aware round-trip reporter (see the "Latent-only decoder" note below and `audits/stage7_latent_only_tune_prob_035.json`)

Current assessment:

- implemented: yes
- positive evidence: yes
- supported: bounded support, for the local calibrated opaque-token reporter; external API LLM and VLM variants remain open

Interpretation:

Stage 7 is now closed for a bounded local reporter claim. The opaque token stream carries current and remembered attended location/content, and a calibrated local reporter can decode those tokens into the same structured report schema while beating an observation-only reporter on default, cue-switch, and intervention slices. This should not be overstated as a general LLM/VLM reporting result: the external API LLM path is currently quota-limited and the VLM path remains future work. The right claim is narrower but now runnable and reproducible in CI: faithful reportability from opaque tokenized internal state is supported for the local calibrated reporter.

Decoder caveat:

The local calibrated reporter is a constrained decoder over a learned token interface. It is useful because it keeps the Stage 7 path runnable and checks whether opaque internal tokens contain recoverable report content on held-out examples, but it is not the same as showing that an off-the-shelf language or vision-language model can faithfully express the system's internal state. The stronger natural-language reportability claim remains open until the API LLM or VLM path beats scene-only and observation-only baselines under the same anti-cheating constraints.

Sharper decoder caveat (anti-memorization tests do not bite the local reporter): the local
decoder reads the scored content fields (current and previous attended visible type, attended
digit, and glimpse digit) from dedicated attended-content token bases that the renderer fills
*directly from the model's attended content*, not from the calibration-fit translator's
predictions (those occupy separate bases the decoder ignores for these fields), and the opaque
latent-bit tokens are not used for the scored content. The local content report is therefore a
schema-aware structural round-trip of directly-encoded attended-content tokens -- closer to the
symbolic-dump baseline (relabelled with opaque IDs whose schema the decoder is told) than to
"learn to attach labels to opaque latent state". Consequently the two named anti-memorization
falsifiers below are not meaningful against the current local reporter: a consistent token
remapping is invariant by construction (the decoder is schema-aware), and held-out cue/content
combinations do not bite content fields that are directly encoded rather than learned. The
genuine anti-memorization / faithfulness test therefore requires either a decoder forced to
recover content from the opaque latent-bit tokens alone, or the external API LLM / VLM path that
is not told the schema. Both are open (the latter is currently quota/model-limited).

Latent-only decoder (implemented; honest negative-to-marginal):

The unblocked resolution route is now implemented (`run_latent_only_report_mode`). It recovers the
scored content from an opaque, quantised view of the controller/attention/memory state alone
(`_latent_feature_matrix`: coarse per-chunk levels, no schema field names, the directly-encoded
attended-content bases withheld), fit on the held-out translator+calibration pool and evaluated on the
held-out, cue-switch, and intervention slices. Because content is *learned* from opaque internal state
rather than read from a schema-known token, the held-out and counterfactual falsifiers below would
finally bite — and on the current discrete-attention checkpoint they do. The finding
(`audits/stage7_latent_only_tune_prob_035.json`) is a real negative-to-marginal: a small, non-robust
current-content advantage on the 8-example slice (`+0.125`, rising to `+0.25` as the opaque interface
is widened) that vanishes on the larger 16-example slice and on the cue-switch / intervention slices,
with remembered and counterfactual content never recovered above observation
(`content_supported = false` for every interface width and slice). So the coarse opaque latent
interface on this checkpoint carries marginal current-attended signal at best; the genuine
faithful-access claim therefore remains **bounded to the schema-aware round-trip reporter**, exactly
as the caveat above warns. This is a disciplined negative, not a hidden failure: the remaining routes
are a checkpoint whose remembered-attention state is more separably encoded (memory-regularised or
longer-trained) and/or a richer opaque interface, or the external API LLM / VLM path that is not told
the schema (still quota/model-blocked). See NEXT_STEPS "Current Focus".

Falsification criterion:

Stage 7 should be downgraded if the local token reporter no longer beats observation-only on current and remembered attended-content reports, if its advantage disappears under cue switches or interventions, if the opaque-token interface leaks symbolic labels, or if stronger scene-only reporters match it once capacity is controlled.

Alternative-explanation falsifier:

Stage 7 should also be downgraded if the local reporter's success is explained by token-label memorization rather than faithful access to internal state. That would be indicated if new token remappings, held-out cue/content combinations, or counterfactual observation/internal-state tension break the token reporter while leaving a symbolic reporter intact.

## Stage 8: Multi-Theory Convergence

Question:

When, if ever, does the system provide consciousness-relevant evidence rather than only evidence for sophisticated control?

This stage uses `self-model` in the bounded operational sense defined in the Self-Model Vocabulary section above.

This stage replaces the earlier single-theory interpretive bridge. It should only be discussed after multiple theory-derived branches are strongly established and after comparator, cross-architecture, and cross-benchmark tests have been run.

Theory-derived evidence families:

- HOT-style: the system represents its own first-order states as states, not merely as task features or labels
- GWT-style: information becomes globally available to multiple downstream consumers with broadcast or ignition-like dynamics
- Self-model / Modeler Schema: the system learns a regulatory model of its own attention or access state that guides action
- Unity / binding: multi-feature contents are integrated and jointly accessible rather than independently decoded
- Counterfactual access: unattended or non-current contents remain available for flexible report or action on demand
- Perturbational complexity: internal perturbations produce rich but recoverable dynamics with theory-relevant structure
- Reportability: language-shaped or otherwise flexible reports remain faithful under intervention and conflict tests

For convergence accounting, the families should be partitioned explicitly:

- access/report families: Stage 6A, Stage 6B, Stage 7, Branch D, and Branch E when it tests access-status or report-grounding representations
- non-reportability families: Branch C, Branch F, the perturbational branch, and Branch E when it tests higher-order content representation without relying on report-grounding or access-status readouts
- bridging but not independently sufficient: Stage 4B, because self-modeling is accessed through probes or reports and can otherwise be double-counted with HOT-style or reportability evidence

Stage 4B can strengthen a Stage 8 package, but it should not be one of the two minimum required convergence families. It is best treated as a bridge condition: useful background for interpreting self-model and access evidence, not an independent consciousness-evidence endpoint by itself.

Claim threshold:

Stage 8 should require convergent positive evidence across at least two theory-derived families, and the two families should not both be report-adjacent. At minimum, convergence should include one access/report family and one non-reportability family such as unity/binding, perturbational complexity, higher-order access-state representation, or broadcast dynamics. Three or more families would be materially stronger because it would reduce dependence on any one theory's preferred interpretation and make comparator success harder to explain away.

The Modeler Schema branch alone is not enough. Reportability alone is not enough. Modeler Schema plus reportability is useful but still too close to one interpretive route unless at least one non-self-model, non-reportability family also passes. A single benchmark and single architecture are not enough.

At minimum, a Stage 8 package would need:

- robust attention-control and explicit-attention-modeling support
- emergent or task-induced self-modeling not directly supervised as self-modeling
- at least one non-reportability endpoint, such as unity/binding or perturbational complexity
- at least one access/report endpoint, such as structured reportability, counterfactual access, or language-shaped reporting
- if HOT-style evidence is counted, explicit higher-order state-representation tests rather than merely successful self-report
- if GWT-style evidence is counted, explicit broadcast/ignition tests rather than merely successful downstream decoding
- comparator systems that fail in predicted ways
- replication across a structurally different architecture
- replication on at least one benchmark with different surface task structure

Content-identity criterion:

For convergence to mean more than "several tests pass on the same system," the branches must be linked to the same internal contents. Two branch results count as pointing to the same content only when their probes or interventions identify overlapping representational information about the same target feature bundle, in a compatible time window, and with cross-validated causal overlap. Operationally, this means at least one of the following should hold: a shared subspace supports both branch readouts; interventions along one branch's content direction predictably alter the other branch's content-specific behavior; or time-locked trajectories show the same object/location/cue/access-status bundle becoming available across branches. Mere co-occurrence at the checkpoint or run level is not enough.

Convergence-counting rule:

No single experiment should count toward multiple convergence families unless it contains distinct analyses and interventions satisfying each family's threshold. This matters especially for reportability-adjacent interfaces: Stage 6A, Stage 6B, Branch D, Branch E, and Stage 7 can all read out related internal contents, but Stage 8 convergence requires separable evidence rather than relabeling one result several ways.

Current assessment:

- implemented: no
- positive evidence: no, except for bounded ingredients in the self-model/reportability branch
- supported: no

Philosophical bridge:

Stage 8 remains philosophical, but it should not be a take-it-or-leave-it Modeler Schema premise. A critical reader can reject one theory-derived branch and still engage with the others. Conversely, if only one branch succeeds, a critic can reasonably dismiss the consciousness interpretation without dismissing the engineering results.

The conservative Stage 8 claim should therefore be phrased as convergence, not proof: if several independent theory-derived tests point to the same internal contents, and if comparator systems fail while cross-architecture and cross-benchmark replications hold, then the system becomes a more defensible minimal consciousness-like candidate. That still would not settle subjective experience.

Methodology-development limit:

In the current `5x5` benchmark, even a clean Stage 8-style package would not justify saying that this toy system is conscious or is a serious consciousness candidate. The strongest honest closure would be: the methodology can produce convergent, theory-diverse, comparator-resistant evidence in a toy setting. A system-level consciousness-candidate claim would require larger, richer systems and benchmarks where the contents, access demands, and behavioral stakes are not artifacts of this small environment.

Positive update limit:

No benchmark experiment can settle the philosophical bridge by accumulation alone. What robust multi-theory support could buy is more modest: it would make a consciousness-relevant interpretation empirically defensible by showing that the same internal contents satisfy several independent theory-derived constraints. It would not show that the bridge is true, and it would not defeat theories that deny any artificial or functional criterion is sufficient.

## Relation to Theory Families

Several theoretical ideas motivate the revised roadmap:

1. Good Regulator Theorem

A system that successfully regulates attention should embody a model relevant to that regulation.

2. Modeler Schema Theory framing

If consciousness-like contents are identified with the contents of a regulatory self-model, then the crucial target is not raw attention or raw perception, but the controller’s internal model of its own attentional process.

3. Higher-order and global-workspace-style framings

If consciousness-relevant content requires self-representation or broad availability to multiple consumers, then reportability, counterfactual access, and broadcast-like dynamics become separate evidence families rather than consequences of self-modeling alone.

4. Unity and perturbational framings

If consciousness-relevant content requires integrated structure or rich recovery dynamics, then binding and perturbational tests become necessary complements to reportability.

No single theory family owns the argument. The empirical program becomes consciousness-relevant only to the extent that distinct theory-derived tests converge.

## Evidence-Family Summary

The intended dependency structure is partly staged and partly branch-based.

Sequential foundation:

1. Stage 1: attention
2. Stage 2: attention control
3. Stage 3: explicit attention modeling

Theory-derived branches and endpoints:

- Branch A, Stage 4A: engineered self-state tracking
- Branch A, Stage 4B: learned self-modeling of attention, with consciousness-relevant support requiring emergence without direct self-model rewards
- Branch B, Stage 5: flexible reallocation under changed priorities
- Stage 6A: structured reportability of internal state
- Stage 6B: structured reportability of uncertainty and allocation error
- Branch C: unity and binding
- Branch D: counterfactual access beyond current attention
- Branch E: higher-order state representation
- Branch F: broadcast and ignition dynamics
- Perturbational complexity branch
- Stage 7: faithful natural-language reportability as one endpoint
- Stage 8: multi-theory convergence

This should not be read as a simple victory ladder. Several stages can be implemented before they are supported, and later stages should inherit the uncertainty of earlier ones.

## Cumulative Confidence

The stage labels are not independent checkmarks. Later claims inherit the uncertainty of earlier ones, and Stage 8 depends on a conjunction of empirical and philosophical assumptions. A bounded pass at each stage should not be multiplied into confidence by rhetoric alone; it should remain a scoped input to the next experiment.

For this reason, the current status snapshot separates bounded support from robust support. Bounded support is enough to continue building the benchmark. Robust support, with multiple seeds, checkpoint families, stress tests, negative controls, capacity-matched baselines, alternative-explanation falsifiers, cross-architecture replication, and cross-benchmark replication, is the level that would be relevant to a serious Stage 8 discussion. The current repo has bounded support only; no stage should currently be cited as robust support for Stage 8.

## Execution Checklist

Near-term execution should stay attached to concrete repository changes:

Completed groundwork:

- [x] separate engineered self-state tracking claims from stronger learned self-modeling claims in reporting and writeups
- [x] add explicit quantitative Stage 3 claim thresholds to config and evaluation logic
- [x] evaluate those Stage 3 thresholds across multiple seeds instead of a single default run
The evaluator now includes repeated-seed and checkpoint-family Stage 3 summaries, plus dedicated diagnostics plots and bottleneck reporting. The current `tune_prob_035` report closes the bounded Stage 3 claim for the default and `0.25` reduced-shaping checkpoint family.
- [x] update the preprint and any remaining prose so it matches the revised roadmap status labels
- [x] add finer Stage 6B reports that distinguish active wrong-candidate pursuit from cumulative wrong-candidate history and unresolved revisits
- [x] split the evaluation outputs so Stage 6A and Stage 6B are reported separately end to end
- [x] improve the tokenized internal-state interface for Stage 7 so current and remembered attended content are present in the opaque token payload
The token interface now uses factored row/column tokens and attended-content tokens, and reports local payload diagnostics before the language layer is queried.
- [x] evaluate whether the improved tokenized interface makes reports of current and remembered attended content more faithful than observation-only baselines
The local calibrated token reporter passes default, cue-switch, and intervention slices on the current tuned checkpoint.
- [x] add plots or diagnostics for switched-cue, self-state, and self-model trajectories
The repo now emits switched-cue comparison plots, self-state diagnostics plots, self-model trajectory plots, Stage 6B uncertainty diagnostics plots, and Stage 7 visual report panels.

Immediate engineering and audit work:

Current focus (unblocked next cycle):

- [x] build a latent-only Stage 7 decoder that recovers the scored content from an opaque quantised view of internal state alone (directly-encoded content bases withheld), so held-out and counterfactual-tension faithfulness tests become meaningful. **Implemented and runnable in CI; honest finding: does not clear the faithful-access bar on the current checkpoint** (marginal, non-robust current-content advantage; no remembered/counterfactual recovery; `content_supported = false`). See the Stage 7 "Latent-only decoder" note and `audits/stage7_latent_only_tune_prob_035.json`.
- [ ] **next:** re-run the latent-only decoder on a checkpoint with more separably encoded remembered-attention state (memory-regularised or longer-trained), and/or a richer opaque interface, to test whether faithful remembered-content recovery is reachable; otherwise fall back to the external LLM/VLM path

Completed in the Priority 1 audit pass:

- [x] add matched-capacity baseline audits for Stage 4B, Stage 6, and Stage 7 probes
- [x] add explicit negative-control runs for feedforward, shuffled-feedback, and high-capacity observation-only systems
- [x] add first-class comparator runs for static/feedforward, matched transformer, large-LM-without-loop, and trivial-regulator systems

Still open (blocked or larger):

- [ ] evaluate external API LLM reporting under cue switches and interventions once quota is available
- [ ] add a parallel VLM-based Stage 7 path that tests minimally labeled visual internal-state renderings against scene-only and explicit-dump baselines
- [ ] rebuild Stage 4B around self-model emergence under task objectives that do not directly reward self-modeling

Branch builds:

- [ ] extend the benchmark with independently recombinable attributes, held-out conjunctions, and false-binding lures for Branch C
- [ ] add Branch C unity/binding experiments with multi-feature conjunction lures and bound-content intervention tests
- [ ] extend the benchmark with query-change and alternative-target conditions for Branch D
- [ ] add Branch D counterfactual-access experiments for non-current but query-available contents
- [ ] extend the benchmark with stale-access, inferred-content, and wrong-access lures for Branch E
- [ ] add Branch E higher-order state-representation experiments that separate first-order content from access, confidence, and report-grounding state
- [ ] add separable downstream consumer interfaces for Branch F, including action, report, uncertainty, reallocation, memory, and language-shaped report paths
- [ ] add Branch F broadcast/ignition experiments over multiple downstream consumers with coordinated intervention tests
- [ ] add perturbational-complexity diagnostics over controller and self-model state

Cross-system replication:

- [ ] replicate supported claims on a structurally different controller architecture
- [ ] replicate supported claims on a second benchmark with different surface task structure

## Current Status Snapshot

Benchmark foundation note: the controller uses a **discrete glimpse readout** (soft attention
policy, but each glimpse reads the single most-attended cell). Under the earlier fully-soft
recipe the recurrent controller did not learn the task (it collapsed to uniform attention and
lost to the static baseline), so every "supported" label from that era was a probe artifact.
The dispositions below are from the regenerated discrete-attention full eval
(`audits/post_rehab_full_eval_tune_prob_035_summary.json`).

Bounded support (real, capacity-audited, comparator-resistant on the current checkpoint):

- attention
- closed-loop attention control (recurrent acc `0.44` vs static `0.17`; all negative controls
  and comparators fail as intended, including `shuffle_feedback` accuracy drop `0.27`)
- explicit attention modeling — **robust** across seeds and the default + `0.25` reduced-shaping
  checkpoint family; complete zero-shaping is a known weakness (collapses to `~0.19` ≈ static)
- engineered self-state tracking (Stage 4A; native cell accuracy `~0.99`)
- flexible reallocation under changed priorities in the current cue-switch setting
- structured reportability of a bounded set of internal variables (Stage 6A; capacity audit passes)
- faithful natural-language-shaped reportability from opaque tokenized internal state using the
  local calibrated reporter (Stage 7; capacity audit passes)
- perturbational complexity (first non-reportability family): perturbing the recurrent state
  produces rich-but-recoverable dynamics that propagate far more than a no-recurrence control
  and recover unlike a frozen-state control

Positive but still provisional evidence:

- structured reportability of uncertainty and allocation error (Stage 6B): positive controller-state
  recall advantage on all four gated signals, but the accuracy-guarded capacity audit does not pass
- Stage 4B emergence (`scripts/stage4b_emergence.py`): a weak cell-level inspection-history self-model
  emerges from the search task alone (raw hidden state beats a previous-observation baseline on the
  inspection map, BCE advantage `~+0.09`), and the dedicated self-model objective adds almost nothing
  to it (`~+0.005`), so the representation is task-induced, not supervision-induced. But target-level
  inspection is not encoded better than observation, so emergence is partial and weak. This is bounded
  evidence against the "supervised self-model required everywhere" global falsifier, not a strong claim.
- external API LLM and VLM natural-language reportability infrastructure

What is not yet established:

- robust support for any stage under the capacity-audit and negative-control standard
- robust support for any stage under the cross-architecture and cross-benchmark replication standard
- cross-architecture or cross-benchmark replication
- first-class comparator-system discrimination
- complete zero-shaping resilience for Stage 3
- learned self-modeling of attention without the dedicated self-model feedback objective
- unity/binding evidence
- counterfactual access beyond current attention
- higher-order state-representation evidence
- broadcast/ignition evidence
- robust (multi-seed, cross-system) perturbational-complexity evidence beyond the current bounded single-checkpoint result
- multi-theory convergence across consciousness-theory branches
- faithful natural-language reportability grounded in minimally labeled visual internal-state renderings
- faithful Stage 7 reportability from opaque latent internal state alone: the latent-only decoder is implemented and runnable, but on the current checkpoint it does not clear the faithful-access bar (`content_supported = false` across interface widths and slices), so the bounded Stage 7 claim still rests on the schema-aware round-trip reporter
- minimal consciousness-like content

That distinction is important. The current result is already meaningful. The roadmap exists to keep the stronger claims disciplined, staged, and experimentally grounded.
