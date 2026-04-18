# Roadmap Toward Evidence of Minimal Consciousness

This document separates the current benchmark result from the larger research goal.

The project already supports a meaningful claim about **closed-loop attention control**. The longer-term goal is more ambitious: to build a sequence of experiments that could count as evidence for **minimal consciousness-like processing** in a tightly bounded, artificial system.

Because that goal is easy to overstate, this roadmap is intentionally conservative. It is meant to discipline the argument, not accelerate it.

## What This Roadmap Is Not Claiming

This roadmap does not aim to establish:

- human-like consciousness
- subjective experience in a strong philosophical sense
- proof rather than interpretation
- open-ended self-awareness outside this benchmark

The narrower goal is to accumulate bounded evidence for increasingly rich attentional regulation, self-modeling, and behaviorally accessible internal content.

## Claim Status Conventions

Each stage should be read using three different labels:

- `implemented`: the experiment or mechanism exists in the repo
- `positive evidence`: current results point in the expected direction
- `supported`: the stage meets a claim threshold strong enough to rely on in the larger argument

A stage should not count as supported merely because a favorable example exists. The stronger standard is reproducible evidence that survives fair baselines, leakage controls, and alternative explanations.

## Evaluation Discipline

The roadmap only works if later stages are protected against easy forms of self-deception.

Across Stages 3 through 7, the following standards should apply:

- each reporter or probe must have a clearly specified input interface
- observation-only baselines must be matched in capacity as closely as possible
- train and evaluation splits must prevent simple memorization of scenes, prompts, or report templates
- benchmark-specific symbolic dumps count only as weak baselines, not as the main claim
- the strongest tests should place observation and internal state under tension so that faithfulness can be distinguished from post-hoc guessing
- all support claims should survive multiple seeds and should be reported with explicit effect sizes or margins over baseline

## Dependency Structure

Stages 1 through 3 are the core sequential foundation. After that, the roadmap branches and then rejoins:

- Stage 4A asks whether the system contains engineered self-state tracking
- Stage 4B asks whether it learns a model of its own attention as an object of regulation
- Stage 5 asks whether attention control remains flexible when priorities change
- Stage 6A asks whether internal state is behaviorally reportable in structured form
- Stage 6B asks whether uncertainty and allocation error are also reportable

Stage 7 depends on strong progress through the earlier stages, and Stage 8 should only be discussed after the reporting stages are in place.

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
- on the current default run, when the direct target-attention supervision term is reduced or removed, useful reallocation becomes weaker but remains positive even in the zero-shaping condition

Current assessment:

- implemented: yes
- positive evidence: yes
- supported: not yet by default; this should remain provisional until quantitative thresholds are fixed in advance and met consistently

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

- explicit numerical thresholds for predictive, intervention, and reduced-shaping evidence
- stronger evidence that the signal reflects structured attentional dynamics rather than generic recurrent state
- a clearer statement of what would falsify the stage

Interpretation:

The current results are promising and materially stronger than Stage 2 alone, but Stage 3 should be treated as **implemented with positive evidence**, not as fully settled, until the claim thresholds are fixed and passed.

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
- supported: yes, as engineered self-state tracking within this benchmark

Interpretation:

This is meaningful progress, but it should not be conflated with a stronger claim that the model has learned its own attentional self-model. The right reading is that the system now contains an explicit internal variable about prior allocation and can behaviorally report it.

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
- interventions on the learned self-model representation should alter both report and downstream reallocation
- the same representation should generalize to changed task conditions rather than only to the original benchmark regime

Current assessment:

- implemented: partially
- positive evidence: not enough yet
- supported: no

Interpretation:

Stage 4B should remain open until the project can show that the controller learns and uses a model of its own attentional process, rather than merely exposing an engineered bookkeeping variable.

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
- supported: yes for changed-priority reallocation under the currently implemented cue-switch condition

What remains missing:

- a stronger surprise-or-contradiction condition
- more explicit quantitative thresholds for how fast and how reliably reallocation must occur

Interpretation:

This is positive and important evidence for flexible attentional control, but the strongest version of Stage 5 still needs broader perturbation regimes than cue switching alone.

## Stage 6A: Structured Reportability of Internal State

This stage depends on progress from Stage 4A and Stage 5, and is strengthened by progress on Stage 4B.

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

Current assessment:

- implemented: yes
- positive evidence: yes
- supported: yes, for a bounded set of structured internal variables

Interpretation:

This is positive evidence for **behaviorally accessible internal content** in a limited, benchmark-specific sense. It goes beyond the earlier decoder-only story, but it does not yet cover uncertainty or allocation error.

## Stage 6B: Structured Reportability of Uncertainty and Allocation Error

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

This stage now has a meaningful foothold in the benchmark. The current evaluator includes native variables for relevant-region inspection, unresolved search, wrong-candidate history, and allocation error, and the new wrong-candidate-history signal can outperform an observation-only baseline on positive-recall style reporting. That is enough to count as bounded positive evidence for Stage 6B, while still falling short of a broad or fully stable support claim.

## Stage 7: Faithful Natural-Language Reportability

Question:

Can the system express its internal attentional and self-model state in flexible natural language, grounded in the actual controller state rather than post-hoc guessing?

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
- the symbolic baseline is strong and currently achieves near-perfect or perfect structured reports on small evaluation slices
- the tokenized internal-state condition is still not successful enough to beat observation-only baselines on current attended location or remembered previous attended content

Current assessment:

- implemented: yes
- positive evidence: partial
- supported: no

Interpretation:

Stage 7 remains open. The blocker currently appears representational rather than infrastructural: the tokenized interface is not yet faithful enough on current and remembered attended semantic content to support a clean natural-language claim.

## Stage 8: Conservative Interpretation as Minimal Consciousness-Like Content

Question:

When, if ever, does the system become a plausible candidate for minimal consciousness-like content?

This stage should only be discussed after the earlier stages are strongly established.

At minimum, the argument would require:

- closed-loop attention control
- evidence for an explicit model of attention
- evidence that this model is a model of the system’s own regulatory state
- structured, behaviorally accessible internal contents
- flexible report and action based on those contents
- natural-language or otherwise flexible reportability that remains faithful under intervention and conflict tests

At that point, one could argue that:

- the system contains a regulatory self-model
- the contents of that self-model are available to guide action and report
- those contents are therefore candidates for minimal consciousness-like contents

This would still be an interpretation, not a proof.

## Relation to Good Regulator and Modeler Schema Ideas

Two theoretical ideas motivate the roadmap:

1. Good Regulator Theorem

A system that successfully regulates attention should embody a model relevant to that regulation.

2. Modeler Schema Theory framing

If consciousness-like contents are identified with the contents of a regulatory self-model, then the crucial target is not raw attention or raw perception, but the controller’s internal model of its own attentional process.

The roadmap is designed to move from a demonstrated control loop toward experimental evidence for that stronger type of internal modeling.

## Structure Summary

The intended dependency structure is:

1. Stage 1: attention
2. Stage 2: attention control
3. Stage 3: explicit attention modeling
4. Branch A, Stage 4A: engineered self-state tracking
5. Branch A, Stage 4B: learned self-modeling of attention
6. Branch B, Stage 5: flexible reallocation under changed priorities
7. Stage 6A: structured reportability of internal state
8. Stage 6B: structured reportability of uncertainty and allocation error
9. Stage 7: faithful natural-language reportability
10. Stage 8: conservative interpretation

This should not be read as a simple victory ladder. Several stages can be implemented before they are supported, and later stages should inherit the uncertainty of earlier ones.

## Execution Checklist

Near-term execution should stay attached to concrete repository changes:

- [x] separate engineered self-state tracking claims from stronger learned self-modeling claims in reporting and writeups
- [x] add explicit quantitative Stage 3 claim thresholds to config and evaluation logic
- [ ] evaluate those Stage 3 thresholds across multiple seeds instead of a single default run
- [ ] update the preprint and any remaining prose so it matches the revised roadmap status labels
- [ ] add stronger allocation-error and uncertainty reports that distinguish missing the target from not yet having inspected the right region
- [x] split the evaluation outputs so Stage 6A and Stage 6B are reported separately end to end
- [ ] improve the tokenized internal-state interface for Stage 7 so current and remembered attended content become more faithful than observation-only baselines
- [ ] evaluate natural-language reporting under cue switches and interventions once the tokenized condition is stable
- [ ] add a parallel VLM-based Stage 7 path that tests minimally labeled visual internal-state renderings against scene-only and explicit-dump baselines
- [ ] add plots or diagnostics for switched-cue, self-state, and self-model trajectories

## Current Status Snapshot

What is already supported:

- attention
- closed-loop attention control
- engineered self-state tracking
- flexible reallocation under changed priorities in the current cue-switch setting
- structured reportability of a bounded set of internal variables

What currently has positive but still provisional evidence:

- explicit attention modeling
- structured reportability of uncertainty and allocation error
- parts of natural-language reportability infrastructure

What is not yet established:

- learned self-modeling of attention
- faithful natural-language reportability grounded in tokenized internal state
- faithful natural-language reportability grounded in minimally labeled visual internal-state renderings
- minimal consciousness-like content

That distinction is important. The current result is already meaningful. The roadmap exists to keep the stronger claims disciplined, staged, and experimentally grounded.
