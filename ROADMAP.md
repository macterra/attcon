# Roadmap Toward Evidence of Minimal Consciousness

This document separates the current benchmark result from the larger research goal.

The project already supports a meaningful claim about **closed-loop attention control**. The longer-term goal is more ambitious: to build a sequence of experiments that could count as evidence for **minimal consciousness-like processing** in a tightly bounded, artificial system.

The safest way to approach that goal is as a staged program.

Stages 1 through 3 are the core sequential foundation. After that, the roadmap branches:

- Stage 4 asks about representational depth: whether the controller models its own attention
- Stage 5 asks about behavioral robustness: whether attention control remains flexible when priorities change

Those two later stages are related, but neither is a strict prerequisite for the other. Stage 6 depends on progress along both branches, and Stage 7 is only worth discussing after the earlier requirements are in place.

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

Interpretation:

This is evidence of **closed-loop attentional regulation**.

It is stronger than ordinary attention, but still not enough to establish an explicit internal model of attention.

## Stage 3: Explicit Attention Modeling

Question:

Does the controller state explicitly model attentional dynamics, rather than merely functioning as useful recurrent memory?

This is the most important next step.

Current status in this repo:

- a predictive probe is now implemented
- on the default run, controller state predicts the next attention map better than a baseline built from current observation alone
- a causal intervention test is now implemented
- on the default run, perturbing controller state causes a measurable shift in the next attention map away from the original cue target and toward an alternate cue target
- reduced-shaping retraining runs are now implemented
- on the current default run, when the direct target-attention supervision term is reduced or removed, useful reallocation becomes weaker but remains positive even in the zero-shaping condition

Interpretation:

This is strong evidence for **explicit attention modeling** within this benchmark.

It is stronger than Stage 2 alone because it now includes predictive, intervention, and reduced-shaping analyses, but it is still not enough to establish a cleanly interpretable model of attention.

Earlier tuning exposed a real tradeoff between switched-priority behavior and reduced-shaping resilience, but the current default checkpoint now supports both.

Required experiments:

1. Predictive probe

Train a readout from controller state at time `t` to predict:

- the next attention map
- target-attention gain at `t+1`
- or the effect of a reallocation

Success criterion:

The controller state predicts future attentional behavior better than baselines built from raw observation alone.

2. Intervention test

Perturb controller state while holding scene and cue fixed.

Best version:

- identify a subspace associated with previous attention or previous observation
- intervene on that subspace only
- measure systematic changes in the next attention allocation

Success criterion:

Future attention changes in a controlled and interpretable way as a function of the perturbation.

3. Reduced shaping-loss condition

Remove or reduce the direct target-attention supervision term.

Success criterion:

Useful reallocation still emerges from task success alone, showing that the controller is not merely optimizing an engineered shaping objective.

What remains missing at this stage:

- sharper quantitative thresholds for what counts as strong enough predictive and reduced-shaping evidence
- sharper quantitative thresholds for what counts as strong enough intervention evidence
- stronger evidence that the predictive and intervention signals reflect structured attentional dynamics rather than generic recurrent state

## Branch A, Stage 4: Self-Modeling of Attention

Question:

Does the system model not just the world, but its own attentional state as an object of regulation?

This requires moving from “uses memory to guide attention” to “tracks attention itself.”

Needed capabilities:

- explicit internal variables that track what has been attended
- explicit representation of what remains uninspected
- uncertainty tied to access rather than just output confidence
- error-correction specifically about allocation mistakes

Good experiment:

- the model must state whether it has already inspected the relevant region
- the answer must depend on internal attention history, not just final task success

Success criterion:

Internal state supports reliable report and correction about the model’s own attentional process.

Current status in this repo:

- the recurrent controller now maintains an explicit inspected-cell state
- that state tracks which cells have already been fixated and which remain uninspected
- the default evaluation now includes a native self-model test over the full inspected/uninspected cell map
- on the default run, the native self-model beats an observation-only probe on both full-map reporting and the binary question of whether the target has already been inspected

Interpretation:

This stage is now supported in a bounded, engineered sense.

The current system does not just expose decoder-friendly hidden state. It now contains an explicit internal variable about prior attentional allocation and can report that variable behaviorally better than an observation-only baseline. That is enough to count as Stage 4 completion for this benchmark, while still falling short of stronger consciousness-style claims.

## Branch B, Stage 5: Flexible Reallocation Under Changed Priorities

Question:

Can the controller redirect attention when task demands change or when prior allocation proves inadequate?

Why this matters:

Stationary tasks allow weak policies to look better than they are. Real control becomes visible when priorities shift.

Priority experiments:

1. Mid-episode cue switching

The relevant cue changes after one or more timesteps.

Success criterion:

The recurrent controller rapidly redirects attention and outperforms static or non-recurrent alternatives.

2. Surprise or contradiction condition

The initially plausible attended region turns out not to contain the relevant target.

Success criterion:

The controller reallocates rather than persisting with the original policy.

Current status in this repo:

- a mid-episode cue-switch evaluation is now implemented
- the default training now mixes stationary and switched-cue episodes
- on the current default run, the recurrent controller now passes this test
- the current checkpoint redirects attention better than the baseline after a mid-episode cue change

Interpretation:

This is now positive evidence.

The current controller can be trained into flexible reallocation under changed priorities, and the latest default checkpoint keeps that result compatible with the current Stage 3 evidence.

## Stage 6: Reportable Internal Content

This stage depends on progress from both Branch A and Branch B.

Question:

Are there structured internal contents available for report that reflect the controller’s regulatory state?

This is where the project begins to approach minimal consciousness-style claims.

Candidate report variables:

- what am I currently attending?
- what type am I searching for?
- have I found the target yet?
- how certain am I that I have inspected the right region?
- which regions remain unresolved?

Success criterion:

The same internal state that guides attention can also support explicit report about attentional status.

Current status in this repo:

- report probes are now implemented
- on the default run, controller state supports stronger readouts than observation alone for current search type and current attended cell
- the current report suite also now supports cumulative target-found reporting and unresolved-region reporting
- the latest default report marks `reportable_internal_content` as supported

Interpretation:

This is now positive evidence for **reportable internal content** in a bounded sense.

The current result supports structured reports about search type, attended cell, target-found status, and unresolved regions. It is still limited and benchmark-specific, but it now goes beyond the earlier partial decoder-only story.

## Stage 7: Natural-Language Reportability

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

Success criterion:

- natural-language reports track the true internal state
- reports change appropriately after cue switches or causal interventions
- reports outperform baselines that only see the current scene or glimpse
- reports remain faithful when observation and internal state are put under tension

Good implementation constraint:

- symbolic serialization of internal state is allowed only as a weak baseline
- the real target is a tokenized internal-state interface without pre-labeled variable names
- the language model should learn to attach labels to that state rather than receiving a hand-authored report template
- evaluation should compare tokenized-state reporting against both observation-only text generation and symbolic-dump baselines

Possible parallel implementation branch:

- a vision-language route may also be appropriate here, because the underlying problem is spatial
- in that branch, the system would render internal attention/self-model state as compact panels or overlays and ask a VLM to report current and remembered attended content
- this should still be held to the same anti-cheating standard:
  - explicit labeled visual dumps count only as weak baselines
  - the stronger target is minimally labeled visual internal-state rendering that still beats scene-only baselines
  - the VLM branch should be treated as complementary to, not a replacement for, the text-token branch

Current status in this repo:

- a natural-language reporting harness is now implemented
- the harness compares three conditions:
  - symbolic internal-state serialization as a weak baseline
  - tokenized internal-state reporting as the real Stage 7 target
  - observation-only reporting as a weaker external baseline
- the symbolic baseline is strong and currently achieves near-perfect or perfect structured reports on small evaluation slices
- the current Stage 7 evaluation is now stricter than a generic state-description task:
  - it asks for current attended content and remembered previous attended content
  - it is evaluated only on non-initial timesteps where previous-attention memory is genuinely required
- the tokenized internal-state condition is still not successful:
  - it can sometimes recover search type, current attended digit, target-match status, and compressed unresolved summaries
  - it still fails to report current attended location/content and remembered previous attended content reliably enough to beat observation-only baselines
- the current blocker is representational rather than infrastructural:
  - the tokenized interface is still not faithful enough on current and remembered attended semantic content to count as a Stage 7 success

Interpretation:

This would be a stronger and more legible form of reportability than the current structured probes, but it should only count if the language output is demonstrably faithful to the real internal state.

In particular, handing a language model a clean symbolic state dump would be too weak to count as the main result. The stronger test is whether language can latch onto tokenized internal state and learn stable labels for the model's own regulatory variables.

Current interpretation:

- Stage 7 is now implemented
- symbolic-state reportability is positive
- tokenized-state reportability is still not strong enough to count as supported
- therefore Stage 7 remains open

## Stage 8: Minimal Consciousness Interpretation

Question:

When, if ever, does the system become a plausible candidate for minimal consciousness-like content?

This stage should only be discussed after the earlier ones are established.

At minimum, the argument would require:

- closed-loop attention control
- evidence for an explicit model of attention
- evidence that this model is a model of the system’s own regulatory state
- structured, behaviorally accessible internal contents
- flexible report and action based on those contents

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
4. Branch A, Stage 4: self-modeling of attention
5. Branch B, Stage 5: flexible reallocation under changed priorities
6. Stage 6: reportable internal content
7. Stage 7: natural-language reportability
8. Stage 8: minimal consciousness interpretation

In other words, Stage 4 and Stage 5 should be read as parallel branches after Stage 3, not as a strict ordered ladder where Stage 4 must be completed before Stage 5 begins.

## Immediate Checklist

The highest-priority next experiments are:

- add stronger allocation-error and uncertainty reports that distinguish missing the target from not yet having inspected the right region
- improve the tokenized internal-state interface for Stage 7 so current and remembered attended content become more faithful than observation-only baselines
- add a parallel VLM-based Stage 7 path that tests minimally labeled visual internal-state renderings against scene-only and explicit-dump baselines
- evaluate natural-language reporting under cue switches and interventions once the tokenized condition is stable
- add plots or diagnostics for switched-cue and self-model trajectories

## Current Status

What is already supported:

- attention
- closed-loop attention control
- explicit attention modeling
- self-modeling of attention
- flexible reallocation under changed priorities, after mixed switched-cue training
- reportable internal content

What is not yet established:

- natural-language reportability grounded in tokenized internal state
- natural-language reportability grounded in minimally labeled visual internal-state renderings
- minimal consciousness-like content

What is currently unstable or tradeoff-limited:

- stronger native report behavior beyond the current explicit inspected-state scaffold
- richer uncertainty and allocation-error reporting beyond the current target-found and unresolved-region variables
- faithful natural-language reporting of current and remembered attended content from tokenized internal state

That distinction is important. The current result is already meaningful. The roadmap exists to keep the stronger claims disciplined and experimentally grounded.
