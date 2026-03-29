# Roadmap Toward Evidence of Minimal Consciousness

This document separates the current benchmark result from the larger research goal.

The project already supports a meaningful claim about **closed-loop attention control**. The longer-term goal is more ambitious: to build a sequence of experiments that could count as evidence for **minimal consciousness-like processing** in a tightly bounded, artificial system.

The safest way to approach that goal is as a staged program.

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
- on the current mixed-training default, when the direct target-attention supervision term is reduced or removed, useful reallocation becomes weaker and the zero-shaping condition is no longer positive

Interpretation:

This is stronger, but still preliminary, evidence for **explicit attention modeling**.

It is stronger than Stage 2 alone because it now includes predictive, intervention, and reduced-shaping analyses, but it is still not enough to establish a cleanly interpretable model of attention.

The latest tuning also shows a real tradeoff: training that improves switched-priority behavior strengthens Stage 5 but weakens the reduced-shaping part of Stage 3.

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

## Stage 4: Self-Modeling of Attention

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

## Stage 5: Flexible Reallocation Under Changed Priorities

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
- the current checkpoint redirects attention better than the baseline after a mid-episode cue change, but this improvement appears to trade off against reduced-shaping resilience

Interpretation:

This is now positive but still qualified evidence.

It suggests the current controller can be trained into more flexible reallocation under changed priorities, but that flexibility does not yet coexist cleanly with every earlier positive signal.

## Stage 6: Reportable Internal Content

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
- controller state also shows a small positive advantage for reporting whether target evidence is currently present in the glimpse

Interpretation:

This is preliminary evidence for **reportable internal content** in a limited sense.

The current result supports simple decoder-based reports from controller state, but not yet flexible, native report behavior from the model itself.

## Stage 7: Minimal Consciousness Interpretation

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

## Immediate Checklist

The highest-priority next experiments are:

- tune or curriculum-train mixed stationary/switching episodes to reduce the Stage 3 versus Stage 5 tradeoff
- add plots or diagnostics for switched-cue trajectories
- strengthen self-modeling-style report tasks beyond simple decoder probes

## Current Status

What is already supported:

- attention
- closed-loop attention control
- preliminary evidence for explicit attention modeling
- flexible reallocation under changed priorities, after mixed switched-cue training
- preliminary evidence for reportable internal content

What is not yet established:

- self-modeling of attention
- minimal consciousness-like content

What is currently unstable or tradeoff-limited:

- reduced-shaping resilience under the mixed switched-cue training default
- a training recipe that keeps Stage 3 and Stage 5 positive at the same time

That distinction is important. The current result is already meaningful. The roadmap exists to keep the stronger claims disciplined and experimentally grounded.
