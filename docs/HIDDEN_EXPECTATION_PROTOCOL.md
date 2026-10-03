# Hidden-expectation intervention diagnostic

Frozen before execution, 2026-09-29. Exploratory follow-up to the
[consistency diagnostic](CONSISTENCY_DIAGNOSTIC_RESULTS.md); no consciousness gate.

For each frozen checkpoint 901/911/921, generate 512 paired eight-step histories
sharing commands, replay trajectories, and glimpse quality, but with opposite
controlled channels. Generate two four-step continuations, one for each channel
owner. Within a continuation, all hidden-state treatments receive exactly the same
observations. Histories consistent and inconsistent with the continuation are
therefore crossed with common sensory evidence.

Use the existing linear command-effect head as a decoder. Decompose the difference
between paired hidden states into that head's row space and null space using SVD.
Compare intact matched history, whole mismatched history, row-space replacement,
null-space replacement, and a fixed-seed random perturbation with the same norm as
the row-space replacement. Do not fit a new head or optimize interventions on
outcomes. Record decoded effects, allocation/access collateral changes, hidden
update magnitudes, and subsequent command predictions under identical inputs.

The null-space intervention must preserve pre-input effect logits within numerical
tolerance; row-space replacement must reproduce the donor effect logits. Check
both on every batch. Record per-seed results and source/checkpoint hashes, preserve
all original models and studies, and verify deterministic replay.

A decoded expectation influencing later estimates establishes a causal pathway,
not necessarily a prediction-error computation. A null-space effect would show
that the decoded forecast is not a sufficient summary of recurrent history.
Collateral changes preclude calling an intervention expectation-specific. This
assay contains no visual bound-state update, so cannot establish that representation
as a Modeler-schema comparison medium. Do not add a new comparator architecture
merely because these diagnostics cannot resolve internal computation.
