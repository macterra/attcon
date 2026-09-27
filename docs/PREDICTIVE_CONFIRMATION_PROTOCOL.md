# Predictive attention confirmation v1: internal model and control

Frozen 2026-09-27 after development seeds 811 and 821, before any confirmation
training. Seed 831 is an additional development replication, not confirmation.
Language pilot is in progress; this protocol freezes only internal prediction and
causal consumption. Language and independent report-character confirmation must
be frozen separately after the pilot. Passing this protocol alone is not the goal.

Architecture and training are unchanged from `predictive_attention.py` and the
1200-update pilot: GRU64, Adam .003, gradient clipping 1, 128 newly generated
12-step episodes each update, seed*100000+update. Confirmation seeds: 901, 911,
921. Evaluation: seed+950000000, 1024 fresh 16-step episodes, exclude first four
steps from predictive metrics. No early stopping or selection of checkpoints.

Every seed must satisfy: allocation and command-effect accuracy >=.99; access
MAE <=.04; controllable-channel identification >=.99; reconstruction Brier no
more than .005 above the oracle simulator probabilities. Report worst access
error without discarding outliers. The oracle is not a learned model control.

At boundary 7, query slot cycles 0–3 in the controlled channel. At least .99 of
commands must select the queried physical slot. Rotate A's command-effect table
one command, preserving all other fields/world/history: at least .99 of commands
must rotate accordingly. Restoration and preservation of allocation/access must
be exact. These interventions are synthetic and need not be states the GRU would
produce. They test causal use and subsequent faithful reporting, not ecological
frequency. Also intervene on access only and allocation only; the current policy
uses the effect table, so these changes must leave its commands invariant. This
explicitly limits the causal claim: access/retention heads share the model but
are not independently shown necessary for control.

Re-evaluate the saved checkpoints exactly; record source and checkpoint hashes.
Run meaningful unit tests and the existing suite. Preserve all outcomes. After a
failure, a changed architecture or training procedure needs a new protocol and
fresh seeds. No prediction gate may be relabeled as evidence of qualia.
