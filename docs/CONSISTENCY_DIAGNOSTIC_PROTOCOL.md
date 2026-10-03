# Existing attention-model consistency diagnostic

Frozen before execution, 2026-09-29. This is an exploratory mechanism diagnostic,
not a new confirmation of consciousness or a replacement for reporting criteria.

Use the existing attention checkpoints 901, 911, 921, with 512 fresh simulated
episodes per checkpoint, 24 steps, and a transition before step 8. Random commands
are shared between conditions to avoid action-induced observation confounds.
Compare unchanged dynamics, a swap of the controlled channel, and a two-slot
offset of the controlled channel's command mapping. The latter mapping was absent
from training; within-episode dynamics changes were also absent from training.

For each condition compare intact recurrent history, history reset just before
the transition, and history shuffled between episodes at that point. Observations
remain identical across these three treatments. Record next-command prediction
accuracy, controlled-channel command success, and executed-command prediction
error before and after each observation. Compare current forecasts with the
physical next-step command table, respecting the transition boundary.

Also corrupt only a decoded forecast while holding history and the next input
fixed. Verify whether subsequent forecasts change. This isolates direct forecast
feedback; it cannot exclude a comparison encoded within recurrent hidden state.
Check unchanged checkpoint parameters, pre-transition equivalence, and common
observations. Archive per-step aggregates, configuration, and source/checkpoint
hashes. No retraining, API calls, or new phenomenological reports.

Interpretation: recovery shows adaptation of state estimates, not necessarily an
explicit mismatch computation or parameter learning. History effects show causal
use of memory but do not by themselves identify prediction-error neurons or an
MSTC comparison representation. Failure on remapped commands limits adaptation
in this assay; it does not falsify MSTC. Trace the bound-state code separately to
determine whether this feedback passes through the reported representation.
