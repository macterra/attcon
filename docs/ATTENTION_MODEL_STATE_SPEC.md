# Attention-model state specification

Registered before new report fitting or held-out evaluation, 2026-09-26.

## Identified object

The existing `RecurrentAttentionController` contains a learned inspection model:

```
recurrent hidden state h
    -> hidden_self_model_head -> sigmoid -> m (25 inspection beliefs)
    -> policy_self_model_head(m) -> b (25 allocation-logit contributions)
attention logits = policy_head(h) + b
```

The tested attention-model state is **m**, not the whole recurrent state, scene,
answer distribution, or explicit inspection bookkeeping. Entry m[i] estimates
whether cell i has been inspected. Its training includes inspection supervision.
The head's state is transiently recomputed at each decision; recurrence carrying
its history resides in h. Its output b is an actual input to allocation selection.
This is a narrow learned model of attention history used for control, not a full
self/attention/object schema or a representation of all aspects of attention.

Source: `src/attcon/models.py`, `RecurrentAttentionController.forward`, immediately
after `hidden_self_model_override` and before softmax/reading the next glimpse.
The authoritative snapshot is before the current action. Physical history is
strictly the fixations before that snapshot. At step zero it is empty.

## Correction to the earlier inventory

`self_model_policy_feedback_weight = 0` disables an auxiliary training loss,
not the forward feedback path. The stored feedback weights are nonzero in the
original seed-7 checkpoint and in seeds 107, 207, 307. Task gradients can reach
them through attention selection. The earlier plan's suggestion that this meant
disabled model-to-policy feedback was incorrect. The evaluation tests its actual
causal contribution; a config label alone does not establish inactivity.

## Systems and source preservation

Use the already independently trained content-memory-v3 GRU controllers with
training seeds 107, 207, 307. Each has 32 hidden units and the original 5x5 task.
These checkpoints predate the new reporting protocol. Their existing task results
are known; all new report contexts and intervention cases are generated after
registration. This is a new reporting audit on old agents, not new agent training
or a claim of independent confirmation of their task performance.

Record source-checkpoint hashes, exact configs, state dictionaries, and the current
model/data source hashes. Archive only the controller and fitted reporter state
dictionaries needed for replay; no experiment source is overwritten. The seed-7
checkpoint is an inventory diagnostic only, not a selectively substituted test seed.

## Report fields and authoritative truth

| Field | Truth from the snapshot | Interpretation |
| --- | --- | --- |
| Inspected belief map | m[i] >= 0.5, threshold fixed | What the model represents as inspected; not necessarily actual history. |
| Model-preferred location | argmax b[i], lowest index tie | Location most promoted by this model's contribution; not necessarily actual focus. |
| Preference shift | Different preferred locations in two successive snapshots | Change in the model's allocation preference; temporal comparison uses two snapshots. |
| Belief persistence | Same cell retains its inspected classification across snapshots | Persistence of an inspection belief; not asserted to be experiential presence. |

All 25 beliefs and the preferred cell form the exact structured report. The
renderer names **model-preferred**, never claims that this is the actual focus.
There is no independently identified representation of phenomenal presence,
vividness, selfhood, color experience, or experienced continuity in this module.
Those candidate correspondences are marked unsupported, not supplied as labels.

## Interventions and causal identity

Use the existing `hidden_self_model_override` hook at step 3, preserving h, scene,
cue, answer content, and actual prior fixations. Apply donor-model states from
other test contexts, an isolated belief flip at a prechosen cell, and neutral
m=0.5 erasure. Feed the changed m through the same trained W to compute b and the
new attention. Evaluate each changed report against changed m, not the environment.
Restoring m must restore allocation and reports. Direct state swaps are within
observed marginal state support but may be inconsistent with recipient h; flips
and neutral erasure are explicitly synthetic interventions.

The actual attention distribution must respond to changing m for causal identity
support; reward improvement, task necessity, and superiority are not required.
The reporter is not allowed access to h, b, or physical history in the state-only
condition: it learns b's preferred cell from m. Direct rendering uses m and W as
an instrumentation baseline, not an independently learned phenomenology result.
