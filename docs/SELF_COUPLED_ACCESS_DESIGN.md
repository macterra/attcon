# Self-coupled access: design note

**Draft, 2026-10-03. Not frozen; nothing built or run.**

## Why

The [specificity test](SPECIFICITY_RESULTS.md) found no difference between reports
of the attention model and reports of the same state described as an external
spotlight and camera. In the current architecture there is no difference to find:
the system's own content never depends on attention, so its "attention model" is
structurally a model of some process that does not gate its contents.

On the theory under test (close to Webb and Graziano's attention schema), the
model that grounds awareness reports is a model of the **system's own access to
its contents**: the relation by which this system holds an object's content to a
degree that depends on its attention. That relation is the candidate
discriminating feature. It must be present in the state, not only in labels,
because the reporter reports whatever structure it is given.

## Pre-specified report feature: self-coupled access

A report asserts that **its own** identification of, or certainty about, a specific
object depends on its attention or access to that object, now or under
redirection. For example: "I can make out the red circle because I am attending to
it; the object on the left is fading because I am not." A correct assertion must
match the state's direction for that object. First-person wording alone does not
qualify, and neither does a statement about an external camera or observer.

## Conditions (same episodes, same attention-model values, same labels)

| Condition | System's own content certainty |
|---|---|
| `coupled` | Encoder distribution mixed toward uniform by the actual access process the attention model forecasts: refreshed by glimpses, decaying when unattended. |
| `decoupled_matched` | Same per-object degradation amounts, permuted across objects within the episode so their marginal uncertainty is matched but no longer tracks modeled access. |
| `current` | v5 contents: full certainty once observed. |

`decoupled_matched` is the key control. It holds overall uncertainty fixed and
breaks only the coupling between the attention model and the system's own
content. All three conditions use the v5 glossary; a neutral-label replicate of
`coupled` vs `decoupled_matched` checks that any difference survives without
attention vocabulary.

## Predictions

- Theory: correct self-coupled access assertions are common in `coupled` and rare
  in `decoupled_matched`, with the same labels in both.
- Disconfirming: similar rates, meaning the reporter asserts the coupling whether
  or not the state contains it, or never asserts it.

## What it requires

1. A content-gating rule in the bound state. This is an engineered architectural
   addition, justified because the theory's distinguishing relation is otherwise
   absent; it does not involve retraining the encoder or the attention model.
2. Extractor v11: v10 plus one structure field for self-coupled access, with new
   positive and negative fixtures, validated before use. All v10 fields unchanged.
3. Scale like specificity_v1: fresh seeds, 24 episodes, about 96–120 reports, cost
   of the same order (tens of dollars at most).

## Open decisions

1. **Gate by the actual access process or by the model's forecast?** Gating by the
   forecast makes the model constitutive of content availability, closer to the
   theory's claim that the model regulates access. It also allows a
   model-only intervention test: change the forecast, and content availability and
   reports should follow. Gating by the actual process is less engineered.
2. **Primary threshold.** Default: same rule as specificity_v1 (25 points, one-sided
   exact paired p < 0.05) on `coupled` vs `decoupled_matched`.

## Limits

A positive result shows that reports pick up self-coupled access when the state
contains it and not when it doesn't. A critic can still call the coupling
engineered and the reporter a reader of supplied structure. A native reporter
trained without phenomenological targets remains the stronger later test. No
result here would establish subjective experience.
