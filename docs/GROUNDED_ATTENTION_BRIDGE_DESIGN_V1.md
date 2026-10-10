# Grounded attention/acquisition bridge: design v1

**Unfrozen proposal, not an executed study or evidence of qualia.** A separate subagent reviewed the architecture and proposed an outcome-grounded bridge; its [full review](../audits/grounded_attention_bridge_design_v1/review.md) and [provenance](../audits/grounded_attention_bridge_design_v1/metadata.json) are preserved. The [new sensory adapter protocol](SENSORY_TASK_ADAPTER_V1_PROTOCOL.md) is a prerequisite under development, not yet a qualified attention model. Human review is optional; shared AI commissioning/model limitations remain.

The existing acquisition simulator assigns quality and decay. The proposed bridge instead predicts events and correctness from executed sensory observations:

1. Four locations hold independently rendered orientations. Commands cause actual image delivery through an unknown location mapping. Each glimpse has fresh sensor noise. World labels remain available only to the evaluator/training targets.
2. A separately frozen sensory adapter supplies orientation evidence. Latest-glimpse memory stores evidence by observed location. An unseen location has a declared guess rule. In a static world, a stored answer's correctness does not decay merely because it ages. Refresh may improve or worsen it; a changing-world/accumulation extension needs its own specification.
3. Randomized training actions produce command, delivered location, sensory evidence and realized outcomes. The attention-control model learns command-to-delivery and current/post-command decision-correctness predictions. Correctness is an event evaluated against the world, not the adapter's confidence used as a target. Chance accuracy is distinct from failure to deliver an image.
4. For held-out command forecasts, clone the pre-command world and memory, execute each command with fresh noise and score actual decisions. The model never sees future branch images, world labels or post-command margins in a pre-command forecast. Split whole worlds, histories and branches together; branch means carry sampling uncertainty.
5. Bind content and modeled consequences by address, retaining their distinct provenance. Do not multiply content by correctness to manufacture clarity. The Controller chooses and executes commands for the declared task. The external Reporter describes represented content, estimated consequences and observed events; it does not control attention.

Qualification must include calibration by sensory condition/history, generalization/adaptation of command mappings, proper-score controls and executed Controller gains over random requests on the declared controllable subset. Difficult/zero-contrast/decoupled conditions remain visible. Exact numerical gates, split sizes, branch counts and stopping rules are not selected by this design review; freeze them before any bridge evaluation.

Selective represented-effect interventions should preserve physical mapping, sensory memory and history; physical mapping changes should initially leave model predictions intact and then induce revision through observed deliveries. Restore exact saved state and noise to test reversibility. Binding-address permutations preserve content and attention marginals, but a Controller wired to consume bindings can change by construction: that alone does not establish qualia or integration.

Use honest, equal-information sensory-confidence/history, unbound address-aware and joint learned predictors. A similarly informed feedforward predictor succeeding does not invalidate an attention model; withholding relevant addresses from controls would create an artificial advantage. Compare attention forecast/hidden, sensory, stored content and bound joint states under matched probe budgets in later appearance-source work. Include naturally reached states alongside arbitrary interventions to check distribution limits.

These engineering tests qualify a causal task substrate only. The full goal still requires an independently grounded appearance/report endpoint, dissociation from decision confidence, selective source intervention/restoration, factual report auditing, powered theory-facing confirmation and broader replication. Enhancement, attenuation, null and criterion-change accounts remain possible. No appearance target or human rating is supplied to the bridge as a training label.

## Prospective sensory-input amendment

The all-condition sensory adapter now has a completed **overall FAIL**: its
classification component passes the specified accuracy/null gates, but the
margin-only correctness instrument fails conditional calibration. Two separate
unblinded boundary reviews ([architecture amendment](../audits/grounded_attention_bridge_design_v1/review_addendum.md),
[independent sensory boundary](../audits/grounded_attention_bridge_design_v1/sensory_boundary_review.md))
agree that a discarded probability instrument is not logically required when no
consumer uses it. This supersedes the first proposal's requirement to qualify
that particular calibrator before using raw sensory evidence. Neither review
rescored the sensory study or qualified a learned bridge.

A fresh bridge may use frozen sensory hidden features and raw task logits from
actually delivered images; exclude the failed correctness probabilities from
inputs, targets, memory updates, Controller utility and Reporter state. Learn
delivery and task-success predictions from independently executed outcomes, then
qualify those forecasts on fresh whole-world/action data and against comparable
admissible source controls. Do not rename another margin-calibration fit an
attention model. Preserve action-contingency learning and revision requirements.
Calibration should be assessed relative to the declared information available
at forecast time. Hidden world orientation and unobserved future sensory strength
may explain diagnostic subgroup errors but cannot be supplied as oracle inputs;
branch means are sampled realized outcomes, not an exact probability known to
the forecaster. Freeze the distinction between qualification strata and latent
world diagnostic strata in the future protocol.

## Executable environment and memory prototype

[World primitives](../scripts/grounded_attention_world_v1.py) and
[four tests](../tests/test_grounded_attention_world.py) now implement independent
Bernoulli orientation labels at four locations, requested-to-delivered mappings,
actual sensory rendering with supplied noise, latest-glimpse raw-logit memory,
explicit horizontal unseen/tie guesses, static memory without invented decay,
and counterfactual command branches. Only command, observed location, raw sensory
logit and observation time are exported as historical observations. Evaluator
correctness is a separate target. A normalized logit transform is explicitly
named **uncalibrated orientation coordinates**, never a posterior or appearance.

The [prototype review](../audits/grounded_attention_bridge_design_v1/prototype_review.md)
and [correction verification](../audits/grounded_attention_bridge_design_v1/prototype_correction_verification.md)
preserve the initial naming issue and its precommit correction. Tests use toy
callbacks and supplied noise; they do not execute a qualified learned system.
Frozen dataclasses do not prevent callers mutating tensor contents. The branch
helper consumes caller noise and cannot certify its independence; an eventual
frozen experiment must sample and record the declared fresh-noise schedule.
A permissive sensor callback is not an information boundary: before learned
execution, freeze and audit an image-only sensory callback with no world-label
closure or future-state access. No attention model, Controller, reporting study
or appearance endpoint is qualified by these primitives.
