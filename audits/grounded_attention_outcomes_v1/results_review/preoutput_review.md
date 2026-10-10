# Independent source review before the registered run

Reviewer: an unblinded AI subagent. No human review or independence from the AI
system is claimed. This review inspected source and generated fixtures only;
no production checkpoint inference or collected study outcome was used.

The current evidence export contains the latest delivered hidden representation,
raw orientation logit and observed flag by location, plus the latest observed
command-delivery row. World labels, contrast, nuisance variables, branch pixels,
future sensor features and correctness are not inputs to `predict`. Correctness
is supplied only to the training, calibration and evaluator functions. The
failed predecessor sensory confidence calibrators are not consumed.

The world uses independent orientation draws rather than a fixed two-horizontal,
two-vertical balance. Nuisance, label, mapping, exploration and branch-noise RNG
streams have distinct seeds across the three study splits and visual models.
All four executed commands map through an actual permutation, so fresh outcomes
can be regrouped by delivered location without an unexecuted target. A branch
updates only that location's stored decision; current correctness remains a
single target per snapshot. Repeated fresh outcomes are scored as Bernoulli
outcomes, and bootstrap units keep whole worlds together.

The structured forecast imposes a specific update-rule decomposition; its
conditional-independence assumptions are tested by the frozen actual-outcome
gates, not guaranteed by the architecture. The joint model receives the same
information with a larger parameter count and need not lose. Sensory/history
ablations remove information and are not selective architectural lesions.
The Controller consumes model predictions, but its query is sampled before
using outcomes or model utility. The Reporter is absent from this study.

Represented row rotation checks causal use of route forecasts and actual task
cost. Physical remapping checks revision after new observed deliveries; the
restoration gate restores model inputs, not an independently executed physical
trajectory. Contradictory intermediate history rows lie outside the static
training family. None of these tests is a subjective appearance endpoint.

An initial runner/assessment signature mismatch was reported before freeze and
corrected: Controller now receives the explicit model kind, avoiding accidental
primary gates on information ablations. No production outcome was available
when that source correction was made.

Generated-fixture validation compared the independent audit formulas with
production formulas for all four model kinds, Controller/bootstrap/corruption,
probability metrics, network forwards, independent world geometry and RGB
rendering. It performed 12,700 value checks with zero discrepancy. This validates
the audit implementation on synthetic fixtures; it is not a study result.

`audit_code.py` reconstructs archived worlds, pixels, supplied noise, frozen
sensory decoding, histories, memory, executed labels, normalization, calibration,
network forecasts, scores, Controller decisions, world bootstrap intervals,
interventions, remap states and gates without importing production assessment
or collection modules. It checks archive/source hashes. It does not refit the
outcome networks; the separately recorded fixed-budget replay does that.

No remaining source blocker was found in the reviewed design. Qualification is
pending the reserved attempt and independent archive reconstruction. Passing
would establish a functional acquisition-outcome prerequisite on synthetic
worlds; the attention-model-as-qualia hypothesis remains unresolved.
