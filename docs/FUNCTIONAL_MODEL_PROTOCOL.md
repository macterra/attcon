# Learned functional model: development protocol

2026-10-04, before execution. Integrate the existing frozen visual encoders
1301/1311/1321 and attention predictors 901/911/921 with the functional comparison.
This is development, not a powered theory confirmation or human-rated report study.

For each model pair, 128 fresh paired scenes have two channels containing the same
four objects. Both channels begin without acquired content. They share commands,
replay motion, and per-step sensor quality. In one history commands select channel
0; in the other they select channel 1. The task readout always uses channel 0.
Thus one condition controls task-used content and the other controls a separate
buffer. Independence here is a specified readout boundary, not an empirical claim
about a phenomenal boundary. Run eight observed steps, then execute each alternative
command from the same physical state with common next-step sensor quality 0.8.

The physical recovery process uses the original decay (0.75) and acquisition rule.
The existing predictor receives only command, observed allocation and glimpses;
it never receives condition identity or readout routing. The frozen encoder reads
the patches. A buffer distribution mixes encoded contents with uniform uncertainty
using recovery probability. This bridge is explicit architecture, not learned
phenomenology. The fixed task readout copies the task buffer distribution for a
queried object; the Controller chooses sampling commands from its modeled effects.

Predict alternatives from the learned effect table and feed each forecast allocation,
known sensor quality, and command into the recurrent predictor from the shared
history. This is model-based simulation, not a roll-forward using simulator owner
truth. Compare the resulting recovery forecasts against physically executed alternatives.

Engineering minima, inherited where applicable from the predictive study: at
least 99% allocation/effect accuracy and controlled-channel identification, at most
0.04 recovery MAE for current and counterfactual states. Report all seeds separately.
Check physical readout routing, matching commands/noise, and exact source replay.
These minima are prerequisites for an interpretable reporter pilot, not evidence
for qualia. Failure remains archived and requires a new development version.

Crossed interventions: swap only the represented command-effect channels, leaving
physical outcomes fixed; restore exactly. Separately swap the physical controlled
channel without new observations, preserving the pre-intervention represented state.
Record errors before and after one new observation; do not require instant model
awareness of an unobserved change or arbitrary quick adaptation.

Reporter-visible schema uses neutral channel/node IDs, original color/shape names,
current and predicted categorical distributions, executed-command histories, and
output distributions. Condition identity, true controlled channel, checkpoint seed,
expected answers and interventions' explanatory names remain evaluator-only.
Predicted and observed fields are explicitly distinguished. Permutations must change
only presentation and preserve all model values. An initial-observation-only control
is identical across conditions. No API calls under this protocol.
