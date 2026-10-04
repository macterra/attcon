# Learned-model integration development results

2026-10-04. The existing visual encoders and recurrent attention predictors now
support the paired functional-access comparison. All six model/condition combinations
pass the [prespecified engineering minima](FUNCTIONAL_MODEL_PROTOCOL.md).
No new prose, human ratings, or consciousness confirmation have been produced.

Three frozen model pairs were tested on 128 paired scenes each (384 underlying
paired scenes). One channel's commands govern task-used content; in the paired
condition they govern a separate buffer. Both begin without acquired content and
share commands, sensor quality, and scene contents. A fixed task readout uses
channel 0. The readout boundary is a functional definition, not a demonstrated
boundary of subjective experience.

| Metric | Range across six model/condition combinations | Minimum |
|---|---:|---:|
| Current allocation accuracy | 100% | 99% |
| Command-effect accuracy | 100% | 99% |
| Controlled-channel identification | 100% | 99% |
| Current recovery MAE | 0.02874–0.03111 | at most 0.04 |
| Counterfactual recovery MAE | 0.02564–0.02754 | at most 0.04 |

Counterfactual prediction uses the learned effect forecast and recurrent state,
not the simulator's channel-owner truth. The next-step sensor quality is a known,
shared 0.8 in this assay. Predicted content probabilities combine learned visual
distributions with recovery forecasts through an explicit uniform-uncertainty
bridge. The bridge and fixed readout remain engineered choices. This integration
uses no phenomenological training labels and requires no retraining.

Swapping only represented command-effect channels changes predicted recovery by
mean absolute 0.24862–0.26129. Restoring them reproduces the predictions exactly.
Physical outcomes are computed separately and remain unchanged. Switching the
physical channel instead preserves the pre-observation model. After one new
observation its recovery error is 0.05287–0.05997. The archived pre-observation
error aggregates all four alternatives; this post-observation value concerns one
executed command, so those two aggregates must not be interpreted as a matched
quantitative improvement contrast. No adaptation acceptance claim uses that comparison.

## Reporting interface and remaining work

Twenty-four illustrative records are archived using neutral node/command identifiers,
observed histories, and clearly marked predicted current and alternative-command
distributions. No condition name, physical owner, or expected classification enters
the payload. Evaluator-only metadata is in the containing archive record and must
be stripped before any reporter request. The fixed output-node role is disclosed;
it supplies legitimate evidence of which information serves the task, not a self label.

The Controller's available action-selection method uses modeled command effects.
This milestone validates its inputs; it does not claim a new autonomous closed-loop
Controller evaluation. The external reporter remains separate. A neutral reporter
pilot must next demonstrate factual readout and attribution based on this evidence,
then test identifier remapping, missing information, and conflicting labels.
Independent rubric review and a powered confirmation remain outstanding.

## Verification

Archive: `audits/functional_model_v1/`, including all tensor states, per-condition
metrics, checkpoint/source hashes, and software version. Reproduce using
`.venv/bin/python scripts/verify_functional_model.py`; it checks every physical and
modeled tensor and all 24 rendered records, not only summary scores.
Three integration tests cover physical selectivity, paired sensor quality, initial
uncertainty, distribution normalization, and readout routing.
