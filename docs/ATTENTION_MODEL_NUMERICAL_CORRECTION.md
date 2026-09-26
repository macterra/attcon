# Numerical evaluation correction

After the three registered runs, seed 207's preference-shift score was stored as
float32 `0.949999988079071`. The exact count is 1216/1280 = 0.95, which meets the
registered inclusive 95% threshold. Float32 rounding incorrectly failed it.

Compute the three correspondence fractions in float64, without changing targets,
predictions, data, thresholds, model weights, reporter fitting, or selection.
Reevaluate all three saved runs and retain the first evaluation JSON files in
`audits/attention_model/numerical_correction/`. Only the seed-207 shift gate changes;
full reporting fidelity still fails in every seed. This is numerical correction,
not threshold relaxation or an additional experiment.

The reevaluation also explicitly records the definitionally exact direct-telemetry
baseline already specified by the protocol. That is a measurement/format baseline,
not independent evidence for phenomenology. No fitting is performed in reevaluation.
