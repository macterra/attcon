# Functional wiring development results

2026-10-04. **Engineering milestone only; no new consciousness-related reports.**

The [specified fixture](FUNCTIONAL_WIRING_DESIGN.md) distinguishes real command
effects on a task's content buffer from effects on a separate buffer. Across 128
paired seeds and all six neutral node permutations, a rule-based observer correctly
identifies all 768 own-access, 768 external-device, and 768 decoupled presentations.
These are 128 underlying paired cases, not 2,304 independent experiments.

All 768 initial-state triplets are identical across conditions. Their condition is
unidentifiable without command consequences. This provides a concrete negative
control against asking a reporter to infer hidden causal wiring from identical data.

Three tests verify physical intervention effects, decision-readout routing, initial
indistinguishability, and invariance under identifier permutations. Exact archive
replay and hashes are checked separately. Results are in
`audits/functional_wiring_v1/summary.json`; full cases include evaluator-only truth
alongside neutral payloads. Future reporter requests must use only `payload`, never
the containing record or archive filename.

The fixture uses synthetic categorical evidence and a fixed readout. It does not
yet include the trained visual encoder, learned attention model, adaptive Controller,
or language reporter. The observer is a sanity check of identifiability, not evidence
for qualia. Integrating a learned represented state and distinguishing state-only
from world-only interventions are the next engineering steps.

The [draft independent-review rubric](FUNCTIONAL_REPORT_RUBRIC.md) and blank
[rating form](FUNCTIONAL_REPORT_RATING_FORM.csv) are prepared. Independent review,
power analysis, frozen confirmation, and human-rated results remain outstanding.
