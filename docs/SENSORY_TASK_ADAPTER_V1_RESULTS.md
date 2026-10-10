# All-condition sensory task adapter v1 results

**Overall registered engineering result: FAIL for all three models.** Orientation decisions improve substantially, while the stand-alone correctness map still fails conditional calibration. The [frozen protocol](SENSORY_TASK_ADAPTER_V1_PROTOCOL.md), failed predecessor and complete raw outputs remain unchanged. No attention model, Controller, Reporter or appearance endpoint is tested by this study.

## Observed classification and calibration

Each frozen visual encoder receives 1,024 training bases, 256 calibration bases and 512 evaluation bases, with one independently assigned orientation and six contrast variants per base. The new 32-unit nonlinear readout trains on all six conditions for a fixed 1,500 full-batch steps. All 1,536 evaluation bases and 9,216 dependent images remain. No evaluation-based tuning, selected seed, new report or participant target is used.

| Visual seed | Highest-contrast hidden accuracy | Conditional base-bootstrap 95% | Zero-contrast hidden accuracy | Hidden pooled Brier / calibration-constant Brier | Overall result |
|---|---:|---:|---:|---:|---|
| 1301 | 99.80% | 99.41–100% | 53.32% | .0597 / .0851 | FAIL |
| 1311 | 99.80% | 99.41–100% | 48.83% | .0671 / .0931 | FAIL |
| 1321 | 99.41% | 98.63–100% | 52.93% | .0647 / .0873 | FAIL |

Physical highest-contrast controls achieve 100%; their primary zero-contrast results fall within the specified chance tolerance. Separately executed identical-null features, scores and probabilities match exactly and yield 50% accuracy for every source. Highest-contrast discrimination, gross-null, duplicate-null and pooled-Brier gates pass in every model. The two conditional calibration gates fail in every model.

| Synthetic contrast level | Hidden accuracy 1301 / 1311 / 1321 | Hidden ten-bin ECE 1301 / 1311 / 1321 |
|---|---|---|
| Zero | .5332 / .4883 / .5293 | .1143 / .1352 / .1185 |
| 2.2 | .9160 / .8965 / .9121 | .0964 / .1075 / .1163 |
| 3.9 | .9980 / 1.0000 / .9922 | .0368 / .0564 / .0557 |
| 5.0 | .9961 / .9980 / .9980 | .0139 / .0273 / .0336 |
| 5.5 | .9961 / .9961 / .9961 | .0082 / .0206 / .0228 |
| 6.9 | .9980 / .9980 / .9941 | .0078 / .0114 / .0161 |

These level names describe the synthetic conversion, not verified participant stimuli. The fixed ECE requirement is ≤.10 at every contrast. Zero-contrast ECE exceeds that requirement in all models. At the lowest positive level, mean predicted correctness minus actual correctness is −.0964, −.1075 and −.1163, outside the fixed ±.075 residual gate. Zero residual also fails for 1311 and 1321. Pooled ECE remains low (.0201, .0178, .0206), again demonstrating why a pooled assessment is insufficient. Probabilities are not qualified for use as an acquisition model, participant confidence, clarity or appearance.

## Source controls and numerical limits

Frozen legacy hidden readouts/calibrators, applied unchanged to this new evaluation set, achieve 91.41%, 93.95%, 94.34% at highest contrast. This is a descriptive comparator. Training count, architecture, condition coverage and numeric scaling change together, so improvement cannot be attributed to coverage alone; the predecessor's original failed verdict is preserved.

New categorical-probability and confidence readouts produce a constant orientation and exactly 50% accuracy at every level, with final BCE about .693147. This does **not** establish absent information. Every dimension falls below the declared 1e-6 RMS floor, so its scale becomes1 and extremely small variation is not amplified. For seed1321 float32 squared RMS underflows to zero, while float64 RMS and distinct-value checks show variation persists. The legacy double-precision scaling differs materially. Equal width/steps/data also do not equal parameter count or information: hidden/logit/category/confidence/physical readouts contain 2,113/321/321/193/97 parameters. No attention-source specificity is inferred from this control outcome.

## Reconstruction, review and preservation

[Results](../audits/sensory_task_adapter_v1/results.json), three compressed raw state archives, coefficients, complete reliability/stratified assessments, loss summaries and bootstrap samples are saved. [Completion hashes](../audits/sensory_task_adapter_v1/completion.json), source snapshots and legacy-archive hashes preserve provenance. Local freeze commit `7f80599` precedes the single reserved run. Both execution and full fixed-budget training/calibration replay reach terminal success; the [replay receipt](../audits/sensory_task_adapter_v1/verification.json) confirms exact reconstruction.

The [independent subagent audit](../audits/sensory_task_adapter_v1/results_review/review.md) uses separately written executable code to regenerate inputs/features, retrain all 15 adapters, refit calibrators, apply legacy fits and recompute all assessments. Archived tensor differences are exactly zero; metrics and gates agree within 2e-12. Its [numerical feature diagnostics](../audits/sensory_task_adapter_v1/results_review/feature_scale_diagnostics.json) preserve the floor/underflow limitations. All 21 predecessor files remain byte-identical to the freeze commit. Five adapter tests and four separately scoped world-prototype tests pass. AI commissioning/shared-model limits and unblinded result-review exposure remain disclosed; this is not an independent empirical replication or human appearance study.

## Prospective boundary decision

The protocol's pass permission is not triggered. Two subsequent unblinded methodological reviews instead support a **different prospective information path**: the [grounded attention bridge](GROUNDED_ATTENTION_BRIDGE_DESIGN_V1.md) may use the explicitly partial frozen sensory-classification component and raw evidence while excluding the failed probability instrument entirely. It must learn action/delivery/realized-success predictions from fresh executed outcomes and qualify those forecasts independently. This changes a future architecture dependency, not the frozen result or gate. No learned bridge is yet qualified; no replacement calibrator is fitted here.

The broader goal remains unachieved. Qualified appearance/report endpoints, source controls, selective attention-model interventions and restoration, factual reporting, theory-facing confirmation and replication remain outstanding. Actual attention-manipulated participant archives are still unqualified. The next experiment is grounded action/outcome learning, not another attempt to rename classifier confidence clarity.

Milestones are saved locally. Authenticated `gh` cannot reach GitHub, so these commits have not been pushed.
