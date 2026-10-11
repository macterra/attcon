# First actual numerical review: failed verification

2026-10-11. Executed by `/root/spatial_native_numerical_execution_review` after the
coordinator confirmed that the first actual results were committed. This is a
separate task allocation within the same AI model team with inherited hypothesis
and project context. It is not independent-family evidence, human endorsement or
empirical replication. The frozen verifier was prepared by a different task; its
unchanged metadata `reviewer` field retains that preparer's identifier,
`/root/spatial_native_task_numerical_auditor`. That label does not identify a new
independent model family or the actual executor of this pass.

The first verifier attempt exited **1**, status **failed_first_verification**,
with **106 failed checks out of 206,070**, and no exception. All three model
engineering feasibility gates reconstructed as passing. Numerical validation
remains failed; utility agreement does not override the first finding.

## Preservation and exact execution

First actual results commit: `614e74ae74ccf1d553f6fc6e0ec318f057a94ed8`.
Registration commit: `3e92bd1d7b9948e27581f5845a8835e537b8feb2`.
Preservation checkout: `/home/david/dev/attcon`.
The shell handle was `63388`; it is terminal, exit 1. The verifier receipt began
at `2026-10-11T13:12:39.232565+00:00`.

```bash
set -o noclobber
/home/david/dev/attcon/.venv/bin/python scripts/verify_spatial_native_task_v1.py --registration audits/spatial_native_task_v1/registration.json --execution audits/spatial_native_task_v1/execution.json --output-dir audits/spatial_native_task_v1/run --preservation-checkout /home/david/dev/attcon --results-commit 614e74ae74ccf1d553f6fc6e0ec318f057a94ed8 --report audits/spatial_native_task_v1/numerical_review/metadata.json --ack-first-results-preserved > audits/spatial_native_task_v1/numerical_review/first_run.log 2>&1
```

The stdout/stderr log and exclusive verifier metadata were retained unchanged.
For publication, the coordinator will preserve the exact metadata bytes in
[`metadata.json.gz`](metadata.json.gz), with a separate manifest binding raw and
compressed SHA256 hashes. Raw `metadata.json` remains unchanged locally; only the
compressed transport artifact is committed. This is byte-identical transport
compression, with no audit rerun, changed finding, filtered checks or replacement
metadata. Decompression must recover the raw SHA256 below and all 206,070 checks.
Transport creation/validation is coordinator-owned, outside this reviewer's three
owned files.
No verifier rerun, fit replay, new world generation, fit retry, source fix,
checkpoint mutation, Git mutation, GitHub/network action or Reporter call was
performed by this reviewer. Subsequent commands only read static source or
summarized the already emitted metadata; they did not rerun numerical reference
or model calculations. Owned writes are only this review, the first log, and the
first metadata.

- verifier SHA256: `83b6c370c0ee19b6a77749fc01a0e06cca196e1e57c0b3786c8ec6084f4988d8`.
- first log SHA256: `38ca12788688aac907922d580bc6a81e1274b19eef6cc2b0a19c7f8102249aca`.
- first metadata SHA256: `6010154b9502daef7f223ac3da4ac9c7854c4c9bc15436b8d987d9a5c93dc338`.

## Scope that completed

The verifier completed all three archived model reconstructions. Current source,
checkpoint and review byte bindings passed (33 checks); actual-result byte
bindings passed (7 checks); preserved-commit bytes and ancestry checks passed
(64 checks). Receipt inventory, namespaces, world layouts, class targets,
fit-only coordinates, initial weights, recorded losses, retained final logits,
frozen-source snapshots and registered checkpoint comparisons were checked.
No source/archive/checkpoint binding failure interrupted reconstruction.

The complete primary denominator is 2,304 item/operator opportunities and 4,608
role actions. Reconstructed duplicate-control role-query counts are sham 1,536,
matched exogenous T2 1,536, restoration 1,536, outer-address permutations 4,608,
and synthetic site permutations 4,608. Wrong, unchanged, tied and erased-content
opportunities remained included.

All 1,599 per-model summary checks passed. All 135 primary assessments based on
the actual emitted fields passed. All 34,560 separately labelled production
Controller replay checks passed. Each encoder passed the exact archived T4
uniform-field check, archived T2 maximum-tie lower bound, and archived T2 zero
margin check. Exact copied/restored/driver-parity outputs, address/site-carriage
controls and unique-maximum action-equivariance checks did not appear in the
failed-check ledger. Checkpoint and retained after/before frozen-weight equality
checks passed for all three encoders.

## Utility findings, conditional on the retained first run

These values reconstruct the emitted native fields/actions after their logits
and fields were compared with independent reference math. They retain the
complete fixed denominator; they are development task/content results.

| Encoder | T1 red targeting | T1 blue targeting | T1 pooled targeting | T1 coarse correct | Raw targeting | T2 pooled targeting | T2 coarse correct | T2 changed with both coarse answers correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1301 | 0.98828125 | 0.984375 | 0.986328125 | 1 | 1 | 0.2265625 | 0.9921875 | 0.9921875 |
| 1311 | 1 | 0.9375 | 0.96875 | 1 | 1 | 0.228515625 | 1 | 1 |
| 1321 | 1 | 0.99609375 | 0.998046875 | 1 | 1 | 0.2421875 | 1 | 1 |

All nine prespecified utility gate comparisons passed: T1 pooled targeting >= .50,
T1 coarse accuracy >= .75, and raw-grid targeting >= .95 for each encoder.
T4 targeting is exactly .0625, coarse correctness 0, and coarse unknown frequency
1 for all three encoders. Those are imposed uniform-field/tie-policy effects.
T2 targeting remains below its structural .25 ceiling. World bootstrap intervals,
per-world values and role-specific fields remain in the metadata; they are
conditional on these fixed models and shared deterministic layout support.

## Failure classification

There are **35 independent-reference discrete-policy discrepancies**, **10
original-path category-logit tolerance failures**, **60 original-path control-delta
tolerance failures**, and **one failed final invariant-verdict check**.

The 35 discrete failures compare selected cells, tie counts, or one synthetic
fine-correct value against fields produced by the independent reference head.
The same retained primary emitted fields pass exact policy reassessment and
Controller replay, and the actual T4 fields pass exact uniformity. Independent
reference logits differ slightly while their tolerance comparisons pass: primary
T2 maximum differences are 4.76837158203125e-7 for all models; T4 maximum differences
are 2.384185791015625e-7, 1.1920928955078125e-7, and 2.384185791015625e-7 for seeds
1301, 1311, and 1321. Exact argmax/tie comparisons can change under these small
perturbations. Static source inspection also shows differing arithmetic paths:
production uses `nn.Linear`, itemwise Controller consumption and emitted fields;
reference math uses batched `x @ weight.T + bias` and float64 field normalization.
The evidence therefore locates these failures in reference-versus-production
arithmetic/discrete-policy agreement. It does **not** show an actual emitted T4
uniformity failure. The precise computational cause has not been isolated by an
additional execution, and these checks remain failed.

The 70 control/output-delta comparisons violate the declared atol=1e-6,
rtol=1e-5 tolerance. Static source inspection shows production encoder suffix
and orientation modules versus independently reconstructed matmul-plus-bias
expressions. Small float32 differences and subtraction of independently computed
baselines are consistent with arithmetic-order roundoff, but this is an inference,
not a demonstrated diagnosis. These are **unresolved numerical invariant
comparisons**, not dismissed production errors and not accepted passes. The largest
failed absolute difference is 0.0001373291015625 in orientation-logit deltas.
Control-delta failures by quantity are:

| Quantity | Failed checks | Largest absolute difference | Total tolerance-violating elements across comparisons |
|---|---:|---:|---:|
| Category logits delta | 15 | 7.62939453125e-6 | 281 |
| Hidden64 delta | 15 | 7.62939453125e-6 | 411 |
| Orientation logits delta | 15 | 0.0001373291015625 | 780 |
| Standardized hidden64 delta | 15 | 4.57763671875e-5 | 1,979 |

The final verdict comparison fails because production reported all software
invariants passing while this verifier had 105 previous failed checks. That is
the required failed numerical-review verdict, not a distinct newly measured
production fault. There is no demonstrated production-origin invariant failure
in this pass; there are unresolved independent reconstruction discrepancies that
prevent numerical approval.

## Complete failed-check ledger

Actual means retained production output; expected means independent reference
output. Every tensor comparison below has matching actual/expected shapes and
dtypes and finite values on both sides. `max abs` and `unequal` quantify their
actual/expected difference; complete individual shapes, dtypes and tolerances are
preserved in `metadata.json`. `Exact` uses zero tolerance. `Tol` uses
atol=1e-6, rtol=1e-5. `Violations` counts the recorded tolerance-violating elements;
for exact integer/bool comparisons in this ledger it also equals the unequal count.
`R` denotes reference arithmetic/discrete-policy discrepancy; `U` denotes an
unresolved control arithmetic comparison; `V` denotes the failed final verdict.
Rows are comparisons, not independent empirical observations; duplicate-control
comparisons can repeat the same item discrepancy.

| Failed check | Class | Comparison | Max abs | Unequal | Violations |
|---|---|---|---:|---:|---:|
| `1301/original_controls/category_logits` | U | Tol | 3.81469726562e-06 | 1034 | 3 |
| `1301/fit_controls/category_logits` | U | Tol | 7.62939453125e-06 | 2110 | 1 |
| `1301/T1/original_path/category_logits` | U | Tol | 3.81469726562e-06 | 1034 | 3 |
| `1301/synthetic_site_permutation/T1/original_path/category_logits` | U | Tol | 3.81469726562e-06 | 1078 | 2 |
| `1301/synthetic_site_permutation/T1/original_path_delta/category_logits` | U | Tol | 5.72204589844e-06 | 1321 | 8 |
| `1301/synthetic_site_permutation/T1/original_path_delta/orientation_hidden` | U | Tol | 3.81469726562e-06 | 3755 | 16 |
| `1301/synthetic_site_permutation/T1/original_path_delta/orientation_logits` | U | Tol | 0.000137329101562 | 196 | 39 |
| `1301/synthetic_site_permutation/T1/original_path_delta/orientation_standardized_hidden` | U | Tol | 3.0517578125e-05 | 1670 | 72 |
| `1301/T2/tie_counts` | R | Exact | 2 | 2 | 2 |
| `1301/T2/original_path_delta/category_logits` | U | Tol | 5.72204589844e-06 | 1317 | 10 |
| `1301/T2/original_path_delta/orientation_hidden` | U | Tol | 3.81469726562e-06 | 3607 | 18 |
| `1301/T2/original_path_delta/orientation_logits` | U | Tol | 0.000106811523438 | 186 | 51 |
| `1301/T2/original_path_delta/orientation_standardized_hidden` | U | Tol | 3.0517578125e-05 | 1485 | 117 |
| `1301/address_permutation/T2/tie_counts` | R | Exact | 2 | 2 | 2 |
| `1301/synthetic_site_permutation/T2/selected_cells` | R | Exact | 1 | 4 | 4 |
| `1301/synthetic_site_permutation/T2/tie_counts` | R | Exact | 3 | 6 | 6 |
| `1301/synthetic_site_permutation/T2/original_path/category_logits` | U | Tol | 3.81469726562e-06 | 1025 | 3 |
| `1301/synthetic_site_permutation/T2/original_path_delta/category_logits` | U | Tol | 5.72204589844e-06 | 1345 | 11 |
| `1301/synthetic_site_permutation/T2/original_path_delta/orientation_hidden` | U | Tol | 3.81469726562e-06 | 3600 | 20 |
| `1301/synthetic_site_permutation/T2/original_path_delta/orientation_logits` | U | Tol | 0.000106811523438 | 184 | 28 |
| `1301/synthetic_site_permutation/T2/original_path_delta/orientation_standardized_hidden` | U | Tol | 3.0517578125e-05 | 1536 | 106 |
| `1301/T4/selected_cells` | R | Exact | 3 | 4 | 4 |
| `1301/T4/tie_counts` | R | Exact | 14 | 5 | 5 |
| `1301/T4/original_path_delta/category_logits` | U | Tol | 3.81469726562e-06 | 1286 | 6 |
| `1301/T4/original_path_delta/orientation_hidden` | U | Tol | 3.81469726562e-06 | 3696 | 19 |
| `1301/T4/original_path_delta/orientation_logits` | U | Tol | 0.000106811523438 | 186 | 34 |
| `1301/T4/original_path_delta/orientation_standardized_hidden` | U | Tol | 3.0517578125e-05 | 1574 | 90 |
| `1301/address_permutation/T4/selected_cells` | R | Exact | 3 | 4 | 4 |
| `1301/address_permutation/T4/tie_counts` | R | Exact | 14 | 5 | 5 |
| `1301/synthetic_site_permutation/T4/selected_cells` | R | Exact | 3 | 4 | 4 |
| `1301/synthetic_site_permutation/T4/tie_counts` | R | Exact | 14 | 5 | 5 |
| `1301/synthetic_site_permutation/T4/original_path_delta/category_logits` | U | Tol | 3.81469726562e-06 | 1286 | 6 |
| `1301/synthetic_site_permutation/T4/original_path_delta/orientation_hidden` | U | Tol | 3.81469726562e-06 | 3696 | 19 |
| `1301/synthetic_site_permutation/T4/original_path_delta/orientation_logits` | U | Tol | 0.000106811523438 | 186 | 34 |
| `1301/synthetic_site_permutation/T4/original_path_delta/orientation_standardized_hidden` | U | Tol | 3.0517578125e-05 | 1574 | 90 |
| `1311/synthetic_site_permutation/T1/original_path/category_logits` | U | Tol | 3.81469726562e-06 | 1121 | 2 |
| `1311/synthetic_site_permutation/T1/original_path_delta/category_logits` | U | Tol | 5.72204589844e-06 | 1414 | 27 |
| `1311/synthetic_site_permutation/T1/original_path_delta/orientation_hidden` | U | Tol | 3.81469726562e-06 | 4776 | 17 |
| `1311/synthetic_site_permutation/T1/original_path_delta/orientation_logits` | U | Tol | 0.0001220703125 | 220 | 52 |
| `1311/synthetic_site_permutation/T1/original_path_delta/orientation_standardized_hidden` | U | Tol | 3.0517578125e-05 | 1571 | 152 |
| `1311/T2/selected_cells` | R | Exact | 1 | 1 | 1 |
| `1311/T2/tie_counts` | R | Exact | 2 | 2 | 2 |
| `1311/T2/original_path_delta/category_logits` | U | Tol | 5.72204589844e-06 | 1459 | 29 |
| `1311/T2/original_path_delta/orientation_hidden` | U | Tol | 3.81469726562e-06 | 4519 | 24 |
| `1311/T2/original_path_delta/orientation_logits` | U | Tol | 9.1552734375e-05 | 207 | 72 |
| `1311/T2/original_path_delta/orientation_standardized_hidden` | U | Tol | 3.0517578125e-05 | 1460 | 179 |
| `1311/address_permutation/T2/selected_cells` | R | Exact | 1 | 1 | 1 |
| `1311/address_permutation/T2/tie_counts` | R | Exact | 2 | 2 | 2 |
| `1311/synthetic_site_permutation/T2/selected_cells` | R | Exact | 3 | 10 | 10 |
| `1311/synthetic_site_permutation/T2/tie_counts` | R | Exact | 3 | 10 | 10 |
| `1311/synthetic_site_permutation/T2/fine_correct` | R | Exact | 1 | 1 | 1 |
| `1311/synthetic_site_permutation/T2/original_path_delta/category_logits` | U | Tol | 5.72204589844e-06 | 1404 | 24 |
| `1311/synthetic_site_permutation/T2/original_path_delta/orientation_hidden` | U | Tol | 3.81469726562e-06 | 4694 | 17 |
| `1311/synthetic_site_permutation/T2/original_path_delta/orientation_logits` | U | Tol | 9.91821289062e-05 | 219 | 76 |
| `1311/synthetic_site_permutation/T2/original_path_delta/orientation_standardized_hidden` | U | Tol | 3.0517578125e-05 | 1456 | 149 |
| `1311/T4/selected_cells` | R | Exact | 3 | 6 | 6 |
| `1311/T4/tie_counts` | R | Exact | 14 | 9 | 9 |
| `1311/T4/original_path_delta/category_logits` | U | Tol | 5.72204589844e-06 | 1427 | 20 |
| `1311/T4/original_path_delta/orientation_hidden` | U | Tol | 3.81469726562e-06 | 4617 | 22 |
| `1311/T4/original_path_delta/orientation_logits` | U | Tol | 0.0001220703125 | 210 | 66 |
| `1311/T4/original_path_delta/orientation_standardized_hidden` | U | Tol | 4.57763671875e-05 | 1481 | 159 |
| `1311/address_permutation/T4/selected_cells` | R | Exact | 3 | 6 | 6 |
| `1311/address_permutation/T4/tie_counts` | R | Exact | 14 | 9 | 9 |
| `1311/synthetic_site_permutation/T4/selected_cells` | R | Exact | 3 | 6 | 6 |
| `1311/synthetic_site_permutation/T4/tie_counts` | R | Exact | 14 | 9 | 9 |
| `1311/synthetic_site_permutation/T4/original_path_delta/category_logits` | U | Tol | 5.72204589844e-06 | 1427 | 20 |
| `1311/synthetic_site_permutation/T4/original_path_delta/orientation_hidden` | U | Tol | 3.81469726562e-06 | 4617 | 22 |
| `1311/synthetic_site_permutation/T4/original_path_delta/orientation_logits` | U | Tol | 0.0001220703125 | 210 | 66 |
| `1311/synthetic_site_permutation/T4/original_path_delta/orientation_standardized_hidden` | U | Tol | 4.57763671875e-05 | 1481 | 159 |
| `1321/original_controls/category_logits` | U | Tol | 4.76837158203e-06 | 1240 | 2 |
| `1321/fit_controls/category_logits` | U | Tol | 5.72204589844e-06 | 2449 | 1 |
| `1321/T1/original_path/category_logits` | U | Tol | 4.76837158203e-06 | 1240 | 2 |
| `1321/synthetic_site_permutation/T1/original_path/category_logits` | U | Tol | 3.81469726562e-06 | 1254 | 2 |
| `1321/synthetic_site_permutation/T1/original_path_delta/category_logits` | U | Tol | 7.62939453125e-06 | 1484 | 21 |
| `1321/synthetic_site_permutation/T1/original_path_delta/orientation_hidden` | U | Tol | 7.62939453125e-06 | 3575 | 30 |
| `1321/synthetic_site_permutation/T1/original_path_delta/orientation_logits` | U | Tol | 0.0001220703125 | 198 | 38 |
| `1321/synthetic_site_permutation/T1/original_path_delta/orientation_standardized_hidden` | U | Tol | 3.0517578125e-05 | 1071 | 135 |
| `1321/T2/selected_cells` | R | Exact | 1 | 2 | 2 |
| `1321/T2/tie_counts` | R | Exact | 2 | 3 | 3 |
| `1321/T2/original_path_delta/category_logits` | U | Tol | 7.62939453125e-06 | 1485 | 28 |
| `1321/T2/original_path_delta/orientation_hidden` | U | Tol | 7.62939453125e-06 | 3453 | 53 |
| `1321/T2/original_path_delta/orientation_logits` | U | Tol | 0.000106811523438 | 193 | 78 |
| `1321/T2/original_path_delta/orientation_standardized_hidden` | U | Tol | 3.0517578125e-05 | 974 | 177 |
| `1321/address_permutation/T2/selected_cells` | R | Exact | 1 | 2 | 2 |
| `1321/address_permutation/T2/tie_counts` | R | Exact | 2 | 3 | 3 |
| `1321/synthetic_site_permutation/T2/selected_cells` | R | Exact | 3 | 12 | 12 |
| `1321/synthetic_site_permutation/T2/tie_counts` | R | Exact | 3 | 9 | 9 |
| `1321/synthetic_site_permutation/T2/original_path_delta/category_logits` | U | Tol | 5.72204589844e-06 | 1488 | 15 |
| `1321/synthetic_site_permutation/T2/original_path_delta/orientation_hidden` | U | Tol | 7.62939453125e-06 | 3532 | 40 |
| `1321/synthetic_site_permutation/T2/original_path_delta/orientation_logits` | U | Tol | 0.000137329101562 | 197 | 52 |
| `1321/synthetic_site_permutation/T2/original_path_delta/orientation_standardized_hidden` | U | Tol | 3.0517578125e-05 | 1045 | 136 |
| `1321/T4/selected_cells` | R | Exact | 3 | 2 | 2 |
| `1321/T4/tie_counts` | R | Exact | 14 | 8 | 8 |
| `1321/T4/original_path_delta/category_logits` | U | Tol | 6.67572021484e-06 | 1480 | 28 |
| `1321/T4/original_path_delta/orientation_hidden` | U | Tol | 5.72204589844e-06 | 3495 | 47 |
| `1321/T4/original_path_delta/orientation_logits` | U | Tol | 0.000137329101562 | 198 | 47 |
| `1321/T4/original_path_delta/orientation_standardized_hidden` | U | Tol | 4.57763671875e-05 | 1033 | 129 |
| `1321/address_permutation/T4/selected_cells` | R | Exact | 3 | 2 | 2 |
| `1321/address_permutation/T4/tie_counts` | R | Exact | 14 | 8 | 8 |
| `1321/synthetic_site_permutation/T4/selected_cells` | R | Exact | 3 | 2 | 2 |
| `1321/synthetic_site_permutation/T4/tie_counts` | R | Exact | 14 | 8 | 8 |
| `1321/synthetic_site_permutation/T4/original_path_delta/category_logits` | U | Tol | 6.67572021484e-06 | 1480 | 28 |
| `1321/synthetic_site_permutation/T4/original_path_delta/orientation_hidden` | U | Tol | 5.72204589844e-06 | 3495 | 47 |
| `1321/synthetic_site_permutation/T4/original_path_delta/orientation_logits` | U | Tol | 0.000137329101562 | 198 | 47 |
| `1321/synthetic_site_permutation/T4/original_path_delta/orientation_standardized_hidden` | U | Tol | 4.57763671875e-05 | 1033 | 129 |
| `results/all_software_invariants_pass` | V | Actual true; expected no prior errors (105 present) | — | — | — |

## Interpretation and limits

A Controller's action on role-location evidence is not a Reporter output. No
attention model A, qualitative report, presentation target, human adjective or
phenomenal rating was introduced by this stage. Even the retained successful
utility results establish only native role/task-content feasibility. The T2
changed/correctly-retained conjunction is descriptive; it does not qualify a
manner of presentation or establish experienced blur. Synthetic site permutation
is an explicit engineering intervention, not a naturally reached attention state.

The source of qualia under the attention-control-model hypothesis remains
unidentified and the full project goal remains unachieved. This failed review
cannot supply confirmation, phenomenal-source specificity, a presentation
counterpart, or human/model-family endorsement.

The loss traces and final weights/logits cannot prove the entire optimizer
trajectory without fit replay. Equal retained checkpoint/before/after weights
cannot rule out transient changes. Registered claims, exclusive paths and the
committed first-run inventory support one recorded attempt but cannot rule out
unrecorded external work. Conditional world intervals do not measure uncertainty
over people, independent model families, or unseen position-layout combinations.
The Controller reenactment uses the production consumer only to test isolation;
it is not an independently implemented consumer and is not evidence for attention
prediction or qualia. All first findings remain preserved without correction.
