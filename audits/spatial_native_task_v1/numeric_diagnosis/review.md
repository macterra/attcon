# First arithmetic diagnosis: all preserved failures explained

2026-10-11. `/root/spatial_native_numerical_execution_review`, separate task
allocation within the same AI model team with inherited hypothesis and project
context. This is not independent-family evidence or human endorsement.

The single committed diagnostic pass completed with exit **0**, status
**diagnosis_complete_all_first_failures_explained**, **106/106 explained**, and
**no unresolved entries**. All 105 original tensor failures reproduced their
recorded shape/dtype, maximum differences, unequal-element counts and
tolerance-violation counts exactly. Independent equations using the native tensor
shape and linear arithmetic matched the archived quantities exactly. The 106th
failure is the original verdict derived from the previous 105 discrepancies.

This is arithmetic diagnosis only. The original verifier and its
**failed_first_verification** result remain unchanged. No passing numerical
revalidation has been performed. No study retry, new world, fit, checkpoint
mutation, source fix, gate change, tolerance change, Reporter call, Git mutation
or GitHub action was performed by this reviewer.

## Bound execution and preserved artifacts

The diagnostic source/design were committed and pushed by the coordinator before
runtime at `13bf32f129a892d77e8f7fcff43054f67c891fd0`. The original actual results
remain at `614e74ae74ccf1d553f6fc6e0ec318f057a94ed8`; the first failed numerical
review remains at `432146ae8f1420b13bacdbb8f00518570d34b95f`. The program verified
current and committed byte bindings before archive loading, including
byte-identical decompression of the full first metadata. It checked the exact
106-failure inventory rather than choosing illustrative failures alone.

```bash
set -o noclobber
/home/david/dev/attcon/.venv/bin/python scripts/diagnose_spatial_native_numeric_reference_v1.py --results-commit 614e74ae74ccf1d553f6fc6e0ec318f057a94ed8 --first-review-commit 432146ae8f1420b13bacdbb8f00518570d34b95f --diagnosis-commit 13bf32f129a892d77e8f7fcff43054f67c891fd0 --report audits/spatial_native_task_v1/numeric_diagnosis/metadata.json --ack-committed-diagnosis > audits/spatial_native_task_v1/numeric_diagnosis/first_run.log 2>&1
```

Shell session `97356` is terminal, exit 0. The new stdout/stderr log and exclusive
metadata remain unchanged. Subsequent commands read and summarize that metadata;
they did not repeat diagnostic inference. Metadata contains complete per-case
comparisons, CPU profiles, all 106 diagnoses and up to three bound violating
witnesses per original tensor failure.

- diagnostic program: 28,132 bytes; SHA256 `da585779a40549ee87a24399bb4a7acb1671cc36f3fba057062d911cebbf5cdd`.
- design: 6,619 bytes; SHA256 `729f55aee87e5fc18cbf936cce74e01f55da269d9844d7bcb78520852e4dd5d6`.
- first diagnostic log: 326 bytes; SHA256 `d813b005172530b94830d20908f0634f69ccf76b73938cbdd864cb45f431f95d`.
- first diagnostic metadata: 1,882,702 bytes; SHA256 `f89788f885e7b3b2f477624dbc657147e6f6460edc9b8c0e4f590b40a842a3e1`.

## Head discrepancy: the tensor shape changes CPU dispatch

The native Controller operates on one `[24,4,4]` map. Its standardized, moved-axis
linear input is `[4,4,24]`, stride `[4,1,16]`, noncontiguous. The first verifier
operates on the full `[64,4,4,4,24]` tensor, stride `[1536,384,4,1,16]`, also
noncontiguous. The observed CPU profiler schedule on retained seed-1301 T4 inputs
shows:

| Arithmetic path | First linear dispatch | Second linear dispatch |
|---|---|---|
| Original batch matmul plus bias | `bmm([1024,4,24],[1024,24,32])`, then `add` | `mm([4096,32],[32,3])`, then `add` |
| Batch F.linear | Same `bmm`, then `add_` | `addmm` with `[4096,32]` input |
| Item matmul plus bias | `mm([16,24],[24,32])`, then `add` | `mm([16,32],[32,3])`, then `add` |
| Item F.linear | Same `[16,24]` `mm`, then `add_` | `addmm` with `[16,32]` input |
| Flattened F.linear | `addmm` with `[4096,24]` input | `addmm` with `[4096,32]` input |

Thus calling F.linear on the full batch does not reproduce itemwise native
arithmetic. A simple flattening change also does not reproduce it. **Both itemwise
F.linear and independently written itemwise matmul-plus-bias match all 18 saved
head cases' archived logits and fields exactly**, with zero policy discrepancies.
The latter preserves an independent explicit equation implementation without
calling the production head or Controller.

By contrast, original batched reference arithmetic and batched F.linear each
produce 24 discrete policy-comparison failures across the 18 fixed cases.
Flattened F.linear produces 27. These totals exclude the repeated address-control
comparisons in the first 35-failure ledger. The original batched reference breaks
symmetry for some exactly identical input-site vectors:

| Encoder | Case | Identical-input ordered pairs | Pairs with differing reference logits | Pairs with differing reference fields |
|---|---|---:|---:|---:|
| 1301 | T2 | 12,288 | 24 | 24 |
| 1301 | synthetic_site_permutation/T2 | 12,288 | 60 | 54 |
| 1301 | T4 | 61,440 | 168 | 168 |
| 1311 | T2 | 12,288 | 32 | 24 |
| 1311 | synthetic_site_permutation/T2 | 12,288 | 66 | 66 |
| 1311 | T4 | 61,440 | 336 | 336 |
| 1321 | T2 | 12,288 | 48 | 48 |
| 1321 | synthetic_site_permutation/T2 | 12,288 | 78 | 78 |
| 1321 | T4 | 61,440 | 224 | 224 |

The matched itemwise paths have **zero** violating pairs in every case, including
synthetic T4. Input equality and tie equality were exact; no tie tolerance was
introduced. T4 reference-logit perturbations are only 1.1920928955078125e-7 to
2.384185791015625e-7, but exact ties are discontinuous. For seed 1301, world 30,
item 2, the actual T4 red/blue tie counts are both 16; the original reference
returns 14 and 2. Matched itemwise equations and float64 equations return 16 and
16. For seed 1311, world 63, item 3, blue-role T2, the actual selected row is 2;
the original reference chooses 3; matched itemwise equations choose 2.

The role-field normalizer is not the source of these discrepancies here. Explicit
exponent/sum normalization on matched itemwise logits also matches archived fields
exactly in all 18 cases. Neither adopting batch F.linear alone nor relaxing exact
ties would address the measured shape-dependent discrepancy correctly.

## Control discrepancy: bias arithmetic and propagated endpoint roundoff

The proposed leading-axis explanation is rejected. Both the original control
reference and the native encoder suffix flatten their input to `[256,384]`
(`[512,384]` for fit maps). The observed control profile shows the original
reference uses separate `mm` and `add` operations, while explicit F.linear uses
`addmm` at each of the four linear layers: encoder hidden64, category logits,
orientation hidden32 and orientation logits. Shapes are matched; operation and
bias-addition paths differ.

Independent flattened F.linear equations match **all 96 archived control
quantity comparisons exactly** across the 24 saved control-map cases. This
includes hidden64, category logits, standardized hidden64 and orientation logits.
For all 70 original control failures, the original separate-matmul arithmetic
reproduces its first failed metrics exactly; matched equations recover the
archived output or delta exactly. All 60 delta failures also have exact matched
changed and original baseline endpoints.

Hidden-layer roundoff propagates through fit-standardization coordinates and the
orientation readout. Subtracting separately rounded changed and baseline outputs
can combine their endpoint errors, while small deltas have less allowance under
the unchanged relative tolerance. A concrete retained witness is seed 1321, T2,
world 0, item 2, orientation-logit delta:

| Quantity | Actual archive / matched F.linear | Original separate-matmul reference |
|---|---:|---:|
| Changed endpoint | 188.10877990722656 | 188.10879516601562 |
| Original endpoint | 181.04481506347656 | 181.0447540283203 |
| Difference | 7.06396484375 | 7.0640411376953125 |

The delta discrepancy is 7.62939453125e-5, exceeding the unchanged reference-based
allowance 7.164041137695313e-5. The independent float64 equation gives
7.063983277067962; it is a useful arithmetic reference but does not equal either
registered float32 path. The diagnostic explains the discrepancy without changing
any archived value or numerical tolerance.

## Higher precision and actual CPU evidence

Float64 equations, both on standardized float32 inputs and full float64
coordinates, preserve all repeated-input symmetries and all discrete policies
in the 18 saved head cases. They do not match the archived float32 logits or
fields exactly. Accordingly, higher precision corroborates the mathematical
symmetry and supplies comparison values; it does not replace registered float32
arithmetic or establish a passing numerical review.

The recorded environment is torch 2.5.1+cpu, four CPU threads, deterministic
algorithms enabled, on Intel Core i7-14700F. Metadata retains full torch build
configuration and tensor layout/dispatch evidence. The profiler observes CPU
ATen operators and their inputs; this diagnosis does not claim to have inspected
vendor microkernel assembly or establish universal behavior on other backends.
The fixed profiler schedule is part of this single diagnostic pass.

## Every first failure accounted for

`H` = head shape/dispatch discrepancy; `C` = control linear-kernel arithmetic;
`V` = original verdict derived from 105 earlier discrepancies. Every H/C row
reproduces the first quantitative metrics and matches archived quantities exactly
with the independent matched equations. H rows additionally match logits and
fields exactly; delta C rows additionally match both endpoints exactly. The full
quantitative first comparison, newly reproduced comparison and bound witness
values are retained for each row in metadata. No failure is omitted or unresolved.

| First failed check | Diagnosis | Original metrics reproduced | Matched archive quantity exact |
|---|---|---|---|
| `1301/original_controls/category_logits` | C | Yes | Yes |
| `1301/fit_controls/category_logits` | C | Yes | Yes |
| `1301/T1/original_path/category_logits` | C | Yes | Yes |
| `1301/synthetic_site_permutation/T1/original_path/category_logits` | C | Yes | Yes |
| `1301/synthetic_site_permutation/T1/original_path_delta/category_logits` | C | Yes | Yes |
| `1301/synthetic_site_permutation/T1/original_path_delta/orientation_hidden` | C | Yes | Yes |
| `1301/synthetic_site_permutation/T1/original_path_delta/orientation_logits` | C | Yes | Yes |
| `1301/synthetic_site_permutation/T1/original_path_delta/orientation_standardized_hidden` | C | Yes | Yes |
| `1301/T2/tie_counts` | H | Yes | Yes |
| `1301/T2/original_path_delta/category_logits` | C | Yes | Yes |
| `1301/T2/original_path_delta/orientation_hidden` | C | Yes | Yes |
| `1301/T2/original_path_delta/orientation_logits` | C | Yes | Yes |
| `1301/T2/original_path_delta/orientation_standardized_hidden` | C | Yes | Yes |
| `1301/address_permutation/T2/tie_counts` | H | Yes | Yes |
| `1301/synthetic_site_permutation/T2/selected_cells` | H | Yes | Yes |
| `1301/synthetic_site_permutation/T2/tie_counts` | H | Yes | Yes |
| `1301/synthetic_site_permutation/T2/original_path/category_logits` | C | Yes | Yes |
| `1301/synthetic_site_permutation/T2/original_path_delta/category_logits` | C | Yes | Yes |
| `1301/synthetic_site_permutation/T2/original_path_delta/orientation_hidden` | C | Yes | Yes |
| `1301/synthetic_site_permutation/T2/original_path_delta/orientation_logits` | C | Yes | Yes |
| `1301/synthetic_site_permutation/T2/original_path_delta/orientation_standardized_hidden` | C | Yes | Yes |
| `1301/T4/selected_cells` | H | Yes | Yes |
| `1301/T4/tie_counts` | H | Yes | Yes |
| `1301/T4/original_path_delta/category_logits` | C | Yes | Yes |
| `1301/T4/original_path_delta/orientation_hidden` | C | Yes | Yes |
| `1301/T4/original_path_delta/orientation_logits` | C | Yes | Yes |
| `1301/T4/original_path_delta/orientation_standardized_hidden` | C | Yes | Yes |
| `1301/address_permutation/T4/selected_cells` | H | Yes | Yes |
| `1301/address_permutation/T4/tie_counts` | H | Yes | Yes |
| `1301/synthetic_site_permutation/T4/selected_cells` | H | Yes | Yes |
| `1301/synthetic_site_permutation/T4/tie_counts` | H | Yes | Yes |
| `1301/synthetic_site_permutation/T4/original_path_delta/category_logits` | C | Yes | Yes |
| `1301/synthetic_site_permutation/T4/original_path_delta/orientation_hidden` | C | Yes | Yes |
| `1301/synthetic_site_permutation/T4/original_path_delta/orientation_logits` | C | Yes | Yes |
| `1301/synthetic_site_permutation/T4/original_path_delta/orientation_standardized_hidden` | C | Yes | Yes |
| `1311/synthetic_site_permutation/T1/original_path/category_logits` | C | Yes | Yes |
| `1311/synthetic_site_permutation/T1/original_path_delta/category_logits` | C | Yes | Yes |
| `1311/synthetic_site_permutation/T1/original_path_delta/orientation_hidden` | C | Yes | Yes |
| `1311/synthetic_site_permutation/T1/original_path_delta/orientation_logits` | C | Yes | Yes |
| `1311/synthetic_site_permutation/T1/original_path_delta/orientation_standardized_hidden` | C | Yes | Yes |
| `1311/T2/selected_cells` | H | Yes | Yes |
| `1311/T2/tie_counts` | H | Yes | Yes |
| `1311/T2/original_path_delta/category_logits` | C | Yes | Yes |
| `1311/T2/original_path_delta/orientation_hidden` | C | Yes | Yes |
| `1311/T2/original_path_delta/orientation_logits` | C | Yes | Yes |
| `1311/T2/original_path_delta/orientation_standardized_hidden` | C | Yes | Yes |
| `1311/address_permutation/T2/selected_cells` | H | Yes | Yes |
| `1311/address_permutation/T2/tie_counts` | H | Yes | Yes |
| `1311/synthetic_site_permutation/T2/selected_cells` | H | Yes | Yes |
| `1311/synthetic_site_permutation/T2/tie_counts` | H | Yes | Yes |
| `1311/synthetic_site_permutation/T2/fine_correct` | H | Yes | Yes |
| `1311/synthetic_site_permutation/T2/original_path_delta/category_logits` | C | Yes | Yes |
| `1311/synthetic_site_permutation/T2/original_path_delta/orientation_hidden` | C | Yes | Yes |
| `1311/synthetic_site_permutation/T2/original_path_delta/orientation_logits` | C | Yes | Yes |
| `1311/synthetic_site_permutation/T2/original_path_delta/orientation_standardized_hidden` | C | Yes | Yes |
| `1311/T4/selected_cells` | H | Yes | Yes |
| `1311/T4/tie_counts` | H | Yes | Yes |
| `1311/T4/original_path_delta/category_logits` | C | Yes | Yes |
| `1311/T4/original_path_delta/orientation_hidden` | C | Yes | Yes |
| `1311/T4/original_path_delta/orientation_logits` | C | Yes | Yes |
| `1311/T4/original_path_delta/orientation_standardized_hidden` | C | Yes | Yes |
| `1311/address_permutation/T4/selected_cells` | H | Yes | Yes |
| `1311/address_permutation/T4/tie_counts` | H | Yes | Yes |
| `1311/synthetic_site_permutation/T4/selected_cells` | H | Yes | Yes |
| `1311/synthetic_site_permutation/T4/tie_counts` | H | Yes | Yes |
| `1311/synthetic_site_permutation/T4/original_path_delta/category_logits` | C | Yes | Yes |
| `1311/synthetic_site_permutation/T4/original_path_delta/orientation_hidden` | C | Yes | Yes |
| `1311/synthetic_site_permutation/T4/original_path_delta/orientation_logits` | C | Yes | Yes |
| `1311/synthetic_site_permutation/T4/original_path_delta/orientation_standardized_hidden` | C | Yes | Yes |
| `1321/original_controls/category_logits` | C | Yes | Yes |
| `1321/fit_controls/category_logits` | C | Yes | Yes |
| `1321/T1/original_path/category_logits` | C | Yes | Yes |
| `1321/synthetic_site_permutation/T1/original_path/category_logits` | C | Yes | Yes |
| `1321/synthetic_site_permutation/T1/original_path_delta/category_logits` | C | Yes | Yes |
| `1321/synthetic_site_permutation/T1/original_path_delta/orientation_hidden` | C | Yes | Yes |
| `1321/synthetic_site_permutation/T1/original_path_delta/orientation_logits` | C | Yes | Yes |
| `1321/synthetic_site_permutation/T1/original_path_delta/orientation_standardized_hidden` | C | Yes | Yes |
| `1321/T2/selected_cells` | H | Yes | Yes |
| `1321/T2/tie_counts` | H | Yes | Yes |
| `1321/T2/original_path_delta/category_logits` | C | Yes | Yes |
| `1321/T2/original_path_delta/orientation_hidden` | C | Yes | Yes |
| `1321/T2/original_path_delta/orientation_logits` | C | Yes | Yes |
| `1321/T2/original_path_delta/orientation_standardized_hidden` | C | Yes | Yes |
| `1321/address_permutation/T2/selected_cells` | H | Yes | Yes |
| `1321/address_permutation/T2/tie_counts` | H | Yes | Yes |
| `1321/synthetic_site_permutation/T2/selected_cells` | H | Yes | Yes |
| `1321/synthetic_site_permutation/T2/tie_counts` | H | Yes | Yes |
| `1321/synthetic_site_permutation/T2/original_path_delta/category_logits` | C | Yes | Yes |
| `1321/synthetic_site_permutation/T2/original_path_delta/orientation_hidden` | C | Yes | Yes |
| `1321/synthetic_site_permutation/T2/original_path_delta/orientation_logits` | C | Yes | Yes |
| `1321/synthetic_site_permutation/T2/original_path_delta/orientation_standardized_hidden` | C | Yes | Yes |
| `1321/T4/selected_cells` | H | Yes | Yes |
| `1321/T4/tie_counts` | H | Yes | Yes |
| `1321/T4/original_path_delta/category_logits` | C | Yes | Yes |
| `1321/T4/original_path_delta/orientation_hidden` | C | Yes | Yes |
| `1321/T4/original_path_delta/orientation_logits` | C | Yes | Yes |
| `1321/T4/original_path_delta/orientation_standardized_hidden` | C | Yes | Yes |
| `1321/address_permutation/T4/selected_cells` | H | Yes | Yes |
| `1321/address_permutation/T4/tie_counts` | H | Yes | Yes |
| `1321/synthetic_site_permutation/T4/selected_cells` | H | Yes | Yes |
| `1321/synthetic_site_permutation/T4/tie_counts` | H | Yes | Yes |
| `1321/synthetic_site_permutation/T4/original_path_delta/category_logits` | C | Yes | Yes |
| `1321/synthetic_site_permutation/T4/original_path_delta/orientation_hidden` | C | Yes | Yes |
| `1321/synthetic_site_permutation/T4/original_path_delta/orientation_logits` | C | Yes | Yes |
| `1321/synthetic_site_permutation/T4/original_path_delta/orientation_standardized_hidden` | C | Yes | Yes |
| `results/all_software_invariants_pass` | V | Original count 105 retained | Derived verdict explained; first status unchanged |

## Remaining work and scientific boundary

These findings justify a separately reviewed arithmetic correction: preserve
native item shapes in the independent head equations, retain exact emitted-policy
checks, and use independently written flattened F.linear equations for the
original control pathways. This diagnosis does not implement that correction or
perform a corrected full verifier pass. The preserved first verifier, log,
metadata, review, production source, fitted heads, original archives, registration,
gates and tolerances remain unchanged.

The first failed verification remains a historical fact. A separately authorized,
labelled correction and full numerical reconstruction would still be needed to
approve the retained results numerically. The diagnosis is not an empirical
replication; all calculations use the same archived fixed-model data and the same
model team with inherited theory context.

Controller is not Reporter. Native role targeting, content relations, engineering
utility, arithmetic equality and imposed T2/T4 policy consequences cannot qualify
presentation manner, identify attention model A as a phenomenal source or
establish qualia. The attention-control-model project's full goal remains
unachieved. No independent-family or human endorsement is supplied.
