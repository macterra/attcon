# Separately labelled corrected numerical reference design

2026-10-11. Prepared by `/root/spatial_native_numerical_execution_review`, a
separate task allocation within the same model team with inherited hypothesis
and project context. No independent-family or human endorsement is supplied.
Preparation permits static inspection and in-memory syntax compilation only.
No actual archive/checkpoint loading or inference may occur until the coordinator
independently reviews, commits and explicitly authorizes this exact source/design.

The executable is `scripts/verify_spatial_native_task_v1_corrected.py`; this is
a separate file, with a corrected-reference schema and labelled statuses. The
original verifier and failed review are unchanged. Future owned outputs in this
directory are exclusive `first_run.log`, `metadata.json` and `review.md`. The
coordinator may transport the full metadata via byte-identical gzip plus a
raw/compressed hash manifest, preserving every check. No Git mutation or GitHub
operation is performed by this reviewer.

## Preserved first evidence and reason for correction

The original actual results remain at
`614e74ae74ccf1d553f6fc6e0ec318f057a94ed8`. The first failed numerical review is
preserved at `432146ae8f1420b13bacdbb8f00518570d34b95f`, with 106 failures across
206,070 checks. See the [first failed review](../numerical_review/review.md),
[unchanged first log](../numerical_review/first_run.log), and byte-identical
[first metadata transport](../numerical_review/metadata.json.gz).

The separately committed arithmetic diagnosis is preserved at
`0f8cf0952d47dca65198a0c46505e3eb9ca3936f`. Its [review](../numeric_diagnosis/review.md),
[metadata](../numeric_diagnosis/metadata.json) and [CPU/input-shape design](../numeric_diagnosis/design.md)
account for all 106 failures. The diagnostic reproduced all 105 tensor-failure
metrics exactly. Native itemwise independent matmul-plus-bias equations match all
18 head cases' archived logits/fields exactly; full-batch F.linear and flattened
F.linear do not. Explicit flattened F.linear control equations match all 96
saved control-quantity comparisons exactly, including all changed/baseline
endpoints underlying the 60 failed delta comparisons.

The observed head reference first layer uses batched `bmm` on noncontiguous
`[64,4,4,4,24]` input, whereas native item consumption uses `mm` on
`[4,4,24]` input. Small batch arithmetic perturbations break some repeated-site
symmetries and exact ties. The control reference and production both already
flatten to `[-1,384]`; there is no leading-axis control correction. Their observed
separate `mm` plus bias versus F.linear `addmm` paths explain propagated control
and baseline-delta roundoff.

## Exact reference changes

Only two arithmetic changes affect reconstruction:

1. `head_reference(..., itemwise=True)` evaluates the original independent
   matmul-plus-separate-bias equations on each native `[4,4,24]` item input before
   stacking outputs. Primary native assessments, consistent-address controls and
   synthetic-site control assessments explicitly request itemwise reconstruction.
   Native logits, fields, ties, actions and control summaries retain every original
   opportunity and comparison. Fitting initial/final logits, initial/final CE and
   raw-grid head assessments keep the original batched default because those
   recorded contexts used batch computations.
2. `controls_reference` uses independently written flattened F.linear equations
   with archived parameters for encoder hidden64, category logits, orientation
   hidden32 and orientation logits. It does not instantiate or call the production
   extractor, suffix, orientation module or control helper. Standardization,
   original-logit controls, sigmoid coordinates and baseline-delta calculations
   preserve the original coordinates and formulas.

Explicit float64 role-minus-background subtraction, exponentiation and sum
normalization remain unchanged. There is no tie tolerance, relaxed threshold,
quantization, clipping, favorable subset, changed gate or dropped check. All
original source equations for maps, block means, CE, statistics, bootstrap,
recorded action reconstruction and separately labelled Controller isolation
reenactment remain unchanged. Core atol=1e-6, rtol=1e-5 and exact discrete/copy
checks are unchanged. The retained registration, production source/checkpoints,
heads, worlds, input/output archives and results are unchanged.

## Additional preservation binding and unchanged check inventory

The CLI retains every original argument and adds three required full preservation
commits: `--reference-commit` binds this exact corrected executable and design
before execution; `--first-review-commit` binds the preserved first failed review;
`--diagnosis-commit` binds the preserved arithmetic diagnosis.

Before torch import or archive loading, the executable verifies current bytes
against committed bytes for the corrected source/design, original verifier,
first failed review/design/log, compressed first metadata and its manifest,
diagnostic program/design/log/review and raw diagnostic metadata. It requires
byte-identical first-metadata decompression with raw SHA256
`6010154b9502daef7f223ac3da4ac9c7854c4c9bc15436b8d987d9a5c93dc338`, and binds
diagnostic metadata SHA256
`f89788f885e7b3b2f477624dbc657147e6f6460edc9b8c0e4f590b40a842a3e1`.
It checks all 106 diagnosed failure names, explained status and original
106-failure / 206,070-check inventory, plus commit ancestry from original results
through first review, diagnosis and corrected reference.

These supplemental binding receipts are recorded separately from the original
core checks. After complete reconstruction, the original ordered list of core
check names must match the first-review list exactly, preserving the same
206,070-check inventory without padding or omission. A mismatch fails the
corrected review. The existing source/checkpoint/review/registration/execution/
results bindings and original-result ancestry checks remain in the core verifier.

The exclusive corrected metadata reservation occurs before preservation checks
or torch imports. Its schema is
`spatial_native_task_v1_corrected_reference_numerical_review`; terminal statuses
are `passed_corrected_reference_revalidation` or
`failed_first_corrected_reference_verification`. The preserved original first
status remains `failed_first_verification`. Stdout/stderr are captured unchanged
in the new exclusive first log. Any failure is retained without automatic source
repair, revalidation retry, fit replay or development-run retry.

## Scope and limitations

This corrected pass reconstructs the retained results only. No new worlds,
rendering, training, optimizer, sweep, checkpoint retry, fallback source, report
target or Reporter call is introduced. Passing would establish numerical
agreement under the reviewed arithmetic, including the complete unconditional
denominators and engineering gates, while leaving the first failure preserved.

Controller is not Reporter. Engineering task utility, arithmetic agreement,
exact restored/copy controls and T2/T4 policy consequences cannot identify
attention model A as the source of qualia, qualify presentation manner or prove
experienced spatial form. All interpretations remain development native
role/task-content semantics; presentation qualification and the full project goal
remain unachieved.

Retained weights/losses cannot prove the entire optimizer trajectory without fit
replay. Equal before/after weights do not rule out transient mutations. Exclusive
registered receipts support one recorded attempt but cannot rule out unrecorded
external work. Conditional world intervals do not establish uncertainty over
people, independent model families or unseen layout combinations. The diagnosis
and corrected reference share model-team/hypothesis context and CPU arithmetic
primitives; the explicit reference equations remain independently written, while
production Controller reenactment is limited to its input isolation contract.

Static preparation validation: in-memory Python syntax compilation passed without
executing the program. AST comparison shows the original and corrected sources
have the same 123 core `audit.check`/`audit.compare`/`audit.tree` call sites, with
identical ASTs for those calls. Only existing `head_reference`,
`controls_reference`, the two native `verify_seed` call sites and `main` differ;
`corrected_reference_provenance` is the single new function. Original verifier
SHA256 remains `83b6c370c0ee19b6a77749fc01a0e06cca196e1e57c0b3786c8ec6084f4988d8`.
No source execution or archive/checkpoint load occurred during this preparation.
