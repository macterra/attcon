# Preserved-failure arithmetic diagnosis design

2026-10-11. Prepared by `/root/spatial_native_numerical_execution_review` within
the same model team and with inherited hypothesis/project context. This is a
separate task allocation, not independent-family or human endorsement. Preparation
uses only static source/receipt reads. No diagnostic inference is permitted before
the coordinator commits this exact program and design and explicitly authorizes
their first execution.

The owned executable is `scripts/diagnose_spatial_native_numeric_reference_v1.py`.
Owned results are this design, `first_run.log`, exclusive `metadata.json` and
`review.md` in this directory. No Git mutation or GitHub operation is performed
by this reviewer. The original verifier, source, registration, archives, tolerances,
engineering gates and first review remain unchanged. The diagnosis cannot relabel
the first failed verification as passed.

## Bound evidence and first-pass preservation

The program requires exact first-results commit
`614e74ae74ccf1d553f6fc6e0ec318f057a94ed8`, first-review preservation commit
`432146ae8f1420b13bacdbb8f00518570d34b95f`, and an explicitly supplied committed
diagnosis program/design revision. It checks current and committed source,
registration, archive, first verifier, first review/log and compressed first
metadata bytes before loading archives or importing torch/reference equations.
The gzip metadata is decompressed and bound to its original raw SHA256 and full
206,070-check / 106-failure inventory. The new exclusive report is reserved before
runtime work; unchanged stdout/stderr go to the new exclusive first diagnostic
log. Exceptions and unresolved findings are preserved without automatic retry,
fallback or correction.

## Hypotheses, including a static correction

The head reference uses higher-rank `x @ weight.T + bias` while the native
Controller consumes one item at a time through `nn.Linear`. The diagnostic will
measure whether the operation, tensor shape, stride or bias-addition path changes
float32 outputs enough to perturb exact tied maxima. It does not assume this is
the cause before executing the declared comparisons.

The suggested control leading-axis hypothesis is not supported by static source:
the existing `controls_reference` already uses `maps.reshape(-1, 384)`, just as
the native encoder suffix flattens the leading item axes. Control diagnosis
therefore measures flattened `F.linear` versus separate matmul then bias at the
same input shape and full high-precision equations. No axis discrepancy is
assumed or asserted.

## Fixed diagnostic calculations

Only the three committed archives are loaded. The original reference module is
imported without running its verifier; its reference math is used to reproduce
the original discrepancies. Separate explicit equations use the archived weights
and fit-only coordinates, without the production head, Controller or source
helpers. No encoder checkpoint loading, world rendering, sampling, optimizer,
fitting, selection by utility gates or study rerun occurs.

For each model's T1/T2/T4 and synthetic-permutation saved W, independently compare:

- Original batched matmul-plus-bias head arithmetic.
- Itemwise `F.linear` with exactly the native input shape and strides.
- Itemwise matmul-plus-bias, batched `F.linear`, and flattened-site `F.linear`,
  isolating operation choice and shape effects.
- Float64 equations on the exact standardized float32 input, and separately
  float64 equations using full float64 coordinates. These are comparison
  references, not changes to the registered float32 task.
- Explicit float64 exponent/sum normalization and separately float64 softmax
  with exact uniform handling. Exact discrete policies are reconstructed without
  relaxed ties or any new tolerance.

All fixed map cases are checked, including cases without an original failure.
The diagnostic checks exact saved-W equality against the original reference
block equation; logits, role fields, actions, ties and fine/coarse policies are
compared with the archived outputs. For every pair of exactly identical input
site vectors it measures differing output logits/fields. This identifies whether
a reference arithmetic path breaks repeated-site symmetry. Higher-precision
agreement alone does not certify the registered float32 output.

For original diagnostic/fit maps, all three primary W maps and three synthetic
maps, compare the original control reference with explicit flattened `F.linear`
and full float64 equations. Subtract the independently reconstructed original
baseline using each path to locate delta amplification and tolerance crossings.
The unchanged numerical tolerance is atol=1e-6, rtol=1e-5; exact discrete/copy
comparisons retain zero tolerance. The report includes endpoint values for up to
three violating elements per failed comparison, complete quantitative comparison
statistics, and original first-check evidence.

All 106 original failing check names receive a diagnosis entry. A tensor failure
is explained only when its first recorded shape/dtype/max-difference/unequal/
violation metrics reproduce exactly and matched explicit linear arithmetic
matches the archived output exactly. Head entries also require exact matched
logits and fields; control deltas also require exact changed and baseline
endpoints. The final derived verdict is explained only by its preserved count of
105 prior discrepancies. Otherwise the item remains explicitly unresolved.

## CPU dispatch evidence and limits

One declared profiler schedule on seed 1301's retained T4 inputs captures CPU
ATen operator dispatch and input shapes for batch/item matmul, batch/item/flattened
linear, and flattened controls with matmul or linear. Shapes, strides, contiguity,
torch/Python versions, CPU identity, build backend configuration and thread counts
are retained. This supplies observed CPU operation evidence, not assumed dispatch
or vendor microkernel assembly. Profiling is part of this fixed diagnosis schedule,
not a retry of the historical study or verifier.

Even complete arithmetic explanation is diagnosis only. A separately authorized
correction and labelled numerical revalidation would still be required. Controller
is not Reporter; utility gates and arithmetic agreement cannot identify attention
model A as a phenomenal source, qualify presentation manner or establish qualia.
The full project goal remains unachieved. Conditional world intervals remain
conditional on the fixed models and shared layout support, without human,
independent-family or unseen-layout generalization endorsement.
