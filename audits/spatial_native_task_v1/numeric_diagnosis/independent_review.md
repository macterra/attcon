# Independent static adjudication of the preserved arithmetic diagnosis

2026-10-11. `/root/spatial_content_manner_critical_review` inspected the committed
diagnosis design, program, review and retained metadata, together with the
original verifier, first failed review and byte-identical compressed first
metadata. This review performed source inspection and JSON consistency checks
only. No trace archive or checkpoint was loaded, no torch/model calculation,
inference, fit, new world, Reporter call, Git mutation or external communication
was performed by this reviewer.

**Verdict: the retained diagnosis supports a narrowly corrected independent
reference implementation. It neither passes the original failed verification
nor supplies a passing full revalidation.** The head correction must preserve
the native item's tensor shape; the control correction concerns matched flattened
linear arithmetic, not a leading-axis defect. No study output, source, gate,
tolerance, exact-tie policy or semantic conclusion needs to change.

## Evidence checked independently from retained JSON

The decompressed first metadata has SHA256
`6010154b9502daef7f223ac3da4ac9c7854c4c9bc15436b8d987d9a5c93dc338`, status
`failed_first_verification`, 206,070 checks and 106 failures. Its original check
records exactly equal the diagnosis's embedded original records. The diagnosis
contains exactly the same 106 unique failed names, without omitted or additional
entries: 35 head/discrete-policy discrepancies, 70 control discrepancies and one
derived final verdict. Every reproduced tensor comparison retains the original
shape, dtype, finiteness, maximum absolute difference, unequal-element count,
tolerance-violation count and compared-element count. All 105 reproduced tensor
comparisons remain failed against the original equations.

For each of those 105 entries, the retained matched-arithmetic comparison passes
exactly against the archived quantity. Head entries additionally retain exact
matched logits and role fields. All 60 delta entries retain exact matches for
both changed and original baseline endpoints. The final verdict entry explains
the preserved count of 105 preceding discrepancies; it is not a newly passed
invariant. All 39 recorded byte bindings agree among expected, current and
committed digests. This reviewer checked those recorded binding equalities and
the current program/original-verifier hashes, without independently reading
archives/checkpoints or repeating their runtime bindings.

Across all 18 saved primary/synthetic head cases, both independently written
itemwise matmul-plus-bias and itemwise F.linear retain exact logits/field matches,
zero discrete-policy discrepancies and zero identical-input symmetry violations.
Explicit exponent/sum normalization on the matched itemwise logits also matches
the archived fields exactly. In contrast, the original batch reference and batch
F.linear each retain 24 policy-comparison failures; flattened F.linear retains
27. These counts concern the fixed diagnosis cases and exclude repeated
address-control failures in the first ledger. All 96 matched control-quantity
comparisons across 24 saved map cases are exact. These metadata counts and
conditions agree with the diagnostic review; no inconsistency was found.

## Why the correction is legitimate

The original head equation applies higher-rank matmul to all items jointly, while
the native targeting consumer applies the shared head to an individual map. The
recorded input layouts differ: native moved-axis input `[4,4,24]` versus the batch
`[64,4,4,4,24]`. The profiler retains different `mm`/`bmm` dispatch and bias-addition
paths. The original batch path breaks some exactly repeated site-vector output
symmetries; the itemwise paths retain them. Very small float32 differences are
sufficient to change an exact maximum or tie count. Resolving this by matching
itemwise equations preserves the actual registered discrete policy, whereas
relaxing ties would replace that policy.

This is a concrete measured reference-versus-consumer arithmetic mismatch. It is
not grounds to copy archived policies as expected values, or to assume batch
F.linear/flattening is equivalent. Itemwise explicit matmul-plus-bias is a useful
correction because it retains independently written equations and the original
normalization/policy reconstruction, without invoking the production head or
Controller. The diagnostic's extra F.linear and high-precision comparisons
corroborate the explanation; they need not become new study computations.

The original controls already flatten leading axes before hidden64, category and
orientation readouts. The diagnosis correctly rejects an axis-change explanation.
At matched flattened shapes, separate matmul-plus-bias versus F.linear dispatch
changes rounded endpoints. Their subsequent subtraction can cross the unchanged
delta tolerance even when each endpoint difference is small. Exact independently
reconstructed changed/baseline endpoints account for every retained delta failure.
Using explicit flattened F.linear equations is a legitimate reference repair to
match the registered native linear operations while keeping source weights,
normalization, readout graph and all control outputs independently assembled.

The profiler demonstrates ATen dispatch and input shapes on this recorded CPU
environment. It does not establish vendor microkernel assembly or universal
backend behavior. “Different observed arithmetic paths explain the retained
discrepancies” is warranted; a stronger unmeasured hardware claim is unnecessary.
Float64 comparison preserves the relevant mathematical symmetries but does not
equal registered float32 outputs and cannot replace their evaluation contract.

## Required guardrails for a corrected full verifier

1. Preserve the first verifier, first failed metadata/log/review, diagnostic first
   pass and their byte bindings unchanged. Label the new executable, receipt,
   log and findings as a corrected reference revalidation. The historical first
   status remains failed even if every later check passes.
2. Make the diff limited to independently assembled head/control arithmetic and
   necessary preservation/provenance plumbing. No expected tensor may be copied
   from the archive, and no production head/Controller helper may replace the
   independent reference. Shared low-level F.linear is acceptable for controls;
   it is not an independent numerical backend or a new experiment.
3. Preserve float32 task coordinates, exact copied/repeated-block/restoration
   comparisons, exact row-major argmax/ties, atol=1e−6 and rtol=1e−5, evaluator
   truth separation, native utility gates and all opportunity denominators.
   No near-tie tolerance, tie collapse, changed target, float64 substitution or
   omitted control comparison is justified by this diagnosis.
4. Handle all original head call sites explicitly. The diagnostic itemwise loop
   is hardcoded to 64 worlds, four items and 24 channels; a corrected verifier
   must also retain the original fit/initial-loss and raw-grid-head comparisons
   with their actual shapes. Their recorded fit/raw sources may use a batched
   consumer path. Do not blindly replace every reference call with the native
   itemwise diagnostic case or accidentally remove those checks. Static review
   must establish the intended shape at each call site before execution.
5. Preserve every original check and require a full inventory-equivalent
   206,070-check reconstruction. Comparing only the 106 former failures, even
   exactly, cannot establish that a changed shared function leaves other
   comparisons valid. Additional preservation checks must be counted separately
   rather than obscuring the original inventory.
6. Bind the corrected executable and approved diff/receipt in a preserved commit,
   verify unchanged study/archive/registration/first-review bytes before runtime,
   and use new exclusive log/report paths. Record backend, thread and tensor
   layout context. A new failure must remain a new retained failure, with no
   automatic favorable-path fallback, refit or study retry.

The currently available diagnosis satisfies the evidential basis for considering
this repair. This adjudication does not approve an unseen corrected executable
or authorize its runtime. Once that source is finalized, the exact diff and its
call-site/inventory preservation can receive a bounded static pre-execution
decision. No human or philosophical-proof blocker is introduced.

## Scientific scope after any successful repair

A passing corrected full revalidation would support numerical reconstruction of
the same retained native-task experiment. It would resolve this arithmetic
reference issue, not create independent data, a replication, another model-family
sample or a new semantic endpoint. Existing conditional-world uncertainty and
fit-trajectory limitations remain. Native targeting evidence and 766/768
coarse-retained field changes would retain their task/content interpretation.

The qualified-presentation negative remains intact: repeated fine-location
evidence, access changes and prescribed filter mechanics are still sufficient
explanations. No learned A or Reporter was introduced by the native run. A
Controller executes targeting; it is distinct from an external Reporter.
Accurate future A-content reports with independently recognized consciousness-like
structure remain a legitimate restricted goal, without requiring proof of private
experience or task necessity. Arithmetic repair alone does not establish that
goal or release a source/report gate.

This adjudication is separate AI task allocation within the same model family
with inherited project context. It is not independent-family evidence, human
endorsement or a numerical rerun by this reviewer.
