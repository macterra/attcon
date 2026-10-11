# Evaluator collector: static review after rejection corrections

2026-10-11. `/root/spatial_content_manner_critical_review` reviewed the corrected
collector and seven handcrafted fixtures against the preserved first D1/D2 review.
This pass performed text/AST/hash inspection only, with no imports, tests, reserved
schedule/render calls, checkpoints, archives, model inference, fits, Reporter,
Git mutation or external communication.

**Verdict: D1 and D2 are resolved in the reviewed source. Approve these bounded
evaluator helpers and handcrafted fixture scope for integration into the later
reviewed runner. No actual A study or full collector/runner execution is approved.**

| Reviewed file | SHA256 |
|---|---|
| Corrected collector | `7007081fe37d5a7a01e9ec139932fd282109550ce87fff122fe44f9fbb8b16cd` |
| Corrected tests | `eb9f2dde703f03e496e2dc38ca4fe921e3efd5cc5f0c97153a29efcdec6f81ea` |

D1 is closed by upfront `_integer_tensor` validation of the complete common
address and symbol arrays before any fork construction or indexing. It requires
CPU long tensors with exact batch×15 shape and address/symbol ranges 0..3/0..2.
A negative common symbol can no longer silently choose the last alternative.
The added fixture tests −1 and 3 while mocking the executor to assert that no fork
is constructed; float address/symbol arrays are also rejected.

D2 is closed by `_mapping_tensor` validation before any bandwidth conversion.
It requires CPU long tensors of shape `[B,4,3]` whose sorted last axis equals
`[1,2,4]` exactly at every address. Fractional inputs, duplicate bandwidths and
other invalid rows cannot pass by integer truncation or valid-band repetition.
Donor mapping indices, addresses and command arrays additionally use the strict
integer/range helper before cloning, lexicographic increment or replay. Negative
and out-of-range donor indices are rejected. The new mapping fixture covers
fractional/float tensors, duplicate rows and donor indices −1/6. Device rejection
is explicit in source; these CPU-only fixtures do not exercise another device.

The corrections leave the prescribed schedule namespaces, six-permutation
balance, mapping-independent command history, same-V/noncumulative executor,
shared actual targets, donor t-execution/t−1-feedback index lists and pre-render
split-string guard intact. Existing positive-path fixtures remain present.
The coordinator reports the separately retained corrected fixture log passing
seven tests, exit 0; this reviewer did not execute or independently replay that
log. The original first source/test/log/review preservation at `036ac46` is
coordinator-supplied history. Original review and metadata still match their
recorded hashes, and this pass does not replace their first defect finding.

Upstream limitations are unchanged. `released_split` is a verified-runner-supplied
string, not an immutable registration/phase-receipt or native acceptance check.
Direct schedule/donor/helper calls are not intrinsically sealed. The returned
episode dictionary is evaluator-only and includes mappings, pixels, truth and
concealed targets. The later runner must enforce permitted source arguments,
delayed observation masks, donor replay/patch/restoration, shared/comparator
ledgers, native corrected acceptance, frozen input hashes and immutable split
release. This static correction pass establishes none of those runtime behaviors.

No native utility gate, train budget, source-semantic requirement or report gate
changes. No learned A or consciousness-report correspondence is tested. The
Controller remains distinct from Reporter; presentation qualification and the
project's restricted source/report goal remain open. This is separate task
allocation within the same model family with inherited context, not independent
family, human or empirical-replication endorsement.
