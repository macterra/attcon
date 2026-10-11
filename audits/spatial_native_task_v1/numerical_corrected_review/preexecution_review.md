# Corrected numerical reference: static pre-execution review

2026-10-11. `/root/spatial_content_manner_critical_review` inspected the exact
original/corrected source diff, corrected design, independent reference call
sites and preservation logic. In-memory syntax compilation and AST comparison
were performed without importing/executing either verifier, torch, checkpoints
or trace archives. No worlds, model inference, fits, Reporter calls, Git mutation
or external communication occurred.

**Verdict: approve these exact source/design bytes for one bounded, separately
labelled full corrected-reference revalidation after their committed preservation
and coordinator release.** No actual execution or passing result is asserted.
The approval covers numerical reconstruction of the retained first experiment;
it does not approve a new study run, A development execution or Reporter release.

| Bound file | SHA256 |
|---|---|
| Corrected verifier | `03381da8405d6ebfec121dc1c8b476e0dfe93eaf48ee536af41c8f6c831e2e56` |
| Corrected design | `1f2c4ed0c6fc5b4fd638be89af6c45b1ad621aebb1d26fc6b2dddf0bb1d4b7b8` |
| Unchanged original verifier | `83b6c370c0ee19b6a77749fc01a0e06cca196e1e57c0b3786c8ec6084f4988d8` |

## Exact arithmetic and call-site assessment

The native head still uses independently written matmul and separate bias
equations. Only primary native, address-remapping and synthetic-site assessments
request itemwise evaluation. Standardization and moved-axis ordering remain
unchanged. For the documented normalized batch layout
`[64,4,4,4,24]`, stride `[1536,384,4,1,16]`, merging the first two axes gives
`[256,4,4,24]`, stride `[384,4,1,16]`; each item consequently retains native
`[4,4,24]`, stride `[4,1,16]`. No flattening of site/channel axes or contiguous
copy is introduced. Output stacking restores the original leading axes and
`[3,4,4]` logits. The itemwise branch is only used for the fixed 4×4 native head.

Fit final-logit/final-CE comparisons, initial CE, and raw-grid diagnostic
assessment retain the original batched default. The correction therefore avoids
applying native itemwise arithmetic to contexts recorded with batch computations,
and does not substitute a 24-channel diagnostic loop into the raw 3-channel head.
All five head-reference call sites were inspected.

The four independent flattened F.linear equations reconstruct encoder hidden64,
category logits, orientation hidden32 and orientation logits using archived
weights. The original `reshape(-1,384)`, ReLU, frozen orientation normalization,
sigmoid coordinates, output reshaping and baseline-delta formulas remain intact.
No production extractor, suffix, readout or control helper replaces these
equations. Shared low-level F.linear is not claimed to be an independent numerical
backend. Float64 role subtraction/exponent/sum normalization remains unchanged;
no softmax substitution, tie tolerance or new rounding is introduced.

## Check inventory and preservation

Static AST comparison independently confirms the same **123 core
`audit.check`/`audit.compare`/`audit.tree` call sites with identical call ASTs**.
`Audit`, `run` and all map, source, normalization, loss, statistic, summary,
record, input-isolation, gate and tolerance routines are unchanged. The only
changed top-level functions are head/control references, `verify_seed` at its
two native-reference calls, and `main`; provenance is the sole new function.
The final result invariant continues to fail whenever any earlier core check
fails. No previous expected value is copied from archived output in place of
independent reconstruction.

The corrected main requires the runtime core check names/count/order to equal
the preserved first metadata's full ordered list. Thus **all 206,070 original
checks**, including formerly passing checks, must complete; a changed inventory
raises a retained verification exception. Supplemental provenance receipts are
outside that count. Current byte, checkpoint, source/review receipt,
registration/execution, actual-result, frozen-source and original-result ancestry
checks remain present. Numerical tolerances are still atol=1e−6, rtol=1e−5;
discrete/copy/restoration comparisons remain exact, and all utility gates and
unconditional opportunity denominators are unchanged.

Before torch import, additional provenance binds corrected source/design,
original verifier/failed review/log/design/metadata transport, and the diagnostic
program/design/log/review/metadata. Exact decompressed first-metadata and diagnosis
hashes, 106 diagnosed names, explained status and results→first-review→diagnosis→
corrected-reference ancestry are required. This identifies the same preserved
first experiment rather than silently evaluating different data. The corrected
reference commit is supplied only after these approved bytes are committed.

The new report is opened exclusively before runtime binding/imports. Use the
design's new exclusive `numerical_corrected_review/metadata.json` and unchanged
stdout/stderr `first_run.log`; the execution command must preserve the latter
with shell noclobber. No alternate-path retry, automatic arithmetic fallback,
source repair, refit or study rerun is approved. Any failure retains the new
labelled failed status and first outputs. The original `failed_first_verification`
metadata, verifier, log and review remain unchanged regardless of this outcome.

## Scope and practical limits

This static approval cannot demonstrate that the corrected equations pass all
archives or that actual tensor/backend behavior follows the retained diagnostic
examples. That is the purpose of the one full revalidation. A passing result
would resolve reference arithmetic reconstruction on the same fixed data, with
the original failure still preserved; it would not constitute independent-data
replication or prove the historical optimizer trajectory.

Native task/content feasibility, the existing negative presentation qualification,
ordinary/filter alternatives and conditional-world uncertainty are unchanged.
Controller remains distinct from Reporter. No A states or report correspondence
were tested here. Accurate future model-content reports with independently
recognized consciousness-like structure remain a separate bounded goal; private
experience proof, task necessity and human endorsement are not imposed.

This is separate AI task allocation within the same model family with inherited
project context. The exact approved hashes and runtime/interpretation limits are
recorded in the accompanying metadata.
