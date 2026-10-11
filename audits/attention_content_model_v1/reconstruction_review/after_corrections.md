# Reconstruction corrections: thirteen-fixture static approval

2026-10-11. Separate review task within the same AI family, with inherited context;
no human or independent-family endorsement. Source/text/hash inspection only.
No imports, compilation, tests, production data/model calls, registration,
generation, artifact loads, fit, optimizer, backward, Reporter or Git operation.

Final author/coordinator bindings match inspected bytes:

| File | SHA-256 |
|---|---|
| [Corrected verifier](../../../scripts/verify_attention_content_model_v1.py) | `11b821bea207f4ef8250fa589148777df9d600bb3dffd01ef600cf4f78a423ab` |
| [Thirteen fixtures](../../../tests/test_verify_attention_content_model_v1.py) | `365103a3973571f88f5058bb205ee6eeae7f7be0498aeb8d9ee386b3f1afa198` |
| [Correction record](../reconstruction/after_corrections.md) | `91ab175af0ada3027060d8578f9213d4c6e1f5ab23376c7543987375890ad040` |
| [Final preparation metadata](../reconstruction/metadata.json) | `966e1c1f219b51b6e11e50ba31933b3335d5424d5224598bbd677e5159005571` |

**V1/V2/F1 are resolved statically. Approve only one first execution of these
thirteen bounded software fixtures after preserving/pushing the exact sources,
correction records and review.** Preserve the first exclusive log/results and
every failure before any repair or separately labeled repetition. No fixture
outcome is asserted here. Actual phase execution, production registration and
complete production closure approval remain unapproved pending first fixture
results and a separate exact-closure approval.

V1 now serializes before writing. If strict JSON rejects raw evidence, it retains
all original checks/errors and the verification exception, adds a separate
serialization traceback, marks failure, and tags nonfinite primitive values with
their explicit type/value. Tensor/bytes/unsupported evidence receives a labeled
encoding rather than removal. Retained input archives are untouched. Write/flush/
close faults retain the original writer traceback and full first report in a new
exclusive `.writer_failure.json` sidecar, leaving the partial primary untouched.
If sidecar writing also fails, the full record and secondary error go to external
stderr capture. A durability failure stays failed; the code cannot guarantee
storage when both filesystem and log transport fail. It makes no primary retry
or overwrite.

V2's existing named step check now requires a finite numeric count exactly300,
rejecting fractional departures even when a continuous numeric comparison would
pass. Numeric singleton tensor conversion is permitted; the helper rejects
multi-element or nonnumeric counts. ATOL1e−6/RTOL1e−5 remain unchanged for
continuous reconstruction, and recorded counters still do not prove a trajectory.
F1 removes `.tensor()` from the already Tensor-valued `role_fields` result.

An independent byte reconstruction removes only the newly added preservation/
exact-count helpers, restores the previous step-comparison call and final
JSON-write integration, and obtains the original verifier SHA-256
`0332633eaef01755b458ec6c7b21b9373f1e949f79b295a3d2471c4141737902`.
Thus all other equations, check sites, schedules, inventories and tolerances are
unchanged. The [first rejection](design_review.md), its metadata and historical
design/draft fixture records remain preserved; no executed failure is relabeled.

The thirteen fixtures import torch and perform numerical comparisons on
handcrafted maps/fresh GRU, decoder and local-head modules. Fixed seeds71/72/73
use isolated RNG state; four CPU threads and deterministic algorithms are set
before fixture work. They include literal block means/point fields/ties, fixture
normalization moments/MSE identities and paired quantile interpolation. Normalizer
moment calculation is not registered source training. No optimizer, backward,
reserved schedule/world, renderer, native archive/checkpoint or Reporter is used.

Binding fixtures use temporary source bytes and mocked Git process streams. CLI
fixtures use temporary report paths: the missing-registration case fails before
native acceptance/loading, and the new malformed-evidence case replaces `run`
entirely. No actual registration, phase, Git subprocess or artifact load occurs.
The added negatives require all checks and original traces to survive NaN/Inf
serialization, fractional terminal steps to fail while ordinary tolerance stays
fixed, and a deliberately partial writer output to remain unchanged beside full
exclusive sidecar evidence. Existing report paths are rejected. The approved
suite covers these thirteen named fixtures only, not the real reconstruction run.

A passing suite would validate these bounded equations and preservation paths,
not authentic phase sealing, source artifact acceptance, complete production
reconstruction, optimizer history or source semantics. A complete approval must
later bind the exact REQUIRED_SOURCES closure, including this corrected verifier,
runner, engine, core, collector, permission/design and transitive local modules.
First actual outputs and immutable receipts then require their own reconstruction.
Literal content, manner and conditional conscious-report correspondence remain
separate. Controller stays distinct from Reporter; reporting remains gated and
the full scientific goal remains unachieved.
