# Reconstruction and production closure: first static review

2026-10-11. Independent task allocation within the same AI family, with inherited
project context; no human or independent-family endorsement. Source/text/hash
inspection only. No imports, compilation, fixtures, actual data/model calls,
registration, world generation, artifact/checkpoint loads, fits, Reporter or Git
operations. Coordinator-reported engineering fixture outcomes were not replayed.

Reviewed [verifier](../../../scripts/verify_attention_content_model_v1.py) SHA-256
`0332633eaef01755b458ec6c7b21b9373f1e949f79b295a3d2471c4141737902`,
its [design](../reconstruction/design.md), and the complete declared production
closure. Exact file hashes are in [metadata](metadata.json). Core bindings are
runner `bd4c41b2…`, engine `4f537c11…`, learned core `a950a422…`, collector
`7007081f…`, permission `8ea6ca99…` and amended design `b64062f3…`.
Also inspected unexecuted [ten-fixture draft](../../../tests/test_verify_attention_content_model_v1.py)
SHA-256 `e610269349758b19352ba010fd3b0bf7dc6fe48103f08c5ff9cd7381a6cd0c3a`.

**Reject these first exact bytes for production approval and fixture execution
until V1/V2/F1 below are preserved and corrected separately.** These are static
software findings, not actual A-study or test failures. Preserve this source and
review before correction. No scientific criterion or numerical tolerance needs
to change. Complete approval additionally awaits reviewed corrected fixtures and
their preserved first outcomes, final closure review and committed registration.

## Concrete required corrections

**V1 — failing diagnostic evidence can prevent retention of the failed report.**
`Audit.check` keeps arbitrary evidence without making it JSON-safe. `Audit.tree`
and `verify_visibility` copy actual primitive values into that evidence. For
example, a retained `current_completion_visible=NaN` fails its required Boolean
comparison, but the check also retains `actual=NaN`. `main` catches reconstruction
exceptions, then executes `json.dump(..., allow_nan=False)` outside that handler.
The NaN causes a serialization exception after reserving the exclusive output;
the promised full failed first JSON can be left incomplete. A malformed tensor
or unsupported metadata object used as primitive evidence has the analogous risk.

Make all diagnostic evidence serializable with explicit faithful tags for
nonfinite/unsupported values, retaining the failing condition and original
exception; do not replace a rejected value with a passing default. Guard final
report serialization and retain its original failure information in the exclusive
first output/log. Add bounded negative fixtures that reject malformed evidence
while writing a parseable failed report and preserving the first result. This
changes preservation only, not the comparisons or opportunity inventory.

**V2 — the independent Adam step-count check is tolerant.**
`verify_fit` compares each terminal optimizer `step` with `300.` through default
`Audit.compare`. With atol1e−6/rtol1e−5, deviations up to approximately .003 are
accepted, although the registered update count is exact and the runner checks
`float(step)==300`. Independently require an exactly300 scalar step value, with
appropriate finite/type/shape validation. Do not apply numeric reconstruction
tolerance to this discrete counter and do not alter the existing numerical
tolerance for weights, maps, logits or fields. Include a tiny fractional-step
negative fixture; recorded terminal steps still do not prove a fitting trajectory.

**F1 — the first native-field fixture uses the wrong return API.**
`test_native_head_itemwise_equations_and_manual_fields` calls
`role_fields(item).tensor()`. The imported `spatial_native_task_v1.role_fields`
returns a `torch.Tensor`, with no `.tensor()` method. Remove that method call
in a separately preserved draft correction. The original ten fixtures remain
unexecuted; this is a pre-execution API defect, not a first test failure.

## Supported source coverage and practical limits

The closure covers the executed local import graph: runner loading
`attcon.bound_content` invokes package initialization/data/models, and
bound_content imports predictive_attention; these files are bound. The package's
lazy train/eval functions are not invoked by these paths. Torch and standard
library dependencies remain environment dependencies rather than local source
files. Actual frozen encoder/H/coordinates are accepted input archives, not
substituted fitted A hashes.

Before tensor loading, the verifier checks complete source approval/registration,
committed current bytes, phase inventories/claims/receipts and ancestry. Native
acceptance pins the corrected first206070-check review and its actual-result
bindings, original result commit and verifier provenance. Heldout requires the
bound passing independent-dev report, same fit/registration lineage and explicit
source semantic dispositions with reporting false. Negative semantics permit
the fixed technical heldout run. This task inspected no live acceptance artifact.

Independent equations cover delivered pixel→V convolution/pool, explicit
same-V block replacement, fit-only coordinates, seed initialization, recorded
fit budget/logs/optimizer artifacts, first-update losses, GRU/decoder evolution,
uniform analytic survivors, current/next assembly and inherited uncertainty.
Donors reconstruct both predetermined schedules with t executor transitions
and t−1 permitted reveals, addressed patching, reached states, restoration,
current/next H fields, scalar/factual effects and null-inclusive summaries.
Original-W controls use itemwise F.linear suffixes; native H uses the reviewed
itemwise [4,4,24] arithmetic. After tolerant field comparison, exact cells/ties/
coarse orders and symbols use archived emitted fields, avoiding reference-rounding
policy errors.

Strict tensor/key/shape inventories cover all sources and donor ledgers. Separately
labeled production-core replay checks exact call hashes, input isolation,
aliases, address remapping, donor/decoder/restoration records and fixed counts.
It is an additional replay of the same implementation, distinct from independent
equation checks. Bootstrap reductions/quantiles and paired differences cover the
registered complete selected/full/inherited and role-specific diagnostics.
Copies/restoration/isolation/discrete policies stay exact; address permutation
uses the approved numerical tolerance. No new science threshold is introduced.

Namespace/schedule checks and approved collector source support separate released
worlds; delivered pixels and frozen extraction are reconstructed without renderer
replay. This cannot prove unrecorded external work or the complete optimizer
trajectory. Fit diagnostics remain fit evidence, fixed engineering seeds are not
an independent-family sample, and intervals are conditional world resamples.
Literal H(A) content, presentation manner and later report correspondence remain
separate assessments. Controller is not Reporter. Passing reconstruction would
not qualify experience or complete the user's conditional faithful-report goal.
