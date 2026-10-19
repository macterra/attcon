# First static source review

Reviewed `scripts/attention_relational_template_property_pairs_v1.py` at SHA-256 `58ef1153b0ac4f4b2aa8a8ba4e42e5bc3a6ae6b87f8de269f867b48dc8aa2560`.

Verdict: **blocked_pending_source_amendment_and_fictional_verification**. Source changed during review: False.

## F1: Temporal identity omits record, archived, feedback and nested represented times

Severity: high; blocking: True; source lines 68–86.

Identity contains only origin_chain[0].represented_time. All inherited origin fields use that same outer time, and peak identity omits the remaining origin chain temporal fields. For a fictional copied_prior_record at represented_time=12 whose record_time changes from11 to10 while its peak set stays equal, the peak property is matched and value_unchanged=True. A nested inherited_origin.represented_time change likewise leaves peak identity and other nested leaves matched. The changed temporal leaf itself is recorded changed, but equal leaves and peaks can still be labeled invariant across a time transition.

Required follow-up: Define the temporal coordinates establishing comparability and preserve the full relevant temporal origin chain without substituting represented_time for record/feedback/archive time. Changed temporal operands must be unmatched/unavailable rather than invariance. Add fictional record-time, inherited represented-time and archived-time transition fixtures.

## F2: Resolved address is not tied to indexed packet/distribution paths

Severity: high; blocking: True; source lines 62–76.

The indexed property path is checked for an equal value; source resolution is separately checked against context. No check proves that packet_path and distribution_packet_path select the node/role named by context. In a fictional packet with equal peak sets at two nodes but different raw distributions, substituting the other node distribution path still passes exact_peak_cells(raw)==unit.value and attaches the wrong raw operand hash. A substituted peak path with equal values likewise passes. The CLI pinned index digest mitigates arbitrary input corruption, but exact binding is not independently verified.

Required follow-up: Resolve the addressed packet node by panel/view/command/node and role and require exact origin/peak/distribution paths and values to be its operands. Reject cross-node, cross-role and cross-panel path substitution even when values coincide.

## F3: Construction failures abort instead of retaining unavailable measurements

Severity: medium; blocking: False; source lines 132–145.

validate_index, validate_export and property_table raise on construction failure. No per-template unavailable record or per-pair unavailable state exists. The successful path retains552/684 and failures do not become nulls, but explicit unavailable measurements within the full inventory are not implemented.

Required follow-up: Implement unavailable construction measurements while retaining all684 pairs, or scope this artifact as conditional on fully validated Stage1 and document the unimplemented construction-failure branch.

## Positive checks

- Matching keys contain no token IDs, positions, character offsets or rendered wording.
- Resolved origin_source_key separates copied W from learned A/ordinary/analytic; context retains panel/view/command/node/role.
- Every scalar nested origin field has an individual value_path.
- Complete exact maxima are independently recomputed from raw4x4 float32 distributions; all ties retained.
- Matched unchanged peaks retain a separate raw-distribution-change flag; missing/source/top-level-represented-time transitions receive None, never unchanged=True.
- Original pair inventory equality,552 tables and684 pairs are required; per-pair rows, source counts and own-baseline restorations retained.
- Restoration equality is nonvacuous and rejects unmatched rows.
- Semantic source meaning and asserted-predicate/endpoint tracking remain unestablished.

## Limits

- Static review only; fictional counterexamples reasoned from source, not executed.
- No assertion that any pinned target property actually suffers these cases.
- Pinned CLI index/export hashes protect exact input bytes; F2 concerns independent semantic address-to-operand validation, not evidence of index corruption.
- No claim of semantic correspondence, independent observations or full Stage3 completion.
- Root added and executed fictional tests concurrently; their source/results were not inspected for this first static review.

No actual target exports/indexes, reference annotations, semantic judgments or project result summaries read. No source/tests/target analysis executed by this reviewer; no network/API/Git calls or implementation edits. Full dependency hashes are in JSON.
