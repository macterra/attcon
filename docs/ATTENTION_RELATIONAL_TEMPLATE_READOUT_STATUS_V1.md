# Template readout implementation and execution status

2026-10-19. The [attribution plan](ATTENTION_RELATIONAL_TEMPLATE_ATTRIBUTION_V1_PLAN.md)
is a frozen source input. Current execution notes belong here, outside its pinned bytes.
All 46 fixed first contexts have ended: 45 produced first artifacts and one ended
without judgments. No first context is replaced or gap-filled.

The [first full collection execution](../audits/attention_relational_report_v1/template_attribution_v1/semantic_review/collection/first_collection_execution.json)
stopped before opening first-review artifacts. Root had appended the implementation
notes below to the frozen plan, causing its hash to differ. The unchanged collector
correctly rejected that lineage. The [diagnosis](../audits/attention_relational_report_v1/template_attribution_v1/semantic_review/collection/first_failure_diagnosis.json)
and [as-executed document](../audits/attention_relational_report_v1/template_attribution_v1/semantic_review/collection/first_failed_execution_plan.md)
are retained. The plan is restored byte-for-byte to its original SHA-256
`5abe470b67693a23e7d3ec80c74e7bd4906175d1317fc577fd3ccfe90e0333b3`;
this page preserves the relocated notes. No collector, manifest, judgment, criterion,
scope or binding changes. The separately recorded operational collection retry
uses the same first judgments after all pinned sources match.

The [second collection execution](../audits/attention_relational_report_v1/template_attribution_v1/semantic_review/collection/second_collection_execution.json)
now completes with exit zero using those unchanged inputs. The
[root inventory](../audits/attention_relational_report_v1/template_attribution_v1/semantic_review/collection/first_collection_inventory.json)
retains 46 contexts, 1,104 reviewer positions and all 684 pairs. Forty-five chunks
pass the frozen mechanical contract; reviewer 1's chunk 09 remains missing in
full. Reviewer 0 retains source judgments for all 148 established original
annotations; reviewer 1 retains 142. The other 404 original template annotations
remain unestablished, and the six established positions lacking the second review
are not gap-filled. This is completeness accounting, not a semantic endpoint.

The complete 51,322,429-byte collection has SHA-256
`cfa555fb0c778c6f3f2ee3fd50434f87265edc4910db03a0385580ced7afcc79`.
Its [verified lossless archive](../audits/attention_relational_report_v1/template_attribution_v1/semantic_review/collection/first_collection.json.archive.json)
stores exactly those bytes in a 5,115,549-byte gzip. Restore the raw file using
`scripts/preserve_large_evidence_v1.py --restore` with that sidecar before analysis.
The qualia-source goal remains unachieved.

## First complete binding inventory

The [first inventory execution](../audits/attention_relational_report_v1/template_attribution_v1/binding_inventory/first_execution.json)
completes with exit zero. Its [complete output](../audits/attention_relational_report_v1/template_attribution_v1/binding_inventory/first_inventory.json)
has SHA-256 `9438c05913719a4e94d7e78341d3a9a629226e40aac323322de49850d4afaa02`
and retains every position and pair. No original binding has all required recorded
support facts true in both first streams. This describes the retained judgments;
it does not establish feature absence, semantic truth, source independence or
absence of experience. Original COD labels remain 135 explicit, 12 ambiguous,
one absent and 404 unestablished. CA remains 12 explicit, five ambiguous,
131 absent and 404 unestablished. No label or scope is upgraded or erased.

This projection keeps each reviewer's selected witnesses and exact atomic source
operands, including unmapped containers. The full collection retains additional
evidence, concerns and diagnostics. Semantic predicate tracking, semantic nulls
and semantic restoration remain unassessed; numerical property comparisons are
separately retained. Independent accounting verification and explicit semantic/
source closure remain necessary for the final readout.

## Readout implementation status

The [binding-inventory utility](../scripts/attention_relational_template_binding_inventory_v1.py)
was prepared before the actual inventory above. Its [ten fictional checks](../audits/attention_relational_report_v1/template_attribution_v1/binding_inventory_source_checks/first_retained_final_source_checks.json)
pass. It joins original scopes, each first
reviewer's chosen evidence, exact atomic property identities and the full pair
inventory without changing first judgments. It retains missing whole chunks,
mixed copied-W/learned-source operands and unresolved atomic mappings. Different
compatible witness paths need not match between reviewers.

The [independent static review](../audits/attention_relational_report_v1/template_attribution_v1/binding_inventory_static_review/first_review.md)
finds no blocking defect under the trusted complete frozen-collector contract.
Its ten fictional checks pass independently. This compact projection must remain
linked to the full retained collection, which preserves unselected/auxiliary
evidence, row concerns and whole-chunk diagnostics; it cannot replace that archive.
The [root disposition](../audits/attention_relational_report_v1/template_attribution_v1/binding_inventory_static_review/root_disposition.json)
accepts both output and interpretation limits without changing the reviewed code.

Recorded support candidates still require semantic and source closure. The
utility leaves semantic predicate tracking, semantic nulls and semantic
restoration unassessed; the separately verified numerical bank supplies their
lower-level property facts. This is preparation for the existing analysis,
not completion of Stage 2 or Stage 3, a new semantic instrument, or evidence for
the theory. All fixed first contexts must actually terminate before collection.
