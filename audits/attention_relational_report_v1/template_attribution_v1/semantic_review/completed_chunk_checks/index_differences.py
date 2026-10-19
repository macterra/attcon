"""Retain explicit first-label/operand differences after both contexts finish.

No semantic grade, majority vote, endpoint, repair or feedback. Equal labels need
not mean equal reasoning. Operand-inventory differences can reflect extra context
or granularity rather than contradictory assertions. Full originals are retained.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from check_pair import ROOT, digest


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def inventory(properties):
    return Counter(encoded({k: p[k] for k in ('unit_id', 'value_path', 'value', 'semantic_role')})
                   for p in properties)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunk', required=True, type=int, choices=range(23))
    args = parser.parse_args()
    base = Path(__file__).parent
    check_path = base / f'first_pair_{args.chunk:02d}.json'
    checked = json.loads(check_path.read_text())
    checks = {c['reviewer_id']: c for c in checked['checks']}
    assert set(checks) == {'reviewer_0', 'reviewer_1'}
    records, dependencies = {}, {str(check_path.relative_to(ROOT)): digest(check_path)}
    for stream, check in checks.items():
        path = ROOT / check['first_artifact']['path']
        assert digest(path) == check['first_artifact']['sha256']
        dependencies[str(path.relative_to(ROOT))] = digest(path)
        records[stream] = {r['id']: r for r in json.loads(path.read_text())['records']}
    first, second = records['reviewer_0'], records['reviewer_1']
    assert set(first) == set(second) and len(first) == 24
    both_valid = all(c['structurally_valid'] for c in checks.values())
    output = []
    for identifier, left in first.items():
        right = second[identifier]
        item = dict(id=identifier, reference_established=left['reference_established'],
            original_statuses_unchanged=True,
            review_comparison_state='both_first_chunks_valid' if both_valid else 'unestablished_whole_chunk',
            explicit_label_differences=[], property_inventory_differences=[])
        if both_valid:
            assert left['reference_established'] == right['reference_established']
            sections = [('component_reviews', 'component_path', ('assessment',)),
                        ('evidence_reviews', 'evidence_path', ('grounding', 'source_meaning')),
                        ('binding_reviews', 'binding_path', ('assessment',))]
            for section, path_key, fields in sections:
                indexed = {encoded(r[path_key]): r for r in right[section]}
                assert set(indexed) == {encoded(r[path_key]) for r in left[section]}
                for a in left[section]:
                    b = indexed[encoded(a[path_key])]
                    for field in fields:
                        if a[field] != b[field]:
                            item['explicit_label_differences'].append(dict(section=section,
                                path=deepcopy(a[path_key]), field=field,
                                reviewer_0=deepcopy(a), reviewer_1=deepcopy(b)))
                    if section == 'evidence_reviews' and inventory(a['properties']) != inventory(b['properties']):
                        item['property_inventory_differences'].append(dict(
                            evidence_path=deepcopy(a[path_key]),
                            reviewer_0_properties=deepcopy(a['properties']),
                            reviewer_1_properties=deepcopy(b['properties']),
                            semantic_disagreement_established=False))
        output.append(item)
    result = dict(schema='attention_relational_completed_first_pair_explicit_differences_v1',
        chunk_number=args.chunk, input_sha256=dependencies,
        indexing_source_sha256=digest(Path(__file__)),
        contract_dispatch_source_sha256=digest(base / 'check_pair.py'),
        both_first_chunks_structurally_valid=both_valid, records=output,
        semantic_adjudication_performed=False, endpoint_computed=False,
        feedback_sent_to_first_reviewers=False,
        equal_labels_do_not_establish_semantic_agreement=True,
        operand_inventory_difference_does_not_alone_establish_semantic_disagreement=True,
        entire_corpus_analysis_complete=False)
    target = base / f'first_pair_{args.chunk:02d}_differences.json'
    with target.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(chunk=args.chunk,
        explicit_label_differences=sum(len(r['explicit_label_differences']) for r in output),
        differing_operand_inventories=sum(len(r['property_inventory_differences']) for r in output))))


if __name__ == '__main__':
    main()
