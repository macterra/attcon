"""Join retained first reviews to original bindings and atomic source properties.

Descriptive bookkeeping, not a semantic grader. In particular, two recorded
supported reviews are candidates for closure, not code-certified consciousness
reports. No new predicate or missing first judgment is supplied by this join.
The CLI requires the complete frozen collection before opening its target rows.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREP = Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation')
MANIFEST_SHA = '0f444a55011dffb7b53e1cb18f45a8304b44282aba9b65216eda30a2e0b094fe'
BANK = Path('audits/attention_relational_report_v1/template_attribution_v1/property_pairs/first_pairs.json')
BANK_SHA = 'c944d239cd52863fefa9331fc91800d048362d38d2fd7ee2659dc3b41d5706a5'
STREAMS = ('reviewer_0', 'reviewer_1')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('SHA-256 mismatch: ' + str(path))
    return json.loads(raw)


def resolve(value, path):
    for step in path:
        value = value[step]
    return value


def key(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def property_operand(ref, units, table):
    """Join exactly the named property; never expand a container to its leaves."""
    unit = units[ref['unit_id']]
    value = resolve(unit['value'], ref['value_path'])
    if key(value) != key(ref['value']):
        raise ValueError('review property differs from exact construction value')
    matches = [(identity, prop) for identity, prop in table.items()
               if prop['construction_unit_id'] == ref['unit_id']
               and prop['identity']['value_path'] == ref['value_path']]
    if len(matches) > 1:
        raise ValueError('nonunique atomic property identity')
    identity, prop = matches[0] if matches else (None, None)
    if prop is not None and key(prop['value']) != key(value):
        raise ValueError('bank property differs from exact reviewed value')
    return dict(first_review_property=deepcopy(ref),
                construction_kind=unit['kind'],
                construction_context=deepcopy(unit.get('context')),
                source_provenance=deepcopy(unit.get('source_resolution')),
                atomic_identity_sha256=identity,
                atomic_identity=None if prop is None else deepcopy(prop['identity']),
                atomic_mapping_state='exact' if prop is not None else 'no_exact_atomic_identity',
                missing_atomic_mapping_is_feature_absence=False)


def binding_inventory(record, judgment, table):
    """Retain each original graph and each first review's specific witnesses."""
    if judgment is None:
        return None  # Whole-chunk failure: no parsed-row salvage.
    if not record['reference_established']:
        if judgment['component_reviews'] or judgment['evidence_reviews'] or judgment['binding_reviews']:
            raise ValueError('unestablished original contains semantic inventories')
        return []
    units = {u['unit_id']: u for u in record['property_table']}
    components = {key(c['component_path']): c for c in judgment['component_reviews']}
    evidence = {key(e['evidence_path']): e for e in judgment['evidence_reviews']}
    bindings = {key(b['binding_path']): b for b in judgment['binding_reviews']}
    result = []
    for original in record['bindings']:
        path = original['path']
        reviewed = bindings[key(path)]
        name = path[1]
        annotation = resolve(record['canonical_first_annotation'], path)
        composite = record['canonical_first_annotation']['composites'][name]
        component = components[key(['composites', name])]
        features = []
        selected = []
        for witness in reviewed['feature_witnesses']:
            feature = witness['feature']
            c = components[key(['features', feature])]
            paths = witness['evidence_paths']
            features.append(dict(feature=feature,
                original_status=c['original_status'], first_component_assessment=c['assessment'],
                first_component_reason=c['notes'], selected_evidence_paths=deepcopy(paths)))
            selected.extend(paths)
        if {f['feature'] for f in features} != set(original['required_features']):
            raise ValueError('first binding does not name original required features')
        selected.extend(reviewed['binding_evidence_paths'])
        unique_paths = {key(p): p for p in selected}
        grounds = []
        for p in unique_paths.values():
            e = evidence[key(p)]
            grounds.append(dict(first_evidence_review=deepcopy(e),
                exact_original_evidence=deepcopy(resolve(record['canonical_first_annotation'], p)),
                operands=[property_operand(ref, units, table) for ref in e['properties']]))
        # These are facts about recorded support. They do not decide adequacy of
        # reasons, compatibility, scope, implicit representation or experience.
        support_facts = dict(
            original_composite_nonabsent=composite['status'] != 'absent',
            composite_application_supported=component['assessment'] == 'supported',
            required_features_nonabsent=all(f['original_status'] != 'absent' for f in features),
            required_feature_applications_supported=all(f['first_component_assessment'] == 'supported' for f in features),
            each_required_feature_has_selected_evidence=all(f['selected_evidence_paths'] for f in features),
            selected_evidence_grounding_supported=all(g['first_evidence_review']['grounding'] == 'supported' for g in grounds),
            binding_compatibility_supported=reviewed['assessment'] == 'supported')
        result.append(dict(binding_path=deepcopy(path), composite=name,
            original_composite_status=composite['status'], original_binding=deepcopy(annotation),
            original_scope=annotation['scope'], original_time=annotation['time'],
            first_composite_review=deepcopy(component), first_binding_review=deepcopy(reviewed),
            required_features=features, selected_evidence=grounds,
            recorded_support_facts=support_facts,
            all_recorded_support_facts_true=all(support_facts.values()),
            semantic_correctness_established_by_code=False,
            source_attributed_endpoint_established_by_code=False))
    return result


def join_positions(records, rows, property_tables):
    indexed = {}
    for row in rows:
        k = (row['reviewer_id'], row['id'])
        if k in indexed:
            raise ValueError('duplicate first-review row')
        indexed[k] = row
    expected = {(s, r['id']) for s in STREAMS for r in records}
    if set(indexed) != expected:
        raise ValueError('incomplete or unexpected first-review opportunities')
    positions = []
    for record in records:
        by_stream = {}
        for stream in STREAMS:
            row = indexed[stream, record['id']]
            for field in ('unit_id', 'text_sha256', 'reference_established'):
                if row[field] != record[field]:
                    raise ValueError('original row binding differs: ' + field)
            if not row['chunk_structurally_valid']:
                if row['first_judgment'] is not None:
                    raise ValueError('invalid whole chunk attempted parsed-row salvage')
                inventories = None
            else:
                if row['first_judgment'] is None:
                    raise ValueError('valid chunk lacks retained first judgment')
                inventories = binding_inventory(record, row['first_judgment'],
                    property_tables[record['id'].removesuffix('_template')])
            by_stream[stream] = dict(independent_context_id=row['independent_context_id'],
                chunk_number=row['chunk_number'], chunk_structurally_valid=row['chunk_structurally_valid'],
                review_state=row['review_state'], bindings=inventories)
        paths = {key(b['path']): b['path'] for b in record['bindings']}
        candidates = []
        for k, path in paths.items():
            matches = []
            for stream in STREAMS:
                bs = by_stream[stream]['bindings']
                matches.append(None if bs is None else next(b for b in bs if key(b['binding_path']) == k))
            if all(b is not None and b['all_recorded_support_facts_true'] for b in matches):
                candidates.append(deepcopy(path))
        positions.append(dict(id=record['id'], case=deepcopy(record['case']),
            reference_established=record['reference_established'],
            original_composite_statuses=None if not record['reference_established'] else {
                name: c['status'] for name, c in record['canonical_first_annotation']['composites'].items()},
            first_streams=by_stream, both_first_streams_recorded_support_candidates=candidates,
            candidates_require_semantic_and_source_closure=True,
            semantic_or_qualia_verdict=None))
    return positions


def join_pairs(indexed, original_pairs, numerical_pairs):
    pairs = []
    for original, numerical in zip(original_pairs, numerical_pairs, strict=True):
        if original != numerical['original_pair']:
            raise ValueError('numerical/semantic original pair inventory differs')
        first, second = (indexed[original[k] + '_template'] for k in ('first_id', 'second_id'))
        pairs.append(dict(original_pair=deepcopy(original),
            first_recorded_support_candidates=deepcopy(first['both_first_streams_recorded_support_candidates']),
            second_recorded_support_candidates=deepcopy(second['both_first_streams_recorded_support_candidates']),
            numerical_property_counts=deepcopy(numerical['counts']),
            numerical_by_actual_source=deepcopy(numerical['by_actual_source']),
            literal_text_equal=numerical['literal_text_equal'],
            semantic_predicate_tracking=None, semantic_true_null=None,
            semantic_exact_restoration=None,
            numerical_facts_do_not_establish_report_meaning=True))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--collection-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    collection = bound(args.collection, args.collection_sha256)
    if collection['schema'] != 'attention_relational_template_review_collection_v1':
        raise ValueError('requires frozen template collection schema')
    if (collection['template_opportunities'], collection['reviewer_opportunities'],
            collection['pair_opportunities'], len(collection['chunks']), len(collection['rows'])) != (552, 1104, 684, 46, 1104):
        raise ValueError('requires complete frozen first collection')
    if not collection['whole_chunk_failure_policy'] or not collection['no_replacement_or_gap_filling']:
        raise ValueError('requires frozen first-attempt policy')
    manifest = bound(ROOT / PREP / 'manifest.json', MANIFEST_SHA)
    if collection['manifest_sha256'] != MANIFEST_SHA:
        raise ValueError('collection/preparation mismatch')
    records = []
    for ref in manifest['chunks']:
        chunk = bound(ROOT / PREP / ref['path'], ref['sha256'])
        records.extend(chunk['records'])
    if [r['id'] for r in records] != manifest['ordered_template_ids']:
        raise ValueError('prepared order differs')
    bank = bound(ROOT / BANK, BANK_SHA)
    positions = join_positions(records, collection['rows'], bank['property_tables'])
    indexed = {p['id']: p for p in positions}
    if len(indexed) != 552 or len(collection['pairs']) != 684 or len(bank['pairs']) != 684:
        raise ValueError('complete position/pair inventory required')
    pairs = join_pairs(indexed, collection['pairs'], bank['pairs'])
    summary = dict(template_opportunities=552, pair_opportunities=684,
        reference_established=sum(p['reference_established'] for p in positions),
        first_chunks_structurally_valid=sum(c['structurally_valid'] for c in collection['chunks']),
        positions_with_both_first_streams_recorded_support_candidates=sum(bool(p['both_first_streams_recorded_support_candidates']) for p in positions),
        both_first_streams_recorded_support_candidate_bindings=sum(len(p['both_first_streams_recorded_support_candidates']) for p in positions),
        original_composite_statuses={name: dict(Counter(p['original_composite_statuses'][name]
            if p['reference_established'] else 'unestablished' for p in positions)) for name in ('COD', 'CA')})
    result = dict(schema='attention_relational_template_binding_inventory_v1',
        analysis_type='descriptive_join_of_retained_first_reviews_no_new_semantic_grader',
        input_collection={'path':str(args.collection), 'sha256':args.collection_sha256},
        prepared_manifest={'path':str(PREP / 'manifest.json'), 'sha256':MANIFEST_SHA},
        numerical_bank={'path':str(BANK), 'sha256':BANK_SHA},
        source_sha256=digest(Path(__file__)), summary=summary, positions=positions, pairs=pairs,
        original_judgments_or_scopes_changed=False, missing_first_judgments_filled=False,
        semantic_endpoint_assessment_complete=False, goal_achieved=False,
        limitations=[
            'Recorded support is a property of first judgments, not semantic truth or source-of-qualia evidence.',
            'Different compatible witness paths are retained; equality of reviewer inventories is not required.',
            'No exact atomic identity for a container is not feature absence; containers are never expanded.',
            'Source provenance describes the estimate, not necessarily the computational producer of its metadata.',
            'Root dispositions and unresolved meaning disputes require separate explicit closure.',
            'Numerical changes, unchanged text and repeated restorations cannot supply missing semantic measurement.',
            'All systems/views remain development data; authored templates are diagnostic instrumentation.'])
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        handle.write('\n')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
