"""Root inventory verification, not an independent semantic review."""
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parents[1]
collection_path = base / 'collection/first_collection.json'
readout_path = base / 'assessment/first_readout.json'
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert digest(collection_path) == '60451ae54822c3e372b4419e5566dddb27fa74ca99dbff298bbe31dcc216ad70'
assert digest(readout_path) == 'ff7cbe1acc00e007e09a66e3563f5437cbd30db468ec3b69eecb4ef5933d658b'
collection = json.loads(collection_path.read_text())
readout = json.loads(readout_path.read_text())
checks = []


def check(name, condition):
    assert condition, name
    checks.append(name)


check('all552position_ids_and_order',
      [r['id'] for r in readout['positions']] ==
      [r['id'] for r in collection['narrator_availability']])
check('all684original_pair_envelopes_and_order',
      [r['original_pair'] for r in readout['pairs']] == collection['pairs'])
check('all_actual_availability_case_and_reference_statuses', all(
    all(a[k] == b[k] for k in ('id', 'case', 'actor_text_available', 'reference_established'))
    for a, b in zip(readout['positions'], collection['narrator_availability'])))
valid = {c['reviewer_id']: c['structurally_valid'] for c in collection['chunks']}
check('first_chunk_validity_preserved', valid == {'reviewer_0': True, 'reviewer_1': False})
candidate_positions = [p for p in readout['positions'] if p['first_stream_judgments']]
check('every_original_candidate_included', len(candidate_positions) == 3 and
      {p['id'] for p in candidate_positions} ==
      {p['id'] for p in collection['candidate_binding_inventory']})
check('invalid_stream_semantics_not_salvaged', all(
    p['first_stream_judgments']['reviewer_1']['semantic_judgments_salvaged'] is False and
    p['first_stream_judgments']['reviewer_1']['state'] == 'unestablished_entire_first_chunk'
    for p in candidate_positions))
check('all_usable_first_meaning_and_ownership_judgments_retained', all(
    p['first_stream_judgments']['reviewer_0']['bindings'][i]['source_meaning'] == b['source_meaning'] and
    p['first_stream_judgments']['reviewer_0']['bindings'][i]['source_ownership'] == b['source_ownership']
    for p in candidate_positions for row in collection['rows']
    if row['reviewer_id'] == 'reviewer_0' and row['id'] == p['id']
    for i, b in enumerate(row['first_judgment']['candidate_binding_reviews'])))
check('no_missing_measurement_converted_to_null_or_tracking', all(
    all(pair[k] is None for k in ('actual_property_identity', 'actual_property_change',
        'asserted_predicate_change', 'selective_tracking', 'true_null',
        'unaffected_assertions_preserved', 'exact_restoration'))
    for pair in readout['pairs']))
check('whole_prose_fidelity_never_upgraded', all(
    p['narrator_fidelity_established'] is False for p in readout['positions']))
check('unknown_reference_and_no_candidate_inventories',
      sum(not p['reference_established'] for p in readout['positions']) == 548 and
      sum(p['reference_established'] and not p['first_stream_judgments']
          for p in readout['positions']) == 1)
result = dict(schema='attention_relational_narrator_source_closure_v1_inventory_verification',
    checks=checks, passed=True, independent_semantic_review=False,
    collection_sha256=digest(collection_path), readout_sha256=digest(readout_path),
    verification_source_sha256=digest(Path(__file__)))
with (base / 'assessment/first_inventory_verification.json').open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
print(json.dumps(result, indent=2))
