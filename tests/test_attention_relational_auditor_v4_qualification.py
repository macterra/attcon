"""Fictional retention/mechanics fixtures, not the24 semantic challenges or key.

No builder, challenge materialization, author notes, hidden key or scientific
target is imported/read. No frozen module is monkeypatched and no API is used.
"""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import attention_relational_auditor_v4_qualification as qualification
import attention_relational_grounded_auditor_v4 as core
import attention_relational_reporting_pipeline_v1 as pipeline
import attention_relational_report_v1 as exporter


def fictional_public_cases():
    red = [[0.0] * 4 for _ in range(4)]
    blue = deepcopy(red)
    red[0][0], blue[3][3] = 1.0, 1.0
    packet = dict(schema=exporter.PACK_SCHEMA, decision_time=12, previous_node='n0', candidate_node='n0',
        queried_role='red', coordinate_system=dict(rows=4, columns=4), derivation={'peaks': 'All exact maxima'},
        candidate_semantics='Conditional supplied record', panels=[dict(panel='p0', candidates=None,
            current=[dict(node='n0', origin=dict(kind='history_estimate', represented_time=12, feedback_through=11),
                          distributions=dict(red=red, blue=blue))])])
    return dict(schema=qualification.PUBLIC_SCHEMA, cases=[dict(case_id=f'private_fixture_title_{i:03d}',
        packet=deepcopy(packet), report_text='Fictional statement.') for i in range(24)])


def prepared_fixture():
    return qualification.prepare(fictional_public_cases(), {'cases': {'path': 'fictional-public.json', 'sha256': 'f' * 64}},
                                 {'fictional_source': 'e' * 64})


def binding_fixture(prepared):
    requests_sha = pipeline.digest(pipeline.encoded(prepared['requests']) + b'\n')
    lineage_sha = pipeline.digest(pipeline.encoded(prepared['lineage']) + b'\n')
    manifest = dict(stage='audit', settings=qualification.settings(), output_root=qualification.RUN_ROOT,
        request_count=48, request_ids=[r['id'] for r in prepared['requests']],
        requests_path='fictional/requests.json', lineage_path='fictional/lineage.json',
        source_sha256={'fictional/requests.json': requests_sha, 'fictional/lineage.json': lineage_sha})
    return qualification.build_binding(manifest, 'a' * 64, prepared['requests'], requests_sha,
                                        prepared['lineage'], lineage_sha, prepared)


def annotation(request):
    value = pipeline.decoded(request['input'].rsplit('\n', 1)[1])
    context = next(c['context_id'] for c in value['source_contexts'] if c['kind'] == 'role_distribution')
    return dict(schema=core.RAW_SCHEMA, unit_id=value['unit_id'], complete_assertion_inventory=True,
        unresolved_ambiguity=False, assertions=[dict(assertion_id='a0', paraphrase='Fictional software judgment.',
            span_ids=[value['spans'][0]['span_id']], source_context_ids=[context],
            verdict='supported', reason='Mechanics fixture only; not a semantic key.')],
        coverage=[dict(span_id=value['spans'][0]['span_id'], assertion_ids=['a0'], classification='assertive',
                       reason='Fictional coverage.')])


def outcome(request, complete=True):
    return dict(id=request['id'], request_sha256=pipeline.runner_request_sha256(request), status='received',
        api_response_status='completed' if complete else 'incomplete',
        response={'status': 'completed' if complete else 'incomplete'},
        output_text=pipeline.encoded(annotation(request)).decode('ascii'))


def collection_fixture(prepared, outcomes=()):
    records = {r['id']: dict(id=r['id'], availability='missing', transport_complete=False,
                            parsed=None, raw_outcome=None, storage_provenance={}) for r in prepared['requests']}
    for raw in outcomes:
        records[raw['id']] = pipeline._response_record(raw['id'], raw, {'kind': 'fictional_fixture'})
    return dict(schema='attention_relational_report_v1_collected_first_responses', retry_performed=False,
        request_count=48, stage_binding=binding_fixture(prepared), records=list(records.values()))


def test_exact_public_wrapper_case_keys_and_count_exclude_key_or_category_fields():
    public = fictional_public_cases()
    assert len(qualification.validate_cases(public)) == 24
    invalid = [public['cases'], dict(public, hidden_key='Fictional forbidden field'),
               dict(public, schema='wrong-schema'), dict(public, cases=public['cases'][:23])]
    extra = deepcopy(public)
    extra['cases'][0]['expected_verdict'] = 'Fictional forbidden field'
    invalid.append(extra)
    duplicate = deepcopy(public)
    duplicate['cases'][1]['case_id'] = duplicate['cases'][0]['case_id']
    invalid.append(duplicate)
    for value in invalid:
        with pytest.raises(ValueError):
            qualification.validate_cases(value)


def test_two_by24_fixed_opaque_ids_order_and_no_private_mapping_in_api_inputs():
    prepared = prepared_fixture()
    qualification.validate_preparation(prepared)
    mapping = prepared['private_grading_mapping']
    assert len(prepared['requests']) == len(mapping) == 48
    assert {a: sum(m['auditor'] == a for m in mapping) for a in qualification.AUDITORS} == {
        'auditor_0': 24, 'auditor_1': 24}
    assert len({m['request_id'] for m in mapping}) == len({m['unit_id'] for m in mapping}) == 48
    assert [r['id'] for r in prepared['requests']] == sorted([m['request_id'] for m in mapping],
        key=lambda identifier: pipeline.digest((qualification.ORDER_SALT + ':' + identifier).encode('ascii')))
    assert prepared == prepared_fixture()
    for request in prepared['requests']:
        value = pipeline.decoded(request['input'].rsplit('\n', 1)[1])
        assert 'private_fixture_title_' not in request['input']
        assert 'auditor_0' not in request['input'] and 'auditor_1' not in request['input']
        assert 'case_id' not in value and 'auditor' not in value and 'key' not in value
        assert value['report_text'] == 'Fictional statement.'
        assert len(value['unit_id']) == 24
    assert prepared['settings']['concurrency'] == 6
    assert prepared['settings']['max_output_tokens'] == 32768
    assert prepared['model_transport'] == dict(verbosity='low', store=False, sdk_max_retries=0, stateless=True)
    assert not prepared['semantic_qualification_assessed'] and not prepared['hidden_key_read']


def test_normal_frozen_raw_collector_uses_actual_new_binding_and_retains_all48(tmp_path):
    prepared = prepared_fixture()
    binding = binding_fixture(prepared)
    root = tmp_path / qualification.RUN_ROOT
    root.mkdir(parents=True)
    request = prepared['requests'][0]
    raw = outcome(request)
    response_path = root / (request['id'] + '.json')
    response_path.write_bytes(pipeline.encoded(raw))
    completion = dict(schema='attention_relational_report_v1_first_batch_completion', retry_performed=False,
        manifest_sha256=binding['manifest_sha256'], request_count=48, outcomes=[dict(id=request['id'],
            response_path=str(response_path), response_sha256=pipeline.digest(response_path.read_bytes()))])
    path = tmp_path / 'fictional-completion.json'
    path.write_bytes(pipeline.encoded(completion))
    collection = pipeline.collect_first_responses(path, pipeline.digest(path.read_bytes()), binding['request_ids'],
        root, repository_root=tmp_path, stage_binding=binding)
    assert collection['stage_binding']['lineage']['schema'] == qualification.VERSION + '_lineage'
    derived = qualification.derive(prepared, collection)
    assert len(derived['rows']) == derived['opportunities'] == 48
    assert sum(r['mechanical_contract_established'] for r in derived['rows']) == 1
    returned = next(r for r in derived['rows'] if r['request_id'] == request['id'])
    assert returned['request_sha256'] == raw['request_sha256']
    assert returned['storage_provenance']['sha256'] == pipeline.digest(response_path.read_bytes())
    assert returned['raw_output_text_sha256'] == pipeline.digest(raw['output_text'].encode('utf-8'))
    assert not derived['semantic_qualification_assessed']
    assert not derived['semantic_qualification_verdict_emitted']


def test_incomplete_and_missing_results_are_unknown_despite_parseable_mechanics():
    prepared = prepared_fixture()
    raw = outcome(prepared['requests'][0], complete=False)
    derived = qualification.derive(prepared, collection_fixture(prepared, [raw]))
    assert len(derived['rows']) == 48
    assert all(not r['mechanical_contract_established'] for r in derived['rows'])
    assert all(r['derived_evidence']['derived_assertions'] is None for r in derived['rows'])
    incomplete = next(r for r in derived['rows'] if r['request_id'] == raw['id'])
    assert incomplete['derived_evidence']['mechanical_valid']  # Before the required transport gate only.
    assert all(c['opportunities'] == c['contract_unestablished'] == 24 for c in derived['by_auditor'].values())
    assert derived['independent_keyed_reviews_required'] and not derived['semantic_qualification_verdict_emitted']


def test_wrong_request_hash_or_foreign_binding_cannot_be_reinterpreted_as_qualification():
    prepared = prepared_fixture()
    raw = outcome(prepared['requests'][0])
    raw['request_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='request hash differs'):
        qualification.derive(prepared, collection_fixture(prepared, [raw]))
    collection = collection_fixture(prepared)
    collection['stage_binding']['measurement_version'] = 'old_measurement'
    with pytest.raises(ValueError, match='actual new qualification binding'):
        qualification.derive(prepared, collection)


def test_wrong_raw_unit_and_missing_inventory_positions_are_not_salvaged():
    prepared = prepared_fixture()
    request = prepared['requests'][0]
    raw = outcome(request)
    value = annotation(request)
    value['unit_id'] = 'wrong-unit'
    raw['output_text'] = pipeline.encoded(value).decode('ascii')
    derived = qualification.derive(prepared, collection_fixture(prepared, [raw]))
    assert not any(r['mechanical_contract_established'] for r in derived['rows'])
    collection = collection_fixture(prepared)
    collection['records'].pop()
    with pytest.raises(ValueError, match='full48 records required'):
        qualification.derive(prepared, collection)


def test_no_semantic_grading_or_favorable_assertion_selection_is_inferred():
    prepared = prepared_fixture()
    request = prepared['requests'][0]
    raw = outcome(request)
    value = annotation(request)
    value['assertions'][0]['paraphrase'] = 'A mechanically legal but potentially semantically wrong interpretation.'
    raw['output_text'] = pipeline.encoded(value).decode('ascii')
    derived = qualification.derive(prepared, collection_fixture(prepared, [raw]))
    assert sum(r['mechanical_contract_established'] for r in derived['rows']) == 1
    assert not derived['semantic_qualification_verdict_emitted']
    assert all('qualification_pass' not in r for r in derived['rows'])


def test_prepared_request_or_case_mutation_is_rejected_before_derivation():
    prepared = prepared_fixture()
    changed = deepcopy(prepared)
    changed['requests'][0]['input'] += ' Changed bytes.'
    with pytest.raises(ValueError, match='qualification preparation differs'):
        qualification.validate_preparation(changed)
    changed = deepcopy(prepared)
    changed['public_cases']['cases'][0]['report_text'] = 'Changed fictional case.'
    with pytest.raises(ValueError, match='qualification preparation differs'):
        qualification.validate_preparation(changed)
