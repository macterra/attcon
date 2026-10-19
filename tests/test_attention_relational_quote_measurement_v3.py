"""Fictional V3 integration fixtures only; no scientific artifacts or API calls.

Minimal typed inventories exercise the validated-input boundary without forging
an old runtime binding or monkeypatching any frozen builder/validator. Production
CLI preparation additionally verifies the pinned real old inputs and lineage.
"""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import attention_relational_quote_measurement_v3 as measurement
import attention_relational_quote_reference_v3 as quotes
import attention_relational_reporting_pipeline_v1 as pipeline


TEXT = 'Fictional red record.'


def fixture_preparation():
    inventory = {'request_ids': [], 'templates': {}, 'provenance': {}}
    records = []
    for number in range(552):
        identifier = f'fiction-{number:04d}'
        inventory['request_ids'].append(identifier)
        inventory['templates'][identifier] = {'report': {'text': TEXT}}
        inventory['provenance'][identifier] = {'case': dict(encoder_seed=1301, world_index=number % 4,
            decision_time=12, source='A', view='A_only', variant='baseline')}
        records.append({'id': identifier, 'transport_complete': number != 1,
                        'parsed': {'schema': pipeline.REPORT_SCHEMA, 'text': '' if number == 2 else TEXT,
                                   'propositions': []}})
    units = measurement.text_units(inventory, {'records': records})
    return measurement.prepare_units(units, {'schema': 'fictional_reference'},
        'Fictional quote-only instructions.',
        dict(manifest_path=measurement.AUDITOR_MANIFEST, manifest_sha256=measurement.AUDITOR_MANIFEST_SHA256,
             output_root=measurement.AUDITOR_ROOT, request_count=537),
        {'fictional_inputs': True}, {'fictional_source': 'f' * 64})


def fixture_binding(prepared):
    requests_sha = pipeline.digest(pipeline.encoded(prepared['requests']) + b'\n')
    lineage_sha = pipeline.digest(pipeline.encoded(prepared['lineage']) + b'\n')
    manifest = dict(stage='rate', settings=prepared['settings'], output_root=measurement.RATE_ROOT,
        requests_path='fictional/requests.json', lineage_path='fictional/lineage.json',
        source_sha256={'fictional/requests.json': requests_sha, 'fictional/lineage.json': lineage_sha},
        request_count=1104, request_ids=[r['id'] for r in prepared['requests']])
    return measurement.build_binding(manifest, 'a' * 64, prepared['requests'], requests_sha,
                                     prepared['lineage'], lineage_sha, prepared)


def annotation(unit, explicit=True):
    entry = dict(quote=TEXT, referent='fictional red', attribute='location',
                 scope='model_organization', time='current')
    b = {k: entry[k] for k in ('referent', 'attribute', 'scope', 'time')}
    b['evidence'] = [deepcopy(entry)]
    status = 'explicit' if explicit else 'absent'
    return dict(schema=quotes.QUOTE_SCHEMA, unit_id=unit['unit_id'],
        features={k: dict(status=status, evidence=[deepcopy(entry)] if explicit else [], rationale='Fictional fixture.')
                  for k in ('C', 'O', 'D', 'A')},
        composites={k: dict(status=status, bindings=[deepcopy(b)] if explicit else [], rationale='Fictional fixture.')
                    for k in ('COD', 'CA')},
        auxiliary={k: dict(status='absent', evidence=[]) for k in pipeline.AUXILIARY}, limitations=[])


def raw_outcome(prepared, unit_index, value=None, complete=True):
    unit = prepared['units'][unit_index]
    request = pipeline._index(prepared['requests'])[unit['unit_id']]
    return dict(id=unit['unit_id'], status='received',
        api_response_status='completed' if complete else 'incomplete',
        response={'status': 'completed' if complete else 'incomplete'},
        request_sha256=pipeline.runner_request_sha256(request),
        output_text=pipeline.encoded(annotation(unit) if value is None else value).decode('ascii'))


def fixture_collection(prepared, outcomes=()):
    records = {r['id']: dict(id=r['id'], availability='missing', transport_complete=False,
                            parsed=None, raw_outcome=None, storage_provenance={}) for r in prepared['requests']}
    for raw in outcomes:
        records[raw['id']] = pipeline._response_record(raw['id'], raw, {'kind': 'fictional_in_memory_fixture'})
    return dict(schema='attention_relational_report_v1_collected_first_responses', request_count=1104,
                stage_binding=fixture_binding(prepared), records=list(records.values()), retry_performed=False)


def factual_rows(prepared, faithful=False):
    return {'all_text_results': [dict(id=u['id'], unit_id=u['unit_id'], text_sha256=u['text_sha256'],
        case=deepcopy(u['case']), template=u['template'], actor_text_available=u['actor_text_available'],
        fidelity_established=True if u['template'] else faithful) for u in prepared['units']]}


def test_instruction_changes_are_only_the_declared_mechanical_edits():
    amended = measurement.instructions()
    assert 'start/end' not in amended and '"start":' not in amended and 'validate offsets' not in amended
    restored = amended.replace('Record each evidence substring exactly as it appears in report_text.',
        'Record each evidence substring exactly as it appears in report_text, with start/end Unicode-codepoint offsets, end exclusive.')
    restored = restored.replace(quotes.QUOTE_SCHEMA, pipeline.RATING_SCHEMA)
    restored = restored.replace('"quote":"literal substring",', '"quote":"literal substring","start":0,"end":1,')
    restored = restored.replace('The program must ground literal quotations and validate schema and evidence consistency;',
                                'The program must validate offsets, schema and evidence consistency;')
    assert restored == pipeline.frozen_instruction('rate')


def test_complete_inventory_new_actual_payloads_ids_order_and_no_source_leakage():
    prepared = fixture_preparation()
    measurement.validate_prepared(prepared)
    assert len(prepared['units']) == len(prepared['requests']) == 1104
    assert sum(u['template'] for u in prepared['units']) == 552
    assert [r['id'] for r in prepared['requests']] == sorted(
        [u['unit_id'] for u in prepared['units']],
        key=lambda unit: pipeline.digest((pipeline.ORDER_SALT + ':' + unit).encode('ascii')))
    for request in prepared['requests']:
        payload = pipeline.decoded(request['input'].rsplit('\n', 1)[1])
        assert set(payload) == {'reference', 'unit_id', 'report_text'}
        assert payload['unit_id'] == request['id']
    assert prepared['lineage']['schema'] == measurement.VERSION + '_lineage'
    assert prepared['settings']['output_schema'] == quotes.response_schema()


def test_generic_raw_collector_preserves_actual_v3_binding_and_all_missing_positions(tmp_path):
    prepared = fixture_preparation()
    binding = fixture_binding(prepared)
    root = tmp_path / measurement.RATE_ROOT
    root.mkdir(parents=True)
    raw = raw_outcome(prepared, 0)
    path = root / (raw['id'] + '.json')
    path.write_bytes(pipeline.encoded(raw))
    completion = dict(schema='attention_relational_report_v1_first_batch_completion', retry_performed=False,
        manifest_sha256=binding['manifest_sha256'], request_count=1104,
        outcomes=[dict(id=raw['id'], response_path=str(path), response_sha256=pipeline.digest(path.read_bytes()))])
    completed = tmp_path / 'fictional-completion.json'
    completed.write_bytes(pipeline.encoded(completion))
    collected = pipeline.collect_first_responses(completed, pipeline.digest(completed.read_bytes()),
        binding['request_ids'], root, repository_root=tmp_path, stage_binding=binding)
    derived = measurement.derive_collection(prepared, collected)
    assert sum(r['established'] for r in derived['rows']) == 1
    assert len(derived['rows']) == 1104
    assert collected['stage_binding']['lineage']['schema'] == measurement.VERSION + '_lineage'
    first = derived['rows'][0]
    assert first['request_sha256'] == raw['request_sha256']
    assert first['storage_provenance']['sha256'] == pipeline.digest(path.read_bytes())
    assert first['raw_output_text_sha256'] == pipeline.digest(raw['output_text'].encode('utf-8'))


def test_wrong_request_or_old_faked_lineage_is_rejected():
    prepared = fixture_preparation()
    raw = raw_outcome(prepared, 0)
    raw['request_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='request hash differs'):
        measurement.derive_collection(prepared, fixture_collection(prepared, [raw]))
    collection = fixture_collection(prepared)
    collection['stage_binding']['measurement_version'] = 'old_measurement'
    with pytest.raises(ValueError, match='honestly bound actual V3'):
        measurement.derive_collection(prepared, collection)


def test_wrong_unit_quote_or_semantic_binding_invalidates_whole_annotation():
    prepared = fixture_preparation()
    for kind in ('unit', 'quote', 'semantic'):
        value = annotation(prepared['units'][0])
        if kind == 'unit':
            value['unit_id'] = 'wrong-unit'
        elif kind == 'quote':
            value['features']['C']['evidence'][0]['quote'] = 'Unprovided text'
        else:
            value['features']['A']['evidence'][0]['time'] = 'future_forecast'
        derived = measurement.derive_collection(prepared, fixture_collection(prepared, [raw_outcome(prepared, 0, value)]))
        assert not derived['rows'][0]['established'] and derived['rows'][0]['canonical_annotation'] is None
        joined = measurement.join_reference(derived, factual_rows(prepared))
        assert set(joined['all_text_results'][0]['features'].values()) == {'unestablished'}
        assert joined['all_text_results'][0]['endpoints']['COD']['status'] == 'unestablished'


def test_incomplete_ratings_and_blank_or_incomplete_actors_never_earn_absent_status():
    prepared = fixture_preparation()
    # Index2 is a narrator whose original response was incomplete; index4 is blank.
    assert not prepared['units'][2]['actor_text_available'] and not prepared['units'][4]['actor_text_available']
    outcomes = [raw_outcome(prepared, 0, complete=False)]
    outcomes += [raw_outcome(prepared, i, annotation(prepared['units'][i], explicit=False)) for i in (2, 4)]
    derived = measurement.derive_collection(prepared, fixture_collection(prepared, outcomes))
    for i in (0, 2, 4):
        assert not derived['rows'][i]['established'] and derived['rows'][i]['canonical_annotation'] is None
    assert derived['rows'][2]['derivation']['valid']  # Contract alone cannot establish actor availability.


def test_wrong_text_or_factual_id_is_rejected_and_no_auditor_fallback_occurs():
    prepared = fixture_preparation()
    derived = measurement.derive_collection(prepared, fixture_collection(prepared, [raw_outcome(prepared, 0)]))
    factual = factual_rows(prepared, faithful=False)
    joined = measurement.join_reference(derived, factual)
    assert joined['all_text_results'][0]['endpoints']['COD']['lower']
    assert not joined['all_text_results'][0]['endpoints']['COD']['faithful_lower']
    assert joined['arm_totals']['narrator']['text_opportunities'] == 552
    assert joined['arm_totals']['template']['text_opportunities'] == 552
    for field, value in [('text_sha256', '0' * 64), ('unit_id', 'wrong-unit')]:
        wrong = deepcopy(factual)
        wrong['all_text_results'][0][field] = value
        with pytest.raises(ValueError, match='exact ID/text/case'):
            measurement.join_reference(derived, wrong)
    wrong = deepcopy(prepared)
    wrong['units'][0]['text'] = 'Changed text.'
    with pytest.raises(ValueError, match='preparation content differs'):
        measurement.validate_prepared(wrong)


def test_missing_inventory_position_is_not_silently_dropped():
    prepared = fixture_preparation()
    collection = fixture_collection(prepared)
    collection['records'].pop()
    with pytest.raises(ValueError, match='full raw V3 inventory differs'):
        measurement.derive_collection(prepared, collection)


def test_selected_whole_auditor_cannot_be_changed_after_collection():
    prepared = fixture_preparation()
    collection = fixture_collection(prepared)
    changed = deepcopy(prepared)
    changed['lineage']['selected_factual_auditor']['manifest_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='honestly bound actual V3'):
        measurement.derive_collection(changed, collection)


def test_wrong_factual_summary_input_version_blocks_before_any_file_reads():
    prepared = fixture_preparation()
    summary = {'preparation_dependencies': {
        'exports': {'path': 'not-read', 'sha256': measurement.origins.EXPORT_SHA256},
        'reports': {'path': 'not-read', 'sha256': measurement.REPORTS_SHA256},
        'audits': {'path': 'not-read', 'sha256': 'f' * 64},
        'ratings': {'path': 'not-read', 'sha256': '0' * 64}}}
    with pytest.raises(ValueError, match='factual summary input versions differ'):
        measurement.verify_factual_summary(summary, prepared)
