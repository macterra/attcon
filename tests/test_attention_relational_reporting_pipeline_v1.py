"""Handcrafted software fixtures only; no scientific archives/API/model execution.

Schema-positive annotations below are fictional validator inputs, not qualified
human references, rater outcomes, narrator reports or scientific matches.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import attention_relational_reporting_pipeline_v1 as pipeline
from attention_relational_report_v1 import PACK_SCHEMA


def packet():
    node = {'node': 'n0', 'origin': {'kind': 'history_estimate'},
            'distributions': {'red': [[1.0]], 'blue': [[1.0]]}}
    return {'panels': [{'panel': 'p0', 'current': [node],
                       'candidates': {'k0': [deepcopy(node)]}}]}


def prose_audit(text='Red peaks here.'):
    return {'schema': pipeline.AUDIT_SCHEMA,
        'claims': [{'quote': text, 'verdict': 'supported', 'reason': 'Fictional fixture source.',
                    'bindings': [{'panel': 'p0', 'view': 'current', 'command': None,
                        'node': 'n0', 'role': 'red', 'basis': 'peaks'}]}],
        'current_covered': True, 'conditional_covered': True, 'unparsed_or_ambiguous': False}


def annotation(text='Red pattern.', unit='u'):
    evidence = {'quote': text, 'start': 0, 'end': len(text), 'referent': 'red',
                'attribute': 'location', 'scope': 'model_organization', 'time': 'current'}
    features = {k: {'status': 'explicit', 'evidence': [deepcopy(evidence)],
                    'rationale': 'Fictional schema-validity fixture, not a semantic key.'}
                for k in ('C', 'O', 'D', 'A')}
    binding = {k: evidence[k] for k in ('referent', 'attribute', 'scope', 'time')}
    binding['evidence'] = [deepcopy(evidence)]
    return {'schema': pipeline.RATING_SCHEMA, 'unit_id': unit, 'features': features,
        'composites': {k: {'status': 'explicit', 'bindings': [deepcopy(binding)], 'rationale': 'Fixture.'}
                       for k in ('COD', 'CA')},
        'auxiliary': {k: {'status': 'absent', 'evidence': []} for k in pipeline.AUXILIARY},
        'limitations': ['Handcrafted schema fixture, not reference qualification.']}


def outcome(identifier, text='Red pattern.', complete=True):
    report = {'schema': pipeline.REPORT_SCHEMA, 'text': text, 'propositions': []}
    return {'id': identifier, 'status': 'received', 'output_text': json.dumps(report),
        'request_sha256': pipeline.runner_request_sha256({'id': identifier, 'input': 'Fixture public request.'}),
        'api_response_status': 'completed' if complete else 'incomplete',
        'response': {'status': 'completed' if complete else 'incomplete'}, 'parsed': report}


def binding(root, identifiers):
    return {'schema': 'attention_relational_report_v1_collection_binding',
        'stage': 'report', 'manifest_sha256': 'a' * 64, 'request_ids': identifiers,
        'requests': [{'id': identifier, 'input': 'Fixture public request.'} for identifier in identifiers],
        'output_root': str(root), 'lineage': {'fictional': True}}


def test_supported_machine_props_do_not_certify_inaccurate_prose():
    audit = prose_audit('The red object is physically present.')
    audit['claims'][0]['verdict'] = 'not_auditable'
    audit['claims'][0]['bindings'] = []
    result = pipeline.validate_prose_audit(packet(), 'The red object is physically present.', audit)
    assert result['valid'] and not result['fidelity_pass']


def test_prose_missing_conditional_and_invented_source_binding_fail():
    audit = prose_audit()
    audit['conditional_covered'] = False
    assert not pipeline.validate_prose_audit(packet(), 'Red peaks here.', audit)['fidelity_pass']
    audit['claims'][0]['bindings'][0]['node'] = 'n9'
    assert not pipeline.validate_prose_audit(packet(), 'Red peaks here.', audit)['valid']


def test_unparsed_prose_and_unhashable_malformed_binding_retain_failure():
    audit = prose_audit()
    audit['unparsed_or_ambiguous'] = True
    assert not pipeline.validate_prose_audit(packet(), 'Red peaks here.', audit)['fidelity_pass']
    audit['claims'][0]['bindings'][0]['panel'] = ['p0']
    assert not pipeline.validate_prose_audit(packet(), 'Red peaks here.', audit)['valid']


def test_unicode_literal_evidence_and_offset_errors_are_distinct():
    text = 'Réd pattern.'
    valid = annotation(text)
    assert pipeline.validate_blind_annotation(text, 'u', valid)['valid']
    invalid = deepcopy(valid)
    invalid['features']['C']['evidence'][0]['end'] += 1
    result = pipeline.validate_blind_annotation(text, 'u', invalid)
    assert not result['valid']
    assert result['evidence_diagnostics'][0]['kind'] == 'quote_present_offset_mismatch'
    invented = deepcopy(valid)
    invented['features']['C']['evidence'][0]['quote'] = 'Unprovided text'
    assert pipeline.validate_blind_annotation(text, 'u', invented)['evidence_diagnostics'][0]['kind'] == 'quote_not_present'


def test_future_feature_cannot_be_promoted_by_current_composite_binding():
    value = annotation()
    value['features']['A']['evidence'][0]['time'] = 'future_forecast'
    result = pipeline.validate_blind_annotation('Red pattern.', 'u', value)
    assert not result['valid']
    assert any('compatible feature evidence for A' in error for error in result['errors'])


def test_missing_explicit_evidence_and_unknown_scope_fail_without_repair():
    value = annotation()
    value['features']['O']['evidence'] = []
    assert not pipeline.validate_blind_annotation('Red pattern.', 'u', value)['valid']
    value = annotation()
    value['composites']['COD']['bindings'][0]['scope'] = 'felt_clarity'
    assert not pipeline.validate_blind_annotation('Red pattern.', 'u', value)['valid']


def test_completed_collector_follows_secondary_path_not_poisoned_primary(tmp_path):
    root = tmp_path / 'stage'
    root.mkdir()
    (root / 'r.json').write_text('{"id":"poisoned"}')
    secondary = root / 'r.secondary.json'
    secondary.write_bytes(pipeline.encoded({'id': 'r', 'first_outcome': outcome('r'),
                                          'primary_write_error': 'OSError'}))
    completion = {'schema': 'attention_relational_report_v1_first_batch_completion',
        'retry_performed': False, 'manifest_sha256': 'a' * 64, 'request_count': 2,
        'outcomes': [{'id': 'r', 'status': 'received', 'response_path': str(secondary),
                      'response_sha256': pipeline.digest(secondary.read_bytes())}]}
    path = root / 'first_completion.json'
    path.write_bytes(pipeline.encoded(completion))
    result = pipeline.collect_first_responses(path, pipeline.digest(path.read_bytes()), ['r', 'missing'], root,
        stage_binding=binding(root, ['r', 'missing']))
    assert result['records'][0]['parsed']['text'] == 'Red pattern.'
    assert result['records'][1]['availability'] == 'missing'


def test_inline_double_storage_failure_retains_complete_first_response(tmp_path):
    root = tmp_path / 'stage'
    root.mkdir()
    envelope = {'id': 'r', 'first_outcome': outcome('r'), 'primary_write_error': 'OSError',
                'secondary_write_error': 'OSError'}
    completion = {'schema': 'attention_relational_report_v1_first_batch_completion',
        'retry_performed': False, 'manifest_sha256': 'a' * 64, 'request_count': 1,
        'outcomes': [{'id': 'r', 'status': 'storage_failure', 'retained_first_outcome_envelope': envelope}]}
    path = root / 'first_completion.json'
    path.write_bytes(pipeline.encoded(completion))
    result = pipeline.collect_first_responses(path, pipeline.digest(path.read_bytes()), ['r'], root,
        stage_binding=binding(root, ['r']))
    assert result['records'][0]['transport_complete']
    assert result['records'][0]['raw_outcome'] == envelope


def test_incomplete_first_response_and_hash_mismatch_are_not_positive(tmp_path):
    received = pipeline._response_record('r', outcome('r', complete=False), {})
    assert received['parsed'] and not received['transport_complete']
    path = tmp_path / 'bad.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        pipeline.bound_json(path, '0' * 64)


def test_duplicate_output_json_keys_and_nonfinite_constants_are_not_repaired():
    for malformed in ('{"text":"first","text":"second"}', '{"value":NaN}'):
        raw = outcome('r')
        raw['output_text'] = malformed
        raw.pop('parsed')
        result = pipeline._response_record('r', raw, {})
        assert result['parsed'] is None and result['parse_error']
        assert result['raw_outcome']['output_text'] == malformed


def test_same_id_foreign_stage_or_foreign_public_request_is_rejected(tmp_path):
    root = tmp_path / 'stage'
    root.mkdir()
    response_path = root / 'r.json'
    response_path.write_bytes(pipeline.encoded(outcome('r')))
    completion = {'schema': 'attention_relational_report_v1_first_batch_completion',
        'retry_performed': False, 'manifest_sha256': 'b' * 64, 'request_count': 1,
        'outcomes': [{'id': 'r', 'status': 'received', 'response_path': str(response_path),
                      'response_sha256': pipeline.digest(response_path.read_bytes())}]}
    path = root / 'first_completion.json'
    path.write_bytes(pipeline.encoded(completion))
    with pytest.raises(ValueError, match='different exact reviewed stage manifest'):
        pipeline.collect_first_responses(path, pipeline.digest(path.read_bytes()), ['r'], root,
            stage_binding=binding(root, ['r']))
    completion['manifest_sha256'] = 'a' * 64
    path.write_bytes(pipeline.encoded(completion))
    foreign = binding(root, ['r'])
    foreign['requests'][0]['input'] = 'Different public payload with same ID.'
    with pytest.raises(ValueError, match='different exact request input'):
        pipeline.collect_first_responses(path, pipeline.digest(path.read_bytes()), ['r'], root, stage_binding=foreign)


def test_export_lineage_change_with_same_request_ids_is_rejected():
    requests = [{'id': 'r', 'input': 'Fixture public request.'}]
    exports = {'requests': requests, 'templates': []}
    lineage = pipeline.build_stage_lineage('report', exports, {'fictional.json': 'a' * 64})
    collection = {'records': [], 'manifest_sha256': 'b' * 64,
        'stage_binding': {'schema': 'attention_relational_report_v1_collection_binding',
            'stage': 'report', 'requests': requests, 'request_ids': ['r'],
            'manifest_sha256': 'b' * 64, 'lineage': lineage}}
    pipeline._validate_collection_lineage(collection, 'report', requests, exports)
    changed_exports = dict(exports, fictional_unseen_source_change=True)
    with pytest.raises(ValueError, match='lineage differs'):
        pipeline._validate_collection_lineage(collection, 'report', requests, changed_exports)


def test_source_blind_rater_requests_have_only_text_reference_and_opaque_id(monkeypatch):
    reference = {'schema': 'fictional_reference_fixture'}
    monkeypatch.setattr(pipeline, 'frozen_reference', lambda: reference)
    monkeypatch.setattr(pipeline, 'frozen_instruction', lambda stage: 'Fixture instructions')
    report = pipeline._response_record('r', outcome('r'), {'secret_source': 'A'})
    collection = {'records': [report]}
    templates = [{'id': 'r', 'report': {'text': 'Literal fictional template.'}}]
    requests = pipeline.build_blind_rater_requests(collection, templates, reference)
    assert len(requests) == 2
    for request in requests:
        payload = json.loads(request['input'].rsplit('\n', 1)[1])
        assert set(payload) == {'reference', 'unit_id', 'report_text'}
        assert payload['unit_id'] == request['id']
        assert 'secret_source' not in request['input']


def test_auditor_request_excludes_provenance_and_missing_positions(monkeypatch):
    monkeypatch.setattr(pipeline, 'frozen_instruction', lambda stage: 'Fixture auditor')
    record = pipeline._response_record('r', outcome('r'), {})
    packets = [{'id': 'r', 'content': packet(), 'provenance': {'source': 'A'}},
               {'id': 'missing', 'content': packet()}]
    requests = pipeline.build_prose_audit_requests({'records': [record]}, packets)
    assert len(requests) == 1
    payload = json.loads(requests[0]['input'].rsplit('\n', 1)[1])
    assert set(payload) == {'packet', 'report_text'}
    assert 'provenance' not in requests[0]['input']


def test_stochastic_restoration_wording_is_not_a_machine_claim_change():
    atom = {'kind': 'peaks', 'panel': 'p0', 'view': 'current', 'command': None,
            'node': 'n0', 'role': 'red', 'cells': [[0, 1]]}
    first = {'text': 'The red estimate peaks at (0,1).', 'propositions': [atom]}
    second = {'text': 'At (0,1), the red field has its maximum.', 'propositions': [deepcopy(atom)]}
    assert first['text'] != second['text']
    changes = pipeline._claim_changes(first, second, 'p0')
    assert not any(changes.values())


def test_exclusive_output_never_replaces_first_artifact(tmp_path):
    path = tmp_path / 'first.json'
    pipeline.exclusive_json(path, {'first': True})
    with pytest.raises(FileExistsError):
        pipeline.exclusive_json(path, {'first': False})
    assert json.loads(path.read_bytes()) == {'first': True}


def test_raw_reference_match_is_not_positive_when_prose_fidelity_fails(monkeypatch):
    # One tiny row, with the full-production inventory validator deliberately
    # mocked. This tests the joining logic, not552-case validation or references.
    content = {'schema': PACK_SCHEMA, 'panels': [{
        'panel': 'p0', 'current': [{'node': 'n' + str(i),
            'derived': {'red': {'peaks': [[0, 0]]}, 'blue': {'peaks': [[0, 1]]}}}
            for i in range(4)], 'candidates': None}]}
    text = 'Red pattern.'
    report = {'schema': pipeline.REPORT_SCHEMA, 'text': text,
              'propositions': pipeline.required_propositions(content)}
    case = {'encoder_seed': 1301, 'world_index': 0, 'decision_time': 12, 'source': 'sensor',
            'view': 'sensor_only', 'variant': 'baseline'}
    inventory = {'request_ids': ['r'], 'packets': {'r': {'content': content}},
        'provenance': {'r': {'case': case}}, 'templates': {'r': {'report': deepcopy(report)}}}
    monkeypatch.setattr(pipeline, 'validate_export', lambda exports, manifest=None: inventory)
    monkeypatch.setattr(pipeline, '_validate_collection_lineage', lambda *args: None)
    raw = outcome('r', text)
    raw['parsed'] = report
    raw['output_text'] = json.dumps(report)
    reports = {'records': [pipeline._response_record('r', raw, {})]}
    bad_prose = prose_audit(text)
    bad_prose['claims'][0]['verdict'] = 'not_auditable'
    bad_prose['claims'][0]['bindings'] = []
    audits = {'records': [{'id': 'r', 'transport_complete': True, 'parsed': bad_prose}]}
    ratings = {'records': [{'id': pipeline.blind_id(identifier), 'transport_complete': True,
        'parsed': annotation(text, pipeline.blind_id(identifier))} for identifier in ('r', 'r_template')]}
    fixture_exports = {
        'requests': [{'id': 'r', 'input': 'Fixture public request.'}],
        'packets': [{'id': 'r', 'content': content}],
        'templates': [{'id': 'r', 'report': deepcopy(report)}],
        'provenance': [{'id': 'r', 'case': case}],
    }
    summary = pipeline.summarize_reporting_results(reports, None, audits, ratings, [], fixture_exports)
    narrator = summary['all_text_results'][0]
    assert narrator['machine_audit']['listed_machine_facts_pass']
    assert narrator['endpoints']['COD']['lower']
    assert not narrator['endpoints']['COD']['faithful_lower']
    assert not narrator['fidelity_established']
    assert summary['interpretation']['experience_established'] is False
    # A syntactically valid possible-match rating of a missing/incomplete actor
    # placeholder cannot contribute even an upper endpoint. Raw labels remain.
    reports['records'][0]['transport_complete'] = False
    missing_rating = annotation('', pipeline.blind_id('r'))
    for feature in missing_rating['features'].values():
        feature.update(status='ambiguous', evidence=[])
    for composite in missing_rating['composites'].values():
        composite.update(status='ambiguous', bindings=[])
    ratings['records'][0]['parsed'] = missing_rating
    missing_summary = pipeline.summarize_reporting_results(reports, None, audits, ratings, [], fixture_exports)
    missing = missing_summary['all_text_results'][0]
    assert missing['reported_composite_labels_before_validation']['COD'] == 'ambiguous'
    assert missing['endpoints']['COD']['status'] == 'unestablished'
    assert not missing['endpoints']['COD']['upper']


def test_list_and_dictionary_prose_verdicts_retain_invalidity_without_crashing():
    for malformed in (['supported'], {'verdict': 'supported'}):
        audit = prose_audit()
        audit['claims'][0]['verdict'] = deepcopy(malformed)
        result = pipeline.validate_prose_audit(packet(), 'Red peaks here.', audit)
        assert not result['valid'] and not result['fidelity_pass']
        assert result['verdict_counts'] == {'invalid': 1}
        assert 'invalid claim verdict or rationale' in result['errors']
        assert audit['claims'][0]['verdict'] == malformed


def test_overflowing_json_exponents_rejected_and_finite_values_accepted():
    for exponent in ('1e309', '-1e309'):
        malformed = '{"value":' + exponent + '}'
        with pytest.raises(ValueError, match='nonfinite JSON exponent'):
            pipeline.decoded(malformed)
        raw = outcome('r')
        raw['output_text'] = malformed
        raw.pop('parsed')
        result = pipeline._response_record('r', raw, {})
        assert result['parsed'] is None and result['parse_error']
        assert result['raw_outcome']['output_text'] == malformed
    assert pipeline.decoded('{"values":[1e2,-2.5e-2,0.0]}') == {'values': [100.0, -0.025, 0.0]}
