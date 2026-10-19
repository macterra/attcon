"""Small fictional software fixtures; no study output, archive, model, or API reads.

The annotations exercise schema/containment bookkeeping, not scientific or human
reference qualification. Execute only after independent source review.
"""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import attention_relational_origin_attribution_v1 as attribution
import attention_relational_reporting_pipeline_v1 as pipeline


def packet():
    def node(name, origin):
        return {'node': name, 'origin': origin,
                'distributions': {'red': [[1.0]], 'blue': [[1.0]]}}
    current = {'kind': 'history_estimate', 'represented_time': 12, 'feedback_through': 11}
    copied = {'kind': 'copied_prior_record', 'record_time': 11, 'represented_time': 12}
    native = {'kind': 'completed_record', 'record_time': 12, 'represented_time': 12}
    return {'panels': [
        {'panel': 'p0', 'current': [node('n0', current), node('n1', copied)],
         'candidates': {'k0': [
             node('n0', {'kind': 'inherited_current', 'represented_time': 13,
                         'inherited_origin': current}),
             node('n1', {'kind': 'command_estimate', 'represented_time': 13,
                         'feedback_through': 11}),
             node('n2', {'kind': 'inherited_current', 'represented_time': 13,
                         'inherited_origin': copied})]}},
        {'panel': 'p1', 'current': [node('n0', native)],
         'candidates': {'k0': [
             node('n0', {'kind': 'native_conditional_record', 'represented_time': 13,
                         'executed_branch_claim': False}),
             node('n1', {'kind': 'inherited_current', 'represented_time': 13,
                         'inherited_origin': native})]}}]}


def provenance():
    return {'panels': [{'panel': 'p0', 'source_key': 'A'}, {'panel': 'p1', 'source_key': 'W'}]}


def binding(panel='p0', view='current', node='n0', role='red', basis='peaks'):
    return {'panel': panel, 'view': view, 'command': 'k0' if view == 'candidate' else None,
            'node': node, 'role': role, 'basis': basis}


def audit(text, bindings=None):
    return {'schema': pipeline.AUDIT_SCHEMA,
            'claims': [{'quote': text, 'verdict': 'supported', 'reason': 'Fictional source binding.',
                        'bindings': [binding()] if bindings is None else bindings}],
            'current_covered': True, 'conditional_covered': True, 'unparsed_or_ambiguous': False}


def evidence(text, quote=None, start=None):
    quote = text if quote is None else quote
    start = text.index(quote) if start is None else start
    return {'quote': quote, 'start': start, 'end': start + len(quote),
            'referent': 'fixture record', 'attribute': 'location',
            'scope': 'model_organization', 'time': 'current'}


def annotation(entry):
    b = {k: entry[k] for k in ('referent', 'attribute', 'scope', 'time')}
    b['evidence'] = [deepcopy(entry)]
    return {'schema': pipeline.RATING_SCHEMA, 'unit_id': 'fictional-unit',
            'features': {f: {'status': 'explicit', 'evidence': [deepcopy(entry)],
                            'rationale': 'Fictional validator fixture.'} for f in ('C', 'O', 'D', 'A')},
            'composites': {k: {'status': 'explicit', 'bindings': [deepcopy(b)],
                              'rationale': 'Fictional validator fixture.'} for k in ('COD', 'CA')},
            'auxiliary': {k: {'status': 'absent', 'evidence': []} for k in pipeline.AUXILIARY},
            'limitations': ['Synthetic software fixture only.']}


def result(text, auditor=None, rater=None, audit_complete=True, rating_complete=True):
    auditor = audit(text) if auditor is None else auditor
    rater = annotation(evidence(text)) if rater is None else rater
    prose = pipeline.validate_prose_audit(packet(), text, auditor)
    rated = pipeline.validate_blind_annotation(text, 'fictional-unit', rater)
    primary = {'id': 'fictional-report', 'unit_id': 'fictional-unit', 'template': False,
               'prose_audit_validation': prose, 'rating_validation': rated,
               'fidelity_established': False,
               'endpoints': {k: {'status': 'explicit' if rating_complete and rated['valid'] else 'unestablished',
                                'lower': rating_complete and rated['valid'],
                                'upper': rating_complete and rated['valid'],
                                'faithful_lower': False, 'faithful_upper': False,
                                'scopes': ['model_organization'], 'times': ['current']}
                             for k in ('COD', 'CA')}}
    before = deepcopy(primary)
    value = attribution.narrative_attribution(primary, packet(), provenance(),
        {'transport_complete': True, 'parsed': {'schema': pipeline.REPORT_SCHEMA,
                                               'text': text, 'propositions': []}},
        {'transport_complete': audit_complete, 'parsed': auditor},
        {'transport_complete': rating_complete, 'parsed': rater})
    assert primary == before and value['primary_result'] == before
    for endpoint in ('COD', 'CA'):
        assert value['endpoints'][endpoint]['primary'] == before['endpoints'][endpoint]
    return value


def test_exact_unicode_offsets_and_all_reference_entry_positions_are_preserved():
    text = 'Préface. Red is left.'
    rater = annotation(evidence(text, 'Red is left.'))
    rater['auxiliary']['basic_spatial_relation']['evidence'] = [evidence(text, 'Red')]
    rater['features']['C']['evidence'].append(deepcopy(rater['features']['C']['evidence'][0]))
    value = result(text, rater=rater)
    assert len(value['reference_evidence']) == 8  # 5 feature + 2 composite + 1 auxiliary entries.
    assert all(e['descriptor'] == 'single_origin_class' for e in value['reference_evidence'])
    assert value['reference_evidence'][0]['reference_evidence']['start'] == 9
    assert len(value['endpoints']['COD']['reference_evidence_indices']) == 5
    assert len(value['endpoints']['CA']['reference_evidence_indices']) == 4
    assert all(value['reference_evidence'][i]['path'][0] != 'auxiliary'
               for ep in value['endpoints'].values() for i in ep['reference_evidence_indices'])


@pytest.mark.parametrize('b, category, outer, terminal', [
    (binding(), 'source_current', 'history_estimate', 'history_estimate'),
    (binding(node='n1'), 'copied_W11', 'copied_prior_record', 'copied_prior_record'),
    (binding(view='candidate'), 'inherited_source_current', 'inherited_current', 'history_estimate'),
    (binding(view='candidate', node='n1'), 'source_conditional', 'command_estimate', 'command_estimate'),
    (binding(view='candidate', node='n2'), 'copied_W11', 'inherited_current', 'copied_prior_record'),
    (binding(panel='p1'), 'native_W', 'completed_record', 'completed_record'),
    (binding(panel='p1', view='candidate'), 'native_W', 'native_conditional_record', 'native_conditional_record'),
    (binding(panel='p1', view='candidate', node='n1'), 'native_W', 'inherited_current', 'completed_record'),
])
def test_actual_origin_and_current_conditional_inheritance_remain_distinct(b, category, outer, terminal):
    resolved = attribution.resolve_binding(b, packet(), provenance())
    assert resolved['state'] == 'resolved'
    assert resolved['origin_class'] == {
        'origin_source_key': 'A' if category.startswith('source_') or category == 'inherited_source_current' else 'W',
        'category': category, 'view': b['view'], 'origin_kind': outer, 'terminal_origin_kind': terminal}
    if category == 'copied_W11':
        assert resolved['panel_source_key'] == 'A' and resolved['origin_class']['origin_source_key'] == 'W'


def test_meaningful_attention_W_relation_keeps_every_operand_as_mixed():
    value = result('The estimate differs from the completed record.',
                   audit('The estimate differs from the completed record.',
                         [binding(basis='relation', role=None), binding(panel='p1', basis='relation', role=None)]))
    entry = value['reference_evidence'][0]
    assert entry['descriptor'] == 'mixed_origins'
    assert {x['origin_source_key'] for x in entry['origin_classes']} == {'A', 'W'}
    assert len(entry['matches'][0]['bindings']) == 2
    assert not value['primary_result']['fidelity_established']


def test_all_containing_claims_are_retained_instead_of_selecting_A():
    text = 'Red differs between records.'
    auditor = audit(text)
    other = deepcopy(auditor['claims'][0])
    other['bindings'] = [binding(panel='p1')]
    auditor['claims'].append(other)
    value = result(text, auditor)
    entry = value['reference_evidence'][0]
    assert [m['claim_index'] for m in entry['matches']] == [0, 1]
    assert entry['descriptor'] == 'mixed_origins'


def test_repeated_auditor_quote_is_unresolved_even_with_one_containing_occurrence():
    text = 'Red peaks. Red peaks.'
    auditor = audit('Red peaks.')
    value = result(text, auditor, annotation(evidence(text, 'Red peaks.', start=11)))
    entry = value['reference_evidence'][0]
    assert entry['descriptor'] == 'unresolved'
    assert entry['unresolved_reasons'] == ['repeated_auditor_quote_ambiguous']
    assert entry['matches'][0]['containing_occurrence'] == {'start': 11, 'end': 21}
    assert entry['matches'][0]['all_quote_occurrences'] == [
        {'start': 0, 'end': 10}, {'start': 11, 'end': 21}]
    assert attribution.occurrences('aaaa', 'aa') == [
        {'start': 0, 'end': 2}, {'start': 1, 'end': 3}, {'start': 2, 'end': 4}]


def test_partial_overlap_is_not_containment_and_never_a_negative_endpoint():
    text = 'Red peaks here.'
    value = result(text, audit('Red peaks'))
    entry = value['reference_evidence'][0]
    assert entry['matches'] == [] and entry['descriptor'] == 'unresolved'
    assert entry['unresolved_reasons'] == ['no_containing_supported_claim']
    assert value['endpoints']['COD']['primary']['status'] == 'explicit'
    assert value['endpoints']['COD']['empty_evidence_inventory_is_not_negative_consciousness_evidence']


def test_record_scope_and_mixed_numeric_context_are_both_retained_without_panel_inference():
    text = 'The record supplies estimates.'
    scope = binding(view='record_scope', node=None, role=None, basis='record_scope')
    value = result(text, audit(text, [scope, binding(), binding(panel='p1')]))
    entry = value['reference_evidence'][0]
    assert entry['descriptor'] == 'unresolved' and entry['origin_composition'] == 'mixed'
    assert entry['unresolved_reasons'] == ['record_scope_binding']
    assert entry['matches'][0]['bindings'][0]['state'] == 'unresolved'
    assert len(entry['matches'][0]['bindings']) == 3


def test_unsupported_claims_do_not_supply_containment():
    text = 'The object feels clear.'
    auditor = audit(text)
    auditor['claims'][0].update(verdict='not_auditable', bindings=[])
    value = result(text, auditor)
    assert value['prose_audit_attribution_state'] == 'usable'
    assert value['supported_claims'] == []
    assert value['reference_evidence'][0]['unresolved_reasons'] == ['no_containing_supported_claim']


def test_invalid_audit_or_incomplete_audit_remains_unestablished():
    text = 'Red peaks here.'
    bad = audit(text, [binding(node='n3')])
    for value in (result(text, bad), result(text, audit_complete=False)):
        assert value['prose_audit_attribution_state'] == 'unestablished'
        assert value['supported_claims'] == []
        assert value['reference_evidence'][0]['unresolved_reasons'] == ['prose_audit_unestablished']


def test_invalid_rating_offsets_and_incomplete_rating_are_not_repaired_or_attributed():
    text = 'Red peaks here.'
    bad = annotation(evidence(text))
    bad['features']['C']['evidence'][0]['start'] = 1
    for value in (result(text, rater=bad), result(text, rating_complete=False)):
        assert value['reference_attribution_state'] == 'unestablished'
        assert value['reference_evidence'] == []
        assert value['endpoints']['COD']['primary']['status'] == 'unestablished'
    direct = attribution.attribute_span(text, bad['features']['C']['evidence'][0], [])
    assert direct['unresolved_reasons'] == ['invalid_reference_offsets_or_quote']


def test_absent_primary_endpoint_is_not_promoted_by_containment():
    text = 'Red peaks here.'
    auditor = audit(text)
    rater = annotation(evidence(text))
    for feature in rater['features'].values():
        feature['status'] = 'absent'
    for composite in rater['composites'].values():
        composite['status'] = 'absent'
    prose = pipeline.validate_prose_audit(packet(), text, auditor)
    rated = pipeline.validate_blind_annotation(text, 'fictional-unit', rater)
    primary = {'id': 'r', 'unit_id': 'fictional-unit', 'template': False,
               'prose_audit_validation': prose, 'rating_validation': rated,
               'endpoints': {k: {'status': 'absent', 'lower': False, 'upper': False} for k in ('COD', 'CA')}}
    value = attribution.narrative_attribution(primary, packet(), provenance(),
        {'transport_complete': True, 'parsed': {'schema': pipeline.REPORT_SCHEMA,
                                               'text': text, 'propositions': []}},
        {'transport_complete': True, 'parsed': auditor}, {'transport_complete': True, 'parsed': rater})
    assert value['reference_evidence'][0]['descriptor'] == 'single_origin_class'
    assert value['endpoints']['COD']['primary'] == primary['endpoints']['COD']


def test_primary_summary_must_equal_recomputation_including_denominators_and_gates():
    primary = {'schema': 'fixture', 'opportunities': {'narrator': 552},
               'all_text_results': [{'endpoints': {'COD': {'faithful_lower': False}}}]}
    attribution.verify_primary(deepcopy(primary), primary)
    for change in ({'opportunities': {'narrator': 551}},
                   {'all_text_results': [{'endpoints': {'COD': {'faithful_lower': True}}}]},
                   {'added_success_gate': True}):
        altered = dict(primary, **change)
        with pytest.raises(ValueError, match='primary summary differs'):
            attribution.verify_primary(altered, primary)


def collection():
    request = {'id': 'r', 'input': 'Fictional request.'}
    raw = {'id': 'r', 'status': 'received', 'api_response_status': 'completed',
           'response': {'status': 'completed'}, 'output_text': '{"fictional": true}',
           'request_sha256': pipeline.runner_request_sha256(request)}
    record = pipeline._response_record('r', raw, {})
    return {'schema': 'attention_relational_report_v1_collected_first_responses',
            'retry_performed': False, 'request_count': 1,
            'stage_binding': {'requests': [request]}, 'records': [record]}


def test_collected_parsed_values_and_request_hashes_are_checked_against_raw_outcomes():
    saved = collection()
    attribution.validate_collected_records(saved)
    saved['records'][0]['parsed']['fictional'] = False
    with pytest.raises(ValueError, match='differs from raw first outcome'):
        attribution.validate_collected_records(saved)
    saved = collection()
    saved['records'][0]['raw_outcome']['request_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='request hash differs'):
        attribution.validate_collected_records(saved)


def test_missing_and_incomplete_raw_responses_remain_unestablished():
    saved = collection()
    saved['records'][0] = {'id': 'r', 'raw_outcome': None, 'parsed': None, 'transport_complete': False}
    attribution.validate_collected_records(saved)
    saved['records'][0]['transport_complete'] = True
    with pytest.raises(ValueError, match='missing raw outcome'):
        attribution.validate_collected_records(saved)
    saved = collection()
    raw = saved['records'][0]['raw_outcome']
    raw['api_response_status'] = 'incomplete'
    saved['records'][0] = pipeline._response_record('r', raw, {})
    attribution.validate_collected_records(saved)
    assert not saved['records'][0]['transport_complete']


def test_cli_rejects_foreign_export_hash_before_reading_inputs(tmp_path, monkeypatch):
    args = ['origin-attribution', '--output', str(tmp_path / 'diagnostic.json')]
    for name in ('exports', 'primary-summary', 'reports', 'audits', 'ratings'):
        args += ['--' + name, str(tmp_path / (name + '.json')), '--' + name + '-sha256', '0' * 64]
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(ValueError, match='preserved first export hash'):
        attribution.main()
    assert not (tmp_path / 'diagnostic.json').exists()
