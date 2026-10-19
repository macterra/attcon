"""Handcrafted V4 mechanics only; no semantic qualification or scientific reads.

The fictional annotations are not rater results, hidden challenge keys, accuracy
measurements, or evidence that mechanically legal support judgments are correct.
"""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import attention_relational_grounded_auditor_v4 as grounded
import attention_relational_report_v1 as exporter
import attention_relational_reporting_pipeline_v1 as pipeline


def grid(cells):
    value = [[0.0] * 4 for _ in range(4)]
    for row, column in cells:
        value[row][column] = 1.0 / len(cells)
    return value


def packet():
    history = dict(kind='history_estimate', represented_time=12, feedback_through=11)
    copied = dict(kind='copied_prior_record', record_time=11, represented_time=12)
    native = dict(kind='completed_record', record_time=12, represented_time=12)

    def node(name, origin):
        fields = dict(red=grid([(1, 1), (1, 2)]), blue=grid([(1, 3)]))
        return dict(node=name, origin=deepcopy(origin), distributions=fields,
                    derived={role: exporter.localization_summary(g) for role, g in fields.items()})

    def inherited(origin):
        return dict(kind='inherited_current', represented_time=13, inherited_origin=deepcopy(origin))

    return dict(schema=exporter.PACK_SCHEMA, decision_time=12, previous_node='n0', candidate_node='n1',
        queried_role='red', coordinate_system=dict(rows=4, columns=4, separate_local_grids=True,
            row_direction='down', column_direction='right'), derivation={'peaks': 'All exact maximizers'},
        candidate_semantics='Conditional supplied record, not an executed branch', panels=[
            dict(panel='p0', current=[node('n0', history), node('n1', copied), node('n2', copied)],
                 candidates={'k0': [node('n0', inherited(history)),
                     node('n1', dict(kind='command_estimate', represented_time=13, feedback_through=11)),
                     node('n2', inherited(copied))]}),
            dict(panel='p1', current=[node('n0', native)], candidates={'k0': [node('n0',
                dict(kind='native_conditional_record', represented_time=13, executed_branch_claim=False))]})])


def bundle(text='One. Two. Three.'):
    request = grounded.build_request('private_scientific_name', text, packet())
    value = pipeline.decoded(request['input'].rsplit('\n', 1)[1])
    return request, value


def raw_annotation(value, selected=None, spans=None):
    if selected is None:
        selected = [next(c['context_id'] for c in value['source_contexts']
                         if c['kind'] == 'role_distribution' and c['panel'] == 'p0'
                         and c['view'] == 'current' and c['node'] == 'n0' and c['role'] == 'red')]
    spans = [value['spans'][0]['span_id']] if spans is None else spans
    return dict(schema=grounded.RAW_SCHEMA, unit_id=value['unit_id'],
        assertions=[dict(assertion_id='a0', paraphrase='Fictional auditor paraphrase; not a literal quotation.',
                         span_ids=spans, source_context_ids=selected, verdict='supported', reason='Fictional mechanics fixture.')],
        coverage=[dict(span_id=s['span_id'], assertion_ids=['a0'] if s['span_id'] in spans else [],
                       classification='assertive' if s['span_id'] in spans else 'nonassertive',
                       reason='Fictional coverage fixture.') for s in value['spans']],
        complete_assertion_inventory=True, unresolved_ambiguity=False)


def derive(request, raw):
    return grounded.derive_annotation(request, raw,
        expected_request_sha256=pipeline.runner_request_sha256(request))


def test_lossless_unicode_sentence_linebreak_and_separator_spans_have_manual_offsets():
    text = 'Réd. Next!\r\nTail?\n'
    spans = grounded.partition_spans(text)
    assert [(s['start'], s['end'], s['text']) for s in spans] == [
        (0, 4, 'Réd.'), (4, 10, ' Next!'), (10, 12, '\r\n'), (12, 17, 'Tail?'), (17, 18, '\n')]
    assert ''.join(s['text'] for s in spans) == text
    assert [s['span_id'] for s in spans] == [f's{i:04d}' for i in range(5)]
    assert grounded.partition_spans('') == []
    assert [s['text'] for s in grounded.partition_spans('Dr. A?B\tX!')] == ['Dr.', ' A?B\tX!']
    assert [s['text'] for s in grounded.partition_spans('\n \t')] == ['\n', ' \t']


def test_every_source_role_origin_and_relation_operand_is_addressed_without_fake_nulls():
    source = packet()
    before = deepcopy(source)
    contexts = grounded.source_contexts(source)
    assert len(contexts) == 35 and len({c['context_id'] for c in contexts}) == 35
    roles = [c for c in contexts if c['kind'] == 'role_distribution']
    assert len(roles) == 16
    for context in roles:
        assert context['role'] in ('red', 'blue')
        assert (context['command'] is None) == (context['view'] == 'current')
        assert context['origin']['represented_time'] == (12 if context['view'] == 'current' else 13)
    red = next(c for c in roles if c['role'] == 'red')
    assert red['calculations']['peaks'] == [[1, 1], [1, 2]]
    copied = next(c for c in roles if c['panel'] == 'p0' and c['view'] == 'current' and c['node'] == 'n2')
    inherited = next(c for c in roles if c['panel'] == 'p0' and c['view'] == 'candidate' and c['node'] == 'n2')
    assert copied['origin']['kind'] == 'copied_prior_record'
    assert inherited['origin']['inherited_origin'] == copied['origin']
    index = {c['context_id']: c for c in contexts}
    for relation in (c for c in contexts if c['kind'] == 'red_blue_peak_relation'):
        assert [index[i]['role'] for i in relation['operands']] == ['red', 'blue']
        assert all(index[i]['origin'] == relation['origin'] for i in relation['operands'])
    for record in (c for c in contexts if c['kind'] == 'record_metadata'):
        assert not {'role', 'node', 'command', 'view'} & set(record)
    assert source == before


def test_context_ids_do_not_depend_on_json_object_key_insertion_order():
    source = packet()
    reversed_keys = pipeline.decoded(pipeline.encoded(source))
    assert grounded.source_contexts(source) == grounded.source_contexts(reversed_keys)


def test_generic_request_has_only_id_input_and_opaque_payload_id_no_hidden_labels():
    request, value = bundle()
    assert set(request) == {'id', 'input'}
    assert request['id'] == 'private_scientific_name'
    assert 'private_scientific_name' not in request['input']
    assert len(value['unit_id']) == 24
    assert value['report_text_sha256'] == pipeline.digest(value['report_text'].encode('utf-8'))
    assert ''.join(s['text'] for s in value['spans']) == value['report_text']
    assert value['packet'] == packet()


def test_static_schema_is_shared_but_optional_enums_narrow_only_actual_addresses():
    request, value = bundle()
    static = grounded.response_schema()
    narrowed = grounded.response_schema([c['context_id'] for c in value['source_contexts']],
                                        [s['span_id'] for s in value['spans']])
    assertion = narrowed['properties']['assertions']['items']['properties']
    assert assertion['source_context_ids']['items']['enum'] == [c['context_id'] for c in value['source_contexts']]
    assert assertion['span_ids']['items']['enum'] == [s['span_id'] for s in value['spans']]
    assert static['properties']['assertions']['items']['properties']['source_context_ids']['items'] == {'type': 'string'}
    static['properties']['schema']['enum'].append('changed caller copy')
    assert grounded.response_schema()['properties']['schema']['enum'] == [grounded.RAW_SCHEMA]
    with pytest.raises(ValueError, match='invalid schema ID inventory'):
        grounded.response_schema(['c0', 'c0'], [])


def test_noncontiguous_exact_evidence_stays_separate_from_auditor_paraphrase():
    request, value = bundle()
    raw = raw_annotation(value, spans=['s0000', 's0002'])
    before = deepcopy(raw)
    derived = derive(request, raw)
    assert derived['mechanical_valid'] and raw == before
    assertion = derived['derived_assertions'][0]
    assert [s['text'] for s in assertion['exact_evidence']] == ['One.', ' Three.']
    assert assertion['assertion']['paraphrase'] not in value['report_text']
    assert all('quote' not in s for s in assertion['exact_evidence'])
    assert derived['semantic_completeness_established'] is False
    assert derived['support_judgments_verified'] is False
    assertion['assertion']['paraphrase'] = 'Changed output copy.'
    assert raw == before


def test_relations_retain_every_selected_cross_panel_role_and_time_operand():
    request, value = bundle()
    contexts = value['source_contexts']
    relation = next(c for c in contexts if c['kind'] == 'red_blue_peak_relation' and c['panel'] == 'p0' and c['view'] == 'current')
    extra = [c['context_id'] for c in contexts if c['kind'] == 'role_distribution'
             and ((c['panel'] == 'p1' and c['view'] == 'current' and c['role'] == 'red')
                  or (c['panel'] == 'p0' and c['view'] == 'candidate' and c['node'] == 'n1' and c['role'] == 'blue'))]
    result = derive(request, raw_annotation(value, selected=[relation['context_id']] + extra))
    operands = result['derived_assertions'][0]['operand_contexts']
    assert len(operands) == 4
    assert {c['context_id'] for c in operands} == set(relation['operands'] + extra)
    assert {c['view'] for c in operands} == {'current', 'candidate'}
    assert {c['panel'] for c in operands} == {'p0', 'p1'}


@pytest.mark.parametrize('kind', ['unknown_span', 'unknown_context', 'missing_coverage', 'duplicate_coverage',
                                  'duplicate_assertion', 'broken_reverse_link', 'nonassertive_with_claim', 'old_schema'])
def test_illegal_ids_and_incomplete_or_inconsistent_coverage_are_whole_invalid(kind):
    request, value = bundle()
    raw = raw_annotation(value)
    if kind == 'unknown_span':
        raw['assertions'][0]['span_ids'] = ['s9999']
    elif kind == 'unknown_context':
        raw['assertions'][0]['source_context_ids'] = ['c9999']
    elif kind == 'missing_coverage':
        raw['coverage'].pop()
    elif kind == 'duplicate_coverage':
        raw['coverage'].append(deepcopy(raw['coverage'][0]))
    elif kind == 'duplicate_assertion':
        raw['assertions'].append(deepcopy(raw['assertions'][0]))
    elif kind == 'broken_reverse_link':
        raw['coverage'][1].update(classification='assertive', assertion_ids=['a0'])
    elif kind == 'nonassertive_with_claim':
        raw['coverage'][0]['classification'] = 'nonassertive'
    else:
        raw['schema'] = pipeline.AUDIT_SCHEMA
    result = derive(request, raw)
    assert not result['mechanical_valid'] and result['derived_assertions'] is None and result['errors']


def test_supported_requires_context_but_unsupported_experience_needs_no_fake_context():
    request, value = bundle()
    raw = raw_annotation(value, selected=[])
    assert not derive(request, raw)['mechanical_valid']
    raw['assertions'][0].update(verdict='not_auditable', paraphrase='An unsupported experiential assertion.')
    result = derive(request, raw)
    assert result['mechanical_valid'] and result['derived_assertions'][0]['resolved_contexts'] == []


def test_complete_coverage_and_legal_ids_do_not_establish_semantic_completeness():
    request, value = bundle()
    raw = raw_annotation(value)
    raw.update(complete_assertion_inventory=False, unresolved_ambiguity=True)
    raw['assertions'][0]['paraphrase'] = 'A mechanically legal but potentially wrong interpretation.'
    result = derive(request, raw)
    assert result['mechanical_valid'] and not result['semantic_completeness_established']
    assert result['complete_assertion_inventory'] is False and result['unresolved_ambiguity'] is True


def test_request_hash_wrong_unit_and_tampered_context_tables_are_rejected():
    request, value = bundle()
    raw = raw_annotation(value)
    with pytest.raises(ValueError, match='request SHA mismatch'):
        grounded.derive_annotation(request, raw, expected_request_sha256='0' * 64)
    raw['unit_id'] = 'wrong-unit'
    assert not derive(request, raw)['mechanical_valid']
    prefix, _ = request['input'].rsplit('\n', 1)
    value['source_contexts'][0]['metadata']['decision_time'] = 99
    tampered = dict(request, input=prefix + '\n' + pipeline.encoded(value).decode('ascii'))
    with pytest.raises(ValueError, match='request/interface content differs'):
        derive(tampered, raw)


def test_empty_text_mechanics_do_not_assess_actor_availability():
    request, value = bundle('')
    raw = dict(schema=grounded.RAW_SCHEMA, unit_id=value['unit_id'], assertions=[], coverage=[],
               complete_assertion_inventory=True, unresolved_ambiguity=False)
    result = derive(request, raw)
    assert result['mechanical_valid'] and result['derived_assertions'] == []
    assert not result['actor_availability_assessed'] and not result['actual_runtime_lineage_verified']
