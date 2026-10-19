"""Fictional target accounting and gate mechanics, never semantic qualification.

The small packets/claims below are invented. The pure assembly fixture has a
fictional proof object, which cannot pass the actual pinned CLI certificate
verifier. No target, qualification answers, hidden key or APIs are read.
"""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import attention_relational_grounded_target_v4 as target
import attention_relational_grounded_auditor_v4 as core
import attention_relational_reporting_pipeline_v1 as pipeline
import attention_relational_report_v1 as exporter


def fictional_packet():
    red = [[0.0] * 4 for _ in range(4)]
    blue = deepcopy(red)
    red[0][0], blue[3][3] = 1.0, 1.0
    node = dict(node='n0', origin=dict(kind='history_estimate', represented_time=12, feedback_through=11),
        distributions=dict(red=red, blue=blue),
        derived={r: exporter.localization_summary(grid) for r, grid in (('red', red), ('blue', blue))})
    return dict(schema=exporter.PACK_SCHEMA, decision_time=12, previous_node='n0', candidate_node='n0',
        queried_role='red', coordinate_system=dict(rows=4, columns=4), derivation={'peaks': 'All exact maxima'},
        candidate_semantics='Conditional supplied record', panels=[dict(panel='p0', candidates=None, current=[node])])


def fixture_units():
    # Preserve the complete12x46 scheduling topology while using small invented packets.
    packet = fictional_packet()
    template = exporter.deterministic_template(packet)
    units = []
    for seed in (1301, 1311, 1321):
        for world in range(4):
            arms = [(s, v, variant) for s in ('A', 'ordinary', 'analytic') for v in ('A_only', 'joined')
                    for variant in exporter.VARIANTS]
            arms += [('W', 'W_only', variant) for variant in exporter.W_VARIANTS]
            arms += [('sensor', 'sensor_only', 'baseline')]
            for source, view, variant in arms:
                identifier = f'fictional_s{seed}_w{world}_{source}_{view}_{variant}'
                complete = len(units) < 537
                text = 'The red peak is at row 0, column 0.' if complete else ''
                report = dict(schema=exporter.REPORT_SCHEMA, text=text,
                              propositions=deepcopy(template['propositions'])) if complete else None
                units.append(dict(id=identifier, case=dict(encoder_seed=seed, world_index=world, decision_time=12,
                    source=source, view=view, variant=variant), packet=deepcopy(packet), text=text,
                    actor_transport_complete=complete, actor_availability='completed' if complete else 'missing',
                    report=report, template=deepcopy(template)))
    return units


def assemble_fixture(units):
    proof = dict(qualification_manifest={'sha256': target.QUALIFICATION_SHA},
        qualification_instructions_sha256=pipeline.digest(core.INSTRUCTIONS.read_bytes()),
        qualification_schema_content_sha256=target.digest(core.response_schema()),
        qualification_core_sha256=pipeline.digest(Path(core.__file__).read_bytes()), fictional_not_a_certificate=True)
    plan = target.reference_plan(dict(stage='rate', output_root='audits/attention_relational_report_v1/quote_reference_v3/rate'),
                                 dict(path='fictional-reference-manifest.json', sha256='f' * 64))
    pairs = pipeline.build_pair_index([dict(id=u['id'], case=u['case']) for u in units])
    return target.assemble(units, pairs, proof, plan, {'fictional_inputs': True}, {'fictional_source': 'e' * 64})


@pytest.fixture(scope='module')
def prepared():
    return assemble_fixture(fixture_units())


def binding(prepared):
    request_sha = target.digest(prepared['requests'])
    lineage_sha = target.digest(prepared['lineage'])
    manifest = dict(stage='audit', settings=target.settings(), output_root=target.RUN_ROOT, request_count=1074,
        request_ids=[r['id'] for r in prepared['requests']], requests_path='fictional/requests.json',
        lineage_path='fictional/lineage.json',
        source_sha256={'fictional/requests.json': request_sha, 'fictional/lineage.json': lineage_sha})
    return target.build_binding(manifest, 'a' * 64, prepared['requests'], request_sha,
                                prepared['lineage'], lineage_sha, prepared)


def annotation(request):
    value = pipeline.decoded(request['input'].rsplit('\n', 1)[1])
    context = next(c['context_id'] for c in value['source_contexts'] if c['kind'] == 'role_distribution')
    return dict(schema=core.RAW_SCHEMA, unit_id=value['unit_id'], complete_assertion_inventory=True,
        unresolved_ambiguity=False, assertions=[dict(assertion_id='a0', paraphrase='Fictional software judgment.',
            span_ids=[value['spans'][0]['span_id']], source_context_ids=[context],
            verdict='supported', reason='Invented fixture; not a semantic qualification assertion.')],
        coverage=[dict(span_id=value['spans'][0]['span_id'], assertion_ids=['a0'], classification='assertive',
                       reason='Invented fixture coverage.')])


def outcome(request, value=None, complete=True):
    return dict(id=request['id'], request_sha256=pipeline.runner_request_sha256(request), status='received',
        api_response_status='completed' if complete else 'incomplete',
        response={'status': 'completed' if complete else 'incomplete'},
        output_text=pipeline.encoded(annotation(request) if value is None else value).decode('ascii'))


def collection(prepared, outcomes=()):
    records = {r['id']: dict(id=r['id'], availability='missing', transport_complete=False,
                            parsed=None, raw_outcome=None, storage_provenance={}) for r in prepared['requests']}
    for raw in outcomes:
        records[raw['id']] = pipeline._response_record(raw['id'], raw, {'kind': 'fictional_fixture'})
    return dict(schema='attention_relational_report_v1_collected_first_responses', retry_performed=False,
        request_count=1074, stage_binding=binding(prepared), records=list(records.values()))


def same_source_requests(prepared):
    source_id = prepared['units'][0]['id']
    requested = {m['request_id'] for m in prepared['private_mapping'] if m['source_id'] == source_id}
    return [r for r in prepared['requests'] if r['id'] in requested]


def test_complete_inventory_private_mapping_and_unchanged_qualified_interface(prepared):
    target.validate_preparation(prepared)
    assert len(prepared['requests']) == 1074
    assert len(prepared['private_mapping']) == 1104 and len(prepared['pair_index']) == 684
    assert {a: sum(m['auditor'] == a for m in prepared['private_mapping']) for a in target.AUDITORS} == {
        'auditor_0': 552, 'auditor_1': 552}
    assert sum(m['request_id'] is None for m in prepared['private_mapping']) == 30
    assert len({m['unit_id'] for m in prepared['private_mapping'] if m['unit_id'] is not None}) == 1074
    assert prepared['settings'] == target.qualification.settings()
    assert prepared['lineage']['reference_join_plan']['name'] == target.JOIN_NAME
    assert prepared['lineage']['reference_join_plan']['earlier_v3_whole_resource_v2_preselection_unchanged']
    assert not prepared['lineage']['reference_join_plan']['join_implemented']
    for request in prepared['requests']:
        value = pipeline.decoded(request['input'].rsplit('\n', 1)[1])
        assert 'fictional_s1301_' not in request['input']
        assert 'auditor_0' not in request['input'] and 'auditor_1' not in request['input']
        assert set(value) == set(core.interface(value['unit_id'], value['report_text'], value['packet']))
        assert request == core.build_request(request['id'], value['report_text'], value['packet'])


def test_missing_audits_preserve_all_denominators_templates_pairs_and_unknowns(prepared):
    result = target.assess(prepared, collection(prepared))
    assert result['narrator_opportunities'] == result['literal_template_opportunities'] == 552
    assert result['audit_opportunities'] == 1104 and result['audit_requests'] == 1074
    assert result['pair_opportunities'] == len(result['pairs']) == 684
    assert result['v4_factual_fidelity_established'] == 0
    assert sum(g['narrator_opportunities'] for g in result['groups'].values()) == 552
    assert all(c['opportunities'] == c['unestablished'] == 552 for c in result['by_auditor'].values())
    assert all(t['literal_template_fidelity_by_construction'] for t in result['literal_templates'])
    assert all(p['source_packet_content_identical'] for p in result['pairs'])  # Null interventions retained.
    unavailable = [r for r in result['rows'] if not r['actor_available']]
    assert len(unavailable) == 15
    assert all(a['availability'] == 'actor_unavailable' for r in unavailable for a in r['auditors'].values())
    assert result['wording_difference_is_not_semantic_change_proof'] and not result['reference_join_implemented']


def test_both_auditors_required_no_favorable_selection_and_machine_gate_recomputed(prepared):
    requests = same_source_requests(prepared)
    result = target.assess(prepared, collection(prepared, [outcome(requests[0])]))
    assert result['v4_factual_fidelity_established'] == 0
    result = target.assess(prepared, collection(prepared, [outcome(r) for r in requests]))
    assert result['v4_factual_fidelity_established'] == 1
    first = result['rows'][0]
    assert all(a['status'] == 'established' for a in first['auditors'].values())
    units = deepcopy(prepared['units'])
    units[0]['report']['propositions'] = []
    changed = assemble_fixture(units)
    result = target.assess(changed, collection(changed, [outcome(r) for r in same_source_requests(changed)]))
    assert result['v4_factual_fidelity_established'] == 0
    assert result['rows'][0]['machine_audit']['error_count'] > 0
    assert all(a['status'] == 'established' for a in result['rows'][0]['auditors'].values())


@pytest.mark.parametrize('failure', ['incomplete', 'wrong_unit', 'incomplete_inventory', 'ambiguity', 'empty', 'contradicted'])
def test_individual_auditor_failures_cannot_be_salvaged(prepared, failure):
    request = same_source_requests(prepared)[0]
    value = annotation(request)
    if failure == 'wrong_unit':
        value['unit_id'] = 'wrong'
    elif failure == 'incomplete_inventory':
        value['complete_assertion_inventory'] = False
    elif failure == 'ambiguity':
        value['unresolved_ambiguity'] = True
    elif failure == 'empty':
        value['assertions'] = []
        value['coverage'][0].update(assertion_ids=[], classification='nonassertive')
    elif failure == 'contradicted':
        value['assertions'][0]['verdict'] = 'contradicted'
    record = pipeline._response_record(request['id'], outcome(request, value, complete=failure != 'incomplete'), {})
    mapping = next(m for m in prepared['private_mapping'] if m['request_id'] == request['id'])
    result = target.auditor_result(request, record, mapping)
    assert result['status'] != 'established'
    if failure in ('incomplete', 'wrong_unit'):
        assert result['status'] == 'unestablished' and result['derived_evidence']['derived_assertions'] is None
    else:
        assert result['status'] == 'not_established'  # A mechanically established annotation still fails the rule.


def test_actor_blank_or_incomplete_stays_unknown_without_request_or_fallback(prepared):
    units = deepcopy(prepared['units'])
    # Exchange which position is available, preserving the fixed537 request denominator.
    units[537].update(text=units[0]['text'], actor_transport_complete=True, report=deepcopy(units[0]['report']))
    units[0].update(text='', actor_transport_complete=True, report=dict(schema=exporter.REPORT_SCHEMA, text='',
        propositions=deepcopy(units[0]['template']['propositions'])))
    changed = assemble_fixture(units)
    assert all(m['request_id'] is None for m in changed['private_mapping'] if m['source_id'] == units[0]['id'])
    result = target.assess(changed, collection(changed))
    assert result['rows'][0]['machine_audit']['listed_machine_facts_pass']
    assert not result['rows'][0]['actor_available'] and not result['rows'][0]['v4_factual_fidelity_established']


def test_foreign_binding_wrong_request_hash_and_missing_records_rejected(prepared):
    request = same_source_requests(prepared)[0]
    raw = outcome(request)
    raw['request_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='request hash differs'):
        target.assess(prepared, collection(prepared, [raw]))
    value = collection(prepared)
    value['stage_binding']['measurement_version'] = 'old-prose-audit'
    with pytest.raises(ValueError, match='actual V4 target collection'):
        target.assess(prepared, value)
    value = collection(prepared)
    value['records'].pop()
    with pytest.raises(ValueError, match='all1074'):
        target.assess(prepared, value)


def test_frozen_normal_collector_retains_actual_target_binding_and_all_first_missing_positions(prepared, tmp_path):
    stage_binding = binding(prepared)
    root = tmp_path / target.RUN_ROOT
    root.mkdir(parents=True)
    request = same_source_requests(prepared)[0]
    response_path = root / (request['id'] + '.json')
    response_path.write_bytes(pipeline.encoded(outcome(request)))
    completion = dict(schema='attention_relational_report_v1_first_batch_completion', retry_performed=False,
        manifest_sha256=stage_binding['manifest_sha256'], request_count=1074, outcomes=[dict(id=request['id'],
            response_path=str(response_path), response_sha256=pipeline.digest(response_path.read_bytes()))])
    path = tmp_path / 'fictional-completion.json'
    path.write_bytes(pipeline.encoded(completion))
    result = pipeline.collect_first_responses(path, pipeline.digest(path.read_bytes()), stage_binding['request_ids'],
        root, repository_root=tmp_path, stage_binding=stage_binding)
    assert len(result['records']) == result['request_count'] == 1074
    assert result['stage_binding']['measurement_version'] == target.VERSION
    assert sum(r['transport_complete'] for r in result['records']) == 1
    returned = pipeline._records(result)[request['id']]
    assert returned['raw_outcome']['request_sha256'] == pipeline.runner_request_sha256(request)
    assert returned['storage_provenance']['sha256'] == pipeline.digest(response_path.read_bytes())


def test_fictional_passing_flags_cannot_authorize_actual_cli_target_preparation(tmp_path):
    cert = dict(schema=target.CERTIFICATE_SCHEMA,
        qualification_manifest=dict(path='never-read.json', sha256='0' * 64), qualification_preparation={},
        qualification_collection={}, qualification_derived={}, qualification_key_sha256=target.KEY_SHA,
        qualification_policy_sha256=target.POLICY_SHA, auditors_qualified={a: True for a in target.AUDITORS},
        independent_first_grades=[], adjudication=None)
    path = tmp_path / 'fictional-not-actual-certificate.json'
    path.write_bytes(pipeline.encoded(cert))
    with pytest.raises(ValueError, match='qualification release bindings differ'):
        target.verify_certificate(path, pipeline.digest(path.read_bytes()))
    with pytest.raises(ValueError, match='pinned original inputs'):
        target.prepare_target({'fictional_inputs': True})


def test_explicit_adjudication_preserves_first_disagreement_without_unanimity_gate():
    final = {a: True for a in target.AUDITORS}
    first = [final, {'auditor_0': False, 'auditor_1': None}]
    hashes = ['a' * 64, 'b' * 64]
    bindings = dict(qualification_manifest_sha256=target.QUALIFICATION_SHA, hidden_key_sha256=target.KEY_SHA,
                    grading_policy_sha256=target.POLICY_SHA)
    target.validate_resolution([final, final], final, hashes, bindings, None)
    with pytest.raises(ValueError, match='negative/uncertain'):
        target.validate_resolution(first, final, hashes, bindings, None)
    resolution = dict(schema=target.ADJUDICATION_SCHEMA, **bindings, independent_first_grade_sha256=hashes,
        auditors_qualified=final, no_unresolved_release_items=True,
        covers_all_negative_and_uncertain_first_judgments=True,
        disagreements=[{'case': 'Fictional differing atom assessments retained.'}],
        resolutions=[{'evidence': 'Fictional external semantic resolution; code does not infer correctness.'}])
    target.validate_resolution(first, final, hashes, bindings, resolution)
    for field, value in [('independent_first_grade_sha256', list(reversed(hashes))),
                         ('no_unresolved_release_items', False), ('resolutions', [])]:
        changed = dict(resolution, **{field: value})
        with pytest.raises(ValueError, match='explicit first-grade'):
            target.validate_resolution(first, final, hashes, bindings, changed)


def test_request_case_or_rule_mutation_is_rejected(prepared):
    changed = deepcopy(prepared)
    changed['requests'][0]['input'] += ' Changed.'
    with pytest.raises(ValueError, match='target preparation differs'):
        target.validate_preparation(changed)
    changed = deepcopy(prepared)
    changed['lineage']['reference_join_plan']['rule']['combination'] = 'pick-best'
    with pytest.raises(ValueError, match='reference-join plan'):
        target.validate_preparation(changed)
