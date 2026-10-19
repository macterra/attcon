"""Entirely fictional V5 inventories/retention; not answers to any real challenge."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import attention_relational_subagent_auditor_v5 as method
from test_attention_relational_subagent_submit_v5 import fictional_request, manual_annotation
from test_attention_relational_template_index_v1 import packet_fixture


def prepared_fixture():
    small = method.pipeline.decoded(fictional_request()['input'].rsplit('\n', 1)[1])['packet']
    large, _ = packet_fixture(full=True)
    short = dict(schema=method.SHORT_SCHEMA, cases=[dict(case_id=f'private_short_fixture_{i}', packet=deepcopy(small),
        report_text='Fictional short statement.') for i in range(24)])
    long = dict(schema=method.LONG_SCHEMA, cases=[dict(case_id=f'private_long_fixture_{i}', packet=deepcopy(large),
        report_text='Fictional ' * 200) for i in range(4)])
    long_sha = method.digest(long)
    approval = dict(schema=method.APPROVAL_SCHEMA, approved=True, independent_review=True, reviewer_id='fictional_independent',
        public_cases_sha256=long_sha, hidden_key_sha256=method.LONG_KEY_SHA, key_review_sha256='b' * 64, mandatory_atom_count=12)
    inputs = dict(short_cases=dict(path='fictional-short.json', sha256=method.digest(short)),
        long_cases=dict(path='fictional-long.json', sha256=long_sha), long_key_review=dict(path='fictional-approval.json', sha256=method.digest(approval)))
    return method.prepare(short, long, approval, inputs, method.source_hashes(), (method.pipeline.ROOT / method.WORKER).read_text())


def publish_batches(prepared, tmp_path):
    paths = {}
    for scheduled in prepared['schedule']:
        bid = scheduled['batch_id']
        path = tmp_path / (bid + '.json')
        path.write_bytes(method.pipeline.encoded(prepared['public_batches'][bid]) + b'\n')
        assert method.pipeline.digest(path.read_bytes()) == scheduled['public_sha256']
        paths[bid] = path
    return paths


def host_receipts(prepared, first_batches=None):
    first_batches = first_batches or {}
    contexts, outcomes = [], []
    for i, scheduled in enumerate(prepared['schedule']):
        bid, agent = scheduled['batch_id'], f'fictional_agent_{i}'
        contexts.append(dict(batch_id=bid, agent_id=agent, fork_turns='none', inherited_parent_model=True,
            model_override=None, runtime_identity={'observed_host_identity': 'fictional fixture; no API model claim'},
            batch_sha256=scheduled['public_sha256'], output_root=scheduled['output_root']))
        outcomes.append(dict(batch_id=bid, agent_id=agent, first_batch=first_batches.get(bid), actual_agent_terminal=True,
            terminal_attestation=dict(agent_id=agent, terminal_observed=True, reason='Fictional terminal attempt, not an observation.')))
    launch = dict(schema=method.VERSION + '_launch_receipt', preparation_sha256='p' * 64,
                  all_inputs_frozen_before_first_tasks=True, contexts=contexts)
    outcome = dict(schema=method.VERSION + '_outcome_receipt', preparation_sha256='p' * 64,
        launch_receipt_sha256=method.digest(launch), no_feedback_before_all_first_contexts_saved_or_terminal=True, contexts=outcomes)
    return launch, outcome


def test_exact_four_batches56_private_positions_same_phase_order_and_no_case_gold_peer_metadata():
    prepared = prepared_fixture()
    method.validate_prepared(prepared)
    assert len(prepared['public_batches']) == 4 and len(prepared['private_mapping']) == 56
    assert [s['request_count'] for s in prepared['schedule']] == [24, 24, 4, 4]
    assert len({m['unit_id'] for m in prepared['private_mapping']}) == 56
    for phase in method.PHASES:
        orders = [[m['case_id'] for m in prepared['private_mapping'] if m['phase'] == phase and m['stream'] == stream] for stream in method.STREAMS]
        assert orders[0] == orders[1]
        assert orders[0] == sorted(orders[0], key=lambda i: (method.pipeline.digest((method.ORDER_SALT + phase + ':' + i).encode()), i))
    for batch in prepared['public_batches'].values():
        assert not {'case_id', 'phase', 'stream', 'hidden_key', 'approval', 'private_mapping'} & set(batch)
        for request in batch['requests']:
            assert 'private_short_fixture_' not in request['input'] and 'private_long_fixture_' not in request['input']
            assert request['id'].startswith('v')
    assert not prepared['semantic_qualification_assessed'] and not prepared['target_release_implemented']


def test_approval_is_required_and_full_long_workload_checks_are_not_semantic_grading():
    prepared = prepared_fixture()
    inputs = prepared['lineage']['inputs']
    for field, value in [('approved', False), ('independent_review', False), ('public_cases_sha256', '0' * 64)]:
        approval = dict(prepared['long_key_approval'], **{field: value})
        with pytest.raises(ValueError, match='approved long-public/key'):
            method.validate_approval(approval, inputs['long_cases']['sha256'])
    wrong = deepcopy(prepared['public_cases']['long'])
    wrong['cases'][0]['report_text'] = 'Too short.'
    with pytest.raises(ValueError, match='180-220'):
        method.validate_public(wrong, 'long')
    wrong = deepcopy(prepared['public_cases']['short'])
    wrong['cases'][0]['expected_verdict'] = 'Not public.'
    with pytest.raises(ValueError, match='key/category'):
        method.validate_public(wrong, 'short')


def test_collection_retains_all56_missing_positions_and_refuses_live_unfinished_context(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    prepared = prepared_fixture()
    launch, outcome = host_receipts(prepared)
    collection = method.collect(prepared, 'p' * 64, launch, method.digest(launch), outcome)
    result = method.derive(prepared, collection)
    assert len(collection['records']) == len(result['rows']) == result['opportunities'] == 56
    assert result['short_opportunities'] == 48 and result['long_opportunities'] == 8
    assert all(not r['mechanical_contract_established'] for r in result['rows'])
    assert all(r['derived_evidence']['derived_assertions'] is None for r in result['rows'])
    assert not result['semantic_qualification_assessed'] and not result['target_release_implemented']
    running = deepcopy(outcome)
    running['contexts'][-1].update(actual_agent_terminal=False, terminal_attestation=None)
    with pytest.raises(ValueError, match='timeout or running'):
        method.collect(prepared, 'p' * 64, launch, method.digest(launch), running)


def test_exact_first_artifact_and_interface_errors_survive_core_mechanics_without_supportedness_feedback(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    prepared = prepared_fixture()
    paths = publish_batches(prepared, tmp_path)
    first = prepared['schedule'][0]
    bid, root = first['batch_id'], Path(first['output_root'])
    request = prepared['public_batches'][bid]['requests'][0]
    annotation = manual_annotation(request, 'contradicted')  # Invented wrong judgment, structurally legal.
    invalid = deepcopy(annotation)
    invalid['unit_id'] = 'wrong'
    with pytest.raises(ValueError):
        method.submitter.submit(paths[bid], root, request['id'], invalid)
    method.submitter.submit(paths[bid], root, request['id'], annotation)
    method.submitter.finalize(paths[bid], root, limitations=['All other fictional positions intentionally missing.'])
    ref = method.descriptor(root / 'first_batch.json', method.pipeline.digest((root / 'first_batch.json').read_bytes()))
    launch, outcome = host_receipts(prepared, {bid: ref})
    collection = method.collect(prepared, 'p' * 64, launch, method.digest(launch), outcome)
    derived = method.derive(prepared, collection)
    assert sum(r['mechanical_contract_established'] for r in derived['rows']) == 1
    retained = next(r for r in derived['rows'] if r['request_id'] == request['id'])
    assert retained['derived_evidence']['derived_assertions'][0]['assertion']['verdict'] == 'contradicted'
    assert collection['contexts'][0]['event_logs']
    assert not derived['semantic_qualification_assessed']
    # A missing artifact is diagnosed, never filled from another request/file.
    (root / (request['id'] + '.first.json')).unlink()
    missing = method.collect(prepared, 'p' * 64, launch, method.digest(launch), outcome)
    assert not any(r['mechanical_contract_established'] for r in method.derive(prepared, missing)['rows'])


def test_foreign_or_duplicate_candidates_and_malformed_first_manifest_are_not_preferred(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    prepared = prepared_fixture()
    paths = publish_batches(prepared, tmp_path)
    scheduled = prepared['schedule'][0]
    bid, root = scheduled['batch_id'], Path(scheduled['output_root'])
    request = prepared['public_batches'][bid]['requests'][0]
    method.submitter.submit(paths[bid], root, request['id'], manual_annotation(request))
    # Another candidate is preserved, but cannot earn favorable selection.
    (root / 'unexpected.first.json').write_bytes((root / (request['id'] + '.first.json')).read_bytes())
    method.submitter.finalize(paths[bid], root)
    ref = method.descriptor(root / 'first_batch.json', method.pipeline.digest((root / 'first_batch.json').read_bytes()))
    launch, outcome = host_receipts(prepared, {bid: ref})
    collected = method.collect(prepared, 'p' * 64, launch, method.digest(launch), outcome)
    assert len(collected['contexts'][0]['annotation_files']) == 2
    assert not any(r['mechanical_contract_established'] for r in method.derive(prepared, collected)['rows'])
    (root / 'first_batch.json').write_bytes(b'{invalid JSON first manifest')
    outcome['contexts'][0]['first_batch']['sha256'] = method.pipeline.digest((root / 'first_batch.json').read_bytes())
    malformed = method.collect(prepared, 'p' * 64, launch, method.digest(launch), outcome)
    assert malformed['contexts'][0]['first_batch_artifact']['parse_error']
    assert len(malformed['records']) == 56


def test_terminal_retention_without_first_manifest_cannot_establish_a_context(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    prepared = prepared_fixture()
    paths = publish_batches(prepared, tmp_path)
    scheduled = prepared['schedule'][0]
    bid, root = scheduled['batch_id'], Path(scheduled['output_root'])
    request = prepared['public_batches'][bid]['requests'][0]
    method.submitter.submit(paths[bid], root, request['id'], manual_annotation(request))
    launch, outcome = host_receipts(prepared)  # Real-work analogue: terminal, but never finalized.
    collection = method.collect(prepared, 'p' * 64, launch, method.digest(launch), outcome)
    result = method.derive(prepared, collection)
    retained = next(r for r in result['rows'] if r['request_id'] == request['id'])
    assert retained['derived_evidence']['mechanical_valid']  # Raw mechanics are separate from first-inventory lineage.
    assert not retained['first_artifact_binding_established'] and not retained['mechanical_contract_established']
    assert retained['derived_evidence']['derived_assertions'] is None
    assert collection['contexts'][0]['annotation_files'] and collection['contexts'][0]['event_logs']
    assert all(not r['mechanical_contract_established'] for r in result['rows'])
