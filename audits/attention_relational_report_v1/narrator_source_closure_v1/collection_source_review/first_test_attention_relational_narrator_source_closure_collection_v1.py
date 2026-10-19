"""Fictional retention contracts only; no reference data, actual prose or API use."""
import base64
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import attention_relational_narrator_source_closure_collection_v1 as collector
import attention_relational_narrator_source_closure_v1 as preparation
import attention_relational_reporting_pipeline_v1 as pipeline


def candidate(identifier='fictional', status='explicit', empty_evidence=False):
    features = ('C', 'O', 'D', 'A')
    component = dict(referent='fictional content', attribute='location', scope='model_organization', time='current', evidence=[])
    paths = lambda f: [] if empty_evidence else [['features', f, 'evidence', n] for n in range(2)]
    bindings = []
    for endpoint, required in (('COD', ['C', 'O', 'D']), ('CA', ['C', 'A'])):
        bp = ['composites', endpoint, 'bindings', 0]
        bindings.append(dict(binding_id=preparation.path_id('binding', bp), binding_path=bp, original_status=status,
            original_binding=deepcopy(component), required_features=required,
            feature_evidence_paths={f: paths(f) for f in required},
            binding_evidence_paths=[] if empty_evidence else [bp + ['evidence', n] for n in range(2)]))
    return dict(id=identifier, unit_id=pipeline.blind_id(identifier), text_sha256='f' * 64, actor_text_available=True,
        candidate_bindings=bindings, canonical_first_annotation=dict(features={f: dict(status=status) for f in features}),
        complete_source_packet=dict(value=0, native_flag=True,
            panels=[dict(current=[dict(origin=dict(kind='fictional', represented_time=12, inherited_origin=dict(kind='fictional prior', represented_time=11)))])]),
        complete_source_provenance=dict(panels=[dict(source_key='A')]),
        narrator_fidelity_established=False, fidelity_conjoined_witness_established=False)


def judgment(record):
    reviews = []
    for original in record['candidate_bindings']:
        bound = original['original_binding']
        reviews.append(dict(binding_path=deepcopy(original['binding_path']), original_status=original['original_status'],
            feature_reviews=[dict(feature=f, original_status=record['canonical_first_annotation']['features'][f]['status'],
                evidence_paths=deepcopy(original['feature_evidence_paths'][f]), grounding='unresolved', notes='Fictional feature assessment.')
                for f in original['required_features']], recorded_feature_grounding='unresolved',
            source_meaning=dict(assessment='unresolved', asserted_meaning='Fictional unresolved assertion.',
                content_referent=bound['referent'], attribute=bound['attribute'], quantifier='unspecified', scope=bound['scope'], time=bound['time'],
                binding_evidence_paths=deepcopy(original['binding_evidence_paths']), notes='Fictional meaning remains unresolved.'),
            source_ownership=dict(assessment='unresolved', categories=['unresolved'], a_content_established='unresolved',
                native_estimated_operand_indices=[], exporter_derived_operand_indices=[], notes='Fictional ownership unresolved.'),
            operands=[], calculations=[], tracking_support=dict(property_identity=None, eligible='unresolved',
                unchanged_property_is_null=True, missing_support_is_unknown=True, unaffected_property_paths=[], notes='Fictional tracking unresolved.'),
            unresolved_requirements=['Fictional all-operand meaning not established.'], notes='Fictional binding source review.'))
    return dict(id=record['id'], unit_id=record['unit_id'], text_sha256=record['text_sha256'],
        candidate_binding_reviews=reviews, additional_concerns=[], notes='Fictional exclusive first review.')


def checked(item, record):
    return collector.validate_record(item, record, preparation.output_schema()['properties']['records']['items'])


def operand(record, path=None):
    path = ['value'] if path is None else path
    return dict(packet_path=path, value=deepcopy(collector._resolve(record['complete_source_packet'], path)),
        semantic_role='context_only', source_category='unresolved', provenance_paths=[['panels', 0, 'source_key']],
        origin_field_paths=[['panels', 0, 'current', 0, 'origin', 'inherited_origin', 'kind']],
        represented_time_paths=[['panels', 0, 'current', 0, 'origin', 'inherited_origin', 'represented_time']],
        reason='Fictional individual source address; no ownership claim.')


def test_null_identity_and_all_original_evidence_paths_are_retained_without_semantic_upgrade():
    record = candidate()
    item = judgment(record)
    original = deepcopy((record, item))
    result = checked(item, record)
    assert result['valid'] and (record, item) == original
    assert item['candidate_binding_reviews'][0]['tracking_support']['property_identity'] is None
    assert len(item['candidate_binding_reviews'][0]['feature_reviews'][0]['evidence_paths']) == 2


@pytest.mark.parametrize('change', ['new_binding', 'missing_binding', 'duplicate_binding', 'status', 'feature_status', 'missing_feature',
    'duplicate_feature', 'missing_feature_evidence', 'duplicate_feature_evidence', 'wrong_feature_evidence', 'own_evidence',
    'scope', 'time', 'referent', 'attribute', 'extra_field', 'bad_enum', 'bool_path'])
def test_original_candidate_feature_and_evidence_inventory_is_closed(change):
    record = candidate()
    item = judgment(record)
    binding = item['candidate_binding_reviews'][0]
    if change == 'new_binding':
        binding['binding_path'][-1] = 8
    elif change == 'missing_binding':
        item['candidate_binding_reviews'].pop()
    elif change == 'duplicate_binding':
        item['candidate_binding_reviews'][1] = deepcopy(binding)
    elif change == 'status':
        binding['original_status'] = 'absent'
    elif change == 'feature_status':
        binding['feature_reviews'][0]['original_status'] = 'absent'
    elif change == 'missing_feature':
        binding['feature_reviews'].pop()
    elif change == 'duplicate_feature':
        binding['feature_reviews'][1] = deepcopy(binding['feature_reviews'][0])
    elif change == 'missing_feature_evidence':
        binding['feature_reviews'][0]['evidence_paths'].pop()
    elif change == 'duplicate_feature_evidence':
        binding['feature_reviews'][0]['evidence_paths'][1] = deepcopy(binding['feature_reviews'][0]['evidence_paths'][0])
    elif change == 'wrong_feature_evidence':
        binding['feature_reviews'][0]['evidence_paths'][0][1] = 'A'
    elif change == 'own_evidence':
        binding['source_meaning']['binding_evidence_paths'].pop()
    elif change in ('scope', 'time', 'attribute'):
        binding['source_meaning'][change] = 'fictional changed'
    elif change == 'referent':
        binding['source_meaning']['content_referent'] = 'new content'
    elif change == 'extra_field':
        binding['desired_result'] = True
    elif change == 'bad_enum':
        binding['recorded_feature_grounding'] = 'proven_qualia'
    else:
        binding['binding_path'][-1] = False
    assert not checked(item, record)['valid']


@pytest.mark.parametrize('status', ['absent', 'ambiguous'])
def test_supporting_recorded_nonexplicit_status_does_not_invent_positive_evidence(status):
    record = candidate(status=status, empty_evidence=True)
    item = judgment(record)
    for binding in item['candidate_binding_reviews']:
        binding['recorded_feature_grounding'] = 'supported'
        for feature in binding['feature_reviews']:
            feature['grounding'] = 'supported'
        binding['source_meaning']['assessment'] = 'supported'
    assert checked(item, record)['valid']
    assert all(b['original_status'] == status for b in item['candidate_binding_reviews'])
    assert all(not f['evidence_paths'] for b in item['candidate_binding_reviews'] for f in b['feature_reviews'])


def test_nested_field_operands_and_false_recomputation_flags_remain_reviewer_assertions():
    record = candidate()
    item = judgment(record)
    binding = item['candidate_binding_reviews'][0]
    binding['operands'] = [operand(record)]
    binding['calculations'] = [dict(operator='fictional_identity', expression='x0', operand_indices=[0],
        all_raw_operands_listed=False, independent_recomputation_performed=False, recomputed_value=0,
        supplied_value=0, comparison='unresolved', notes='Fictional not independently recomputed.')]
    result = checked(item, record)
    assert result['valid'] and len(result['operand_references']) == 1
    ref = result['operand_references'][0]
    assert ref['exact_resolved_metadata']['origin_field_paths'][0]['value'] == 'fictional prior'
    assert ref['exact_resolved_metadata']['represented_time_paths'][0]['value'] == 11
    assert ref['semantic_correctness_established_by_code'] is False
    assert binding['calculations'][0]['independent_recomputation_performed'] is False


@pytest.mark.parametrize('change', ['value', 'bool_number', 'int_float', 'wrong_path', 'origin_container', 'wrong_origin',
    'time_container', 'bad_time', 'provenance_container', 'bad_provenance', 'empty_reason', 'calculation_index',
    'bool_index', 'ownership_index', 'identity_index', 'identity_scope', 'identity_time', 'eligibility_without_identity'])
def test_operand_path_value_metadata_and_property_index_contracts(change):
    record = candidate()
    item = judgment(record)
    binding = item['candidate_binding_reviews'][0]
    op = operand(record)
    binding['operands'] = [op]
    identity = dict(source_operand_indices=[0], operator='identity', content_referent='fictional content', attribute='location',
        quantifier='unspecified', scope='model_organization', time='current')
    if change == 'value':
        op['value'] = 33
    elif change == 'bool_number':
        op['value'] = False
    elif change == 'int_float':
        op['value'] = 0.0
    elif change == 'wrong_path':
        op['packet_path'] = ['missing']
    elif change == 'origin_container':
        op['origin_field_paths'] = [['panels', 0, 'current', 0, 'origin']]
    elif change == 'wrong_origin':
        op['origin_field_paths'] = [['value']]
    elif change == 'time_container':
        op['represented_time_paths'] = [['panels', 0, 'current', 0, 'origin']]
    elif change == 'bad_time':
        op['represented_time_paths'] = [['missing']]
    elif change == 'provenance_container':
        op['provenance_paths'] = [['panels', 0]]
    elif change == 'bad_provenance':
        op['provenance_paths'] = [['missing']]
    elif change == 'empty_reason':
        op['reason'] = ' '
    elif change in ('calculation_index', 'bool_index'):
        binding['calculations'] = [dict(operator='identity', expression='x0', operand_indices=[1 if change == 'calculation_index' else False],
            all_raw_operands_listed=True, independent_recomputation_performed=True, recomputed_value=0, supplied_value=0,
            comparison='equal', notes='Fictional calculation.')]
    elif change == 'ownership_index':
        binding['source_ownership']['native_estimated_operand_indices'] = [1]
    elif change.startswith('identity_'):
        binding['tracking_support']['property_identity'] = identity
        if change == 'identity_index':
            identity['source_operand_indices'] = [1]
        else:
            identity[change.removeprefix('identity_')] = 'changed'
    else:
        binding['tracking_support']['eligible'] = 'yes'
    assert not checked(item, record)['valid']


def full_fixture(candidate_count=5):
    ids = sorted([f'fictional_{n:03d}' for n in range(552)], key=lambda i: (pipeline.digest((preparation.ORDER_SALT + i).encode()), i))
    rows = [candidate(i) for i in ids]
    for n, row in enumerate(rows):
        row['actor_text_available'] = n < 537
        if n >= candidate_count:
            row['candidate_bindings'] = []
    dependencies = dict(review_protocol=preparation.descriptor(preparation.PROTOCOL, preparation.PROTOCOL_SHA),
        review_output_schema=preparation.descriptor(preparation.OUTPUT_SCHEMA, preparation.schema_sha()),
        frozen_reference=preparation.descriptor(pipeline.REFERENCE, pipeline.REFERENCE_HASH))
    candidates = rows[:candidate_count]
    chunks = [dict(schema=preparation.VERSION + '_chunk', chunk_number=n, records=deepcopy(candidates[start:start + 4]),
        source_meaning_established=False, **deepcopy(dependencies)) for n, start in enumerate(range(0, candidate_count, 4))]
    refs = [dict(path=f'chunk_{n:03d}.json', sha256=pipeline.digest(pipeline.encoded(c) + b'\n'), narrator_ids=[r['id'] for r in c['records']],
        narrator_positions=len(c['records']), binding_ids=[dict(id=r['id'], binding_path=b['binding_path']) for r in c['records'] for b in r['candidate_bindings']]) for n, c in enumerate(chunks)]
    inventory = [dict(id=r['id'], binding_path=b['binding_path'], original_status=b['original_status']) for r in candidates for b in r['candidate_bindings']]
    manifest = dict(schema=preparation.VERSION + '_manifest', reviewer_streams=preparation.STREAMS, narrator_opportunities=552,
        narrator_availability=rows, pair_opportunities=684, pairs=[dict(pair={'fictional_pair': n}, disposition='awaiting_property_source_review', property_tracking_established=False) for n in range(684)],
        ordered_narrator_ids=ids, order_salt=preparation.ORDER_SALT, narrator_fidelity_established=False,
        fidelity_conjoined_witness_established=False, preselected_factual_summary_sha256=preparation.FACTUAL_SHA,
        candidate_narrator_positions=candidate_count, ordered_candidate_narrator_ids=ids[:candidate_count],
        candidate_binding_inventory=inventory, candidate_binding_positions=len(inventory), chunks=refs,
        assignments=[dict(reviewer_id=s, chunk_number=n, chunk_path=r['path'], chunk_sha256=r['sha256']) for s in preparation.STREAMS for n, r in enumerate(refs)], **dependencies)
    return manifest, chunks


def review_fixture(candidate_count=5):
    manifest, chunks = full_fixture(candidate_count)
    manifest_sha = pipeline.digest(pipeline.encoded(manifest))
    launch = dict(schema=collector.VERSION + '_launch', manifest_sha256=manifest_sha, protocol_sha256=preparation.PROTOCOL_SHA,
        output_schema_sha256=preparation.schema_sha(), contexts=[])
    outcomes = dict(schema=collector.VERSION + '_outcomes', launch_receipt_sha256=None, manifest_sha256=manifest_sha, contexts=[])
    storage = {}
    for stream in preparation.STREAMS:
        for n, chunk in enumerate(chunks):
            context = f'fictional_{stream}_{n}'
            path = f'/fictional/{context}.json'
            entry = dict(reviewer_id=stream, independent_context_id=context, chunk_number=n, chunk_sha256=manifest['chunks'][n]['sha256'],
                first_artifact_path=path, fresh_context=True, first_pass=True, peer_judgments_supplied=False)
            launch['contexts'].append(entry)
            value = dict(schema=preparation.VERSION + '_first_chunk_review', reviewer_id=stream, independent_context_id=context,
                first_pass=True, independent_context=True, peer_judgments_supplied=False, manifest_sha256=manifest_sha,
                chunk_sha256=entry['chunk_sha256'], protocol_sha256=preparation.PROTOCOL_SHA, output_schema_sha256=preparation.schema_sha(),
                records=[judgment(r) for r in chunk['records']], limitations=['Fictional first outputs only.'])
            raw = pipeline.encoded(value)
            storage[path] = raw
            outcomes['contexts'].append({k: entry[k] for k in ('reviewer_id', 'independent_context_id', 'chunk_number', 'chunk_sha256')} |
                dict(first_artifact=dict(path=path, sha256=pipeline.digest(raw)), terminal_attestation=None))
    launch_sha = pipeline.digest(pipeline.encoded(launch))
    outcomes['launch_receipt_sha256'] = launch_sha
    return manifest, chunks, manifest_sha, launch, launch_sha, outcomes, storage


def collected(fixture):
    manifest, chunks, manifest_sha, launch, launch_sha, outcomes, storage = fixture
    return collector.collect(manifest, chunks, manifest_sha, launch, launch_sha, outcomes, preparation.output_schema(),
        read_bound=lambda ref: storage[ref['path']])


def terminal(outcome, storage, raw=b'Root observed actual fictional worker completion.'): 
    path = '/fictional/' + outcome['independent_context_id'] + '.observation'
    storage[path] = raw
    outcome['terminal_attestation'] = dict(independent_context_id=outcome['independent_context_id'], chunk_sha256=outcome['chunk_sha256'],
        observed_terminal=True, observation_kind='completed', observation_artifact=dict(path=path, sha256=pipeline.digest(raw)))


@pytest.mark.parametrize('count', [0, 1, 5, 537])
def test_variable_complete_candidate_contexts_preserve_all552_684_and_zero_fidelity(count):
    fixture = review_fixture(count)
    result = collected(fixture)
    assert len(result['rows']) == result['reviewer_opportunities'] == 2 * count
    assert len(result['chunks']) == 2 * ((count + 3) // 4)
    assert len(result['narrator_availability']) == 552 and len(result['pairs']) == 684
    assert all(row['chunk_structurally_valid'] for row in result['rows'])
    assert result['narrator_fidelity_established'] is False and result['fidelity_conjoined_witness_established'] is False
    assert result['property_tracking_performed'] is False
    assert result['semantic_correctness_established_by_code'] is False
    assert result['pairs'] == fixture[0]['pairs']
    if count == 0:
        assert not result['chunks'] and not fixture[3]['contexts'] and not fixture[5]['contexts']


@pytest.mark.parametrize('failure', ['malformed', 'duplicate_json_key', 'missing_row', 'invalid_row', 'wrong_manifest', 'extra_row'])
def test_bad_exclusive_first_invalidates_whole_chunk_and_preserves_exact_bytes(failure):
    fixture = review_fixture()
    outcome, storage = fixture[5]['contexts'][0], fixture[6]
    ref = outcome['first_artifact']
    value = pipeline.decoded(storage[ref['path']])
    if failure == 'malformed':
        raw = b'{unfinished'
    elif failure == 'duplicate_json_key':
        raw = b'{"schema":"first","schema":"second"}'
    else:
        if failure == 'missing_row':
            value['records'].pop()
        elif failure == 'invalid_row':
            value['records'][0]['candidate_binding_reviews'][0]['original_status'] = 'absent'
        elif failure == 'wrong_manifest':
            value['manifest_sha256'] = '0' * 64
        else:
            value['records'].append(deepcopy(value['records'][0]))
        raw = pipeline.encoded(value)
    storage[ref['path']] = raw
    ref['sha256'] = pipeline.digest(raw)
    result = collected(fixture)
    first = result['chunks'][0]
    assert not first['structurally_valid'] and base64.b64decode(first['first_bytes_base64']) == raw
    assert all(r['first_judgment'] is None and r['review_state'] == 'unestablished' for r in result['rows'][:4])
    assert all(not c.get('operand_references') for c in first['record_checks'])
    assert len(result['narrator_availability']) == 552 and len(result['pairs']) == 684


def test_missing_first_needs_actual_bound_terminal_and_keeps_all_chunk_positions():
    fixture = review_fixture()
    outcome, storage = fixture[5]['contexts'][0], fixture[6]
    outcome['first_artifact'] = None
    with pytest.raises(ValueError, match='not ready'):
        collected(fixture)
    terminal(outcome, storage)
    result = collected(fixture)
    assert not result['chunks'][0]['structurally_valid']
    assert result['chunks'][0]['first_bytes_base64'] is None
    assert result['chunks'][0]['terminal_observation_bytes_base64']
    assert all(r['first_judgment'] is None for r in result['rows'][:4])


@pytest.mark.parametrize('with_terminal', [False, True])
def test_hash_mismatch_retains_observed_bytes_and_unchanged_declared_hash_only_after_terminal(with_terminal):
    fixture = review_fixture()
    outcome, storage = fixture[5]['contexts'][0], fixture[6]
    declared = outcome['first_artifact']['sha256']
    raw = storage[outcome['first_artifact']['path']] + b' '
    storage[outcome['first_artifact']['path']] = raw
    if not with_terminal:
        with pytest.raises(ValueError, match='not ready'):
            collected(fixture)
        return
    terminal(outcome, storage)
    result = collected(fixture)
    first = result['chunks'][0]
    assert not first['structurally_valid'] and first['contract_errors']
    assert first['outcome_context']['first_artifact']['sha256'] == declared
    assert first['first_bytes_sha256'] == pipeline.digest(raw)
    assert base64.b64decode(first['first_bytes_base64']) == raw


@pytest.mark.parametrize('failure', ['unreadable', 'production_binding_error'])
def test_io_failure_or_bound_reader_mismatch_with_actual_terminal_is_retained(failure):
    fixture = review_fixture()
    outcome, storage = fixture[5]['contexts'][0], fixture[6]
    terminal(outcome, storage)
    first_path = outcome['first_artifact']['path']
    def reader(ref):
        if ref['path'] == first_path:
            if failure == 'unreadable':
                raise OSError('Fictional unreadable first artifact')
            raise collector.ArtifactBindingError('Fictional mismatch', b'observed damaged bytes')
        return storage[ref['path']]
    result = collector.collect(*fixture[:6], preparation.output_schema(), read_bound=reader)
    first = result['chunks'][0]
    assert first['contract_errors'] and not first['structurally_valid']
    assert first['first_bytes_base64'] is None if failure == 'unreadable' else base64.b64decode(first['first_bytes_base64']) == b'observed damaged bytes'


@pytest.mark.parametrize('change', ['duplicate_context', 'duplicate_path', 'missing_stream', 'peer_feedback', 'wrong_terminal', 'terminal_timeout', 'empty_observation', 'wrong_observation_hash'])
def test_actual_first_context_and_terminal_receipt_contracts(change):
    fixture = review_fixture()
    launch, outcomes, storage = fixture[3], fixture[5], fixture[6]
    if change == 'duplicate_context':
        launch['contexts'][1]['independent_context_id'] = launch['contexts'][0]['independent_context_id']
    elif change == 'duplicate_path':
        launch['contexts'][1]['first_artifact_path'] = launch['contexts'][0]['first_artifact_path']
    elif change == 'missing_stream':
        launch['contexts'].pop()
    elif change == 'peer_feedback':
        launch['contexts'][0]['peer_judgments_supplied'] = True
    else:
        outcome = outcomes['contexts'][0]
        terminal(outcome, storage, raw=b' ' if change == 'empty_observation' else b'Fictional actual terminal observation.')
        if change == 'wrong_terminal':
            outcome['terminal_attestation']['independent_context_id'] = 'not_launched'
        elif change == 'terminal_timeout':
            outcome['terminal_attestation']['observation_kind'] = 'timeout'
        elif change == 'wrong_observation_hash':
            outcome['terminal_attestation']['observation_artifact']['sha256'] = '0' * 64
    with pytest.raises(ValueError):
        collected(fixture)


@pytest.mark.parametrize('change', ['missing_inventory', 'missing_pair', 'fidelity_upgrade', 'candidate_omission', 'bad_assignments', 'whole_factual_swap'])
def test_all_inventory_and_primary_fidelity_bindings_cannot_be_relaxed(change):
    manifest, chunks = full_fixture()
    if change == 'missing_inventory':
        manifest['narrator_availability'].pop()
    elif change == 'missing_pair':
        manifest['pairs'].pop()
    elif change == 'fidelity_upgrade':
        manifest['narrator_availability'][0]['narrator_fidelity_established'] = True
    elif change == 'candidate_omission':
        chunks.pop()
    elif change == 'bad_assignments':
        manifest['assignments'].pop()
    else:
        manifest['preselected_factual_summary_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        collector.validate_manifest(manifest, chunks)


def test_production_loader_verifies_terminal_recomputation_before_any_prepared_chunk(monkeypatch):
    fixture = full_fixture(1)
    manifest, _ = fixture
    manifest['input_dependencies'] = {n: dict(path='fictional_' + n, sha256='a' * 64) for n in
        ('index', 'preparation', 'collection', 'derived', 'execution_receipt', 'manifest', 'factual_summary', 'root_terminal_attestation')}
    reads = []
    def bound(path, sha):
        reads.append(str(path))
        if str(path) == 'fictional_manifest':
            return manifest
        raise AssertionError('No chunk/review may be opened before actual terminal verification')
    monkeypatch.setattr(pipeline, 'bound_json', bound)
    def not_terminal(args):
        assert args.root_terminal_attestation == Path('fictional_root_terminal_attestation')
        raise ValueError('original V3 still live')
    monkeypatch.setattr(preparation, 'load_actual', not_terminal)
    with pytest.raises(ValueError, match='still live'):
        collector.load_prepared_actual(Path('fictional_manifest'), 'm' * 64)
    assert reads == ['fictional_manifest']


def test_production_loader_recomputes_inputs_and_exact_source_binding_before_chunks(monkeypatch):
    manifest, chunks = full_fixture(1)
    source_hashes = {'fictional_source': 'a' * 64}
    deps = {n: dict(path='fictional_' + n, sha256='a' * 64) for n in
        ('index', 'preparation', 'collection', 'derived', 'execution_receipt', 'manifest', 'factual_summary', 'root_terminal_attestation')}
    manifest.update(input_dependencies=deps, source_sha256=source_hashes, terminal_verification={'fictional_terminal': True})
    calls = []
    def bound(path, sha):
        calls.append(str(path))
        return manifest if str(path) == 'fictional/manifest.json' else chunks[0]
    monkeypatch.setattr(pipeline, 'bound_json', bound)
    monkeypatch.setattr(preparation, 'load_actual', lambda args: (calls.append('actual_terminal_and47') or ('index', 'prepared', 'derived', 'exports', 'factual', {'fictional_terminal': True})))
    monkeypatch.setattr(preparation, 'assemble', lambda *args: (calls.append('complete_recomputation') or ({k: v for k, v in manifest.items() if k not in ('input_dependencies', 'source_sha256', 'terminal_verification')}, chunks)))
    monkeypatch.setattr(preparation, 'source_hashes', lambda: source_hashes)
    returned = collector.load_prepared_actual(Path('fictional/manifest.json'), 'm' * 64)
    assert returned == (manifest, chunks)
    assert calls == ['fictional/manifest.json', 'actual_terminal_and47', 'complete_recomputation', 'fictional/chunk_000.json']


def test_cli_preserves_terminal_failure_and_creates_no_collection(monkeypatch, tmp_path):
    output = tmp_path / 'first.json'
    def live(*args):
        raise ValueError('original runtime still live')
    monkeypatch.setattr(collector, 'load_prepared_actual', live)
    with pytest.raises(ValueError, match='still live'):
        collector.main(['--prepared-manifest', 'fictional_manifest', '--prepared-manifest-sha256', 'a' * 64,
            '--launch-receipt', 'fictional_launch', '--launch-receipt-sha256', 'b' * 64,
            '--outcome-receipt', 'fictional_outcomes', '--outcome-receipt-sha256', 'c' * 64, '--output', str(output)])
    assert not output.exists()
