"""Wholly fictional first-retention contracts; never target review judgments."""
import base64
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import attention_relational_template_review_collection_v1 as collector
import attention_relational_template_review_v1 as preparation
import attention_relational_reporting_pipeline_v1 as pipeline
from test_attention_relational_template_review_v1 import index_row, measured


@pytest.fixture
def schema():
    # This frozen structural schema is allowed; no actual annotations are read.
    return collector.frozen_output_schema()


def established_record():
    row = index_row()
    return preparation.build_record(row, measured(row))


def judgment(record):
    return dict(id=record['id'], unit_id=record['unit_id'], text_sha256=record['text_sha256'],
        reference_established=record['reference_established'],
        component_reviews=[dict(component_path=deepcopy(c['path']), original_status=c['recorded_status'],
            assessment='unsupported', notes='Fictional reviewer asserts a scope concern.') for c in record['components']],
        evidence_reviews=[dict(evidence_path=deepcopy(e['path']), grounding='unresolved', source_meaning='mixed_sources',
            properties=[], notes='Fictional reviewer leaves the operand relation unresolved.') for e in record['evidence']],
        binding_reviews=[dict(binding_path=deepcopy(b['path']),
            feature_witnesses=[dict(feature=f, evidence_paths=[deepcopy(next(e['path'] for e in record['evidence']
                if e['evidence_id'] == eid)) for eid in b['candidate_feature_evidence_ids'][f][:1]]) for f in b['required_features']],
            binding_evidence_paths=[deepcopy(next(e['path'] for e in record['evidence'] if e['evidence_id'] == eid)) for eid in b['own_evidence_ids']],
            assessment='unsupported', notes='Fictional reviewer asserts incompatible content/scope/time.') for b in record['bindings']],
        additional_concerns=[], notes='Fictional retained first judgment.' if record['reference_established'] else
            'unestablished: fictional original first annotation was unavailable.')


def record_check(item, record, schema):
    return collector.validate_record(item, record, schema['properties']['records']['items'])


def test_record_preserves_original_status_duplicate_evidence_paths_and_judgments(schema):
    record = established_record()
    item = judgment(record)
    original = deepcopy((item, record))
    checked = record_check(item, record, schema)
    assert checked['valid'] and (item, record) == original
    assert len(item['component_reviews']) == 12 and len(item['evidence_reviews']) == 42
    assert record['evidence'][0]['recorded_evidence'] == record['evidence'][2]['recorded_evidence']
    assert item['evidence_reviews'][0]['evidence_path'] != item['evidence_reviews'][2]['evidence_path']
    assert {c['assessment'] for c in item['component_reviews']} == {'unsupported'}
    assert {c['grounding'] for c in item['evidence_reviews']} == {'unresolved'}
    assert all(c['original_status'] == original['recorded_status'] for c, original in zip(item['component_reviews'], record['components']))


@pytest.mark.parametrize('change', ['extra', 'enum', 'status', 'missing_component', 'duplicate_evidence', 'missing_binding', 'bool_path'])
def test_closed_schema_and_complete_inventories_reject_changes(change, schema):
    record = established_record()
    item = judgment(record)
    if change == 'extra':
        item['component_reviews'][0]['new_assurance'] = True
    elif change == 'enum':
        item['evidence_reviews'][0]['source_meaning'] = 'A_success'
    elif change == 'status':
        item['component_reviews'][0]['original_status'] = 'absent'
    elif change == 'missing_component':
        item['component_reviews'].pop()
    elif change == 'duplicate_evidence':
        item['evidence_reviews'][1] = deepcopy(item['evidence_reviews'][0])
    elif change == 'missing_binding':
        item['binding_reviews'].pop()
    else:
        item['evidence_reviews'][0]['evidence_path'][-1] = True
    assert not record_check(item, record, schema)['valid']


def property_ref(unit, path, role='content_operand'):
    return dict(unit_id=unit['unit_id'], value_path=path,
        value=deepcopy(collector._resolve(unit['value'], path)), semantic_role=role,
        reason='Fictional full-text binding explanation relates this exact operand to the quoted assertion.')


def test_precise_nested_origin_and_complete_peak_set_outside_overlap_remain_asserted(schema):
    record = established_record()
    item = judgment(record)
    origin = next(u for u in record['property_table'] if u['kind'] == 'origin' and 'inherited_origin' in u['value'])
    leaf = next(k for k, v in origin['value']['inherited_origin'].items() if not isinstance(v, (dict, list)))
    peak = next(u for u in record['property_table'] if u['kind'] == 'role_peaks')
    refs = [property_ref(origin, ['inherited_origin', leaf], 'processing_relation_operand'), property_ref(peak, [])]
    # First evidence quotes only the header; explicit full-text reasons allow
    # external operands. Structural acceptance does not certify these reasons.
    item['evidence_reviews'][0]['properties'] = refs
    checked = record_check(item, record, schema)
    assert checked['valid']
    nested, complete = checked['property_references']
    assert nested['value_path'] == ['inherited_origin', leaf]
    assert nested['packet_value_path'] == origin['packet_path'] + ['inherited_origin', leaf]
    assert nested['source_provenance'] == origin['source_resolution']
    assert nested['value'] == origin['value']['inherited_origin'][leaf]
    assert 'inherited_origin' not in nested['value_path'][1:]
    assert complete['value_path'] == [] and complete['value'] == peak['value']
    assert not nested['overlaps_quoted_evidence'] and not complete['overlaps_quoted_evidence']
    assert nested['semantic_relation_asserted_by_reviewer_only']
    assert item['evidence_reviews'][0]['source_meaning'] == 'mixed_sources'


@pytest.mark.parametrize('change', ['wrong_unit', 'wrong_path', 'wrong_value', 'boolean_number', 'bad_role', 'empty_reason'])
def test_exact_property_identity_path_type_value_role_and_reason(change, schema):
    record = established_record()
    item = judgment(record)
    unit = next(u for u in record['property_table'] if u['kind'] == 'role_peaks')
    ref = property_ref(unit, [0, 0])
    assert type(ref['value']) is int
    if change == 'wrong_unit':
        ref['unit_id'] = 'not_supplied'
    elif change == 'wrong_path':
        ref['value_path'] = ['not_supplied']
    elif change == 'wrong_value':
        ref['value'] += 100
    elif change == 'boolean_number':
        # Use a dedicated fictional scalar equal under Python ==; JSON type
        # identity must still reject False as numeric zero or True as one.
        unit['value'][0][0] = 1
        ref['value'] = True
    elif change == 'bad_role':
        ref['semantic_role'] = 'actual_A_success'
    else:
        ref['reason'] = ' '
    item['evidence_reviews'][0]['properties'] = [ref]
    assert not record_check(item, record, schema)['valid']


@pytest.mark.parametrize('change', ['wrong_feature', 'wrong_feature_path', 'wrong_binding_path', 'empty_supported_feature'])
def test_binding_witness_ids_must_belong_to_original_feature_and_own_binding(change, schema):
    record = established_record()
    item = judgment(record)
    binding = item['binding_reviews'][0]
    if change == 'wrong_feature':
        binding['feature_witnesses'][0]['feature'] = 'A'
    elif change == 'wrong_feature_path':
        binding['feature_witnesses'][0]['evidence_paths'] = [['features', 'A', 'evidence', 0]]
    elif change == 'wrong_binding_path':
        binding['binding_evidence_paths'][0] = item['binding_reviews'][1]['binding_evidence_paths'][0]
    else:
        binding['assessment'] = 'supported'
        binding['feature_witnesses'][0]['evidence_paths'] = []
    assert not record_check(item, record, schema)['valid']


def test_unestablished_original_has_empty_inventories_and_unchanged_diagnostics(schema):
    row = index_row()
    record = preparation.build_record(row, measured(row, complete=False))
    item = judgment(record)
    before = deepcopy(record['reference_measurement'])
    assert record_check(item, record, schema)['valid']
    assert record['reference_measurement'] == before
    item['component_reviews'] = [dict(component_path=['features', 'C'], original_status='explicit', assessment='supported', notes='Invented salvage.')]
    assert not record_check(item, record, schema)['valid']


def full_fixture():
    """552 deliberately unavailable fictional positions, no original data."""
    ids = sorted([f'fictional_{i:03d}_template' for i in range(552)],
                 key=lambda i: (pipeline.digest((preparation.ORDER_SALT + i).encode()), i))
    dependencies = dict(review_protocol=preparation.descriptor(preparation.PROTOCOL, preparation.PROTOCOL_SHA),
        review_output_schema=preparation.descriptor(preparation.OUTPUT_SCHEMA, preparation.OUTPUT_SCHEMA_SHA),
        frozen_reference=preparation.descriptor(pipeline.REFERENCE, pipeline.REFERENCE_HASH))
    records = [dict(id=i, unit_id=pipeline.blind_id(i), text_sha256='f' * 64, reference_established=False,
        components=[], evidence=[], bindings=[], property_table=[],
        reference_measurement=dict(established=False, derivation={'valid': False, 'errors': ['Fictional unavailable first.']})) for i in ids]
    chunks = [dict(schema=preparation.VERSION + '_chunk', chunk_number=n, ordered_template_ids=ids[n*24:n*24+24],
        records=deepcopy(records[n*24:n*24+24]), **deepcopy(dependencies)) for n in range(23)]
    manifest = dict(schema=preparation.VERSION + '_manifest', order_salt=preparation.ORDER_SALT,
        ordered_template_ids=ids, reviewer_streams=['reviewer_0', 'reviewer_1'],
        template_opportunities=552, narrator_opportunities=552,
        narrator_availability=[dict(id=f'fictional_{i:03d}', established=False, actor_text_available=i < 500,
            derivation={'valid': False, 'errors': ['Fictional narrator availability diagnostic.']}) for i in range(552)],
        pair_opportunities=684, pairs=[dict(id=f'fictional_pair_{i}', endpoints=[i, i+1]) for i in range(684)],
        reference_lineage={'fictional': True}, **dependencies,
        chunks=[dict(path=f'chunk_{n:02d}.json', positions=24, ordered_template_ids=c['ordered_template_ids'],
            sha256=pipeline.digest(pipeline.encoded(c) + b'\n')) for n, c in enumerate(chunks)])
    manifest_sha, launch_sha = preparation.digest(manifest), 'a' * 64
    launched, outcomes, files = [], [], {}
    for stream in manifest['reviewer_streams']:
        for n, chunk in enumerate(chunks):
            context = f'fictional_{stream}_{n}'
            path = f'/fictional/{context}/exclusive_first.json'
            binding = dict(reviewer_id=stream, independent_context_id=context,
                manifest_sha256=manifest_sha, chunk_sha256=manifest['chunks'][n]['sha256'],
                protocol_sha256=preparation.PROTOCOL_SHA, output_schema_sha256=preparation.OUTPUT_SCHEMA_SHA)
            value = dict(schema='attention_relational_template_semantic_review_v1_first_chunk_grade',
                first_pass=True, independent_context=True, records=[judgment(r) for r in chunk['records']], limitations=['Wholly fictional.'], **binding)
            raw = pipeline.encoded(value) + b'\n'
            files[path] = raw
            launched.append(dict(reviewer_id=stream, independent_context_id=context, chunk_number=n,
                chunk_sha256=binding['chunk_sha256'], first_artifact_path=path,
                fresh_context=True, first_pass=True, peer_judgments_supplied=False))
            outcomes.append(dict(reviewer_id=stream, independent_context_id=context, chunk_number=n,
                chunk_sha256=binding['chunk_sha256'], first_artifact=dict(path=path, sha256=pipeline.digest(raw)), terminal_attestation=None))
    launch = dict(schema=collector.VERSION + '_launch', manifest_sha256=manifest_sha,
        protocol_sha256=preparation.PROTOCOL_SHA, output_schema_sha256=preparation.OUTPUT_SCHEMA_SHA, contexts=launched)
    outcome = dict(schema=collector.VERSION + '_outcomes', launch_receipt_sha256=launch_sha, manifest_sha256=manifest_sha, contexts=outcomes)
    return manifest, chunks, manifest_sha, launch, launch_sha, outcome, files


def read_files(files):
    def read(ref):
        if ref['path'] not in files:
            raise FileNotFoundError(ref['path'])
        raw = files[ref['path']]
        assert pipeline.digest(raw) == ref['sha256']
        return raw
    return read


def run_collection(fixture, schema):
    *arguments, files = fixture
    return collector.collect(*arguments, schema, read_bound=read_files(files))


def set_first(fixture, raw):
    outcome, files = fixture[-2:]
    ref = outcome['contexts'][0]['first_artifact']
    files[ref['path']] = raw
    ref['sha256'] = pipeline.digest(raw)


def terminal_for_first(fixture):
    entry = fixture[-2]['contexts'][0]
    raw = b'Fictional root tool observation: this exact fictional reviewer context terminated.\n'
    path = '/fictional/actual_terminal_observation.txt'
    fixture[-1][path] = raw
    entry['terminal_attestation'] = dict(independent_context_id=entry['independent_context_id'],
        chunk_sha256=entry['chunk_sha256'], observed_terminal=True, observation_kind='failed',
        observation_artifact=dict(path=path, sha256=pipeline.digest(raw)))


def test_complete552x2_retention_preserves_all_narrator_availability_pairs_and_firsts(schema):
    fixture = full_fixture()
    before = deepcopy(fixture)
    retained = run_collection(fixture, schema)
    assert fixture == before
    assert len(retained['chunks']) == 46 and len(retained['rows']) == 1104
    assert all(c['structurally_valid'] for c in retained['chunks'])
    assert all(r['review_state'] == 'unestablished' for r in retained['rows'])
    assert all(r['first_judgment']['notes'].startswith('unestablished') for r in retained['rows'])
    for stream in ('reviewer_0', 'reviewer_1'):
        assert [r['id'] for r in retained['rows'] if r['reviewer_id'] == stream] == fixture[0]['ordered_template_ids']
    assert retained['narrator_availability'] == fixture[0]['narrator_availability']
    assert retained['pairs'] == fixture[0]['pairs'] and len(retained['pairs']) == 684
    assert not retained['semantic_correctness_established_by_code']
    assert base64.b64decode(retained['chunks'][0]['first_bytes_base64']) == next(iter(fixture[-1].values()))


@pytest.mark.parametrize('failure', ['malformed', 'duplicate_json_key', 'incomplete', 'extra_field', 'wrong_id', 'wrong_chunk'])
def test_any_first_chunk_failure_retains_all24_unestablished_and_original_bytes(failure, schema):
    fixture = full_fixture()
    first = fixture[-2]['contexts'][0]['first_artifact']
    raw = fixture[-1][first['path']]
    if failure == 'malformed':
        raw = b'{fictional bad JSON\xff'
    elif failure == 'duplicate_json_key':
        raw = b'{"schema":"one","schema":"two"}'
    else:
        value = pipeline.decoded(raw)
        if failure == 'incomplete':
            value['records'].pop()
        elif failure == 'extra_field':
            value['records'][0]['new_semantic_key'] = True
        elif failure == 'wrong_id':
            value['records'][0]['id'] = value['records'][1]['id']
        else:
            value['chunk_sha256'] = '0' * 64
        raw = pipeline.encoded(value)
    set_first(fixture, raw)
    retained = run_collection(fixture, schema)
    chunk = retained['chunks'][0]
    assert not chunk['structurally_valid']
    assert base64.b64decode(chunk['first_bytes_base64']) == raw
    assert chunk['parse_errors'] or chunk['schema_errors'] or chunk['contract_errors']
    assert all(r['first_judgment'] is None and r['review_state'] == 'unestablished' for r in retained['rows'][:24])
    assert all(c['structurally_valid'] for c in retained['chunks'][1:])
    assert len(retained['rows']) == 1104


def test_missing_first_requires_actual_terminal_and_keeps_24_positions(schema):
    fixture = full_fixture()
    path = fixture[-2]['contexts'][0]['first_artifact']['path']
    del fixture[-1][path]
    with pytest.raises(ValueError, match='collection not ready'):
        run_collection(fixture, schema)
    terminal_for_first(fixture)
    retained = run_collection(fixture, schema)
    assert retained['chunks'][0]['first_bytes_base64'] is None
    assert retained['chunks'][0]['parse_errors'] == ['missing exclusive first artifact']
    assert len(retained['rows'][:24]) == 24 and all(r['first_judgment'] is None for r in retained['rows'][:24])
    assert retained['chunks'][0]['terminal_observation_bytes_base64']


def test_unsaved_nonterminal_descriptor_only_is_not_ready(schema):
    fixture = full_fixture()
    fixture[-2]['contexts'][0]['first_artifact'] = None
    with pytest.raises(ValueError, match='collection not ready'):
        run_collection(fixture, schema)
    terminal_for_first(fixture)
    terminal = fixture[-2]['contexts'][0]['terminal_attestation']
    del fixture[-1][terminal['observation_artifact']['path']]
    with pytest.raises(FileNotFoundError):
        run_collection(fixture, schema)


@pytest.mark.parametrize('change', ['reused_context', 'reused_path', 'missing_launch', 'missing_outcome', 'wrong_chunk', 'not_fresh', 'peer_supplied', 'false_terminal'])
def test_root_receipts_require46_unique_fresh_bound_contexts(change, schema):
    fixture = full_fixture()
    launch, outcomes = fixture[3], fixture[5]
    if change == 'reused_context':
        launch['contexts'][1]['independent_context_id'] = launch['contexts'][0]['independent_context_id']
    elif change == 'reused_path':
        launch['contexts'][1]['first_artifact_path'] = launch['contexts'][0]['first_artifact_path']
    elif change == 'missing_launch':
        launch['contexts'].pop()
    elif change == 'missing_outcome':
        outcomes['contexts'].pop()
    elif change == 'wrong_chunk':
        outcomes['contexts'][0]['chunk_sha256'] = '0' * 64
    elif change == 'not_fresh':
        launch['contexts'][0]['fresh_context'] = False
    elif change == 'peer_supplied':
        launch['contexts'][0]['peer_judgments_supplied'] = True
    else:
        terminal_for_first(fixture)
        outcomes['contexts'][0]['terminal_attestation']['observed_terminal'] = False
    with pytest.raises(ValueError):
        run_collection(fixture, schema)


def test_whole_chunk_property_failure_does_not_salvage_valid_record(schema):
    fixture = full_fixture()
    # Use a realistic established fictional first record in one chunk, then
    # bind a bad property. The other23 valid records must also be unestablished.
    record = established_record()
    old = fixture[1][0]['records'][0]
    for name in ('id', 'unit_id', 'text_sha256'):
        record[name] = old[name]
    fixture[1][0]['records'][0] = record
    chunk = fixture[1][0]
    sha = pipeline.digest(pipeline.encoded(chunk) + b'\n')
    fixture[0]['chunks'][0]['sha256'] = sha
    for context in fixture[3]['contexts'] + fixture[5]['contexts']:
        if context['chunk_number'] == 0:
            context['chunk_sha256'] = sha
    # Both first artifact bindings must reflect the changed fictional packet.
    for entry in fixture[5]['contexts']:
        if entry['chunk_number'] == 0:
            ref = entry['first_artifact']
            value = pipeline.decoded(fixture[-1][ref['path']])
            value['chunk_sha256'] = sha
            value['records'][0] = judgment(record)
            if entry['reviewer_id'] == 'reviewer_0':
                unit = next(u for u in record['property_table'] if u['kind'] == 'role_peaks')
                prop = property_ref(unit, [])
                prop['value'] = 'wrong exact value'
                value['records'][0]['evidence_reviews'][0]['properties'] = [prop]
            raw = pipeline.encoded(value)
            fixture[-1][ref['path']] = raw
            ref['sha256'] = pipeline.digest(raw)
    retained = run_collection(fixture, schema)
    assert not retained['chunks'][0]['structurally_valid']
    assert all(r['first_judgment'] is None for r in retained['rows'][:24])
    assert retained['chunks'][23]['structurally_valid']
    other = retained['rows'][552]
    assert other['first_judgment']['component_reviews'][0]['assessment'] == 'unsupported'
    assert other['first_judgment']['evidence_reviews'][0]['grounding'] == 'unresolved'
    assert other['review_state'] == 'retained_structurally_valid_judgment'
    assert not other['semantic_correctness_established_by_code']


def test_cli_checks_production_preparation_before_any_first_judgment_load(tmp_path, monkeypatch):
    events = []
    def stop(path, sha):
        events.append(('actual_terminal_and_preparation', path, sha))
        raise ValueError('fictional still-live original session')
    def forbidden(*args, **kwargs):
        pytest.fail('CLI loaded first judgments before original actual terminal/preparation verification')
    monkeypatch.setattr(collector, 'load_prepared_actual', stop)
    monkeypatch.setattr(collector, 'frozen_output_schema', forbidden)
    monkeypatch.setattr(pipeline, 'bound_json', forbidden)
    args = ['--prepared-manifest', '/fictional/prepared_manifest.json', '--prepared-manifest-sha256', 'a'*64,
        '--launch-receipt', '/fictional/launch.json', '--launch-receipt-sha256', 'b'*64,
        '--outcome-receipt', '/fictional/outcomes.json', '--outcome-receipt-sha256', 'c'*64,
        '--output', str(tmp_path / 'first_collection.json')]
    with pytest.raises(ValueError, match='still-live'):
        collector.main(args)
    assert len(events) == 1 and not (tmp_path / 'first_collection.json').exists()
