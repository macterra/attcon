"""Fictional preparation/geometry/terminal contracts, not semantic review results."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import attention_relational_template_review_v1 as review
from test_attention_relational_template_index_v1 import packet_fixture
import attention_relational_report_v1 as exporter
import attention_relational_reporting_pipeline_v1 as pipeline


def index_row():
    packet, provenance = packet_fixture()
    identifier = 'fictional_template'
    provenance['id'] = 'fictional'
    item = review.construction.index_template(packet, exporter.deterministic_template(packet), provenance)
    return dict(id=identifier, source_id='fictional', unit_id=pipeline.blind_id(identifier), case={'fictional': True},
                panel_provenance=provenance['panels'], index=item)


def raw_annotation(row):
    # Labels are invented software inputs. Passing the frozen structural
    # evaluator does not establish their semantic correctness.
    entries = [dict(quote=q, referent='fictional_pattern', attribute='location', scope='model_organization', time='current')
               for q in (review.construction.HEADER, row['index']['text'], review.construction.HEADER)]
    binding = dict(referent='fictional_pattern', attribute='location', scope='model_organization', time='current', evidence=deepcopy(entries))
    return dict(schema=review.quotes.QUOTE_SCHEMA, unit_id=row['unit_id'],
        features={n: dict(status='explicit', evidence=deepcopy(entries), rationale='Fictional fixture only.') for n in ('C', 'O', 'D', 'A')},
        auxiliary={n: dict(status='absent', evidence=deepcopy(entries)) for n in pipeline.AUXILIARY},
        composites={n: dict(status='explicit', bindings=[deepcopy(binding), deepcopy(binding)], rationale='Fictional.') for n in ('COD', 'CA')},
        limitations=['Not a semantic key or target annotation.'])


_DEFAULT_RAW = object()


def measured(row, complete=True, raw=_DEFAULT_RAW):
    raw = raw_annotation(row) if raw is _DEFAULT_RAW else raw
    grounded = review.quotes.derive_annotation(row['index']['text'], row['unit_id'], raw,
                                              expected_text_sha256=row['index']['text_sha256'])
    annotation = grounded.pop('canonical_annotation')
    established = complete and grounded['valid']
    return dict(id=row['id'], unit_id=row['unit_id'], case=deepcopy(row['case']), template=True,
        text_sha256=row['index']['text_sha256'], actor_text_available=True, transport_complete=complete,
        request_sha256='a' * 64, raw_response_content_sha256='b' * 64, raw_output_text_sha256='c' * 64,
        storage_provenance={'fictional': True}, derivation=grounded, established=established,
        canonical_annotation=annotation if established else None)


def test_all_components_bindings_duplicate_paths_and_unit_references_are_retained_once():
    row = index_row()
    measurement = measured(row)
    assert measurement['established']
    original = deepcopy((row, measurement))
    record = review.build_record(row, measurement)
    assert (row, measurement) == original
    assert len(record['components']) == 12 and len(record['bindings']) == 4
    assert len(record['evidence']) == 42
    assert len({e['evidence_id'] for e in record['evidence']}) == 42
    c = [e for e in record['evidence'] if e['path'][:2] == ['features', 'C']]
    assert c[0]['recorded_evidence'] == c[2]['recorded_evidence']
    assert c[0]['path'] != c[2]['path'] and c[0]['evidence_id'] != c[2]['evidence_id']
    assert all('unit' not in g and set(g) >= {'unit_id', 'overlap_start', 'overlap_end'}
               for e in record['evidence'] for g in e['geometry']['intersections'])
    assert all(not b['compatibility_established'] for b in record['bindings'])
    assert all(len(b['candidate_feature_evidence_ids'][f]) == 3 for b in record['bindings'] for f in b['required_features'])
    assert not record['semantic_attribution_established']


def test_header_mixed_geometry_and_nested_origin_fields_do_not_establish_meaning():
    row = index_row()
    record = review.build_record(row, measured(row))
    first, mixed = record['evidence'][:2]
    table = {u['unit_id']: u for u in record['property_table']}
    assert [table[g['unit_id']]['kind'] for g in first['geometry']['intersections']] == ['header']
    assert table[first['geometry']['intersections'][0]['unit_id']]['source_resolution'] is None
    sources = {table[g['unit_id']]['source_resolution']['origin_class']['origin_source_key']
               for g in mixed['geometry']['intersections'] if table[g['unit_id']]['source_resolution'] is not None}
    assert sources == {'A', 'W'}
    inherited = next(u for u in record['property_table'] if u['kind'] == 'origin' and u['value']['kind'] == 'inherited_current')
    assert inherited['value']['inherited_origin'] == inherited['source_resolution']['origin_chain'][1]
    assert all(e['geometry']['semantic_attribution'] == 'unresolved_geometry_only' for e in record['evidence'])


@pytest.mark.parametrize('failure', ['missing', 'incomplete', 'invalid', 'repeated_quote'])
def test_unestablished_reference_preserves_diagnostics_without_any_semantic_salvage(failure):
    row = index_row()
    raw = raw_annotation(row)
    if failure == 'missing':
        raw = None
    elif failure == 'invalid':
        raw['unit_id'] = 'wrong'
    elif failure == 'repeated_quote':
        raw['features']['C']['evidence'][0]['quote'] = 'p0/'
    measurement = measured(row, complete=failure != 'incomplete', raw=raw)
    record = review.build_record(row, measurement)
    assert record['review_state'] == 'unestablished' and not record['reference_established']
    assert record['canonical_first_annotation'] is None
    assert record['components'] == record['evidence'] == record['bindings'] == []
    assert record['reference_measurement']['derivation'] == measurement['derivation']
    assert record['property_table']
    if failure == 'repeated_quote':
        assert any(d.get('occurrence_count', 0) > 1 for d in measurement['derivation']['quote_diagnostics'])


def test_wrong_text_or_identity_fails_instead_of_repairing_offsets_or_binding():
    row = index_row()
    measurement = measured(row)
    for key, value in [('text_sha256', '0' * 64), ('unit_id', 'wrong'), ('case', {'different': True})]:
        with pytest.raises(ValueError, match='ID/text/case'):
            review.build_record(row, dict(measurement, **{key: value}))


def test_terminal_metadata_contract_rejects_fake_completion_release_log_and_interrupt_claims():
    command = ['.venv/bin/python', 'scripts/run_attention_relational_reports_v1.py', '--manifest', review.MANIFEST,
               '--commit', review.RELEASE, '--output-root', review.measurement.RATE_ROOT]
    execution = dict(command=command, preservation_commit=review.RELEASE, stage='rate', retry_performed=False,
                     exit_code=0, log_sha256='a' * 64)
    assert review.verify_terminal(execution, 'a' * 64, True) == 'normal'
    for changed, log, exists in [(dict(execution, exit_code=None), 'a' * 64, True),
                                 (dict(execution, preservation_commit='0' * 40), 'a' * 64, True),
                                 (execution, 'b' * 64, True), (execution, 'a' * 64, False),
                                 (dict(execution, retry_performed=True), 'a' * 64, True)]:
        with pytest.raises(ValueError):
            review.verify_terminal(changed, log, exists)
    stopped = dict(execution, exit_code=-2)
    terminal = dict(schema='attention_relational_quote_v3_terminal_receipt', terminal_verified=True, stage='rate',
        tool_session_id=123, manifest_sha256=review.MANIFEST_SHA, output_root=review.measurement.RATE_ROOT,
        execution_receipt_sha256='e' * 64, child_exit_code=-2)
    assert review.verify_terminal(stopped, 'a' * 64, False, terminal, 'e' * 64) == 'interrupted'
    for wrong in (None, dict(terminal, execution_receipt_sha256='f' * 64), dict(terminal, terminal_verified=False)):
        with pytest.raises(ValueError):
            review.verify_terminal(stopped, 'a' * 64, False, wrong, 'e' * 64)
    with pytest.raises(ValueError):
        review.verify_terminal(stopped, 'a' * 64, True, terminal, 'e' * 64)
    # Pure fictional terminal metadata never substitutes for the production
    # loader's pinned actual first index/manifest and preserved runtime files.
    with pytest.raises(ValueError, match='pinned first index'):
        review.load_actual(SimpleNamespace(index_sha256='0' * 64, manifest_sha256=review.MANIFEST_SHA,
                                           manifest=Path(review.MANIFEST)))


def full_fictional_inputs():
    exports = dict(requests=[], packets=[], provenance=[], templates=[])
    instruction = pipeline.frozen_instruction('narrator')
    for seed in (1301, 1311, 1321):
        for world in range(4):
            arms = [(s, v, a) for s in exporter.SOURCE_KEYS for v in ('A_only', 'joined') for a in exporter.VARIANTS]
            arms += [('W', 'W_only', a) for a in exporter.W_VARIANTS] + [('sensor', 'sensor_only', 'baseline')]
            for source, view, variant in arms:
                packet, provenance = packet_fixture(source=source, joined=view == 'joined', full=True)
                identifier = f's{seed}_w{world:02d}_t12_{source}_{view}_{variant}'
                provenance.update(id=identifier, case=dict(encoder_seed=seed, world_index=world, decision_time=12,
                    source=source, view=view, variant=variant))
                exports['requests'].append(dict(id=identifier, input=instruction + '\n' + pipeline.encoded(packet).decode('ascii')))
                exports['packets'].append(dict(id=identifier, content=packet, content_sha256=review.digest(packet)))
                exports['provenance'].append(provenance)
                exports['templates'].append(dict(id=identifier, report=exporter.deterministic_template(packet)))
    index = review.construction.build_index(exports)
    units, rows = [], []
    for number, row in enumerate(index['templates']):
        for template in (False, True):
            identifier = row['id'] if template else row['source_id']
            text = row['index']['text'] if template else ('Fictional narrator.' if number < 537 else '')
            unit = dict(id=identifier, unit_id=pipeline.blind_id(identifier), text=text, case=deepcopy(row['case']),
                template=template, text_sha256=pipeline.digest(text.encode('utf-8')), actor_text_available=template or bool(text.strip()))
            units.append(unit)
            rows.append({k: deepcopy(v) for k, v in unit.items() if k != 'text'} | dict(transport_complete=False,
                established=False, canonical_annotation=None, derivation={'valid': False, 'errors': ['Fictional missing first outcome']}))
    prepared = review.measurement.prepare_units(units, {'fictional_reference': True}, 'Fictional instructions.',
        {'fictional_selection': True}, {'fictional_inputs': True}, {'fictional_source': 'd' * 64})
    derived = dict(schema=review.measurement.VERSION + '_derived_collection', preparation_content_sha256=review.digest(prepared),
                   raw_collection_content_sha256='e' * 64, lineage=deepcopy(prepared['lineage']), rows=rows)
    return index, prepared, derived


def test_full23x24_order_includes_all_invalid_positions_and_all_narrator_availability():
    index, prepared, derived = full_fictional_inputs()
    manifest, chunks = review.assemble(index, prepared, derived)
    assert len(chunks) == 23 and all(len(c['records']) == 24 for c in chunks)
    expected = sorted([r['id'] for r in index['templates']], key=lambda i: (pipeline.digest((review.ORDER_SALT + i).encode()), i))
    assert manifest['ordered_template_ids'] == [r['id'] for c in chunks for r in c['records']] == expected
    assert len(manifest['narrator_availability']) == 552
    assert sum(r['actor_text_available'] for r in manifest['narrator_availability']) == 537
    assert len(manifest['pairs']) == 684 and not manifest['semantic_review_performed']
    assert all(r['review_state'] == 'unestablished' and not r['components'] and not r['evidence'] and not r['bindings']
               for c in chunks for r in c['records'])
    assert all(ref['sha256'] == pipeline.digest(pipeline.encoded(chunk) + b'\n') for ref, chunk in zip(manifest['chunks'], chunks))
    assert manifest['review_output_schema']['sha256'] == review.OUTPUT_SCHEMA_SHA
    origin = next(u for u in index['templates'][0]['index']['units'] if u['kind'] == 'origin')
    chain = origin['source_resolution']['origin_chain']
    origin['source_resolution']['origin_chain'] = []
    with pytest.raises(ValueError, match='source property/origin metadata'):
        review.validate_index(index)
    origin['source_resolution']['origin_chain'] = chain
    derived['rows'].pop()
    with pytest.raises(ValueError, match='full1104'):
        review.assemble(index, prepared, derived)
