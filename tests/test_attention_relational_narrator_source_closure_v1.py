"""Invented inputs only: inventory/terminal/schema contracts, not meaning accuracy."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import attention_relational_narrator_source_closure_v1 as closure
import attention_relational_reporting_pipeline_v1 as pipeline
from test_attention_relational_template_review_v1 import full_fictional_inputs
from test_attention_relational_template_index_v1 import packet_fixture


def annotation(unit, status='explicit', bindings=True):
    entry = dict(quote=unit['text'], referent='invented', attribute='location', scope='model_organization', time='current')
    bound = dict(referent='invented', attribute='location', scope='model_organization', time='current', evidence=[deepcopy(entry), deepcopy(entry)])
    raw = dict(schema=closure.template.quotes.QUOTE_SCHEMA, unit_id=unit['unit_id'],
        features={n: dict(status='explicit', evidence=[deepcopy(entry), deepcopy(entry)], rationale='Fictional.') for n in ('C', 'O', 'D', 'A')},
        auxiliary={n: dict(status='absent', evidence=[]) for n in pipeline.AUXILIARY},
        composites={n: dict(status=status, bindings=[deepcopy(bound), deepcopy(bound)] if bindings else [], rationale='Fictional.') for n in ('COD', 'CA')},
        limitations=['Invented input; no semantic accuracy claim.'])
    result = closure.template.quotes.derive_annotation(unit['text'], unit['unit_id'], raw, expected_text_sha256=unit['text_sha256'])
    assert result['valid']
    canonical = result.pop('canonical_annotation')
    return dict({k: deepcopy(v) for k, v in unit.items() if k != 'text'}, established=True,
                canonical_annotation=canonical, derivation=result, transport_complete=True)


@pytest.fixture(scope='module')
def corpus():
    index, prepared, derived = full_fictional_inputs()
    instruction = pipeline.frozen_instruction('narrator')
    exports = dict(requests=[], packets=[], provenance=[], templates=[])
    for row in index['templates']:
        case, identifier = row['case'], row['source_id']
        packet, provenance = packet_fixture(source=case['source'], joined=case['view'] == 'joined', full=True)
        provenance.update(id=identifier, case=deepcopy(case))
        exports['requests'].append(dict(id=identifier, input=instruction + '\n' + pipeline.encoded(packet).decode('ascii')))
        exports['packets'].append(dict(id=identifier, content=packet, content_sha256=closure.digest(packet)))
        exports['provenance'].append(provenance)
        exports['templates'].append(dict(id=identifier, report=closure.template.construction.exporter.deterministic_template(packet)))
    units = pipeline._index(prepared['units'])
    positions = [i for i, r in enumerate(derived['rows']) if not r['template']]
    for n, pos in enumerate(positions[:10]):
        unit = units[derived['rows'][pos]['id']]
        # All three original statuses, including absent records with bindings.
        derived['rows'][pos] = annotation(unit, status='absent' if n == 9 else ('explicit', 'ambiguous', 'absent')[n % 3], bindings=n != 9)
    facts = dict(all_text_results=[{k: deepcopy(v) for k, v in row.items()
        if k in ('id', 'unit_id', 'case', 'template', 'text_sha256', 'actor_text_available')} | dict(fidelity_established=False)
        for row in derived['rows']])
    return index, prepared, derived, exports, facts


def test_complete_inventory_all_statuses_duplicates_and_two_stream_assignments(corpus):
    manifest, chunks = closure.assemble(*corpus)
    assert len(manifest['narrator_availability']) == 552 and len(manifest['pairs']) == 684
    assert sum(r['actor_text_available'] for r in manifest['narrator_availability']) == 537
    assert manifest['candidate_narrator_positions'] == 9 and manifest['candidate_binding_positions'] == 36
    assert [len(c['records']) for c in chunks] == [4, 4, 1]
    assert len(manifest['assignments']) == 6
    assert {b['original_status'] for b in manifest['candidate_binding_inventory']} == {'explicit', 'ambiguous', 'absent'}
    ordered = sorted(manifest['ordered_narrator_ids'], key=lambda i: (pipeline.digest((closure.ORDER_SALT + i).encode()), i))
    assert manifest['ordered_narrator_ids'] == ordered
    assert [r['id'] for c in chunks for r in c['records']] == manifest['ordered_candidate_narrator_ids']
    for stream in closure.STREAMS:
        assert [a['chunk_number'] for a in manifest['assignments'] if a['reviewer_id'] == stream] == [0, 1, 2]
    record = chunks[0]['records'][0]
    assert len(record['candidate_bindings']) == 4
    assert len(record['evidence']) == 16  # 8 feature entries plus 8 binding entries.
    assert len({str(e['evidence_path']) for e in record['evidence']}) == 16
    assert record['evidence'][0]['recorded_evidence'] == record['evidence'][1]['recorded_evidence']
    assert record['evidence'][0]['evidence_path'] != record['evidence'][1]['evidence_path']
    assert all(r['disposition'] == 'awaiting_property_source_review' for r in manifest['pairs'])
    assert not manifest['source_meaning_established'] and not manifest['fidelity_conjoined_witness_established']
    assert manifest['first_artifact_policy']['gap_filling_allowed'] is False
    for ref, chunk in zip(manifest['chunks'], chunks):
        assert ref['sha256'] == pipeline.digest(pipeline.encoded(chunk) + b'\n')


def test_unavailable_unestablished_and_no_original_binding_are_explicit(corpus):
    manifest, _ = closure.assemble(*corpus)
    states = [r['inventory_disposition'] for r in manifest['narrator_availability']]
    assert states.count('unavailable_narrator') == 15
    assert states.count('candidate') == 9 and states.count('no_original_binding') == 1
    assert states.count('unestablished_reference') == 527
    for record in manifest['narrator_availability']:
        if not record['reference_established']:
            assert record['canonical_first_annotation'] is None
            assert record['components'] == record['evidence'] == record['candidate_bindings'] == []


def test_precise_nested_origins_maps_and_exporter_summaries_have_no_semantic_upgrade(corpus):
    index, prepared, derived, exports, _ = corpus
    unit = next(u for u in prepared['units'] if not u['template'] and u['case']['source'] == 'A')
    measured = pipeline._index(derived['rows'])[unit['id']]
    inv = pipeline.validate_export(exports)
    idx = next(r['index'] for r in index['templates'] if r['source_id'] == unit['id'])
    record = closure.build_record(unit, measured, inv['packets'][unit['id']]['content'], inv['provenance'][unit['id']], idx)
    addresses = record['source_addresses']
    assert any(a['representation'] == 'exporter_derived_summary' for a in addresses)
    assert any('distributions' in a['packet_path'] for a in addresses)
    assert any('origin' in a['packet_path'] and 'represented_time' in a['packet_path'] for a in addresses)
    assert all(a['semantic_ownership'] == 'unresolved' for a in addresses)
    classes = {p['source_resolution']['origin_class']['category'] for p in record['declared_source_property_index']}
    assert 'copied_W11' in classes and 'source_current' in classes and 'source_conditional' in classes
    assert not record['semantic_review_performed'] and not record['narrator_fidelity_established']


def test_exporter_peak_relation_outside_derived_is_tagged_without_upgrading_raw_maps():
    packet, _ = packet_fixture(joined=False)
    node = packet['panels'][0]['current'][0]
    node['red_blue_peak_relation'] = closure.template.construction.exporter.peak_relation(
        node['derived']['red']['peaks'], node['derived']['blue']['peaks'])
    addresses = closure.source_addresses(packet)
    relation = [a for a in addresses if 'red_blue_peak_relation' in a['packet_path']]
    raw = [a for a in addresses if 'distributions' in a['packet_path']]
    assert relation and all(a['representation'] == 'exporter_derived_summary' for a in relation)
    assert raw and all(a['representation'] == 'supplied_packet_field' for a in raw)
    assert all(a['semantic_ownership'] == 'unresolved' for a in addresses)


@pytest.mark.parametrize('change', ['case', 'text', 'annotation', 'source'])
def test_exact_record_identity_and_no_label_salvage(corpus, change):
    index, prepared, derived, exports, _ = corpus
    unit = deepcopy(next(u for u in prepared['units'] if not u['template']))
    measured = deepcopy(pipeline._index(derived['rows'])[unit['id']])
    inv = pipeline.validate_export(exports)
    packet, provenance = deepcopy(inv['packets'][unit['id']]['content']), deepcopy(inv['provenance'][unit['id']])
    idx = next(r['index'] for r in index['templates'] if r['source_id'] == unit['id'])
    if change == 'case':
        measured['case'] = {}
    elif change == 'text':
        unit['text'] += 'Changed.'
    elif change == 'annotation':
        measured['established'] = False
    else:
        packet['previous_node'] = 'wrong'
    with pytest.raises(ValueError):
        closure.build_record(unit, measured, packet, provenance, idx)


@pytest.mark.parametrize('change', ['missing_row', 'fidelity_upgrade', 'wrong_fact_identity', 'missing_pair', 'missing_source'])
def test_whole_denominators_and_preselected_fidelity_are_not_relaxed(corpus, change):
    index, prepared, derived, exports, facts = deepcopy(corpus)
    if change == 'missing_row':
        facts['all_text_results'].pop()
    elif change == 'fidelity_upgrade':
        next(r for r in facts['all_text_results'] if not r['template'])['fidelity_established'] = True
    elif change == 'wrong_fact_identity':
        facts['all_text_results'][0]['text_sha256'] = '0' * 64
    elif change == 'missing_pair':
        index['pairs'].pop()
    else:
        exports['packets'].pop()
    with pytest.raises(ValueError):
        closure.assemble(index, prepared, derived, exports, facts)


def test_schema_closed_separate_meaning_ownership_calculation_and_property_identity():
    schema = closure.output_schema()
    def walk(node):
        if isinstance(node, dict):
            if node.get('type') == 'object':
                assert node['additionalProperties'] is False and set(node['required']) == set(node['properties'])
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)
    walk(schema)
    record = schema['properties']['records']
    assert record['minItems'] == 1 and record['maxItems'] == 4
    binding = record['items']['properties']['candidate_binding_reviews']['items']['properties']
    assert {'recorded_feature_grounding', 'source_meaning', 'source_ownership', 'operands', 'calculations', 'tracking_support'} <= set(binding)
    assert 'independent_recomputation_performed' in binding['calculations']['items']['properties']
    assert pipeline.bound_json(pipeline.ROOT / closure.OUTPUT_SCHEMA, closure.schema_sha()) == schema


def loader_args():
    return SimpleNamespace(manifest_sha256=closure.template.MANIFEST_SHA, manifest=Path(closure.template.MANIFEST),
        root_terminal_attestation=Path('fictional_root_terminal.json'), root_terminal_attestation_sha256='a' * 64,
        execution_receipt_sha256='e' * 64)


@pytest.mark.parametrize('error', ['not_terminal', 'wrong_session', 'incomplete47', 'different_execution', 'different_manifest'])
def test_root_terminal_and_all47_attestation_precede_any_label_loader(monkeypatch, error):
    args = loader_args()
    sources = {f'fictional_{n}': str(n) for n in range(47)}
    receipt = dict(schema=closure.VERSION + '_root_terminal', terminal_observed=True, tool_session_id=5286,
        manifest_sha256=closure.template.MANIFEST_SHA, execution_receipt_sha256='e' * 64, runtime_source_sha256=sources)
    if error == 'not_terminal':
        receipt['terminal_observed'] = False
    elif error == 'wrong_session':
        receipt['tool_session_id'] = 9000
    elif error == 'incomplete47':
        sources.pop('fictional_46')
    elif error == 'different_execution':
        receipt['execution_receipt_sha256'] = 'f' * 64
    else:
        receipt['manifest_sha256'] = '0' * 64
    label_loader = Mock(side_effect=AssertionError('must not load labels'))
    monkeypatch.setattr(closure.template, 'load_actual', label_loader)
    monkeypatch.setattr(pipeline, 'bound_json', lambda p, sha: dict(source_sha256=sources) if p == args.manifest else receipt)
    with pytest.raises(ValueError, match='terminal observation'):
        closure.load_actual(args)
    label_loader.assert_not_called()


def test_runtime_source_mismatch_stops_before_reference_output(monkeypatch, tmp_path):
    args = loader_args()
    paths = []
    for n in range(47):
        path = tmp_path / f'p{n}'
        path.write_text('fictional')
        paths.append(path)
    sources = {str(p): pipeline.digest(p.read_bytes()) for p in paths}
    sources[str(paths[-1])] = '0' * 64
    receipt = dict(schema=closure.VERSION + '_root_terminal', terminal_observed=True, tool_session_id=5286,
        manifest_sha256=closure.template.MANIFEST_SHA, execution_receipt_sha256='e' * 64, runtime_source_sha256=sources)
    label_loader = Mock(side_effect=AssertionError('must not load labels'))
    monkeypatch.setattr(closure.template, 'load_actual', label_loader)
    monkeypatch.setattr(pipeline, 'bound_json', lambda p, sha: dict(source_sha256=sources) if p == args.manifest else receipt)
    with pytest.raises(ValueError, match='all47 frozen runtime'):
        closure.load_actual(args)
    label_loader.assert_not_called()


def test_successful_terminal_checks_delegate_then_recompute_all_inputs(monkeypatch, tmp_path):
    args = loader_args()
    args.preparation, args.preparation_sha256 = Path('fictional_preparation.json'), 'p' * 64
    args.factual_summary, args.factual_summary_sha256 = Path('fictional_factual.json'), closure.FACTUAL_SHA
    sources = {}
    for n in range(47):
        p = tmp_path / str(n)
        p.write_text('fictional')
        sources[str(p)] = pipeline.digest(b'fictional')
    receipt = dict(schema=closure.VERSION + '_root_terminal', terminal_observed=True, tool_session_id=5286,
        manifest_sha256=closure.template.MANIFEST_SHA, execution_receipt_sha256='e' * 64, runtime_source_sha256=sources)
    prepared = dict(lineage=dict(inputs=dict(exports=dict(path='fictional_exports', sha256='x' * 64))))
    calls = []
    real_bound = pipeline.bound_json
    def bound(path, sha):
        if path == args.manifest:
            return dict(source_sha256=sources)
        if path == args.root_terminal_attestation:
            return receipt
        if path == pipeline.ROOT / closure.OUTPUT_SCHEMA:
            return real_bound(path, sha)
        calls.append(str(path))
        return {'fictional': str(path)}
    monkeypatch.setattr(pipeline, 'bound_json', bound)
    monkeypatch.setattr(closure.template, 'load_actual', lambda a: (calls.append('terminal_loader') or ('index', prepared, 'derived', 'terminal')))
    monkeypatch.setattr(closure.measurement, 'load_preparation', lambda *a: (calls.append('recompute_preparation') or prepared))
    monkeypatch.setattr(closure.measurement, 'verify_factual_summary', lambda *a: (calls.append('recompute_whole_factual') or 'factual'))
    result = closure.load_actual(args)
    assert calls == ['terminal_loader', 'recompute_preparation', 'fictional_exports', 'fictional_factual.json', 'recompute_whole_factual']
    assert result == ('index', prepared, 'derived', {'fictional': 'fictional_exports'}, 'factual', 'terminal')
