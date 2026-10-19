"""Invented construction fixtures only; no target/reference outputs or API use."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import attention_relational_template_index_v1 as indexer
import attention_relational_report_v1 as exporter
import attention_relational_reporting_pipeline_v1 as pipeline


def node(name, origin):
    red = [[0.0] * 4 for _ in range(4)]
    blue = deepcopy(red)
    red[0][0] = red[2][3] = 0.5
    blue[3][3] = 1.0
    return dict(node=name, origin=deepcopy(origin), distributions=dict(red=red, blue=blue),
        derived={r: exporter.localization_summary(grid) for r, grid in (('red', red), ('blue', blue))})


def packet_fixture(source='A', joined=True, full=False):
    count = 4 if full else 2

    def panel(name, key):
        current = [node(f'n{i}', exporter._origin(key, i, 'current', 0, 1)) for i in range(count)]
        candidates = None if key == 'sensor' else {command: [
            node(f'n{i}', exporter._origin(key, i, 'candidate', 0, 1)) for i in range(count)] for command in exporter.COMMANDS}
        return dict(panel=name, current=current, candidates=candidates)

    panels = [panel('p0', source)]
    sources = [dict(panel='p0', source_key=source)]
    if joined:
        panels.append(panel('p1', 'W'))
        sources.append(dict(panel='p1', source_key='W'))
    packet = dict(schema=exporter.PACK_SCHEMA, decision_time=12, previous_node='n0', candidate_node='n1',
        queried_role='red', coordinate_system=dict(rows=4, columns=4, row_direction='down', column_direction='right',
            separate_local_grids=True), candidate_semantics='conditional supplied record or estimate, not an executed branch', panels=panels)
    return packet, dict(id='fictional_fixture', case={'fictional': True}, panels=sources)


def indexed(packet, provenance):
    return indexer.index_template(packet, exporter.deterministic_template(packet), provenance)


def property_at(unit, packet):
    value = packet
    for component in unit['packet_path']:
        value = value[component]
    return value


def test_independent_ties_exact_renderer_complete_partition_and_packet_paths():
    packet, provenance = packet_fixture()
    original = deepcopy((packet, provenance))
    result = indexed(packet, provenance)
    assert (packet, provenance) == original
    assert result['text'] == exporter.deterministic_template(packet)['text']
    assert result['text_sha256'] == pipeline.digest(result['text'].encode('utf-8'))
    assert result['exact_text_verified'] and result['exact_propositions_verified']
    assert result['independent_map_maxima_verified'] and result['complete_character_partition_verified']
    assert not result['semantic_source_attribution_established']
    units = result['units']
    assert ''.join(u['text'] for u in units) == result['text']
    assert sum(u['end'] - u['start'] for u in units) == len(result['text'])
    for number, unit in enumerate(units):
        assert unit['start'] == (units[number - 1]['end'] if number else 0)
        assert result['text'][unit['start']:unit['end']] == unit['text']
        if unit['kind'] in ('origin', 'role_peaks'):
            assert property_at(unit, packet) == unit['value']
            assert unit['source_resolution']['state'] == 'resolved'
    red = [u for u in units if u['kind'] == 'role_peaks' and u['context']['role'] == 'red']
    assert all(u['value'] == [[0, 0], [2, 3]] for u in red)
    assert all(u['text'].endswith('={(0,0),(2,3)}') for u in red)
    assert all(u['text'] == ' ' and u['source_resolution'] is None for u in units if u['kind'] == 'separator')
    header = units[0]
    assert header['kind'] == 'header' and header['source_resolution'] is None and header['packet_path'] is None
    assert header['global_packet_metadata'][0]['value'] == packet['coordinate_system']
    assert [v['candidate_commands'] for v in header['candidate_view_availability']] == [list(exporter.COMMANDS)] * 2


@pytest.mark.parametrize('change', ['upward', 'missing_axis', 'missing_coordinates', 'wrong_shape', 'nonlocal',
                                   'missing_conditional', 'wrong_conditional'])
def test_header_requires_complete_frozen_coordinates_and_conditional_declarations(change):
    packet, provenance = packet_fixture()
    if change == 'upward':
        packet['coordinate_system']['row_direction'] = 'up'
    elif change == 'missing_axis':
        del packet['coordinate_system']['column_direction']
    elif change == 'missing_coordinates':
        del packet['coordinate_system']
    elif change == 'wrong_shape':
        packet['coordinate_system']['rows'] = 5
    elif change == 'nonlocal':
        packet['coordinate_system']['separate_local_grids'] = False
    elif change == 'missing_conditional':
        del packet['candidate_semantics']
    else:
        packet['candidate_semantics'] = 'An executed future branch.'
    # The frozen renderer still emits the same header, so matching its text
    # alone must not establish the global declarations checked by this index.
    report = exporter.deterministic_template(packet)
    with pytest.raises(ValueError, match='header .* declaration differs'):
        indexer.index_template(packet, report, provenance)


def test_sensor_header_retains_no_candidate_view_without_fabricating_a_forecast():
    packet, provenance = packet_fixture(source='sensor', joined=False)
    result = indexed(packet, provenance)
    header = result['units'][0]
    assert header['source_resolution'] is None
    assert header['candidate_view_availability'] == [dict(panel='p0', packet_path=['panels', 0, 'candidates'], candidate_commands=None)]
    assert packet['panels'][0]['candidates'] is None
    assert all(u['context']['view'] == 'current' for u in result['units'] if u['kind'] in ('origin', 'role_peaks'))
    assert not result['semantic_source_attribution_established']


def test_actual_hybrid_sources_and_recursive_origins_preserve_every_time_field():
    packet, provenance = packet_fixture(joined=False)
    candidate = packet['panels'][0]['candidates']['k0'][0]
    history = dict(kind='history_estimate', represented_time=12, feedback_through=11)
    candidate['origin'] = dict(kind='inherited_current', represented_time=14, inherited_origin=dict(
        kind='inherited_current', represented_time=13, inherited_origin=history))
    result = indexed(packet, provenance)
    source_units = [u for u in result['units'] if u['source_resolution'] is not None]
    classes = [u['source_resolution']['origin_class'] for u in source_units]
    assert any(c['origin_source_key'] == 'W' and c['category'] == 'copied_W11' for c in classes)
    assert any(c['origin_source_key'] == 'A' and c['category'] == 'source_current' for c in classes)
    assert any(c['origin_source_key'] == 'A' and c['category'] == 'source_conditional' for c in classes)
    inherited = next(u for u in source_units if u['context'] == dict(panel='p0', view='candidate', command='k0', node='n0', role='red'))
    chain = inherited['source_resolution']['origin_chain']
    assert [o['represented_time'] for o in chain] == [14, 13, 12]
    assert chain[-1]['feedback_through'] == 11
    assert inherited['source_resolution']['origin_class']['category'] == 'inherited_source_current'


@pytest.mark.parametrize('source', ['ordinary', 'analytic', 'W', 'sensor'])
def test_alternative_sources_resolve_from_provenance_not_panel_name(source):
    packet, provenance = packet_fixture(source=source, joined=False)
    result = indexed(packet, provenance)
    resolved = [u['source_resolution'] for u in result['units'] if u['source_resolution'] is not None]
    assert all(r['panel_source_key'] == source for r in resolved)
    assert any(r['origin_class']['origin_source_key'] == source for r in resolved)
    if source in ('ordinary', 'analytic'):
        assert any(r['origin_class']['category'] == 'copied_W11' for r in resolved)


def test_unicode_origin_escapes_and_general_partition_offsets_are_exact_codepoints():
    packet, provenance = packet_fixture(joined=False)
    packet['panels'][0]['current'][0]['origin']['fictional_note'] = 'Réd🙂'
    result = indexed(packet, provenance)
    origin = next(u for u in result['units'] if u['kind'] == 'origin')
    assert origin['value']['fictional_note'] == 'Réd🙂'
    assert '\\u00e9' in origin['text'] and '\\ud83d\\ude42' in origin['text']
    # Frozen template text is ASCII because origin JSON escapes Unicode. The
    # partition/overlap algorithms still use codepoints, demonstrated manually.
    text, units = indexer.partition_units([dict(kind='header', text='É🙂'), dict(kind='fictional', text='β')])
    assert text == 'É🙂 β'
    assert [(u['start'], u['end']) for u in units] == [(0, 2), (2, 3), (3, 4)]
    overlap = indexer.overlap_units(dict(text=text, units=units), 1, 4, '🙂 β')
    assert [m['unit']['kind'] for m in overlap['matches']] == ['header', 'separator', 'fictional']
    assert overlap['semantic_attribution'] == 'unresolved_geometry_only'


def test_collusive_derived_peaks_and_rendered_report_tampering_cannot_replace_map_truth():
    packet, provenance = packet_fixture()
    packet['panels'][0]['current'][0]['derived']['red']['peaks'] = [[1, 1]]
    report = exporter.deterministic_template(packet)  # Both old helpers share this wrong derived field.
    assert report['propositions'][0]['cells'] == [[1, 1]]
    with pytest.raises(ValueError, match='independent exact map maxima'):
        indexer.index_template(packet, report, provenance)


def test_text_propositions_origin_and_distribution_tampering_are_distinct():
    packet, provenance = packet_fixture()
    report = exporter.deterministic_template(packet)
    with pytest.raises(ValueError, match='complete reconstruction'):
        indexer.index_template(packet, dict(report, text=report['text'] + ' '), provenance)
    wrong = deepcopy(report)
    wrong['propositions'][0]['cells'] = [[1, 1]]
    with pytest.raises(ValueError, match='complete map maxima'):
        indexer.index_template(packet, wrong, provenance)
    wrong_packet = deepcopy(packet)
    wrong_packet['panels'][0]['current'][0]['origin'] = dict(kind='completed_record', record_time=12, represented_time=12)
    with pytest.raises(ValueError, match='actual source'):
        indexed(wrong_packet, provenance)
    for grid in ([[[0.0] * 4] * 4], [[0.1] * 4 for _ in range(4)], [[float('nan')] * 4 for _ in range(4)]):
        with pytest.raises(ValueError):
            indexer.exact_peak_cells(grid)


def test_every_overlap_partial_boundary_header_and_repeated_occurrence_stays_geometry():
    packet, provenance = packet_fixture()
    result = indexed(packet, provenance)
    units = result['units']
    peak = next(u for u in units if u['kind'] == 'role_peaks')
    partial = indexer.overlap_units(result, peak['start'] + 1, peak['end'] - 1)
    assert len(partial['matches']) == 1 and partial['matches'][0]['partial_unit']
    assert partial['matches'][0]['unit_contains_evidence']
    end = peak['end'] + 2
    crossing = indexer.overlap_units(result, peak['end'] - 2, end)
    assert len(crossing['matches']) == 3 and crossing['boundary_crossing']
    header = indexer.overlap_units(result, 0, len(indexer.HEADER), indexer.HEADER)
    assert len(header['matches']) == 1 and header['matches'][0]['unit']['source_resolution'] is None
    repeated = indexer.overlap_units(result, peak['end'] - 5, peak['end'] - 2)
    assert len(repeated['all_exact_quote_occurrences']) > 1
    assert all(r['semantic_attribution'] == 'unresolved_geometry_only' for r in (partial, crossing, header, repeated))
    with pytest.raises(ValueError, match='mismatch'):
        indexer.overlap_units(result, peak['start'], peak['end'], 'Wrong quote')


def test_source_change_and_exact_restoration_use_addressed_properties_not_offsets():
    packet, provenance = packet_fixture()
    baseline = indexed(packet, provenance)
    changed_packet = deepcopy(packet)
    changed_node = changed_packet['panels'][0]['current'][0]
    changed_node['distributions']['red'][0][0] = 0.0
    changed_node['distributions']['red'][1][1] = 0.5
    changed_node['derived']['red'] = exporter.localization_summary(changed_node['distributions']['red'])
    changed = indexed(changed_packet, provenance)
    properties = lambda result: {tuple(u['packet_path']): u['value'] for u in result['units'] if u['packet_path'] is not None}
    before, after = properties(baseline), properties(changed)
    assert {k for k in before if before[k] != after[k]} == {('panels', 0, 'current', 0, 'derived', 'red', 'peaks')}
    assert indexed(deepcopy(packet), deepcopy(provenance)) == baseline
    # An unchanged map/peak case remains a genuine indexed null.
    assert properties(indexed(deepcopy(packet), provenance)) == before


def test_complete_fictional552_inventory_fixed684_pairs_and_token_denominators():
    exports = dict(requests=[], packets=[], provenance=[], templates=[])
    instruction = pipeline.frozen_instruction('narrator')
    for seed in (1301, 1311, 1321):
        for world in range(4):
            arms = [(s, v, variant) for s in exporter.SOURCE_KEYS for v in ('A_only', 'joined') for variant in exporter.VARIANTS]
            arms += [('W', 'W_only', variant) for variant in exporter.W_VARIANTS]
            arms += [('sensor', 'sensor_only', 'baseline')]
            for source, view, variant in arms:
                packet, provenance = packet_fixture(source=source, joined=view == 'joined', full=True)
                identifier = f's{seed}_w{world:02d}_t12_{source}_{view}_{variant}'
                provenance.update(id=identifier, case=dict(encoder_seed=seed, world_index=world, decision_time=12,
                    source=source, view=view, variant=variant))
                exports['requests'].append(dict(id=identifier, input=instruction + '\n' + pipeline.encoded(packet).decode('ascii')))
                exports['packets'].append(dict(id=identifier, content=packet, content_sha256=indexer.content_sha(packet)))
                exports['provenance'].append(provenance)
                exports['templates'].append(dict(id=identifier, report=exporter.deterministic_template(packet)))
    result = indexer.build_index(exports)
    assert result['template_opportunities'] == len(result['templates']) == 552
    assert result['counts'] == dict(header=552, origin=12720, role_peaks=25440, separator=38160)
    assert result['partition_units'] == 76872
    assert result['pair_opportunities'] == len(result['pairs']) == 684
    assert result['pairs'] == pipeline.build_pair_index(exports['provenance'])
    assert not result['reference_labels_read'] and not result['semantic_source_attribution_established']
