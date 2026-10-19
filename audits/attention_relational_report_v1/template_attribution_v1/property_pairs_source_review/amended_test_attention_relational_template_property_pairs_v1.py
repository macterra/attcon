"""Fictional archived-property comparisons only; no target semantic judgments."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import attention_relational_template_property_pairs_v1 as pairs
import attention_relational_report_v1 as exporter
from test_attention_relational_template_index_v1 import packet_fixture, indexed


def fixture(packet=None, provenance=None):
    if packet is None:
        packet, provenance = packet_fixture()
    row = dict(index=indexed(packet, provenance))
    return row, packet, provenance


def table(packet=None, provenance=None):
    row, packet, provenance = fixture(packet, provenance)
    return pairs.property_table(row, packet, provenance)


def test_identical_tables_preserve_all_tied_peaks_and_individual_origin_fields():
    original = table()
    rows = pairs.compare(original, deepcopy(original))
    assert all(r['state'] == 'matched' and r['value_unchanged'] for r in rows)
    assert pairs.all_equal(rows, 'value_unchanged')
    assert pairs.all_equal(rows, 'raw_distribution_unchanged')
    assert any(p['value'] == [[0, 0], [2, 3]] for p in original.values())
    assert all(type(p['value']) is not dict for p in original.values())
    assert all(p['identity']['value_path'] for p in original.values()
               if p['identity']['kind'] == 'origin')
    assert all(not r['source_meaning_established'] and r['asserted_predicate_tracking'] is None for r in rows)


def test_token_ids_offsets_and_table_order_do_not_define_property_identity():
    row, packet, provenance = fixture()
    original = pairs.property_table(row, packet, provenance)
    for unit in row['index']['units']:
        unit['unit_id'] = 'arbitrary_' + unit['unit_id']
        unit['start'] += 99
        unit['end'] += 99
    row['index']['units'].reverse()
    changed = pairs.property_table(row, packet, provenance)
    assert set(original) == set(changed)
    assert pairs.all_equal(pairs.compare(original, changed), 'value_unchanged')


def test_same_peak_set_with_changed_distribution_is_a_true_peak_null():
    packet, provenance = packet_fixture()
    original = table(packet, provenance)
    grid = [[0.0] * 4 for _ in range(4)]
    grid[0][0] = grid[2][3] = 0.375
    grid[0][1] = grid[3][1] = 0.125
    node = packet['panels'][0]['current'][1]
    node['distributions']['red'] = grid
    node['derived']['red'] = exporter.localization_summary(grid)
    changed = table(packet, provenance)
    compared = pairs.compare(original, changed)
    assert pairs.all_equal(compared, 'value_unchanged')
    assert not pairs.all_equal(compared, 'raw_distribution_unchanged')
    assert sum(r['peak_null_with_raw_distribution_change'] is True for r in compared) == 1


def test_changed_peak_property_is_not_a_null():
    packet, provenance = packet_fixture()
    original = table(packet, provenance)
    grid = [[0.0] * 4 for _ in range(4)]
    grid[1][2] = 1.0
    node = packet['panels'][0]['current'][1]
    node['distributions']['red'] = grid
    node['derived']['red'] = exporter.localization_summary(grid)
    rows = pairs.compare(original, table(packet, provenance))
    assert sum(r['value_unchanged'] is False for r in rows) == 1
    assert not any(r['peak_null_with_raw_distribution_change'] is True for r in rows)


def test_source_transition_from_copied_w_to_a_is_unmatched_not_invariance():
    packet, provenance = packet_fixture()
    original = table(packet, provenance)
    packet['panels'][0]['current'][0]['origin'] = deepcopy(packet['panels'][0]['current'][1]['origin'])
    changed = table(packet, provenance)
    rows = pairs.compare(original, changed)
    missing = [r for r in rows if r['state'] != 'matched']
    assert missing and {r['state'] for r in missing} == {'first_only', 'second_only'}
    assert all(r['value_unchanged'] is None and r['raw_distribution_unchanged'] is None for r in missing)
    assert not pairs.all_equal(rows, 'value_unchanged')
    sources = {p['identity']['actual_source'] for p in original.values()}
    assert sources == {'A', 'W'}


def test_represented_time_transition_is_unmatched_not_invariance():
    packet, provenance = packet_fixture()
    original = table(packet, provenance)
    packet['panels'][0]['current'][1]['origin']['represented_time'] += 1
    rows = pairs.compare(original, table(packet, provenance))
    assert any(r['state'] != 'matched' for r in rows)
    assert all(r['value_unchanged'] is None for r in rows if r['state'] != 'matched')


def test_missing_property_is_unknown_and_no_empty_comparison_proves_restoration():
    original = table()
    key = next(iter(original))
    missing = deepcopy(original)
    del missing[key]
    row = next(r for r in pairs.compare(original, missing) if r['property_identity_sha256'] == key)
    assert row['state'] == 'first_only' and row['value_unchanged'] is None
    assert not pairs.all_equal([], 'value_unchanged')
    assert not pairs.all_equal(pairs.compare(original, {}), 'raw_distribution_unchanged')


def test_origin_fields_do_not_control_claim_about_raw_peak_operand_restoration():
    packet, provenance = packet_fixture()
    original = table(packet, provenance)
    packet['panels'][0]['current'][1]['origin']['extra_origin_field'] = 'fictional'
    compared = pairs.compare(original, table(packet, provenance))
    assert not pairs.all_equal(compared, 'value_unchanged')
    assert pairs.all_equal(compared, 'raw_distribution_unchanged')


def test_duplicate_actual_property_identity_is_rejected():
    row, packet, provenance = fixture()
    unit = next(u for u in row['index']['units'] if u['kind'] == 'role_peaks')
    row['index']['units'].append(deepcopy(unit))
    with pytest.raises(ValueError, match='duplicate actual source'):
        pairs.property_table(row, packet, provenance)


def test_jointly_tampered_derived_and_indexed_peaks_are_not_self_verification():
    row, packet, provenance = fixture()
    unit = next(u for u in row['index']['units'] if u['kind'] == 'role_peaks')
    pairs.resolve(packet, unit['packet_path'])[:] = [[3, 2]]
    unit['value'] = [[3, 2]]
    row['index']['packet_content_sha256'] = pairs.construction.content_sha(packet)
    with pytest.raises(ValueError, match='independent complete exact maxima'):
        pairs.property_table(row, packet, provenance)


def test_same_panel_with_changed_provenance_cannot_retain_old_source_identity():
    row, packet, provenance = fixture()
    provenance['panels'][0]['source_key'] = 'ordinary'
    row['index']['provenance_content_sha256'] = pairs.construction.content_sha(provenance)
    with pytest.raises(ValueError, match='actual source provenance'):
        pairs.property_table(row, packet, provenance)


def test_exact_property_value_binding_rejects_type_change():
    row, packet, provenance = fixture()
    unit = next(u for u in row['index']['units'] if u['kind'] == 'role_peaks')
    unit['value'][0][0] = float(unit['value'][0][0])
    with pytest.raises(ValueError, match='indexed exact property value differs'):
        pairs.property_table(row, packet, provenance)


@pytest.mark.parametrize('field', ['record_time', 'feedback_through', 'archived_origin_time', 'shown_at'])
def test_all_individual_origin_times_affect_comparability(field):
    packet, provenance = packet_fixture()
    origin = packet['panels'][0]['current'][0]['origin']
    origin[field] = 11
    before = table(packet, provenance)
    origin[field] = 10
    after = table(packet, provenance)
    keys = lambda t: {k for k, p in t.items() if p['identity']['context']['panel'] == 'p0'
        and p['identity']['context']['view'] == 'current' and p['identity']['context']['node'] == 'n0'}
    assert keys(before) and keys(after) and keys(before).isdisjoint(keys(after))


def test_inherited_nested_time_change_cannot_leave_peaks_comparable():
    packet, provenance = packet_fixture()
    origin = packet['panels'][0]['candidates']['k0'][0]['origin']
    assert origin['kind'] == 'inherited_current'
    before = table(packet, provenance)
    origin['inherited_origin']['represented_time'] += 1
    after = table(packet, provenance)
    keys = lambda t: {k for k, p in t.items() if p['identity']['context']['panel'] == 'p0'
        and p['identity']['context']['command'] == 'k0' and p['identity']['context']['node'] == 'n0'}
    assert keys(before).isdisjoint(keys(after))


@pytest.mark.parametrize('field', ['packet_path', 'distribution_packet_path'])
def test_equal_values_at_wrong_node_cannot_supply_addressed_operand(field):
    row, packet, provenance = fixture()
    unit = next(u for u in row['index']['units'] if u['kind'] == 'role_peaks'
        and u['context']['panel'] == 'p0' and u['context']['view'] == 'current'
        and u['context']['node'] == 'n1' and u['context']['role'] == 'red')
    assert unit[field][3] == 1
    unit[field][3] = 0
    with pytest.raises(ValueError, match='exact addressed operand'):
        pairs.property_table(row, packet, provenance)


def test_equal_red_blue_values_do_not_allow_cross_role_raw_operand_substitution():
    packet, provenance = packet_fixture()
    node = packet['panels'][0]['current'][1]
    node['distributions']['blue'] = deepcopy(node['distributions']['red'])
    node['derived']['blue'] = exporter.localization_summary(node['distributions']['blue'])
    row, packet, provenance = fixture(packet, provenance)
    unit = next(u for u in row['index']['units'] if u['kind'] == 'role_peaks'
        and u['context']['panel'] == 'p0' and u['context']['view'] == 'current'
        and u['context']['node'] == 'n1' and u['context']['role'] == 'red')
    unit['distribution_packet_path'][-1] = 'blue'
    with pytest.raises(ValueError, match='exact addressed operand'):
        pairs.property_table(row, packet, provenance)
