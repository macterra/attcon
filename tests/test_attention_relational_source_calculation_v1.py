"""Analytic fictional examples; no consciousness/report qualification."""
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import attention_relational_source_calculation_v1 as calc


def point(row, column, weight=1.0):
    grid = [[0.0] * 4 for _ in range(4)]
    grid[row][column] = weight
    return grid


def test_complete_tied_maxima_and_analytic_scalar_values():
    grid = [[1 / 16] * 4 for _ in range(4)]
    assert calc.calculate('maximal_cells', [grid]) == [[r, c] for r in range(4) for c in range(4)]
    assert calc.calculate('peak_probability', [grid]) == 1 / 16
    assert calc.calculate('entropy_nats', [grid]) == pytest.approx(math.log(16))
    assert calc.calculate('expected_row', [grid]) == 1.5
    assert calc.calculate('expected_column', [grid]) == 1.5


def test_point_and_unnormalized_supplied_mass_are_not_renormalized():
    grid = point(2, 3, 0.5)
    assert calc.calculate('maximal_cells', [grid]) == [[2, 3]]
    assert calc.calculate('expected_row', [grid]) == 1.0
    assert calc.calculate('expected_column', [grid]) == 1.5
    assert calc.calculate('entropy_nats', [grid]) == -0.5 * math.log(0.5)
    assert calc.calculate('entropy_nats', [point(0, 0)]) == 0


def test_relations_use_all_raw_red_blue_ties_cartesian_product():
    red, blue = point(0, 0, 0.5), point(0, 0, 0.5)
    red[3][3] = blue[3][3] = 0.5
    assert calc.calculate('red_blue_peak_relation', [red, blue]) == dict(
        horizontal=['left_of', 'same_column', 'right_of'], vertical=['above', 'same_row', 'below'],
        coincident_possible=True, coincident_all=False)
    assert calc.calculate('red_blue_peak_relation', [point(2, 1), point(2, 1)])['coincident_all'] is True
    assert calc.calculate('red_blue_peak_relation', [point(0, 3), point(2, 1)]) == dict(
        horizontal=['right_of'], vertical=['above'], coincident_possible=False, coincident_all=False)


def test_corrupted_exporter_summaries_cannot_change_raw_calculation():
    source = dict(distributions=dict(red=point(1, 2), blue=point(3, 0)),
                  derived=dict(red=dict(peaks=[[0, 0]])),
                  red_blue_peak_relation=dict(horizontal=['left_of']))
    record = calc.calculation_record(source, 'invented hash', 'maximal_cells', [['distributions', 'red']])
    assert record['value'] == [[1, 2]] and record['operands'][0]['value'] == point(1, 2)
    assert record['operand_path_syntax_checked'] is True
    assert record['source_packet_or_native_origin_verified'] is False
    assert record['report_or_source_semantics_assessed'] is False
    with pytest.raises(ValueError):
        calc.calculation_record(source, 'invented hash', 'maximal_cells', [['derived', 'red', 'peaks']])
    source['derived']['fake_grid'] = point(0, 0)
    with pytest.raises(ValueError):
        calc.calculation_record(source, 'invented hash', 'maximal_cells', [['derived', 'fake_grid']])


def test_relation_rejects_cross_node_or_reversed_operand_paths():
    source = dict(first=dict(distributions=dict(red=point(0, 0), blue=point(1, 1))),
                  second=dict(distributions=dict(blue=point(2, 2))))
    for paths in ([['first', 'distributions', 'red'], ['second', 'distributions', 'blue']],
                  [['first', 'distributions', 'blue'], ['first', 'distributions', 'red']]):
        with pytest.raises(ValueError):
            calc.calculation_record(source, 'invented', 'red_blue_peak_relation', paths)


@pytest.mark.parametrize('bad', [True, -0.1, 1.1, float('inf'), float('nan'), '0.5'])
def test_invalid_probability_literals_rejected(bad):
    grid = point(1, 2)
    grid[0][0] = bad
    with pytest.raises(ValueError):
        calc.calculate('maximal_cells', [grid])


@pytest.mark.parametrize('operation, grids', [('unknown', [point(0, 0)]),
    ('red_blue_peak_relation', [point(0, 0)]), ('expected_row', [point(0, 0), point(1, 1)]),
    ('maximal_cells', [[[1.0]]])])
def test_operation_shape_and_complete_operand_counts(operation, grids):
    with pytest.raises(ValueError):
        calc.calculate(operation, grids)


def test_operand_path_types_preserve_bool_distinction():
    with pytest.raises(ValueError):
        calc.resolve([point(0, 0)], [True])


def test_decoy_distribution_paths_cannot_establish_native_origin():
    source = dict(decoy=dict(distributions=dict(red=point(0, 0))))
    record = calc.calculation_record(source, 'invented', 'maximal_cells', [['decoy', 'distributions', 'red']])
    assert record['value'] == [[0, 0]]
    assert record['source_packet_or_native_origin_verified'] is False


def test_cli_binds_input_bytes_and_refuses_first_output_replacement(tmp_path, monkeypatch):
    source, output = tmp_path / 'fictional.json', tmp_path / 'first.json'
    calc.pipeline.exclusive_json(source, dict(distributions=dict(red=point(1, 3))))
    args = ['calculate', '--input', str(source), '--input-sha256', '0' * 64,
            '--operation', 'maximal_cells', '--operand-path', '["distributions","red"]', '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(ValueError):
        calc.main()
    assert not output.exists()
    args[4] = calc.pipeline.digest(source.read_bytes())
    calc.main()
    before = output.read_bytes()
    result = calc.pipeline.decoded(before)
    assert result['value'] == [[1, 3]] and result['source_packet_or_native_origin_verified'] is False
    with pytest.raises(ValueError):
        calc.main()
    assert output.read_bytes() == before
