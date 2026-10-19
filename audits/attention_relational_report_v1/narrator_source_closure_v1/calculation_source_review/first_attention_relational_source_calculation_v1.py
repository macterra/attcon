"""Independent arithmetic on exact supplied source grids; no semantic grading.

No exporter calculation is called and stored derived summaries are not operands.
The CLI binds its input bytes and names every raw operand path/value. Results
describe supplied numbers only; they establish no ownership, appearance, actual
execution, report entailment or consciousness-related endpoint.
"""
import argparse
import math
from pathlib import Path

import attention_relational_reporting_pipeline_v1 as pipeline


VERSION = 'attention_relational_source_calculation_v1'
OPERATIONS = ('maximal_cells', 'peak_probability', 'entropy_nats', 'expected_row',
              'expected_column', 'red_blue_peak_relation')


def resolve(value, path):
    pipeline.require(type(path) is list, 'exact JSON operand path required')
    for part in path:
        if type(value) is dict and type(part) is str and part in value:
            value = value[part]
        elif type(value) is list and type(part) is int and 0 <= part < len(value):
            value = value[part]
        else:
            raise ValueError('operand path does not resolve exactly')
    return value


def grid_values(grid):
    pipeline.require(type(grid) is list and len(grid) == 4
        and all(type(row) is list and len(row) == 4 for row in grid), 'complete 4x4 source grid required')
    values = [(r, c, grid[r][c]) for r in range(4) for c in range(4)]
    pipeline.require(all(type(p) in (int, float) and 0 <= p <= 1 and math.isfinite(p)
        for _, _, p in values), 'finite nonnegative probability literals required; booleans forbidden')
    return values


def maxima(grid):
    values = grid_values(grid)
    largest = max(p for _, _, p in values)
    return [[r, c] for r, c, p in values if p == largest]


def calculate(operation, grids):
    pipeline.require(operation in OPERATIONS and len(grids) == (2 if operation == 'red_blue_peak_relation' else 1),
                     'operation and complete raw operand inventory required')
    first = grid_values(grids[0])
    if operation == 'maximal_cells':
        return maxima(grids[0])
    if operation == 'peak_probability':
        return max(p for _, _, p in first)
    if operation == 'entropy_nats':
        return -math.fsum(p * math.log(p) for _, _, p in first if p != 0)
    if operation == 'expected_row':
        return math.fsum(r * p for r, _, p in first)
    if operation == 'expected_column':
        return math.fsum(c * p for _, c, p in first)
    red, blue = maxima(grids[0]), maxima(grids[1])
    combinations = [(r, b) for r in red for b in blue]
    horizontal = {'left_of' if r[1] < b[1] else 'right_of' if r[1] > b[1] else 'same_column'
                  for r, b in combinations}
    vertical = {'above' if r[0] < b[0] else 'below' if r[0] > b[0] else 'same_row'
                for r, b in combinations}
    return dict(horizontal=[v for v in ('left_of', 'same_column', 'right_of') if v in horizontal],
        vertical=[v for v in ('above', 'same_row', 'below') if v in vertical],
        coincident_possible=any(r == b for r, b in combinations),
        coincident_all=all(r == b for r, b in combinations))


def calculation_record(source, source_sha, operation, operand_paths):
    pipeline.require(all(type(path) is list and len(path) >= 2
        and path[-2] == 'distributions' and path[-1] in ('red', 'blue')
        and 'derived' not in path and 'red_blue_peak_relation' not in path
        for path in operand_paths), 'exact raw distribution fields required, not stored summaries')
    if operation == 'red_blue_peak_relation':
        pipeline.require(len(operand_paths) == 2 and operand_paths[0][:-1] == operand_paths[1][:-1]
            and [p[-1] for p in operand_paths] == ['red', 'blue'],
            'relation requires red then blue distributions at the same source node')
    grids = [resolve(source, path) for path in operand_paths]
    return dict(schema=VERSION, input_sha256=source_sha, operator=operation,
        operands=[dict(path=path, value=value) for path, value in zip(operand_paths, grids)],
        value=calculate(operation, grids), exporter_derived_values_used_as_operands=False,
        renormalization_performed=False, report_or_source_semantics_assessed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--operation', choices=OPERATIONS, required=True)
    parser.add_argument('--operand-path', action='append', required=True,
                        help='Exact JSON path to one full raw source grid; repeat for red then blue')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pipeline.require(not args.output.exists(), 'refusing replacement of first calculation artifact')
    source = pipeline.bound_json(args.input, args.input_sha256)
    paths = [pipeline.decoded(path) for path in args.operand_path]
    record = calculation_record(source, args.input_sha256, args.operation, paths)
    record['input_path'] = str(args.input)
    record['calculation_source_sha256'] = pipeline.digest(Path(__file__).read_bytes())
    pipeline.exclusive_json(args.output, record)


if __name__ == '__main__':
    main()
