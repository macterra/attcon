"""Stage1 literal construction provenance; no API/reference judgments or attribution.

Independently reconstruct the frozen renderer and exact maxima from its supplied
float32 maps. Geometry locates constructed properties, never the property that
made a reference feature apply. All original sources, nulls and pairs are kept.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import struct

import attention_relational_origin_attribution_v1 as attribution
import attention_relational_report_v1 as exporter
import attention_relational_reporting_pipeline_v1 as pipeline
import inventory_attention_relational_origins_v1 as origins


VERSION = 'attention_relational_template_index_v1'
EXPORT_SHA256 = '4f5f0bc2fa78b3b24b3580f52e306f20df0f3011d079b8139a2bc1f029050837'
PLAN = 'docs/ATTENTION_RELATIONAL_TEMPLATE_ATTRIBUTION_V1_PLAN.md'
HEADER = 'Rows increase down; columns right. Current and conditional candidate peaks follow.'
PARAMETERS = dict(offset_unit='Python Unicode code points; start inclusive, end exclusive',
    partition='One fixed header, every origin/role-peak token, every joining space separately; no character omitted.',
    peaks='All exact maxima of supplied lossless float32 values; no normalization or rounding.',
    source='Actual provenance source_key plus recursively resolved declared origin, never panel name alone.',
    nested_origin='A whole origin token does not attribute every nested field; later semantic review must select exact property paths.',
    pair_identity='Match source/address/property paths, never character offsets; all original fixed pairs retained.',
    interpretation='Construction provenance and overlap geometry only; semantic attribution remains unresolved.')


def content_sha(value):
    return pipeline.digest(pipeline.encoded(value))


def exact_peak_cells(grid):
    """Independent4x4 maximum/tie computation, including float32/mass checks."""
    pipeline.require(type(grid) is list and len(grid) == 4 and all(type(row) is list and len(row) == 4 for row in grid),
                     'distribution must be4x4')
    values = [value for row in grid for value in row]
    pipeline.require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values),
                     'invalid finite unit-range distribution value')
    pipeline.require(list(struct.unpack('<16f', struct.pack('<16f', *values))) == values,
                     'distribution is not lossless float32; rounding is forbidden')
    pipeline.require(abs(math.fsum(values) - 1.0) <= 1e-6 + 1e-12, 'distribution mass differs from one')
    largest = max(values)
    return [[i // 4, i % 4] for i, value in enumerate(values) if value == largest]


def partition_units(tokens):
    """Deterministic concatenation, no substring search or occurrence selection."""
    units, cursor = [], 0
    for token in tokens:
        pipeline.require(type(token.get('text')) is str and bool(token['text']), 'empty/nonstring construction token')
        if units:
            units.append(dict(unit_id=f'u{len(units):04d}', kind='separator', text=' ', start=cursor, end=cursor + 1,
                              packet_path=None, value=' ', context=None, source_resolution=None))
            cursor += 1
        unit = deepcopy(token)
        unit.update(unit_id=f'u{len(units):04d}', start=cursor, end=cursor + len(token['text']))
        units.append(unit)
        cursor = unit['end']
    text = ''.join(unit['text'] for unit in units)
    pipeline.require(all(unit['start'] == (units[i - 1]['end'] if i else 0)
        and text[unit['start']:unit['end']] == unit['text'] for i, unit in enumerate(units))
        and cursor == len(text), 'complete character partition differs')
    return text, units


def index_template(packet, report, provenance):
    """Pure independent reconstruction of one complete packet/template pair.

    Does not call deterministic_template or use required_propositions as truth.
    Global header conventions have no source operand; spaces acquire no source
    from adjacent tokens. Each property retains its actual path/value/context.
    """
    pipeline.require(packet.get('schema') == exporter.PACK_SCHEMA and type(packet.get('panels')) is list,
                     'neutral numerical packet required')
    pipeline.require(type(report) is dict and set(report) == {'schema', 'text', 'propositions'}
        and report['schema'] == exporter.REPORT_SCHEMA and type(report['text']) is str, 'exact literal-report schema required')
    panel_names = [p['panel'] for p in packet['panels']]
    pipeline.require(1 <= len(panel_names) <= 2 and len(panel_names) == len(set(panel_names)) and set(panel_names) <= {'p0', 'p1'}
        and set(panel_names) == {p['panel'] for p in provenance['panels']}
        and len(provenance['panels']) == len(panel_names), 'exact panel provenance required')
    global_metadata = [dict(packet_path=[k], value=deepcopy(packet[k]))
                       for k in ('coordinate_system', 'candidate_semantics') if k in packet]
    tokens = [dict(kind='header', text=HEADER, packet_path=None, value=HEADER,
        context=dict(view='global_convention'), source_resolution=None, global_packet_metadata=global_metadata,
        construction='Authored fixed renderer convention; not a source-content or semantic attribution.')]
    atoms, seen = [], set()
    for pi, panel in enumerate(packet['panels']):
        views = [('current', None, panel['current'], ['panels', pi, 'current'])]
        if panel['candidates'] is not None:
            pipeline.require(set(panel['candidates']) == set(exporter.COMMANDS), 'complete candidate-command inventory required')
            views += [('candidate', command, panel['candidates'][command], ['panels', pi, 'candidates', command])
                      for command in exporter.COMMANDS]
        for view, command, nodes, path in views:
            for ni, node in enumerate(nodes):
                context = dict(panel=panel['panel'], view=view, command=command, node=node['node'])
                key = (context['panel'], view, command, node['node'])
                pipeline.require(node['node'] in exporter.NODES and key not in seen, 'invalid/duplicate node context')
                seen.add(key)
                node_path = path + [ni]
                prefix = '/'.join([panel['panel'], view] + ([] if command is None else [command]) + [node['node']])
                binding = dict(**context, role=None, basis='origin')
                resolution = attribution.resolve_binding(binding, packet, provenance)
                pipeline.require(resolution['state'] == 'resolved', 'origin/provenance cannot resolve actual source')
                origin_json = json.dumps(node['origin'], sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
                tokens.append(dict(kind='origin', text=prefix + '/origin=' + origin_json,
                    packet_path=node_path + ['origin'], value=deepcopy(node['origin']), context=dict(**context, role=None),
                    source_resolution=resolution))
                pipeline.require(set(node['distributions']) == set(exporter.ROLES), 'both role distributions required')
                for role in exporter.ROLES:
                    cells = exact_peak_cells(node['distributions'][role])
                    supplied = node['derived'][role]['peaks']
                    pipeline.require(pipeline.encoded(supplied) == pipeline.encoded(cells),
                                     'stored derived peaks differ from independent exact map maxima')
                    atom = dict(kind='peaks', **context, role=role, cells=deepcopy(cells))
                    atoms.append(atom)
                    binding = dict(**context, role=role, basis='peaks')
                    resolved = attribution.resolve_binding(binding, packet, provenance)
                    pipeline.require(resolved['state'] == 'resolved', 'peak provenance cannot resolve actual source')
                    rendered = prefix + '/' + role + '={' + ','.join('(' + str(r) + ',' + str(c) + ')' for r, c in cells) + '}'
                    tokens.append(dict(kind='role_peaks', text=rendered, packet_path=node_path + ['derived', role, 'peaks'],
                        value=deepcopy(cells), distribution_packet_path=node_path + ['distributions', role],
                        context=dict(**context, role=role), source_resolution=resolved))
    text, units = partition_units(tokens)
    pipeline.require(text == report['text'], 'literal template text differs from independent complete reconstruction')
    pipeline.require(pipeline.encoded(atoms) == pipeline.encoded(report['propositions']),
                     'template propositions differ from independent complete map maxima')
    pipeline.require(0 < len(text.split()) <= exporter.WORD_LIMIT, 'literal template word budget differs')
    counts = Counter(u['kind'] for u in units)
    pipeline.require(counts['header'] == 1 and counts['role_peaks'] == 2 * counts['origin']
        and counts['separator'] == len(tokens) - 1, 'template token accounting differs')
    return dict(schema=VERSION + '_template', text=text, text_sha256=pipeline.digest(text.encode('utf-8')),
        packet_content_sha256=content_sha(packet), provenance_content_sha256=content_sha(provenance),
        required_propositions_content_sha256=content_sha(atoms), required_peak_count=len(atoms),
        counts=dict(counts), units=units, exact_text_verified=True, exact_propositions_verified=True,
        independent_map_maxima_verified=True, complete_character_partition_verified=True,
        semantic_source_attribution_established=False)


def overlap_units(index, start, end, quote=None):
    """List every intersecting construction unit; containment never earns meaning."""
    text = index['text']
    pipeline.require(type(start) is int and type(end) is int and 0 <= start < end <= len(text), 'invalid exact overlap offsets')
    pipeline.require(quote is None or type(quote) is str and text[start:end] == quote, 'overlap quote/offset mismatch')
    matches = []
    for unit in index['units']:
        left, right = max(start, unit['start']), min(end, unit['end'])
        if left < right:
            matches.append(dict(unit=deepcopy(unit), overlap_start=left, overlap_end=right, overlap_text=text[left:right],
                unit_contains_evidence=unit['start'] <= start and end <= unit['end'],
                evidence_contains_unit=start <= unit['start'] and unit['end'] <= end,
                partial_unit=left != unit['start'] or right != unit['end']))
    return dict(start=start, end=end, quote=text[start:end], matches=matches,
        all_exact_quote_occurrences=attribution.occurrences(text, text[start:end]),
        boundary_crossing=len(matches) > 1, semantic_attribution='unresolved_geometry_only')


def build_index(exports):
    inventory = pipeline.validate_export(exports)
    rows, totals, cases = [], Counter(), Counter()
    for identifier in inventory['request_ids']:
        provenance = inventory['provenance'][identifier]
        indexed = index_template(inventory['packets'][identifier]['content'], inventory['templates'][identifier]['report'], provenance)
        rows.append(dict(id=identifier + '_template', source_id=identifier, unit_id=pipeline.blind_id(identifier + '_template'),
            case=deepcopy(provenance['case']), panel_provenance=deepcopy(provenance['panels']), index=indexed))
        totals.update(indexed['counts'])
        cases[(provenance['case']['encoder_seed'], provenance['case']['world_index'])] += 1
    pairs = pipeline.build_pair_index(exports['provenance'])
    pipeline.require(len(rows) == 552 and len(pairs) == 684 and len(cases) == 12 and set(cases.values()) == {46}
        and totals == Counter(header=552, role_peaks=25440, origin=12720, separator=38160),
        'fixed552/25440/12720/684 construction inventory differs')
    return dict(schema=VERSION, parameters=deepcopy(PARAMETERS), parameters_content_sha256=content_sha(PARAMETERS),
        template_opportunities=552, counts=dict(totals), partition_units=sum(totals.values()),
        ordered_source_ids=inventory['request_ids'], templates=rows, pair_opportunities=684, pairs=pairs,
        reference_labels_read=False, semantic_source_attribution_established=False,
        independent_observations_claimed=False, changes_registered_endpoint_or_qualification=False,
        interpretation='Authored construction provenance only; source dependence does not identify a quale-generating locus.')


def source_hashes():
    paths = [Path(m.__file__) for m in (exporter, pipeline, attribution, origins)] + [Path(__file__)]
    paths += [pipeline.ROOT / PLAN, pipeline.ROOT / 'tests/test_attention_relational_template_index_v1.py']
    return {str(p.resolve().relative_to(pipeline.ROOT)): pipeline.digest(p.read_bytes()) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exports', type=Path, required=True)
    parser.add_argument('--exports-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pipeline.require(args.exports_sha256 == EXPORT_SHA256, 'only the pinned first export is allowed')
    pipeline.require(not args.output.exists(), 'refusing replacement of first construction index')
    result = build_index(pipeline.bound_json(args.exports, args.exports_sha256))
    result.update(input_dependencies=dict(exports=dict(path=str(args.exports), sha256=args.exports_sha256)),
                  source_sha256=source_hashes(), no_api_or_model_calls=True,
                  chronology='Stage1 implementation follows the published template-attribution plan; no reference judgments read.')
    pipeline.exclusive_json(args.output, result)


if __name__ == '__main__':
    main()
