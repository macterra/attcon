"""All archived template/source property comparisons; no semantic endpoint grading.

Implements the numerical part of Stage3 of the existing template-attribution plan.
Read no reference annotations or source-review judgments. Identity uses resolved
source, addressed context, represented time and the actual property, never token
position, offset or wording. Missing/differently sourced/timed properties are
unmatched, not unchanged. Peak-set nulls and changes in their raw distributions
are separate. This bank cannot establish an asserted predicate or qualia.

This utility is conditional on the complete validated Stage1 index and exports.
An input/construction binding failure aborts bank production; it never releases a
partial bank or calls unavailable rows nulls. It does not implement the complete
Stage3 failure/semantic assessment, which must retain the full original inventory
and the actual failed execution if this conditional numerical stage cannot run.
"""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

import attention_relational_template_index_v1 as construction
import attention_relational_template_review_v1 as preparation
import attention_relational_reporting_pipeline_v1 as pipeline


VERSION = 'attention_relational_template_property_pairs_v1'
TIME_FIELDS = ('represented_time', 'record_time', 'feedback_through',
               'shown_at', 'archived_origin_time', 'decision_time')


def same(a, b):
    return pipeline.encoded(a) == pipeline.encoded(b)


def resolve(value, path):
    for part in path:
        if type(value) is dict:
            pipeline.require(type(part) is str and part in value, 'exact property key missing')
        elif type(value) is list:
            pipeline.require(type(part) is int and 0 <= part < len(value), 'exact property index missing')
        else:
            raise ValueError('property path enters scalar')
        value = value[part]
    return value


def leaves(value, path=()):
    """Every individual origin field; no containing object earns all its fields."""
    if type(value) is dict:
        pipeline.require(bool(value), 'empty origin object has no individual fields')
        for key in sorted(value):
            yield from leaves(value[key], path + (key,))
    elif type(value) is list:
        pipeline.require(bool(value), 'empty origin list has no individual fields')
        for index, item in enumerate(value):
            yield from leaves(item, path + (index,))
    else:
        yield list(path), value


def addressed_path(packet, context):
    panels = [(i, p) for i, p in enumerate(packet['panels']) if p['panel'] == context['panel']]
    pipeline.require(len(panels) == 1, 'exact addressed panel required')
    pi, panel = panels[0]
    if context['view'] == 'current':
        pipeline.require(context['command'] is None, 'current address cannot contain a command')
        prefix, nodes = ['panels', pi, 'current'], panel['current']
    else:
        pipeline.require(context['view'] == 'candidate' and panel['candidates'] is not None
            and context['command'] in panel['candidates'], 'exact addressed candidate command required')
        prefix = ['panels', pi, 'candidates', context['command']]
        nodes = panel['candidates'][context['command']]
    matches = [i for i, n in enumerate(nodes) if n['node'] == context['node']]
    pipeline.require(len(matches) == 1, 'exact addressed node required')
    return prefix + matches


def property_table(row, packet, provenance):
    indexed = row['index']
    pipeline.require(indexed['packet_content_sha256'] == construction.content_sha(packet)
        and indexed['provenance_content_sha256'] == construction.content_sha(provenance),
        'indexed packet/provenance bytes differ')
    properties = {}
    for unit in indexed['units']:
        if unit['kind'] not in ('origin', 'role_peaks'):
            continue
        path = unit['packet_path']
        node_path = addressed_path(packet, unit['context'])
        role = unit['context']['role']
        expected = node_path + (['origin'] if unit['kind'] == 'origin' else ['derived', role, 'peaks'])
        pipeline.require(same(path, expected), 'indexed property path differs from exact addressed operand')
        pipeline.require(same(resolve(packet, path), unit['value']), 'indexed exact property value differs')
        binding = dict(**unit['context'], basis='origin' if unit['kind'] == 'origin' else 'peaks')
        source = construction.attribution.resolve_binding(binding, packet, provenance)
        pipeline.require(source['state'] == 'resolved' and same(source, unit['source_resolution']),
            'actual source provenance/origin differs')
        origin = source['origin_chain'][0]
        represented = origin.get('represented_time')
        pipeline.require(type(represented) is int, 'individual represented_time required')
        temporal_coordinates = []
        for time_path, time_value in leaves(origin):
            if time_path[-1] in TIME_FIELDS:
                pipeline.require(type(time_value) is int, 'individual temporal origin field must be integer')
                temporal_coordinates.append(dict(path=time_path, value=time_value))
        pipeline.require(type(packet['decision_time']) is int, 'exact decision_time required')
        raw_path, raw_sha = None, None
        if unit['kind'] == 'role_peaks':
            raw_path = unit['distribution_packet_path']
            pipeline.require(same(raw_path, node_path + ['distributions', role]),
                'raw distribution path differs from exact addressed operand')
            raw = resolve(packet, raw_path)
            pipeline.require(same(construction.exact_peak_cells(raw), unit['value']),
                'peak property differs from independent complete exact maxima')
            raw_sha = construction.content_sha(raw)
            values = [([], unit['value'])]
        else:
            pipeline.require(type(unit['value']) is dict, 'individual origin fields required')
            values = list(leaves(unit['value']))
        for relative, value in values:
            identity = dict(actual_source=source['origin_class']['origin_source_key'],
                context=deepcopy(unit['context']), represented_time=represented,
                decision_time=packet['decision_time'],
                origin_category=source['origin_class']['category'],
                origin_kind=source['origin_class']['origin_kind'],
                terminal_origin_kind=source['origin_class']['terminal_origin_kind'],
                temporal_coordinates=deepcopy(temporal_coordinates),
                kind=unit['kind'], operator='exact_argmax' if unit['kind'] == 'role_peaks' else 'identity',
                value_path=relative)
            key = construction.content_sha(identity)
            pipeline.require(key not in properties, 'duplicate actual source/address/property identity')
            properties[key] = dict(identity=identity, value=deepcopy(value),
                construction_unit_id=unit['unit_id'], packet_value_path=path + relative,
                raw_distribution_packet_path=raw_path, raw_distribution_content_sha256=raw_sha,
                source_resolution=deepcopy(source))
    return properties


def compare(first, second):
    """Compare all identities. None signals missing operands, never a null."""
    rows = []
    for key in sorted(set(first) | set(second)):
        a, b = first.get(key), second.get(key)
        state = ('matched' if a is not None and b is not None else
                 'first_only' if a is not None else 'second_only')
        comparable = state == 'matched'
        peak = (a or b)['identity']['kind'] == 'role_peaks'
        unchanged = same(a['value'], b['value']) if comparable else None
        raw_unchanged = (a['raw_distribution_content_sha256'] == b['raw_distribution_content_sha256']
                         if comparable and peak else None)
        rows.append(dict(property_identity_sha256=key, state=state,
            property_kind=(a or b)['identity']['kind'],
            value_unchanged=unchanged, raw_distribution_unchanged=raw_unchanged,
            peak_null_with_raw_distribution_change=(unchanged and not raw_unchanged)
                if comparable and peak else None,
            source_meaning_established=False, asserted_predicate_tracking=None))
    return rows


def comparison_counts(rows):
    return dict(total=len(rows), matched=sum(r['state'] == 'matched' for r in rows),
        unmatched=sum(r['state'] != 'matched' for r in rows),
        value_unchanged=sum(r['value_unchanged'] is True for r in rows),
        value_changed=sum(r['value_unchanged'] is False for r in rows),
        peak_null_with_raw_distribution_change=sum(r['peak_null_with_raw_distribution_change'] is True for r in rows))


def all_equal(rows, field):
    """No vacuous or missing-operand restoration success."""
    relevant = ([r for r in rows if r['property_kind'] == 'role_peaks']
                if field == 'raw_distribution_unchanged' else rows)
    return bool(relevant) and all(r['state'] == 'matched' for r in relevant) and all(
        r[field] is True for r in relevant)


def assemble(index, exports):
    preparation.validate_index(index)
    inventory = pipeline.validate_export(exports)
    pipeline.require(index['ordered_source_ids'] == inventory['request_ids']
        and index['pairs'] == pipeline.build_pair_index(exports['provenance']),
        'all original source identities/pairs required')
    tables, baselines, by_id = {}, {}, {}
    for row in index['templates']:
        source_id = row['source_id']
        packet = inventory['packets'][source_id]['content']
        provenance = inventory['provenance'][source_id]
        pipeline.require(same(row['case'], provenance['case']) and
            same(row['index']['text'], inventory['templates'][source_id]['report']['text']),
            'indexed original case/text differs')
        tables[source_id] = property_table(row, packet, provenance)
        by_id[source_id] = row
        if row['case']['variant'] == 'baseline':
            key = tuple(row['case'][k] for k in ('encoder_seed', 'world_index', 'source', 'view'))
            pipeline.require(key not in baselines, 'duplicate baseline case')
            baselines[key] = source_id
    pairs = []
    for pair in index['pairs']:
        first, second = pair['first_id'], pair['second_id']
        comparison = compare(tables[first], tables[second])
        baseline = baselines[tuple(pair['case'])]
        restoration = None
        if by_id[second]['case']['variant'].endswith('_restored'):
            restored = compare(tables[baseline], tables[second])
            restoration = dict(baseline_id=baseline,
                second_indexed_properties_equal_baseline=all_equal(restored, 'value_unchanged'),
                second_raw_peak_operands_equal_baseline=all_equal(restored, 'raw_distribution_unchanged'),
                second_literal_text_equal_baseline=same(by_id[baseline]['index']['text'], by_id[second]['index']['text']),
                comparison_counts=comparison_counts(restored))
        source_counts = {}
        for source in sorted({v['identity']['actual_source'] for t in (tables[first], tables[second]) for v in t.values()}):
            source_counts[source] = comparison_counts([r for r in comparison if
                (tables[first].get(r['property_identity_sha256']) or
                 tables[second][r['property_identity_sha256']])['identity']['actual_source'] == source])
        pairs.append(dict(original_pair=deepcopy(pair), properties=comparison,
            counts=comparison_counts(comparison), by_actual_source=source_counts,
            literal_text_equal=same(by_id[first]['index']['text'], by_id[second]['index']['text']),
            restoration_to_own_baseline=restoration,
            semantic_endpoint_tracking_established=False))
    pipeline.require(len(tables) == 552 and len(pairs) == 684, 'complete552/684 property bank required')
    grouped = {}
    for kind in sorted({p['original_pair']['kind'] for p in pairs}):
        counter = Counter()
        for pair in pairs:
            if pair['original_pair']['kind'] == kind:
                counter.update(pair['counts'])
        grouped[kind] = dict(pair_count=sum(p['original_pair']['kind'] == kind for p in pairs),
                             property_counts=dict(counter))
    return dict(schema=VERSION, template_opportunities=len(tables), pair_opportunities=len(pairs),
        property_tables=tables, pairs=pairs, summary_by_pair_kind=grouped,
        reference_or_source_review_judgments_read=False,
        semantic_endpoint_tracking_performed=False, complete_semantic_analysis=False,
        new_independent_observations=False, no_api_or_model_calls=True,
        limitations=[
            'Conditional on complete validated Stage1 inputs; binding/construction failure aborts this bank, '
            'not the original opportunity inventory. Full unavailable-measurement accounting is a separate assessment.',
            'Numerical/source comparison only; no report meaning or asserted predicate is established.',
            'Same peak set can be a true property null while its raw distribution changes.',
            'Source or represented-time changes yield unmatched identities, not invariance.',
            'Literal renderer tracking checks instrumentation, not spontaneous reporting or experience.',
            'Origin metadata uses the archived supplied provenance, not a new native-archive reconstruction.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--exports', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    pipeline.require(not args.output.exists(), 'refusing replacement of first property bank')
    result = assemble(pipeline.bound_json(args.index, preparation.INDEX_SHA),
                      pipeline.bound_json(args.exports, construction.EXPORT_SHA256))
    result['input_dependencies'] = dict(
        index=dict(path=str(args.index), sha256=preparation.INDEX_SHA),
        exports=dict(path=str(args.exports), sha256=construction.EXPORT_SHA256))
    modules = (construction, preparation, pipeline, construction.attribution,
               construction.exporter, construction.origins)
    result['source_sha256'] = {str(Path(m.__file__).resolve().relative_to(pipeline.ROOT)):
        pipeline.digest(Path(m.__file__).read_bytes()) for m in modules}
    result['source_sha256'][str(Path(__file__).resolve().relative_to(pipeline.ROOT))] = pipeline.digest(Path(__file__).read_bytes())
    pipeline.exclusive_json(args.output, result)


if __name__ == '__main__':
    main()
