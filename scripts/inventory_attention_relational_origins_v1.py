"""Count source origins in the preserved export; never reads narrator outputs.

This descriptive inventory changes no registered endpoint or acceptance rule.
Each atom is one required role/distribution peak-set reporting opportunity.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


EXPORT_SHA256 = '4f5f0bc2fa78b3b24b3580f52e306f20df0f3011d079b8139a2bc1f029050837'


def classify(origin):
    inherited = origin['kind'] == 'inherited_current'
    while origin['kind'] == 'inherited_current':
        origin = origin['inherited_origin']
    kind = origin['kind']
    if kind == 'history_estimate':
        return 'inherited_source_current' if inherited else 'source_current'
    if kind == 'command_estimate':
        return 'source_conditional'
    if kind == 'copied_prior_record':
        return 'copied_W11'
    if kind in ('completed_record', 'counterfactual_native_snapshot', 'native_conditional_record'):
        return 'native_W'
    if kind == 'fixed_observation_record':
        return 'sensor'
    raise ValueError('unknown origin: ' + kind)


def inventory(export):
    packets = export['packets']
    provenance = {p['id']: p for p in export['provenance']}
    totals = Counter()
    by_source = defaultdict(Counter)
    by_format = defaultdict(Counter)
    rows = []
    for packet, record in zip(packets, export['provenance'], strict=True):
        # The exporter preserves packet and provenance in matching fixed order.
        assert record == provenance[record['id']]
        assert packet['id'] == record['id']
        sources = {p['panel']: p['source_key'] for p in record['panels']}
        counts = Counter()
        for panel in packet['content']['panels']:
            source = sources[panel['panel']]
            contexts = [('current', panel['current'])]
            if panel['candidates'] is not None:
                contexts += [('candidate', nodes) for nodes in panel['candidates'].values()]
            for view, nodes in contexts:
                for node in nodes:
                    origin = classify(node['origin'])
                    # Both red and blue distributions each require one peak-set atom.
                    assert set(node['distributions']) == {'red', 'blue'}
                    totals[origin] += 2
                    by_source[source][origin] += 2
                    by_format[(record['case']['view'], source, view)][origin] += 2
                    counts[origin] += 2
        rows.append({'id': record['id'], 'case': record['case'], 'origin_atoms': dict(counts)})
    assert len(rows) == 552 and sum(totals.values()) == 25440
    assert sum(by_source['A'][k] for k in
               ('source_current', 'source_conditional', 'inherited_source_current')) == 2100
    return {
        'schema': 'attention_relational_report_v1_origin_inventory',
        'input_sha256': EXPORT_SHA256,
        'reads_narrator_auditor_or_rater_outputs': False,
        'changes_registered_endpoints_or_thresholds': False,
        'unit': 'required role/distribution peak-set opportunity, including repeated records',
        'independent_observations_claimed': False,
        'total_atoms': sum(totals.values()),
        'origin_atoms': dict(totals),
        'by_panel_source': {k: dict(v) for k, v in by_source.items()},
        'by_format_source_view': [dict(format=k[0], panel_source=k[1], view=k[2],
                                      origin_atoms=dict(v)) for k, v in sorted(by_format.items())],
        'reports': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exports', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.exports.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPORT_SHA256:
        raise ValueError('export bytes differ from the preserved first export')
    result = inventory(json.loads(raw))
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'total_atoms': result['total_atoms'],
                      'attention_estimate_atoms': 2100}))


if __name__ == '__main__':
    main()
