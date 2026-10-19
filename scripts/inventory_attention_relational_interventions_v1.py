"""Descriptive source-only manipulation check; never reads generated outputs.

Developed after a read-only exploratory count during narrator collection.
This preserves all 684 registered pairs and changes no endpoint or denominator.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from inventory_attention_relational_origins_v1 import EXPORT_SHA256, classify
from attention_relational_reporting_pipeline_v1 import build_pair_index


def fields(content, provenance):
    sources = {p['panel']: p['source_key'] for p in provenance['panels']}
    result = {}
    for panel in content['panels']:
        contexts = [('current', None, panel['current'])]
        if panel['candidates'] is not None:
            contexts += [('candidate', cmd, nodes) for cmd, nodes in panel['candidates'].items()]
        for view, command, nodes in contexts:
            for node in nodes:
                for role in ('red', 'blue'):
                    key = (panel['panel'], view, command, node['node'], role)
                    category = classify(node['origin'])
                    source = sources[panel['panel']]
                    actual = source if category in ('source_current', 'source_conditional',
                        'inherited_source_current') else 'W' if category != 'sensor' else 'sensor'
                    result[key] = (actual, category, node['distributions'][role],
                                   node['derived'][role]['peaks'])
    return result


def inventory(exports):
    provenance = {p['id']: p for p in exports['provenance']}
    values = {p['id']: fields(p['content'], provenance[p['id']]) for p in exports['packets']}
    pairs = build_pair_index(exports['provenance'])
    assert len(pairs) == 684
    groups = defaultdict(Counter)
    donor_cases = defaultdict(dict)
    rows = []
    for pair in pairs:
        first, second = pair['first_id'], pair['second_id']
        a, b = values[first], values[second]
        assert a.keys() == b.keys()
        changed, peaks = [], []
        for key, left in a.items():
            right = b[key]
            assert left[:2] == right[:2]
            item = {'context': list(key), 'origin_source': left[0], 'origin_category': left[1]}
            if left[2] != right[2]:
                changed.append(item)
            if left[3] != right[3]:
                peaks.append(item)
        case = provenance[first]['case']
        group = groups[(pair['kind'], case['source'], case['view'])]
        group.update(pairs=1, pairs_with_distribution_changes=bool(changed),
                     pairs_with_peak_changes=bool(peaks), changed_distribution_atoms=len(changed),
                     changed_peak_atoms=len(peaks), fully_null_fields_and_peaks=not changed and not peaks)
        if pair['kind'] in ('current_donor', 'prospective_donor'):
            key = (case['encoder_seed'], case['world_index'])
            state = (bool(changed), bool(peaks))
            same_case = donor_cases[(pair['kind'], case['source'])]
            assert key not in same_case or same_case[key] == state
            same_case[key] = state
        if pair['kind'].endswith('baseline_to_restored'):
            assert not changed and not peaks
        assert not any(x['origin_category'] == 'copied_W11' for x in changed)
        rows.append(dict(pair, distribution_changes=changed, peak_changes=peaks))
    unique = []
    for (kind, source), cases in sorted(donor_cases.items()):
        assert len(cases) == 12
        unique.append(dict(kind=kind, source=source, archive_cases=12,
            cases_with_distribution_changes=sum(x[0] for x in cases.values()),
            cases_with_peak_changes=sum(x[1] for x in cases.values())))
    return {'schema': 'attention_relational_report_v1_source_intervention_inventory',
        'input_sha256': EXPORT_SHA256, 'reads_generated_outputs': False,
        'changes_registered_endpoints_or_denominators': False,
        'repeated_views_and_inherited_atoms_are_not_independent_observations': True,
        'interpretation': 'Peak-set nulls do not imply unchanged distributions or absent reportable organization.',
        'pair_count': len(rows), 'donor_unique_archive_cases': unique,
        'groups': [dict(kind=k[0], source=k[1], view=k[2], **v) for k, v in sorted(groups.items())],
        'pairs': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exports', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.exports.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPORT_SHA256:
        raise ValueError('not the preserved first export')
    result = inventory(json.loads(raw))
    result['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), Path(__file__).with_name('inventory_attention_relational_origins_v1.py'),
                  Path(__file__).with_name('attention_relational_reporting_pipeline_v1.py'))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'pair_count': result['pair_count'],
                      'donor_unique_archive_cases': result['donor_unique_archive_cases']}))


if __name__ == '__main__':
    main()
