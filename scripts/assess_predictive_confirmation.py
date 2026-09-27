#!/usr/bin/env python3
"""Score frozen factual criteria and package prose for independent assessment."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from assess_predictive_language_pilot import FIELDS, matches

ROOT = Path('audits/predictive_attention/report_confirmation_v1')


def main():
    requests = json.loads((ROOT / 'requests.json').read_text())
    parsed, req_by_id, groups, review, unblinding = {}, {}, defaultdict(list), [], []
    totals = defaultdict(int)
    for req in requests:
        path = ROOT / (req['id'] + '.json')
        result = json.loads(path.read_text()) if path.exists() else {'status': 'missing'}
        p = result.get('parsed', {})
        parsed[req['id']] = p; req_by_id[req['id']] = req
        correctness = {k: bool(p) and matches(p.get(k), req['truth'][k], k) for k in FIELDS}
        groups[req['condition']].append({'id': req['id'], 'seed': req['seed'], 'style': req['style'],
            'complete': bool(p), 'exact': all(correctness.values()), 'fields': correctness,
            'original_model_exact': bool(p) and all(matches(p.get(k), req['original_truth'][k], k) for k in FIELDS)})
        usage = result.get('response', {}).get('usage') or {}
        for key in ('input_tokens', 'output_tokens', 'total_tokens'):
            totals[key] += usage.get(key, 0)
        if p:
            blind_id = hashlib.sha256(('confirmation-review-v1:' + req['id']).encode()).hexdigest()[:12]
            review.append({'blind_id': blind_id, 'report': p['report'], 'manner_of_access_0_2': '',
                'graded_presentation_0_2': '', 'agent_object_relation_0_2': '', 'report_character': '',
                'unsupported_experiential_assertion': '', 'notes': ''})
            unblinding.append({'blind_id': blind_id, 'request_id': req['id'], 'source': req['source'], 'truth': req['truth']})
    primary = sum((groups[k] for k in ('model', 'allocation', 'access', 'effects')), [])
    gates, seed_style = {}, {}
    for seed in (901, 911, 921):
        for style in ('neutral', 'styled'):
            rows = [r for r in primary if r['seed'] == seed and r['style'] == style]
            accuracy = sum(r['exact'] for r in rows) / len(rows)
            label = f'{seed}_{style}'
            seed_style[label] = {'correct': sum(r['exact'] for r in rows), 'total': len(rows), 'accuracy': accuracy}
            gates[label] = accuracy >= .95
    paired, restoration = [], []
    for req in requests:
        if req['condition'] not in ('allocation', 'access', 'effects', 'restored'):
            continue
        baseline_id = f"{req['seed']}_{req['episode']}_model_{req['style']}"
        base, p = parsed[baseline_id], parsed[req['id']]
        baseline_truth = req_by_id[baseline_id]['truth']
        accurate_pair = bool(base) and bool(p) and all(matches(base.get(k), baseline_truth[k], k) and matches(p.get(k), req['truth'][k], k) for k in FIELDS)
        if req['condition'] == 'restored':
            restoration.append(accurate_pair and all(base.get(k) == p.get(k) for k in FIELDS))
        else:
            paired.append(accurate_pair)
    gates['paired_intervention_accuracy'] = sum(paired) / len(paired) >= .95
    gates['restoration_exact'] = all(restoration)
    gates['constant_and_missing_correct'] = all(r['exact'] for k in ('constant', 'objects_only') for r in groups[k])
    summary = {'status': 'mechanical confirmation; report-character assessment pending',
        'gates': gates, 'all_mechanical_gates_pass': all(gates.values()), 'seed_style_primary': seed_style,
        'paired_intervention_correct': sum(paired), 'paired_intervention_total': len(paired),
        'restored_exact': sum(restoration), 'restored_total': len(restoration),
        'usage': dict(totals), 'estimated_usd': (totals['input_tokens'] * .25 + totals['output_tokens'] * 2) / 1000000,
        'conditions': {k: {'n': len(rows), 'completed': sum(r['complete'] for r in rows),
            'exact': sum(r['exact'] for r in rows), 'original_model_exact': sum(r['original_model_exact'] for r in rows), 'cases': rows} for k, rows in groups.items()}}
    (ROOT / 'assessment.json').write_text(json.dumps(summary, indent=2) + '\n')
    packet = ROOT / 'review'; packet.mkdir(exist_ok=True); review.sort(key=lambda r: r['blind_id'])
    with (packet / 'ratings.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(review[0])); w.writeheader(); w.writerows(review)
    (packet / 'unblinding_key.json').write_text(json.dumps(unblinding, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'conditions'}, indent=2))
    for k, v in summary['conditions'].items():
        print(k, {key: val for key, val in v.items() if key != 'cases'})


if __name__ == '__main__':
    main()
