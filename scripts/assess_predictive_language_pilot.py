#!/usr/bin/env python3
"""Mechanical fidelity and blinded review preparation; no automated qualia verdict."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path('audits/predictive_attention/language_pilot_v1')
FIELDS = ('focal', 'access_now', 'access_after_delay', 'responsive_channel')


def matches(value, target, key):
    if value is None or target is None:
        return value is target
    if key.startswith('access'):
        return isinstance(value, (int, float)) and abs(value - target) <= .01
    return type(value) is type(target) and value == target


def main():
    requests = json.loads((ROOT / 'requests.json').read_text())
    groups, review, key_rows = defaultdict(list), [], []
    tokens = defaultdict(int)
    for req in requests:
        path = ROOT / (req['id'] + '.json')
        result = json.loads(path.read_text()) if path.exists() else {'status': 'missing'}
        parsed = result.get('parsed', {})
        match = {k: matches(parsed.get(k), req['truth'][k], k) for k in FIELDS}
        if not parsed:
            match = dict.fromkeys(FIELDS, False)
        original = {k: matches(parsed.get(k), req['original_truth'][k], k) for k in FIELDS}
        groups[req['condition']].append({'id': req['id'], 'status': result['status'],
            'complete': bool(parsed), 'supplied_exact': all(match.values()),
            'original_exact': bool(parsed) and all(original.values()), 'fields': match})
        usage = result.get('response', {}).get('usage') or {}
        for k in ('input_tokens', 'output_tokens', 'total_tokens'):
            tokens[k] += usage.get(k, 0)
        if parsed:
            blind_id = hashlib.sha256(('blind-review-v1:' + req['id']).encode()).hexdigest()[:12]
            review.append({'blind_id': blind_id, 'report': parsed['report'],
                'manner_of_access_0_2': '', 'graded_presentation_0_2': '',
                'agent_object_relation_0_2': '', 'report_character': '',
                'unsupported_experiential_assertion': '', 'notes': ''})
            key_rows.append({'blind_id': blind_id, 'request_id': req['id'], 'source': req['source'],
                             'truth': req['truth']})
    summary = {'status': 'pilot; independent review pending', 'tokens': dict(tokens), 'conditions': {}}
    for condition, rows in groups.items():
        summary['conditions'][condition] = {'requests': len(rows),
            'completed': sum(r['complete'] for r in rows),
            'supplied_state_exact': sum(r['supplied_exact'] for r in rows) / len(rows),
            'original_model_exact': sum(r['original_exact'] for r in rows) / len(rows),
            'field_accuracy': {k: sum(r['fields'][k] for r in rows) / len(rows) for k in FIELDS},
            'cases': rows}
    summary['estimated_usd_at_registered_prices'] = (tokens['input_tokens'] * .25 + tokens['output_tokens'] * 2) / 1000000
    (ROOT / 'assessment.json').write_text(json.dumps(summary, indent=2) + '\n')
    review.sort(key=lambda x: x['blind_id'])
    packet = ROOT / 'review'
    packet.mkdir(exist_ok=True)
    if review:
        with (packet / 'ratings.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(review[0])); w.writeheader(); w.writerows(review)
    (packet / 'unblinding_key.json').write_text(json.dumps(key_rows, indent=2) + '\n')
    (packet / 'README.md').write_text('''# Independent report review\n\nStatus: ratings pending. `ratings.csv` contains anonymized reports in a fixed\nshuffled order. Use the rubric at `docs/PREDICTIVE_REPORT_RUBRIC.md`; do not inspect\nthe unblinding key or source records until the initial ratings are complete.\nRecord your identity or pseudonym, relevant expertise, and any prior exposure\nto conditions. Return a separate copy rather than overwrite the blank template.\n\nThe key is public for reproducibility, so blinding depends on the reviewer not\nopening it. After first-pass ratings, a facilitator can reveal source records\nwithout condition labels for prose-fidelity ratings. These files contain no\nhuman ratings and do not establish subjective experience.\n''')
    print(json.dumps({k: v for k, v in summary.items() if k != 'conditions'}, indent=2))
    for condition, values in summary['conditions'].items():
        print(condition, {k: v for k, v in values.items() if k != 'cases'})


if __name__ == '__main__':
    main()
