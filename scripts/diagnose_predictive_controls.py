#!/usr/bin/env python3
"""Post-hoc descriptive breakdown; does not alter any registered gate."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from assess_predictive_language_pilot import FIELDS, matches

root = Path('audits/predictive_attention/report_confirmation_v2')
requests = json.loads((root / 'requests.json').read_text())
result = {'status': 'post-hoc descriptive breakdown; no new gate', 'conditions': {}}
for condition in ('physical', 'history_predictor', 'shuffled', 'constant', 'objects_only'):
    rows = [r for r in requests if r['condition'] == condition]
    counts = dict.fromkeys(FIELDS, 0)
    for req in rows:
        parsed = json.loads((root / (req['id'] + '.json')).read_text())['parsed']
        for key in FIELDS:
            counts[key] += matches(parsed[key], req['original_truth'][key], key)
    result['conditions'][condition] = {'reports': len(rows), 'matches_original_A_by_field': counts}
result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
Path('audits/predictive_attention/control_content_diagnostic.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result['conditions'], indent=2))
