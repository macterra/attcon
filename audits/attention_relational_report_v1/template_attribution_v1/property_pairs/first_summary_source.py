"""Root numerical summary only; no semantic endpoint adjudication."""
from collections import Counter
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parent
path = base / 'first_pairs.json'
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(path) == 'c944d239cd52863fefa9331fc91800d048362d38d2fd7ee2659dc3b41d5706a5'
data = json.loads(path.read_text())
restorations = [p['restoration_to_own_baseline'] for p in data['pairs']
                if p['restoration_to_own_baseline'] is not None]
per_case = {}
for pair in data['pairs']:
    case = str(pair['original_pair']['case'][:2])
    if case not in per_case:
        per_case[case] = dict(pair_count=0, property_counts=Counter())
    per_case[case]['pair_count'] += 1
    per_case[case]['property_counts'].update(pair['counts'])
summary = dict(schema='attention_relational_template_property_pairs_v1_first_summary',
    input=dict(path=str(path.relative_to(Path.cwd())), sha256=digest(path)),
    summary_source_sha256=digest(Path(__file__)),
    template_opportunities=data['template_opportunities'],
    pair_opportunities=data['pair_opportunities'],
    indexed_property_count=sum(len(t) for t in data['property_tables'].values()),
    indexed_property_count_by_actual_source=dict(Counter(
        p['identity']['actual_source'] for t in data['property_tables'].values() for p in t.values())),
    pair_property_counts=dict(sum((Counter(p['counts']) for p in data['pairs']), Counter())),
    summary_by_pair_kind=data['summary_by_pair_kind'],
    per_archived_case=per_case,
    second_restored_endpoint_checks=len(restorations),
    restoration_fields={k: dict(Counter(str(p[k]) for p in restorations)) for k in
        ('second_indexed_properties_equal_baseline', 'second_raw_peak_operands_equal_baseline',
         'second_literal_text_equal_baseline')},
    source_or_time_transition_is_not_invariance=True,
    independent_numerical_review_complete=False,
    semantic_endpoint_tracking_performed=False,
    complete_semantic_analysis=False,
    new_independent_observations=False,
    goal_achieved=False)
with (base / 'first_summary.json').open('x') as handle:
    json.dump(summary, handle, indent=2)
    handle.write('\n')
print(json.dumps({k: summary[k] for k in ('template_opportunities', 'pair_opportunities',
    'indexed_property_count', 'pair_property_counts', 'second_restored_endpoint_checks')}, indent=2))
