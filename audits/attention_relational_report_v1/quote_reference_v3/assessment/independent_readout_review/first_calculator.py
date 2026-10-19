"""Independent numerical readout; no project imports or text-semantic judgments."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json

ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / 'audits/attention_relational_report_v1/quote_reference_v3'
OUT = Path(__file__).parent
EXPECTED = {
    'derived/first_derived.json': 'd4d950c9eca3034a8a30e4883b6f55b0b8812af4540c310ed0a3e0b8a25d6dc7',
    'assessment/first_assessment.json': '18523e2a2e08562b50b69438d9125a30f845307abb4afdf7781b5cab84f18a76',
}
SCOPES = ['model_organization', 'model_access', 'claimed_perceptual_appearance', 'display_or_sensor_property', 'unclear']

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def load(relative):
    raw = (BASE / relative).read_bytes()
    if relative in EXPECTED:
        assert sha(raw) == EXPECTED[relative], relative
    return json.loads(raw), sha(raw)

derived, _ = load('derived/first_derived.json')
joined, _ = load('assessment/first_assessment.json')
diagnostic, diagnostic_sha = load('assessment/first_reference_diagnostic.json')
grades = {r['id']: r for r in joined['all_text_results']}
assert len(grades) == len(joined['all_text_results']) == len(derived['rows']) == 1104
assert len({r['id'] for r in derived['rows']}) == 1104

checks = []
def check(name, actual, expected):
    checks.append({'check': name, 'pass': actual == expected, 'actual': actual, 'expected': expected})

stats = {'narrator': Counter(), 'template': Counter()}
totals = {'narrator': Counter(), 'template': Counter()}
groups = defaultdict(Counter)
feature_counts = {'narrator': defaultdict(Counter), 'template': defaultdict(Counter)}
patterns, quote_instances, compatibility_instances = Counter(), Counter(), Counter()
quote_units = Counter()
all_binding_scopes = {'narrator': defaultdict(Counter), 'template': defaultdict(Counter)}
all_binding_times = {'narrator': defaultdict(Counter), 'template': defaultdict(Counter)}
established_narrators = []
worlds = set()
for row in derived['rows']:
    arm = 'template' if row['template'] else 'narrator'
    worlds.add((row['case']['encoder_seed'], row['case']['world_index']))
    target = grades[row['id']]
    for field in ('unit_id', 'text_sha256', 'case', 'template', 'actor_text_available'):
        assert target[field] == row[field], (row['id'], field)
    assert target['reference_established'] is row['established']
    assert target['rating_transport_complete'] is row['transport_complete']
    # Frozen literal fidelity versus preselected unestablished whole narrator fidelity.
    assert target['fidelity_established'] is row['template']
    assert row['established'] is bool(row['transport_complete'] and row['actor_text_available'] and row['derivation']['valid'])
    ann = row['canonical_annotation']
    assert (ann is not None) is row['established']
    stats[arm].update(opportunities=1, transport_complete=int(row['transport_complete']),
        actor_available=int(row['actor_text_available']), internally_valid_derivation=int(row['derivation']['valid']),
        established_reference=int(row['established']))
    stats[arm]['internally_valid_unavailable'] += int(row['derivation']['valid'] and not row['actor_text_available'])
    patterns[tuple(sorted(set(row['derivation']['errors'])))] += 1
    unit_error_names = set()
    for entry in row['derivation']['quote_diagnostics']:
        quote_instances.update(entry['errors'])
        unit_error_names.update(entry['errors'])
    quote_units.update(unit_error_names)
    semantic = row['derivation']['semantic_validation']
    if semantic:
        compatibility_instances.update(semantic['errors'])
    count = Counter(text_opportunities=1, valid_reference=int(row['established']),
        reference_unestablished=int(not row['established']), fidelity_established=int(row['template']),
        rating_transport_complete=int(row['transport_complete']))
    for feature in ('C', 'O', 'D', 'A'):
        status = ann['features'][feature]['status'] if ann else 'unestablished'
        assert target['features'][feature] == status
        feature_counts[arm][feature][status] += 1
    for endpoint in ('COD', 'CA'):
        composite = ann['composites'][endpoint] if ann else None
        status = composite['status'] if composite else 'unestablished'
        scopes = sorted({b['scope'] for b in composite['bindings']}) if composite else []
        times = sorted({b['time'] for b in composite['bindings']}) if composite else []
        if composite:
            all_binding_scopes[arm][endpoint].update(b['scope'] for b in composite['bindings'])
            all_binding_times[arm][endpoint].update(b['time'] for b in composite['bindings'])
        lower, upper = status == 'explicit', status in ('explicit', 'ambiguous')
        endpoint_value = dict(status=status, lower=lower, upper=upper,
            faithful_lower=lower and row['template'], faithful_upper=upper and row['template'], scopes=scopes, times=times)
        assert target['endpoints'][endpoint] == endpoint_value, (row['id'], endpoint)
        count[endpoint + '_status_' + status] += 1
        for metric in ('lower', 'upper', 'faithful_lower', 'faithful_upper'):
            flag = endpoint_value[metric]
            count[endpoint + '_' + metric] += int(flag)
            for scope in SCOPES:
                count[endpoint + '_' + scope + '_' + metric] += int(flag and scope in scopes)
    totals[arm].update(count)
    key = '|'.join(str(row['case'][field]) for field in ('encoder_seed', 'source', 'view', 'variant')) + '|' + arm
    groups[key].update(count)
    if row['established'] and not row['template']:
        established_narrators.append(dict(id=row['id'], case=row['case'], composites={name:
            dict(status=c['status'], binding_count=len(c['bindings']), scopes=sorted({b['scope'] for b in c['bindings']}))
            for name, c in ann['composites'].items()}))

for arm in stats:
    check(arm + ' denominator', stats[arm]['opportunities'], 552)
    check(arm + ' all joined totals', dict(totals[arm]), joined['arm_totals'][arm])
    base_stats = {key: value for key, value in stats[arm].items() if key != 'internally_valid_unavailable'}
    check(arm + ' diagnostic opportunity counts', base_stats, diagnostic['opportunity_counts'][arm])
    rates = dict(text_denominator=552)
    for endpoint in ('COD', 'CA'):
        for metric in ('lower', 'upper', 'faithful_lower', 'faithful_upper'):
            rates[endpoint + '_' + metric + '_rate'] = totals[arm][endpoint + '_' + metric] / 552
    check(arm + ' rates', rates, joined['arm_rates'][arm])
check('all joined groups', {key: dict(value) for key, value in groups.items()}, joined['groups'])
for key, value in groups.items():
    denominator = value['text_opportunities']
    rates = dict(text_denominator=denominator)
    for endpoint in ('COD', 'CA'):
        for metric in ('lower', 'upper', 'faithful_lower', 'faithful_upper'):
            rates[endpoint + '_' + metric + '_rate'] = value[endpoint + '_' + metric] / denominator
    assert rates == joined['group_rates'][key], key
all_totals = Counter()
for value in totals.values():
    all_totals.update(value)
check('joined grand totals', dict(all_totals), joined['totals'])
check('diagnostic error-pattern units', [dict(errors=list(key), unit_count=value) for key, value in sorted(patterns.items())], diagnostic['unit_error_patterns'])
check('diagnostic quote error instances', dict(quote_instances), diagnostic['quote_error_instances'])
check('diagnostic compatibility error instances', dict(compatibility_instances), diagnostic['frozen_compatibility_error_instances'])
check('complete established narrator identities/statuses/scopes', established_narrators, diagnostic['complete_established_narrator_inventory'])
check('unchanged supplementary version policy', (joined['original_primary_results_replaced'], joined['no_itemwise_auditor_selection'], joined['supplementary_same_12_cases']), (False, True, True))
check('factual summary dependency', joined['input_dependencies']['factual_summary']['sha256'], '4d78e6ac95fb318a491ba9ca689252525d3892aaa1541d8e0f78fee429d5e5c2')
check('independent development cases', len(worlds), 12)
diagnostic_source = (BASE / 'assessment/first_reference_diagnostic.py').read_bytes()
check('diagnostic source binding', sha(diagnostic_source), diagnostic['source_sha256'])
result = dict(schema='independent_v3_numerical_readout_review_v1',
    inputs={**EXPECTED, 'assessment/first_reference_diagnostic.json': diagnostic_sha},
    calculator_sha256=sha(Path(__file__).read_bytes()), arm_counts={key: dict(value) for key, value in stats.items()},
    arm_totals={key: dict(value) for key, value in totals.items()},
    feature_counts={arm: {key: dict(value) for key, value in features.items()} for arm, features in feature_counts.items()},
    all_established_binding_scope_instances={arm: {key: dict(value) for key, value in features.items()} for arm, features in all_binding_scopes.items()},
    all_established_binding_time_instances={arm: {key: dict(value) for key, value in features.items()} for arm, features in all_binding_times.items()},
    error_pattern_unit_counts=[dict(errors=list(key), units=value) for key, value in sorted(patterns.items())],
    quote_error_instances=dict(quote_instances), quote_error_unit_counts=dict(quote_units),
    compatibility_error_instances=dict(compatibility_instances), established_narrators=established_narrators,
    checks=checks, all_checks_pass=all(item['pass'] for item in checks),
    semantic_meaning_judged=False, raw_timeout_subtype_or_archive_revalidated=False,
    factual_supportedness_regraded=False, target_texts_or_quote_meanings_interpreted=False)
with (OUT / 'first_computation.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps(dict(all_checks_pass=result['all_checks_pass'], checks=len(checks), arm_counts=result['arm_counts'],
    error_patterns=result['error_pattern_unit_counts'], quote_error_units=result['quote_error_unit_counts'],
    all_binding_scopes=result['all_established_binding_scope_instances']), sort_keys=True))
assert result['all_checks_pass'], 'independent readout mismatch; preserve first output'
