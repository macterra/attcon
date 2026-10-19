"""Describe the complete frozen first derivation; no repairs or semantic grading."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json

BASE = Path('audits/attention_relational_report_v1/quote_reference_v3')


def load(path, expected):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    return json.loads(raw)


derived = load(BASE / 'derived/first_derived.json', 'd4d950c9eca3034a8a30e4883b6f55b0b8812af4540c310ed0a3e0b8a25d6dc7')
assessment = load(BASE / 'assessment/first_assessment.json', '18523e2a2e08562b50b69438d9125a30f845307abb4afdf7781b5cab84f18a76')
assert len(derived['rows']) == len(assessment['all_text_results']) == 1104
arms = defaultdict(Counter)
patterns, grounding, compatibility = Counter(), Counter(), Counter()
valid_narrators = []
for row in derived['rows']:
    arm = 'template' if row['template'] else 'narrator'
    arms[arm].update(opportunities=1, transport_complete=int(row['transport_complete']),
        actor_available=int(row['actor_text_available']), internally_valid_derivation=int(row['derivation']['valid']),
        established_reference=int(row['established']))
    errors = row['derivation']['errors']
    patterns[tuple(sorted(set(errors)))] += 1
    for item in row['derivation']['quote_diagnostics']:
        grounding.update(item['errors'])
    semantic = row['derivation']['semantic_validation']
    if semantic:
        compatibility.update(semantic['errors'])
    if row['established'] and not row['template']:
        annotation = row['canonical_annotation']
        valid_narrators.append(dict(id=row['id'], case=row['case'],
            composites={name: dict(status=value['status'], binding_count=len(value['bindings']),
                scopes=sorted({b['scope'] for b in value['bindings']}))
                for name, value in annotation['composites'].items()}))
result = dict(schema='attention_relational_quote_v3_first_complete_reference_diagnostic',
    inputs=dict(derived_sha256='d4d950c9eca3034a8a30e4883b6f55b0b8812af4540c310ed0a3e0b8a25d6dc7',
        assessment_sha256='18523e2a2e08562b50b69438d9125a30f845307abb4afdf7781b5cab84f18a76'),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    opportunity_counts={k: dict(v) for k, v in arms.items()},
    unit_error_patterns=[dict(errors=list(k), unit_count=v) for k, v in sorted(patterns.items())],
    quote_error_instances=dict(grounding), frozen_compatibility_error_instances=dict(compatibility),
    complete_established_narrator_inventory=valid_narrators,
    arm_totals=assessment['arm_totals'], arm_rates=assessment['arm_rates'],
    error_instance_counts_are_not_independent_report_counts=True,
    internally_valid_but_unavailable_actor_is_not_established_reference=True,
    raw_annotations_repaired_or_salvaged=False, source_semantics_reviewed=False,
    narrator_fidelity_established=False, independent_cases=12,
    new_system_confirmation_or_consciousness_result=False)
with (BASE / 'assessment/first_reference_diagnostic.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps(result['opportunity_counts'], sort_keys=True))
