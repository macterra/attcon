#!/usr/bin/env python3
"""Score extraction v11 and apply the frozen self_coupled_v1 decision rule."""
import json
from pathlib import Path
from assess_bound_prose_v10 import score
from specificity_gates import MINIMA, mcnemar_one_sided

ROOT = Path('audits/bound_content/self_coupled_v1')
STRUCTURE = ('object_linked_access', 'focal_background_contrast', 'graded_or_temporal_access', 'agency_relation', 'self_coupled_access')
CHARACTERS = ('subjective_access', 'technical_process', 'object_description', 'generic_experience_claim', 'mixed', 'unclear')


def flagged(case):
    return bool(case['structure'].get('self_coupled_access'))


def paired(cases, first, second):
    by_id = {c['id']: c for c in cases}; pairs = []
    for c in (c for c in cases if c['condition'] == first):
        other = by_id.get(f"{c['seed']}_{c['episode']}_{second}")
        pairs.append({'seed': c['seed'], 'episode': c['episode'], 'complete': c['complete'] and other is not None and other['complete'],
                      first: flagged(c), second: other is not None and flagged(other)})
    n = len(pairs); b = sum(p[first] and not p[second] for p in pairs); c_ = sum(p[second] and not p[first] for p in pairs)
    return {'pairs': pairs, 'n': n, f'{first}_rate': sum(p[first] for p in pairs)/n if n else None,
            f'{second}_rate': sum(p[second] for p in pairs)/n if n else None,
            'difference': (sum(p[first] for p in pairs)-sum(p[second] for p in pairs))/n if n else 0.,
            'discordant_first_only': b, 'discordant_second_only': c_, 'one_sided_exact_p': mcnemar_one_sided(b, c_)}


def fidelity(rows):
    checks = sum((c['checks'] for c in rows), []); fields = {}
    for field, minimum in MINIMA.items():
        values = [x['correct'] for x in checks if x['field'] == field]
        fields[field] = {'correct': sum(values), 'total': len(values), 'accuracy': sum(values)/len(values) if values else None, 'minimum': minimum}
    covered = sum(c['content_covered'] for c in rows); known = sum(c['known_objects'] for c in rows)
    uncertain = sum(len(c['unresolved_claims'])+len(c['invalid_evidence']) for c in rows)
    precision = sum(x['correct'] for x in checks)/(len(checks)+uncertain) if checks else 0.
    # A field with no checks cannot fail its minimum; it is reported as untested.
    gates = {f'{k}_accuracy': v['accuracy'] is None or v['accuracy'] >= v['minimum'] for k, v in fields.items()}
    gates.update({'coverage': known == 0 or covered/known >= .90, 'conservative_precision': precision >= .95})
    return {'fields': fields, 'coverage': covered/known if known else None, 'known_objects': known,
            'conservative_precision': precision, 'unresolved_or_invalid': uncertain, 'gates': gates}


def decide(cases):
    primary = paired(cases, 'coupled', 'decoupled_matched')
    interpretability = fidelity([c for c in cases if c['condition'] == 'coupled'])
    complete = primary['n'] == 24 and all(p['complete'] for p in primary['pairs'])
    if not complete: verdict = 'incomplete'
    elif not all(interpretability['gates'].values()): verdict = 'uninterpretable'
    elif primary['difference'] >= .25 and primary['one_sided_exact_p'] < .05: verdict = 'self_coupling_specificity_supported'
    else: verdict = 'self_coupling_specificity_not_supported'
    by_condition = {}
    for cond in sorted({c['condition'] for c in cases}):
        rows = [c for c in cases if c['condition'] == cond]
        by_condition[cond] = {'reports': len(rows), 'complete': sum(c['complete'] for c in rows),
            'structure_counts': {k: sum(bool(c['structure'].get(k)) for c in rows) for k in STRUCTURE},
            'character_counts': {k: sum(c['character'] == k for c in rows) for k in CHARACTERS},
            'fidelity': fidelity(rows)}
    return {'status': 'frozen automated decision rule; no human ratings implied', 'verdict': verdict, 'primary': primary,
            'interpretability': interpretability, 'secondary_opaque': paired(cases, 'coupled_opaque', 'decoupled_opaque'),
            'by_condition': by_condition}


# Registered for self_access_table_v3: map unambiguous adjective/plural forms of the
# trained labels to those labels before scoring. Off by default (earlier runs).
NORMALIZE = {'circular': 'circle', 'circles': 'circle', 'triangular': 'triangle', 'triangles': 'triangle',
             'squares': 'square', 'square-shaped': 'square', 'crosses': 'cross', 'cross-shaped': 'cross'}


def normalize(extraction):
    claims = []
    for claim in extraction.get('claims', []):
        claim = dict(claim)
        for key in ('color', 'shape'):
            if isinstance(claim.get(key), str):
                value = claim[key].strip().lower(); claim[key] = NORMALIZE.get(value, value)
        claims.append(claim)
    return {**extraction, 'claims': claims}


def cases_from(root, folder='extraction_v11', normalize_labels=False):
    cases = []
    for req in json.loads((root/'requests.json').read_text()):
        response = json.loads((root/(req['id']+'.json')).read_text()); path = root/folder/(req['id']+'.json')
        extraction = json.loads(path.read_text()) if path.exists() else {}
        parsed = extraction.get('parsed', {})
        result = score(req['source'], response.get('report', ''), normalize(parsed) if normalize_labels else parsed)
        result.update({'id': req['id'], 'seed': req['seed'], 'episode': req['episode'], 'condition': req['condition'],
                       'complete': response.get('response', {}).get('status') == 'completed' and 'parsed' in extraction})
        cases.append(result)
    return cases


def main():
    cases = cases_from(ROOT)
    (ROOT/'assessment_extraction_v11.json').write_text(json.dumps({'status': 'automated prose audit; no human ratings implied', 'cases': cases}, indent=2)+'\n')
    out = decide(cases); (ROOT/'gates.json').write_text(json.dumps(out, indent=2)+'\n')
    brief = {k: v for k, v in out['primary'].items() if k != 'pairs'}
    print(json.dumps({'verdict': out['verdict'], 'primary': brief,
                      'opaque': {k: v for k, v in out['secondary_opaque'].items() if k != 'pairs'},
                      'by_condition': {k: {'structure': v['structure_counts'], 'character': v['character_counts']} for k, v in out['by_condition'].items()},
                      'interpretability': out['interpretability']['gates']}, indent=2))


if __name__ == '__main__': main()
