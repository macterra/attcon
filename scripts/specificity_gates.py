#!/usr/bin/env python3
"""Apply the frozen specificity_v1 decision rule to the automated prose audit."""
import json
from math import comb
from pathlib import Path

ROOT = Path('audits/bound_content/specificity_v1')
STRUCTURE = ('object_linked_access', 'focal_background_contrast', 'graded_or_temporal_access', 'agency_relation')
CHARACTERS = ('subjective_access', 'technical_process', 'object_description', 'generic_experience_claim', 'mixed', 'unclear')
MINIMA = {'color': .98, 'shape': .98, 'focal': .95, 'most_recoverable': .95, 'access_trend': .95, 'under_own_control': .95, 'command_next': .95}


def primary(case):
    s = case['structure']
    return bool(s.get('object_linked_access') and s.get('graded_or_temporal_access') and case['character'] in ('subjective_access', 'mixed'))


def mcnemar_one_sided(b, c):
    """P(X >= b) for X ~ Binomial(b + c, 1/2): exact one-sided paired test."""
    n = b+c
    return 1. if n == 0 else sum(comb(n, k) for k in range(b, n+1))/2**n


def decide(cases):
    by_id = {c['id']: c for c in cases}
    model = [c for c in cases if c['condition'] == 'model']
    # Interpretability: v5 fidelity minima, pooled over the three seeds, on `model`.
    checks = sum((c['checks'] for c in model), [])
    fields = {}
    for field, minimum in MINIMA.items():
        values = [x['correct'] for x in checks if x['field'] == field]
        fields[field] = {'correct': sum(values), 'total': len(values), 'accuracy': sum(values)/len(values) if values else 0., 'minimum': minimum}
    covered = sum(c['content_covered'] for c in model); known = sum(c['known_objects'] for c in model)
    uncertain = sum(len(c['unresolved_claims'])+len(c['invalid_evidence']) for c in model)
    precision = sum(x['correct'] for x in checks)/(len(checks)+uncertain) if checks else 0.
    fidelity = {f'{k}_accuracy': v['accuracy'] >= v['minimum'] for k, v in fields.items()}
    fidelity.update({'coverage': known > 0 and covered/known >= .90, 'conservative_precision': precision >= .95})
    # Primary paired contrast: model vs external, by episode.
    pairs = []
    for c in model:
        other = by_id.get(f"{c['seed']}_{c['episode']}_external")
        pairs.append({'seed': c['seed'], 'episode': c['episode'], 'complete': c['complete'] and other is not None and other['complete'],
                      'model': primary(c), 'external': other is not None and primary(other)})
    complete = all(p['complete'] for p in pairs) and len(pairs) == 24
    b = sum(p['model'] and not p['external'] for p in pairs); c_ = sum(p['external'] and not p['model'] for p in pairs)
    n = len(pairs); difference = (sum(p['model'] for p in pairs)-sum(p['external'] for p in pairs))/n if n else 0.
    p_value = mcnemar_one_sided(b, c_)
    if not complete: verdict = 'incomplete'
    elif not all(fidelity.values()): verdict = 'uninterpretable'
    elif difference >= .25 and p_value < .05: verdict = 'specificity_supported'
    else: verdict = 'specificity_not_supported'
    conditions = sorted({c['condition'] for c in cases})
    by_condition = {}
    for cond in conditions:
        rows = [c for c in cases if c['condition'] == cond]
        by_condition[cond] = {'reports': len(rows), 'complete': sum(c['complete'] for c in rows), 'primary_conjunction': sum(primary(c) for c in rows),
            'structure_counts': {k: sum(bool(c['structure'].get(k)) for c in rows) for k in STRUCTURE},
            'character_counts': {k: sum(c['character'] == k for c in rows) for k in CHARACTERS}}
    return {'status': 'frozen automated decision rule; no human ratings implied', 'verdict': verdict,
            'primary': {'model_rate': sum(p['model'] for p in pairs)/n if n else None, 'external_rate': sum(p['external'] for p in pairs)/n if n else None,
                        'difference': difference, 'discordant_model_only': b, 'discordant_external_only': c_, 'one_sided_exact_p': p_value,
                        'threshold_difference': .25, 'threshold_p': .05, 'pairs': pairs},
            'interpretability': {'fields': fields, 'coverage': covered/known if known else None, 'conservative_precision': precision,
                                 'unresolved_or_invalid': uncertain, 'gates': fidelity},
            'by_condition': by_condition}


def main():
    audit = json.loads((ROOT/'assessment_extraction_v10.json').read_text())
    out = decide(audit['cases'])
    (ROOT/'gates.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({k: out[k] for k in ('verdict', 'by_condition')} | {'primary': {k: v for k, v in out['primary'].items() if k != 'pairs'}}, indent=2))


if __name__ == '__main__': main()
