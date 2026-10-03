#!/usr/bin/env python3
"""Score extraction v11 and apply the frozen self_access_table_v1 decision rule."""
import json
from pathlib import Path
from self_coupled_gates import STRUCTURE, CHARACTERS, paired, fidelity, cases_from

ROOT = Path('audits/bound_content/self_access_table_v1')


def verdict_for(contrast):
    return contrast['difference'] >= .25 and contrast['one_sided_exact_p'] < .05


def decide(cases):
    primary = paired(cases, 'coupled_table', 'independent_table')
    interpretability = fidelity([c for c in cases if c['condition'] == 'coupled_table'])
    complete = primary['n'] == 24 and all(p['complete'] for p in primary['pairs'])
    if not complete: verdict = 'incomplete'
    elif not all(interpretability['gates'].values()): verdict = 'uninterpretable'
    elif verdict_for(primary): verdict = 'self_access_reporting_supported'
    else: verdict = 'self_access_reporting_not_supported'
    dissociation = paired(cases, 'coupled_table', 'coupled_table_swapped')
    table_effect = paired(cases, 'coupled_table', 'coupled_no_table')
    by_condition = {}
    for cond in sorted({c['condition'] for c in cases}):
        rows = [c for c in cases if c['condition'] == cond]
        by_condition[cond] = {'reports': len(rows), 'complete': sum(c['complete'] for c in rows),
            'structure_counts': {k: sum(bool(c['structure'].get(k)) for c in rows) for k in STRUCTURE},
            'character_counts': {k: sum(c['character'] == k for c in rows) for k in CHARACTERS}, 'fidelity': fidelity(rows)}
    return {'status': 'frozen automated decision rule; no human ratings implied', 'verdict': verdict, 'primary': primary,
            'interpretability': interpretability,
            'secondary_dissociation': {**dissociation, 'meets_primary_threshold': verdict_for(dissociation)},
            'secondary_table_effect': {**table_effect, 'meets_primary_threshold': verdict_for(table_effect)},
            'by_condition': by_condition}


def main():
    cases = cases_from(ROOT)
    (ROOT/'assessment_extraction_v11.json').write_text(json.dumps({'status': 'automated prose audit; no human ratings implied', 'cases': cases}, indent=2)+'\n')
    out = decide(cases); (ROOT/'gates.json').write_text(json.dumps(out, indent=2)+'\n')
    strip = lambda d: {k: v for k, v in d.items() if k != 'pairs'}
    print(json.dumps({'verdict': out['verdict'], 'primary': strip(out['primary']), 'dissociation': strip(out['secondary_dissociation']),
                      'table_effect': strip(out['secondary_table_effect']),
                      'by_condition': {k: {'structure': v['structure_counts'], 'character': v['character_counts']} for k, v in out['by_condition'].items()},
                      'interpretability': out['interpretability']['gates']}, indent=2))


if __name__ == '__main__': main()
