#!/usr/bin/env python3
"""Apply the frozen self_label_v1 decision rule (replication, then self versus label)."""
import json
from pathlib import Path
from self_coupled_gates import CHARACTERS, fidelity, cases_from
from specificity_gates import mcnemar_one_sided

ROOT = Path('audits/bound_content/self_label_v1')
CONFIG = Path('configs/bound_content/self_label_v1.json')
EPISODES = 48
FLAGS = ('object_linked_access', 'focal_background_contrast', 'graded_or_temporal_access', 'agency_relation',
         'self_coupled_access', 'stated_content_dependence')


def contrast(cases, first, second, key):
    by_id = {c['id']: c for c in cases}; pairs = []
    for c in (c for c in cases if c['condition'] == first):
        other = by_id.get(f"{c['seed']}_{c['episode']}_{second}")
        pairs.append({'seed': c['seed'], 'episode': c['episode'], 'complete': c['complete'] and other is not None and other['complete'],
                      'first': bool(c['structure'].get(key)), 'second': other is not None and bool(other['structure'].get(key))})
    n = len(pairs); b = sum(p['first'] and not p['second'] for p in pairs); c_ = sum(p['second'] and not p['first'] for p in pairs)
    difference = (sum(p['first'] for p in pairs)-sum(p['second'] for p in pairs))/n if n else 0.
    p_value = mcnemar_one_sided(b, c_)
    return {'first': first, 'second': second, 'flag': key, 'n': n, 'first_count': sum(p['first'] for p in pairs),
            'second_count': sum(p['second'] for p in pairs), 'difference': difference, 'discordant_first_only': b,
            'discordant_second_only': c_, 'one_sided_exact_p': p_value, 'meets_threshold': difference >= .25 and p_value < .05,
            'complete': n == EPISODES and all(p['complete'] for p in pairs), 'pairs': pairs}


def decide(cases):
    replication = contrast(cases, 'coupled_table', 'independent_table', 'self_coupled_access')
    label_dependence = contrast(cases, 'coupled_table', 'external_table', 'stated_content_dependence')
    label_attribution = contrast(cases, 'coupled_table', 'external_table', 'self_coupled_access')
    interpretability = fidelity([c for c in cases if c['condition'] == 'coupled_table'])
    if not (replication['complete'] and label_dependence['complete']): verdict = 'incomplete'
    elif not all(interpretability['gates'].values()): verdict = 'uninterpretable'
    elif not replication['meets_threshold']: verdict = 'self_access_reporting_not_replicated'
    elif label_dependence['meets_threshold']: verdict = 'replicated_and_self_specific_beyond_label'
    else: verdict = 'replicated_attribution_follows_label'
    by_condition = {}
    for cond in sorted({c['condition'] for c in cases}):
        rows = [c for c in cases if c['condition'] == cond]
        by_condition[cond] = {'reports': len(rows), 'complete': sum(c['complete'] for c in rows),
            'structure_counts': {k: sum(bool(c['structure'].get(k)) for c in rows) for k in FLAGS},
            'character_counts': {k: sum(c['character'] == k for c in rows) for k in CHARACTERS}, 'fidelity': fidelity(rows)}
    return {'status': 'frozen automated decision rule; no human ratings implied', 'verdict': verdict,
            'replication': replication, 'label_dependence': label_dependence, 'label_attribution': label_attribution,
            'interpretability': interpretability, 'by_condition': by_condition}


def main():
    config = json.loads(CONFIG.read_text())
    cases = cases_from(ROOT, f"extraction_{config['extractor']}", config.get('normalize_labels', False))
    (ROOT/f"assessment_extraction_{config['extractor']}.json").write_text(json.dumps({'status': 'automated prose audit; no human ratings implied', 'cases': cases}, indent=2)+'\n')
    out = decide(cases); (ROOT/'gates.json').write_text(json.dumps(out, indent=2)+'\n')
    strip = lambda d: {k: v for k, v in d.items() if k != 'pairs'}
    print(json.dumps({'verdict': out['verdict'], 'replication': strip(out['replication']), 'label_dependence': strip(out['label_dependence']),
                      'label_attribution': strip(out['label_attribution']), 'interpretability': out['interpretability']['gates'],
                      'by_condition': {k: {'flags': v['structure_counts'], 'character': v['character_counts']} for k, v in out['by_condition'].items()}}, indent=2))


if __name__ == '__main__': main()
