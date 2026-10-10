"""Grade retained first passes against an explicitly author-proposed fixture key."""
import hashlib
import json
from pathlib import Path

ROOT = Path('audits/qualitative_report_annotation_v1')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess():
    freeze = json.loads((ROOT/'registration.json').read_text())
    for path, sha in freeze['sha256'].items():
        assert digest(path) == sha, path
    assert digest('docs/QUALITATIVE_REPORT_CORRESPONDENCE_V1_DESIGN.md') == freeze['design_sha256']
    fixtures = json.loads((ROOT/'fixtures.json').read_text())
    fields = fixtures['field_order']
    key = {c['id']: c for c in fixtures['cases']}
    ratings, results, hashes = {}, {}, {}
    for rater in ('R1', 'R2'):
        path = ROOT/'raters'/rater/'first_pass.json'
        raw = json.loads(path.read_text())
        ratings[rater] = {c['id']: c for c in raw['cases']}
        assert set(ratings[rater]) == set(key) and len(raw['cases']) == 24
        differences, false_credit, protected_ambiguities = {}, [], []
        for identifier, expected in key.items():
            case = ratings[rater][identifier]
            assert case['text'] == expected['rater_visible']['text']
            labels = case['labels']
            assert set(labels) == set(fields)
            assert set(labels.values()) <= {'explicit', 'ambiguous', 'absent'}
            mismatch = {f: {'author_proposed': expected['author_only']['proposed_labels'][f],
                            'first_pass': labels[f]} for f in fields
                        if labels[f] != expected['author_only']['proposed_labels'][f]}
            if mismatch:
                differences[identifier] = mismatch
            if identifier in ('Q01', 'Q02', 'Q03', 'Q04', 'Q05'):
                for field in ('M_access', 'M_presentation', 'M', 'P', 'P_access', 'P_presentation'):
                    if labels[field] == 'explicit':
                        false_credit.append({'id': identifier, 'field': field})
                    elif labels[field] == 'ambiguous':
                        protected_ambiguities.append({'id': identifier, 'field': field})
        correct = 24-len(differences)
        results[rater] = {'cases': 24, 'full_tuples_correct': correct,
                          'full_tuple_accuracy': correct/24,
                          'author_key_disagreements': differences,
                          'protected_cases': 5, 'protected_explicit_false_credit': false_credit,
                          'protected_ambiguities': protected_ambiguities,
                          'qualification_pass': correct >= 22 and not false_credit}
        hashes[str(path)] = digest(path)
    peer = {identifier: {f: {'R1': ratings['R1'][identifier]['labels'][f],
                             'R2': ratings['R2'][identifier]['labels'][f]} for f in fields
                        if ratings['R1'][identifier]['labels'][f] != ratings['R2'][identifier]['labels'][f]}
            for identifier in key}
    peer = {k: v for k, v in peer.items() if v}
    return {'kind': 'author_key_application_qualification', 'raters': results,
            'both_raters_qualify': all(r['qualification_pass'] for r in results.values()),
            'inter_rater_disagreements': peer, 'inter_rater_matching_fields': 312-sum(map(len, peer.values())),
            'inter_rater_fields': 312, 'full_matching_cases': 24-len(peer),
            'first_pass_sha256': hashes, 'registration_sha256': digest(ROOT/'registration.json'),
            'assessment_code_sha256': digest(__file__),
            'author_key_is_validated_gold_standard': False,
            'authentic_anchors_are_provisional_not_scored_as_trials': True,
            'source_fidelity_qualified': False, 'source_counterpart_qualified': False,
            'new_target_reports': 0, 'consciousness_related_correspondence_established': False,
            'goal_achieved': False}


if __name__ == '__main__':
    result = assess()
    with (ROOT/'qualification.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'both_raters_qualify': result['both_raters_qualify'],
                      'correct': {k: v['full_tuples_correct'] for k, v in result['raters'].items()},
                      'matching_fields': result['inter_rater_matching_fields'], 'fields': 312}))
