"""Frozen complete-case grading, including canonical scopes and exact evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path('audits/qualitative_report_annotation_v2')
SCOPES = {'display', 'sensor', 'model_presentation', 'model_representation_access',
          'claimed_experiential', 'unclear'}
VALUES = {'explicit', 'ambiguous', 'absent'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_errors(case, text, fields):
    errors = []
    if set(case) != {'id', 'labels', 'scope_labels', 'evidence', 'notes', 'role_scope_notes'}:
        errors.append('case schema fields differ')
    labels = case.get('labels', {})
    if not isinstance(labels, dict) or set(labels) != set(fields) or not set(labels.values()) <= VALUES:
        errors.append('invalid labels')
    scopes = case.get('scope_labels', [])
    if not isinstance(scopes, list) or any(not isinstance(s, str) for s in scopes):
        errors.append('invalid scope array')
    elif scopes != sorted(set(scopes)) or not set(scopes) <= SCOPES:
        errors.append('noncanonical, duplicate or unsorted scopes')
    if not isinstance(case.get('notes'), str) or not isinstance(case.get('role_scope_notes'), str):
        errors.append('invalid notes')
    evidence = case.get('evidence', [])
    if not isinstance(evidence, list):
        return errors+['invalid evidence array']
    for index, span in enumerate(evidence):
        try:
            assert set(span) == {'fields', 'start', 'end', 'quote', 'referent', 'attribute', 'time', 'polarity'}
            assert isinstance(span['fields'], list) and span['fields']
            assert len(span['fields']) == len(set(span['fields']))
            assert set(span['fields']) <= set(fields) | {'scope_labels'}
            assert type(span['start']) is int and type(span['end']) is int
            assert 0 <= span['start'] <= span['end'] <= len(text)
            assert all(isinstance(span[k], str) for k in ('quote', 'referent', 'attribute', 'time'))
            assert text[span['start']:span['end']] == span['quote']
            assert span['polarity'] in {'affirmed', 'negated', 'mixed', 'unclear', 'not_applicable'}
        except (AssertionError, TypeError, KeyError):
            errors.append(f'invalid exact evidence span {index}')
    return errors


def assess():
    registration = json.loads((ROOT/'registration.json').read_text())
    for path, sha in registration['sha256'].items():
        assert digest(path) == sha, path
    fixtures = json.loads((ROOT/'fixtures.json').read_text())
    fields = fixtures['field_order']
    results, ratings, hashes = {}, {}, {}
    for rater in ('R1', 'R2'):
        path = ROOT/'raters'/rater/'first_pass.json'
        raw = json.loads(path.read_text())
        header_valid = (set(raw) == {'schema_version', 'rater_id', 'exposure', 'cases'} and
                        raw['schema_version'] == 'qualitative-report-annotations-v2' and
                        raw['rater_id'] == rater and isinstance(raw['exposure'], dict))
        cases = raw.get('cases', [])
        ids = [c.get('id') for c in cases]
        ratings[rater] = {c.get('id'): c for c in cases}
        exact_inventory = len(cases) == 24 and set(ids) == {c['id'] for c in fixtures['cases']} and len(set(ids)) == 24
        differences, guards, label_correct, full_correct = {}, [], 0, 0
        for expected in fixtures['cases']:
            identifier = expected['id']
            key = expected['author_only']
            case = ratings[rater].get(identifier, {})
            errors = record_errors(case, expected['rater_visible']['text'], fields)
            if not header_valid or not exact_inventory:
                errors.append('invalid header/inventory')
            labels = case.get('labels', {})
            mismatch = {f: {'author_proposed': key['proposed_labels'][f], 'first_pass': labels.get(f)}
                        for f in fields if key['proposed_labels'][f] != labels.get(f)}
            label_correct += not bool(mismatch)
            scopes = case.get('scope_labels', [])
            scope_equal = scopes == sorted(key['proposed_scopes'])
            match = not errors and not mismatch and scope_equal
            full_correct += match
            if not match:
                differences[identifier] = {'labels': mismatch, 'schema_errors': errors,
                                          'scopes': {'author_proposed': key['proposed_scopes'], 'first_pass': scopes}}
            if identifier in ('Q01', 'Q02', 'Q03', 'Q04', 'Q05'):
                guards.extend({'id': identifier, 'field': f} for f in
                              ('M_access', 'M_presentation', 'M', 'P', 'P_access', 'P_presentation')
                              if labels.get(f) == 'explicit')
        results[rater] = {'cases': 24, 'label_only_correct': label_correct,
                          'full_cases_correct': full_correct, 'full_case_accuracy': full_correct/24,
                          'disagreements': differences, 'protected_cases': 5,
                          'protected_explicit_false_credit': guards,
                          'qualification_pass': full_correct >= 22 and not guards}
        hashes[str(path)] = digest(path)
    peer = {c['id']: {'labels_differ': ratings['R1'].get(c['id'], {}).get('labels') != ratings['R2'].get(c['id'], {}).get('labels'),
                     'scopes_differ': ratings['R1'].get(c['id'], {}).get('scope_labels') != ratings['R2'].get(c['id'], {}).get('scope_labels')}
            for c in fixtures['cases']}
    return {'kind': 'canonical_scope_complete_case_qualification_v2', 'raters': results,
            'both_raters_qualify': all(r['qualification_pass'] for r in results.values()),
            'inter_rater_differences': {k: v for k, v in peer.items() if any(v.values())},
            'first_pass_sha256': hashes, 'assessment_code_sha256': digest(__file__),
            'registration_sha256': digest(ROOT/'registration.json'), 'author_key_validated': False,
            'reused_synthetic_cases': 24, 'new_target_reports': 0, 'source_fidelity_qualified': False,
            'source_counterpart_qualified': False, 'goal_achieved': False}


if __name__ == '__main__':
    value = assess()
    with (ROOT/'qualification.json').open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'both_raters_qualify': value['both_raters_qualify'],
                      'full_case_correct': {k: v['full_cases_correct'] for k, v in value['raters'].items()}}))
