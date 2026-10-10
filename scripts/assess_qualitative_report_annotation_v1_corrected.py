"""Correct scope-inclusive grading; retain the original insufficient assessment."""
import json
from assess_qualitative_report_annotation_v1 import ROOT, assess, digest


def corrected_assessment():
    result = assess()
    fixtures = json.loads((ROOT/'fixtures.json').read_text())
    key = {c['id']: c['author_only']['proposed_scopes'] for c in fixtures['cases']}
    peer_scopes = {}
    rater_scopes = {}
    for rater in ('R1', 'R2'):
        cases = json.loads((ROOT/'raters'/rater/'first_pass.json').read_text())['cases']
        scopes = {c['id']: c['scope_labels'] for c in cases}
        rater_scopes[rater] = scopes
        mismatches = {identifier: {'author_proposed': expected, 'first_pass': scopes[identifier]}
                      for identifier, expected in key.items()
                      if set(scopes[identifier]) != set(expected)}
        original = result['raters'][rater]
        failed = set(original['author_key_disagreements']) | set(mismatches)
        original['label_only_full_tuples_correct'] = original['full_tuples_correct']
        original['label_only_accuracy'] = original['full_tuple_accuracy']
        original['scope_disagreements'] = mismatches
        original['full_cases_correct'] = 24-len(failed)
        original['full_case_accuracy'] = (24-len(failed))/24
        original['qualification_pass'] = len(failed) <= 2 and not original['protected_explicit_false_credit']
    for identifier in key:
        if set(rater_scopes['R1'][identifier]) != set(rater_scopes['R2'][identifier]):
            peer_scopes[identifier] = {r: rater_scopes[r][identifier] for r in ('R1', 'R2')}
    result.update(kind='corrected_scope_inclusive_author_key_qualification',
                  both_raters_qualify=all(r['qualification_pass'] for r in result['raters'].values()),
                  scope_label_inter_rater_disagreements=peer_scopes,
                  original_assessment_invalid=True,
                  original_assessment_sha256=digest(ROOT/'qualification.json'),
                  corrected_assessment_code_sha256=digest(__file__),
                  posthoc_scope_aliases_accepted=False,
                  correction_reason='Original parent scorer omitted the frozen scope-set requirement; scope vocabulary/delivery were inconsistent. First passes, protocol and original failed assessment are retained.')
    return result


if __name__ == '__main__':
    result = corrected_assessment()
    with (ROOT/'qualification_corrected.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'both_raters_qualify': result['both_raters_qualify'],
                      'scope_inclusive_correct': {r: v['full_cases_correct'] for r, v in result['raters'].items()}}))
