"""Check exact first-grade inventories and summarize; never grade semantics anew.

Stdlib only. Reads pinned first artifacts, validates complete identities and null
dispositions, compares both first semantic judgments, and writes exclusively.
No annotation repair, reclassification, model calls or target release.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


BASE = Path('audits/attention_relational_report_v1/subagent_auditor_v5')
QUAL = BASE / 'qualification'
PINNED = {
    str(BASE / 'qualification_grader_instructions.txt'): 'c149840def2a4e1347673ce34710f5c972a797592f1c636451b6e624e845b6c0',
    str(QUAL / 'preparation/preparation.json'): '85c8660dbf3ea1e196bddfa10d0f8b35d2211b7c44cfe55cea2abb7bab4f81e6',
    str(QUAL / 'collection/first_collection.json'): 'dbbea70ddceab7235581c1e219dd72ae8c0fcca48ee1b5dbf58f832ff9aa81d7',
    str(QUAL / 'derived/first_derived.json'): '548d44f2af12689336591e2c50d2c72a1c51c04b48596c1165b27501b36f7234',
    'audits/attention_relational_report_v1/grounded_auditor_v4/qualification/first_materialization/hidden_atomic_key.json': 'c2dda44ac8b49bd0f076d12e22f2b7499ae714bfcc8da382e34c7648e8fab067',
    str(BASE / 'long_qualification/first_materialization/hidden_atomic_key.json'): '3aab8f32ed5101b33dcfc299fd9c4768079af8646e274f13bfec01656db204f5',
}
GRADES = {
    'reviewer_a': '34b772ef8d878a65c64a3f42644018f78de5c93eb98ac545d18e0f3cfa455f17',
    'reviewer_b': '0871a48b1428560a2ea32b4a509d19b2de8e295afc36113e5c1e69e8caf56beb',
}
FIELDS = ('coverage_established', 'verdict_correct',
          'all_required_operands_present', 'every_linked_reason_supported')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, expected):
    require(sha(path) == expected, 'Input bytes differ: ' + str(path))
    text = Path(path).read_text()
    return json.loads(text) if Path(path).suffix == '.json' else text


def index(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[f] for f in fields)
        require(key not in result, 'Duplicate inventory identity: ' + repr(key))
        result[key] = row
    return result


def summarize(rows):
    return {field: dict(Counter('unestablished' if r[field] is None else
            'true' if r[field] is True else 'false' for r in rows)) for field in FIELDS}


def assess():
    inputs = {p: load(p, h) for p, h in PINNED.items()}
    prepared = inputs[str(QUAL / 'preparation/preparation.json')]
    collected = inputs[str(QUAL / 'collection/first_collection.json')]
    derived = inputs[str(QUAL / 'derived/first_derived.json')]
    keys = {
        'short': inputs['audits/attention_relational_report_v1/grounded_auditor_v4/qualification/first_materialization/hidden_atomic_key.json'],
        'long': inputs[str(BASE / 'long_qualification/first_materialization/hidden_atomic_key.json')],
    }
    cases = {phase: {c['case_id']: c for c in key['cases']} for phase, key in keys.items()}
    positions = index(derived['rows'], ('request_id',))
    expected = {(r['request_id'], a['assertion_id']): (r, a)
                for r in derived['rows']
                for a in cases[r['phase']][r['case_id']]['mandatory_atomic_assertions']}
    invalid = {key for key, (r, _) in expected.items() if not r['mechanical_contract_established']}
    raw = {r['request_id']: r['annotation_file']['parsed'] for r in collected['records']}
    actual = {(rid, a['assertion_id']): a for rid, annotation in raw.items()
              for a in annotation['assertions']}
    require(len(positions) == 56 and len(expected) == 830 and len(actual) == 918,
            'Complete first opportunity inventories differ')
    grades, primary, diagnostic, summaries = {}, {}, {}, {}
    for reviewer, expected_sha in GRADES.items():
        path = QUAL / 'grading' / reviewer / 'first_grade.json'
        grade = grades[reviewer] = load(path, expected_sha)
        require(grade.get('input_hashes', grade.get('input_sha256')) == PINNED,
                'Grader input bindings differ: ' + reviewer)
        pg = index(grade['positions'], ('request_id',))
        ag = primary[reviewer] = index(grade['atomic_grades'], ('request_id', 'gold_assertion_id'))
        dg = diagnostic[reviewer] = index(grade['diagnostic_raw_atomic_grades'], ('request_id', 'gold_assertion_id'))
        require(set(pg) == set(positions) and set(ag) == set(expected) and set(dg) == invalid,
                'Missing or extra first-grade identities: ' + reviewer)
        for identity, position in pg.items():
            source = positions[identity]
            require(all(position[k] == source[k] for k in ('phase', 'stream', 'case_id',
                    'mechanical_contract_established', 'first_artifact_binding_established')),
                    'Position/source identity or eligibility differs')
        for identity, atom in ag.items():
            source, gold = expected[identity]
            require(all(atom[k] == source[k] for k in ('phase', 'stream', 'case_id'))
                    and atom['expected_verdict'] == gold['expected_verdict'],
                    'Atom source/gold identity differs')
            require(all(atom[f] is None or type(atom[f]) is bool for f in FIELDS),
                    'Atomic grades must be booleans or unestablished')
            require(all(atom[f] is None for f in FIELDS) if identity in invalid
                    else all(type(atom[f]) is bool for f in FIELDS),
                    'Invalid annotations were upgraded or eligible judgments omitted')
            require(all((identity[0], aid) in actual for aid in atom['actual_assertion_ids']),
                    'Grade names an absent first assertion')
        for identity, atom in dg.items():
            require(atom['diagnostic_only'] is True and
                    atom.get('upgrades_primary_eligibility', False) is False and
                    atom.get('primary_eligible', False) is False and
                    all(type(atom[f]) is bool for f in FIELDS),
                    'Raw diagnostic upgraded primary evidence or is incomplete')
        assertion_field = 'assertion_id' if reviewer == 'reviewer_a' else 'actual_assertion_id'
        checks = index(grade['all_actual_assertion_checks'], ('request_id', assertion_field))
        require(set(checks) == set(actual), 'Actual assertion/reason inventory incomplete')
        require(all(checks[k]['actual_assertion'] == actual[k] for k in actual),
                'Grader actual assertion bytes/content were changed')
        require(grade['whole_version_pass'] is False and all(
            grade['gates'][s]['short']['pass'] is True and
            grade['gates'][s]['long']['pass'] is False
            for s in ('stream_0', 'stream_1')), 'Recorded first gate findings differ')
        summaries[reviewer] = dict(
            grade_path=str(path), grade_sha256=expected_sha,
            positions=len(pg), mandatory_atom_opportunities=len(ag),
            actual_assertions_checked=len(checks), primary=summarize(ag.values()),
            diagnostic_raw=summarize(dg.values()), gates=grade['gates'],
            primary_operand_failures=[dict(request_id=r['request_id'], stream=r['stream'],
                case_id=r['case_id'], gold_assertion_id=r['gold_assertion_id'],
                actual_assertion_ids=r['actual_assertion_ids'],
                missing_required_operands=r['missing_required_operands'])
                for r in ag.values() if r['all_required_operands_present'] is False])
    disagreements = [dict(mode=mode, request_id=k[0], gold_assertion_id=k[1], field=f,
                         reviewer_a=table['reviewer_a'][k][f], reviewer_b=table['reviewer_b'][k][f])
        for mode, table in (('primary', primary), ('diagnostic_raw', diagnostic))
        for k in table['reviewer_a'] for f in FIELDS
        if table['reviewer_a'][k][f] is not table['reviewer_b'][k][f]]
    return dict(schema='attention_relational_subagent_auditor_v5_first_assessment',
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        assessor_source_sha256=sha(__file__), input_sha256=PINNED,
        first_grade_sha256=GRADES, complete_planned_positions=56,
        mechanical_valid_positions=sum(r['mechanical_contract_established'] for r in derived['rows']),
        primary_atom_opportunities=830, primary_unestablished_atom_opportunities=len(invalid),
        reviewer_summaries=summaries, atomic_field_disagreements=disagreements,
        source_truth_adjudication_performed=False,
        limitation='Checks inventories, bindings and first-grade agreement; does not automate semantic truth or establish population reliability.',
        qualification_pass=False, target_release_authorized=False,
        annotation_or_grade_repaired=False, automatic_successor_commissioned=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing first assessment replacement')
    result = assess()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(qualification_pass=result['qualification_pass'],
        positions=result['complete_planned_positions'],
        unestablished_atoms=result['primary_unestablished_atom_opportunities'],
        atomic_field_disagreements=len(result['atomic_field_disagreements']))))
