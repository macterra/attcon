"""Replay descriptive AI annotations; validation does not certify semantics."""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path

from check_manual_fact_annotations import validate

OUT = Path('audits/ai_review_v1/summary.json')
DIMENSIONS = ('content_identity', 'selection_contrast', 'content_prominence',
              'recovery_forecast', 'identity_certainty', 'presentation_manner',
              'response_confidence', 'action_access_dependence', 'access_attribution',
              'temporal_availability', 'experiential_wording')


def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def summarize():
    sources = [Path(__file__)]
    result = {'kind': 'descriptive_AI_development_review', 'human_required': False,
              'human_reviews_received': 0, 'confirmation': False,
              'audit_qualified': False, 'semantic_completeness_certified': False,
              'scientific_independence_established': False,
              'prior_verdicts_rescored': False, 'factual': {}}
    for name, root, inputs in (
        ('original_development', Path('audits/ai_review_v1'),
         ['synthetic_cases.json', 'whole_prose_cases.json']),
        ('integrated_task_cases', Path('audits/ai_task_audit_review_v1'), ['cases.json'])):
        records = {}
        for filename in inputs:
            p = root/filename; sources.append(p)
            for record in json.loads(p.read_text()):
                assert record['id'] not in records
                records[record['id']] = record
        reviewers = {}; rows = {}
        for reviewer in ('A', 'B'):
            folder = root/('reviewer_'+reviewer)
            claim_file = folder/'claims.csv'; metadata_file = folder/'reviewer_metadata.csv'
            inventory_file = folder/'report_inventory.json'
            sources.extend([claim_file, metadata_file, inventory_file])
            rows[reviewer] = csv_rows(claim_file)
            metadata = csv_rows(metadata_file)
            assert all(row['auditor_id'] == reviewer for row in rows[reviewer])
            seen = {row['report_id'] for row in rows[reviewer]}
            assert seen == set(records), (name, reviewer, seen ^ set(records))
            reviewers[reviewer] = validate(rows[reviewer], metadata, records)
        per_report = []
        for rid in sorted(records):
            per_report.append({'report_id': rid, 'reviewers': {
                r: {'claim_rows': len(selected := [x for x in rows[r] if x['report_id'] == rid]),
                    'outcomes': dict(Counter(x['outcome'] for x in selected))}
                for r in ('A', 'B')}})
        result['factual'][name] = {'reports': len(records), 'reviewers': reviewers,
            'per_report_inventories': per_report,
            'claim_agreement_calculated': False,
            'reason': 'Row inventories differ in granularity, scope and extra assertions; row totals are not semantic agreement or precision.'}

    root = Path('audits/ai_character_review_v1')
    report_file = root/'reports.json'; sources.append(report_file)
    records = {r['report_id']: r['report'] for r in json.loads(report_file.read_text())}
    ratings = {}
    for rater in ('S1', 'S2'):
        p = root/('rater_'+rater)/'ratings.csv'; sources.append(p)
        rows = csv_rows(p); assert len(rows) == len(records)
        ratings[rater] = {x['report_id']: x for x in rows}
        assert set(ratings[rater]) == set(records)
        for row in rows:
            assert row['rater_id'] == rater and row['rubric_version'] == '3'
            assert all(row[d] in ('absent', 'explicit', 'ambiguous') for d in DIMENSIONS)
            evidence = json.loads(row['evidence_quotes'])
            supplied = set()
            for quote in evidence:
                assert quote['dimension'] in DIMENSIONS
                assert quote['quote'] and quote['quote'] in records[row['report_id']]
                supplied.add(quote['dimension'])
            assert {d for d in DIMENSIONS if row[d] != 'absent'} <= supplied
            assert isinstance(json.loads(row['presentation_referents']), list)
            assert isinstance(json.loads(row['access_referents']), list)
    by_dimension = {}; disagreements = []; same = 0
    for d in DIMENSIONS:
        by_dimension[d] = {r: dict(Counter(row[d] for row in ratings[r].values()))
                           for r in ratings}
        for rid in sorted(records):
            labels = {r: ratings[r][rid][d] for r in ratings}
            if len(set(labels.values())) == 1: same += 1
            else: disagreements.append({'report_id': rid, 'dimension': d, 'labels': labels})
    total = len(records)*len(DIMENSIONS)
    result['character'] = {'reports': len(records), 'underlying_development_episodes': 2,
        'by_dimension': by_dimension, 'dimension_labels_equal': same,
        'dimension_label_pairs': total, 'descriptive_label_agreement': same/total,
        'disagreements': disagreements,
        'character': {r: dict(Counter(x['character'] for x in ratings[r].values())) for r in ratings},
        'technical_alternative': {r: dict(Counter(x['technical_alternative'] for x in ratings[r].values())) for r in ratings},
        'qualification_verdict': 'not_assessed',
        'limit': 'Raw label agreement on repeated development texts; not independent samples, feature validation, semantics certification or a frozen pass gate.'}
    result['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sources}
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(); result = summarize()
    data = (json.dumps(result, indent=2)+'\n').encode()
    if args.verify:
        assert OUT.read_bytes() == data
    else:
        if OUT.exists(): raise SystemExit('refusing overwrite of descriptive summary')
        OUT.write_bytes(data)
    print(('Exactly verified' if args.verify else 'Saved')+
          ' AI review counts and raw character disagreements; no qualification verdict')


if __name__ == '__main__': main()
