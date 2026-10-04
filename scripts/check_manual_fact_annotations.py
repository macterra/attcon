"""Validate submitted annotations; structural validity never qualifies the audit."""
import argparse
import csv
import json
import zipfile
from pathlib import Path
from build_functional_review_packet import ASSETS, pointer


def load_cases():
    records={}
    for archive_name,files in (
        ('synthetic_factual_audit.zip',('cases.json',)),
        ('whole_prose_development_audit.zip',('calibration.json','held_out_prose.json'))):
        with zipfile.ZipFile(ASSETS/archive_name) as archive:
            for file in files:
                for case in json.loads(archive.read(file)):
                    assert case['id'] not in records
                    records[case['id']]=case
    return records


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return [{k:v for k,v in row.items()} for row in csv.DictReader(stream)
                if any(v for v in row.values())]


def validate(claims,metadata,records):
    people={}
    for row in metadata:
        reviewer=row['reviewer_id']
        if not reviewer or reviewer in people:raise ValueError('missing or repeated reviewer ID')
        if row['role'] not in ('author','independent','other'):raise ValueError('unknown reviewer role')
        people[reviewer]=row
    seen=set();counts={};roles={}
    for row in claims:
        reviewer=row['auditor_id'];report_id=row['report_id'];claim_id=row['claim_id']
        if reviewer not in people:raise ValueError('claim has no reviewer metadata')
        if report_id not in records:raise ValueError('unknown report ID')
        if not claim_id or (reviewer,report_id,claim_id) in seen:raise ValueError('missing or duplicate claim ID')
        seen.add((reviewer,report_id,claim_id))
        if row['procedure_version']!='1':raise ValueError('unknown procedure version')
        start,end=int(row['quote_start']),int(row['quote_end']);report=records[report_id]['report']
        if not 0<=start<end<=len(report) or report[start:end]!=row['exact_quote']:
            raise ValueError('quote/span does not match unchanged report')
        if not row['proposition'].strip():raise ValueError('empty proposition')
        if row['outcome'] not in ('entailed','contradicted','unsupported','ambiguous'):
            raise ValueError('unknown factual outcome')
        paths=json.loads(row['source_paths'])
        if not isinstance(paths,list) or not all(isinstance(p,str) for p in paths):
            raise ValueError('source_paths must be a JSON array of strings')
        if row['outcome']=='entailed' and not paths:raise ValueError('entailed claim needs source evidence')
        for path in paths:pointer(records[report_id]['payload'],path)
        counts[row['outcome']]=counts.get(row['outcome'],0)+1
        roles[people[reviewer]['role']]=roles.get(people[reviewer]['role'],0)+1
    return {'structurally_valid_claim_rows':len(claims),'declared_reviewers':len(people),
        'outcome_counts':counts,'claim_rows_by_declared_role':roles,
        'independence_verified':False,'whole_report_coverage_verified':False,
        'audit_qualified':False,
        'status':'annotations_received_for_review' if claims else 'awaiting_annotations',
        'limit':'Checks quotes, IDs and source paths only; cannot certify semantic judgment, completeness or independence.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--claims',required=True)
    parser.add_argument('--metadata',required=True);args=parser.parse_args()
    result=validate(read_csv(args.claims),read_csv(args.metadata),load_cases())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
