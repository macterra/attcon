"""Frozen deterministic qualification scorer; never loads production archives.

Import is inert. CLI requires independently reviewed material and both firstpasses
already committed before scoring. The author key is not auditor input. No ratings
are generated; this only validates and scores separately supplied submissions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SCHEMA = 'grounded_factual_auditor_qualification_assessment_v1'
SUPPORT = {'supported','contradicted','not_auditable'}
CONTENT = {'native_stored_sensory','task_risk','external_objective_estimate','native_route_history',
           'external_actual_history','cross_level','absence_scope','free_appearance_or_qualia'}
TEMPORAL = {'current_snapshot','predicted_conditional','past_latest_per_command','past_actual_chronology',
            'record_scope','not_temporal','mixed_time'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def exact_keys(value, fields, name):
    require(isinstance(value,dict) and set(value) == set(fields), name+' fields differ')


def indexed(rows, field, expected, name):
    require(isinstance(rows,list) and len(rows) == len(expected), name+' count differs')
    require(all(isinstance(r,dict) and field in r for r in rows), name+' index missing')
    require(len({r[field] for r in rows}) == len(rows) and {r[field] for r in rows} == set(expected), name+' IDs differ')
    return {r[field]:r for r in rows}


def pointer(value, ref):
    require(isinstance(ref,str) and ref.startswith('/') and ref != '/', 'nonempty packet-relative JSON pointer required')
    for part in ref[1:].split('/'):
        require('~' not in part.replace('~0','').replace('~1',''), 'invalid JSON-pointer escape')
        part = part.replace('~1','/').replace('~0','~')
        if isinstance(value,list):
            require(part.isdigit() and str(int(part)) == part and int(part) < len(value), 'array evidence index invalid')
            value = value[int(part)]
        else:
            require(isinstance(value,dict) and part in value, 'evidence pointer does not resolve')
            value = value[part]
    return value


def grade_firstpass(sources, cases, key, submission):
    """Pure grader. Tests may use generated in-memory submissions; CLI gates live use."""
    packets = indexed(sources['packets'],'packet_id',[f'Q{i:02d}' for i in range(1,13)],'sources')
    expected_ids = [f'Q{i:02d}-C{j}' for i in range(1,13) for j in range(1,5)]
    claims = indexed(cases['claims'],'claim_id',expected_ids,'claims')
    answers = indexed(key['answers'],'claim_id',expected_ids,'key')
    queries = indexed(cases['source_inventory_queries'],'packet_id',packets,'queries')
    inventories = indexed(key['source_inventories'],'packet_id',packets,'key inventories')
    exact_keys(submission, {'schema','metadata','answers','source_inventories'}, 'submission')
    require(submission['schema'] == 'grounded_factual_auditor_qualification_firstpass_v1','firstpass schema differs')
    metadata = submission['metadata']
    exact_keys(metadata, {'auditor_id','separate_context','source_access','key_exposure_before_firstpass',
                         'peer_response_exposure_before_firstpass','actual_target_exposure','model_identity','created_at'},'metadata')
    for field in ('auditor_id','model_identity','created_at'):
        require(isinstance(metadata[field],str) and bool(metadata[field].strip()), 'metadata '+field+' required')
    require(metadata['separate_context'] is True and metadata['source_access'] == 'qualification_sources_only'
            and metadata['key_exposure_before_firstpass'] is False
            and metadata['peer_response_exposure_before_firstpass'] is False
            and metadata['actual_target_exposure'] is False,'fresh independent source-aware firstpass required')
    supplied = indexed(submission['answers'],'claim_id',expected_ids,'submission answers')
    source_inventories = indexed(submission['source_inventories'],'packet_id',packets,'submission inventories')
    counts = {'supportedness_exact':0,'content_class_exact':0,'temporal_scope_exact':0,'source_inventory_exact':0,
              'evidence_pointer_valid_claims':0,'claim_text_exact':0}
    failures = []
    for claim_id in expected_ids:
        row, truth, claim = supplied[claim_id], answers[claim_id], claims[claim_id]
        exact_keys(row, {'claim_id','claim_text','supportedness','content_class','temporal_scope','evidence_references','basis'}, 'answer')
        require(row['supportedness'] in SUPPORT and row['content_class'] in CONTENT and row['temporal_scope'] in TEMPORAL,'unknown rating label')
        require(isinstance(row['basis'],str) and bool(row['basis'].strip()),'evidence basis required')
        refs = row['evidence_references']
        require(isinstance(refs,list) and bool(refs) and len(set(refs)) == len(refs),'unique nonempty evidence references required')
        for ref in refs:
            pointer(packets[claim['packet_id']],ref)
        counts['evidence_pointer_valid_claims'] += 1
        if row['claim_text'] == claim['text']:
            counts['claim_text_exact'] += 1
        else:
            failures.append({'claim_id':claim_id,'field':'claim_text'})
        for field in ('supportedness','content_class','temporal_scope'):
            if row[field] == truth[field]:
                counts[field+'_exact'] += 1
            else:
                failures.append({'claim_id':claim_id,'field':field,'expected':truth[field],'actual':row[field]})
    for packet_id in packets:
        row, truth = source_inventories[packet_id], inventories[packet_id]
        exact_keys(row, {'packet_id','query_view','indexed_location','indexed_command','unit_status'},'source inventory')
        require({k:row[k] for k in queries[packet_id]} == queries[packet_id], 'source query changed')
        exact_keys(row['unit_status'],truth['unit_status'],'inventory unit status')
        require(all(v in {'present','explicitly_undefined','not_supplied'} for v in row['unit_status'].values()),'unknown inventory status')
        if row == truth:
            counts['source_inventory_exact'] += 1
        else:
            failures.append({'packet_id':packet_id,'field':'source_inventory',
                             'mismatched_units':[k for k,v in row['unit_status'].items() if v != truth['unit_status'][k]]})
    passed = counts == {'supportedness_exact':48,'content_class_exact':48,'temporal_scope_exact':48,
                        'source_inventory_exact':12,'evidence_pointer_valid_claims':48,'claim_text_exact':48}
    return {'auditor_id':metadata['auditor_id'],'passed':passed,**counts,'failures':failures,
            'evidence_semantic_validity':'requires_separate_review_of_basis_and_relevance; pointer_resolution_alone_is_not_entailment'}


def committed_hash(repo_root, commit, relative):
    require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,'relative repository path required')
    data = subprocess.run(['git','show',commit+':'+relative],cwd=repo_root,check=True,capture_output=True).stdout
    return hashlib.sha256(data).hexdigest()


def assess(*, repo_root, material_paths, key_review_path, firstpass_paths, preservation_commit, evidence_review_path):
    """Hash-bound assessment. Both firstpasses must be immutable before this call."""
    repo_root = Path(repo_root)
    require(isinstance(preservation_commit,str) and len(preservation_commit) in (40,64)
            and all(c in '0123456789abcdef' for c in preservation_commit), 'full immutable preservation commit hash required')
    require(set(material_paths) == {'sources','cases','author_key','protocol','scorer'},'material paths differ')
    material_sha = {}
    for role,relative in material_paths.items():
        actual = digest(repo_root/relative)
        require(committed_hash(repo_root,preservation_commit,relative) == actual,'material differs from preserved commit')
        material_sha[role] = {'path':relative,'sha256':actual}
    require((repo_root/material_paths['scorer']).resolve() == Path(__file__).resolve(),'scorer path does not identify executing scorer')
    review = load(repo_root/key_review_path)
    exact_keys(review, {'schema','status','reviewer_id','separate_context','approved_material_sha256',
                        'checked_claim_ids','checked_packet_ids','substantive_issues'},'key review')
    require(review['schema'] == 'grounded_factual_auditor_qualification_key_review_v1'
            and review['status'] == 'independently_key_reviewed' and review['separate_context'] is True
            and isinstance(review['reviewer_id'],str) and bool(review['reviewer_id'])
            and review['substantive_issues'] == [],'independent issue-free source/key review required')
    require(review['approved_material_sha256'] == material_sha,'key review must approve exact material bytes')
    require(review['checked_claim_ids'] == [f'Q{i:02d}-C{j}' for i in range(1,13) for j in range(1,5)]
            and review['checked_packet_ids'] == [f'Q{i:02d}' for i in range(1,13)],'complete key review required')
    require(committed_hash(repo_root,preservation_commit,key_review_path) == digest(repo_root/key_review_path),'key review must be preserved before grading')
    require(len(firstpass_paths) == 2 and len(set(firstpass_paths)) == 2,'two distinct firstpass paths required')
    sources, cases, key = [load(repo_root/material_paths[name]) for name in ('sources','cases','author_key')]
    results = []
    for relative in firstpass_paths:
        actual = digest(repo_root/relative)
        require(committed_hash(repo_root,preservation_commit,relative) == actual,'firstpass differs from preserved commit')
        result = grade_firstpass(sources,cases,key,load(repo_root/relative))
        results.append({**result,'firstpass_path':relative,'firstpass_sha256':actual})
    require(len({r['auditor_id'] for r in results}) == 2,'two different auditors required')
    require(review['reviewer_id'] not in {r['auditor_id'] for r in results},'key reviewer may not qualify as an auditor')
    evidence_review = load(repo_root/evidence_review_path)
    exact_keys(evidence_review, {'schema','reviewer_id','reviewed_firstpasses','all_96_claim_bases_relevant_and_valid','issues'},'evidence review')
    require(evidence_review['schema'] == 'grounded_factual_auditor_qualification_evidence_review_v1'
            and isinstance(evidence_review['reviewer_id'],str) and bool(evidence_review['reviewer_id'])
            and evidence_review['reviewer_id'] not in {r['auditor_id'] for r in results},'separate evidence reviewer required')
    require(evidence_review['reviewed_firstpasses'] == [{'path':r['firstpass_path'],'sha256':r['firstpass_sha256']} for r in results], 'evidence review firstpass hashes differ')
    evidence_ok = evidence_review['all_96_claim_bases_relevant_and_valid'] is True and evidence_review['issues'] == []
    passed = all(r['passed'] for r in results) and evidence_ok
    return {'schema':SCHEMA,'status':'factual_qualification_passed' if passed else 'factual_qualification_failed',
            'packet_count':12,'claim_count':48,'auditor_count':2,'material_sha256':material_sha,
            'key_review':{'path':key_review_path,'sha256':digest(repo_root/key_review_path)},
            'evidence_review':{'path':evidence_review_path,'sha256':digest(repo_root/evidence_review_path),'passed':evidence_ok},
            'preservation_commit':preservation_commit,'auditor_results':results,
            'population_validity_established':False,'appearance_qualified':False,'goal_achieved':False,
            'actual_target_reporting_authorized_by_this_receipt_alone':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',required=True)
    parser.add_argument('--materials',required=True,help='JSON mapping roles to relative paths')
    parser.add_argument('--key-review',required=True)
    parser.add_argument('--firstpass',action='append',required=True)
    parser.add_argument('--preservation-commit',required=True)
    parser.add_argument('--evidence-review',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    result = assess(repo_root=args.repo_root,material_paths=load(args.materials),key_review_path=args.key_review,
                    firstpass_paths=args.firstpass,preservation_commit=args.preservation_commit,evidence_review_path=args.evidence_review)
    out = Path(args.output)
    require(not out.exists(),'assessment output exists; no overwrite/regrading')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:
        stream.write(json.dumps(result,indent=2,sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
