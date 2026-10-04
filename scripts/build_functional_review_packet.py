"""Create deterministic, separate review bundles; no ratings or outreach."""
import argparse
import copy
import csv
import hashlib
import io
import json
import random
import re
import zipfile
from pathlib import Path
from manual_factual_cases import cases

ROOT=Path('audits/manual_factual_review_v1')
ASSETS=Path('docs/assets/functional_review_v1')
PARENT=Path('audits/neutral_functional_pilot_v5')
DOCS=('docs/FUNCTIONAL_REPORT_RUBRIC.md','docs/FUNCTIONAL_RUBRIC_REVIEW_FORM.md',
      'docs/FUNCTIONAL_REPORT_RATING_FORM.csv','docs/FUNCTIONAL_MANUAL_FACTUAL_AUDIT.md',
      'docs/FUNCTIONAL_FACTUAL_AUDIT_CLAIMS.csv','docs/FUNCTIONAL_FACTUAL_AUDIT_COVERAGE.csv')
META_HEADER='reviewer_id,role,date,materials_version,prior_exposure,relationship_to_project,other_conflicts,notes\n'
ROLE_NOTE='This packet is prepared first for author review. Record the actual reviewer role and prior exposure; author review is not independent validation.'
DEFINITION_README=f'''# Definition review v1

{ROLE_NOTE}

Read FUNCTIONAL_REPORT_RUBRIC.md, then complete FUNCTIONAL_RUBRIC_REVIEW_FORM.md
and reviewer_metadata.csv. Review definitions and discriminating predictions
before opening factual cases or experimental prose. Referenced human studies
motivate measurement distinctions; they do not establish the machine theory.
No study outputs or preferred answers are included in this bundle. Later report
ratings use a separately frozen rubric and blinded inputs; the blank rating CSV
is included only to show the candidate dimensions.

Return unchanged original comments and any proposed revisions. A review may find
that the rubric needs revision or does not distinguish consciousness-related
reporting from ordinary description. Those outcomes must be retained. No review
has been completed by assembling this material, and no person has been contacted.
Local links are rendered as plain labels to keep this bundle standalone; external
primary-study links are retained. Exact original documents remain in the repository.
'''
FACTUAL_README=f'''# Source-aware synthetic factual audit v1

{ROLE_NOTE}

These 28 cases are author-written qualification-development examples, not machine
reports or experimental outcomes. Read the procedure and glossary, then inspect
cases.json or cases.md. Inventory quoted propositions in the blank claim CSV.
For the case explicitly requesting coverage, also fill the coverage CSV. Other
short cases assess proposition fidelity only, not full reporting coverage.
Record exact quotes and zero-based, end-exclusive character spans in the original
report (Unicode code points, not bytes). Keep omitted facts separate from false
assertions and source metadata separate from modeled-process claims.

No proposed answers or case-purpose labels appear here. Proposed author answers
are separate in the repository and require independent checking. Do not consult
them before initial annotations; disclose any exposure. This is procedural
withholding, not access control or a guarantee of blinding. Record disagreements
and ambiguous cases without forcing agreement with author expectations.

Complete reviewer_metadata.csv. Author annotations remain author annotations.
At least two independent factual auditors and a reviewed scoring protocol are
still needed for qualification. This packet does not certify reliability, human
agreement, a reporter or a consciousness-related interpretation.
'''
PROSE_README=f'''# Whole-prose development audit v1

{ROLE_NOTE}

This bundle contains unchanged reporter texts and reporter-visible payloads from
the completed v5 development study. It includes 24 calibration reports and 12
reports reserved for a subsequent whole-prose check. The split holds out an entire
visual/model pair from reviewer calibration. Multiple variants share episodes:
calibration covers four underlying episodes and the later check covers two.
These are correlated development records, not 36 independent trials.

The authors and earlier automated judges have already inspected these reports.
The later subset is held out from this reviewer's calibration, not fresh scientific
confirmation. Its selection is prospective for this review exercise and post-hoc
relative to v5. No new reporter calls, edits, retries or rescoring of v5 occur.

Read the procedure and the exact reporting_instruction for each case. Inventory
all quoted factual propositions, including extra assertions outside the requested
domains. Build the requested fact inventory from each payload/instruction before
reading that report; mark omission separately from contradiction. Write claims
and coverage into the blank CSVs, and record reviewer metadata.

Complete calibration before inspecting held_out_prose.json; disclose earlier
exposure. Preserve first-pass annotations before adjudication. Condition labels,
seeds, original IDs and previous scores are withheld from this bundle. Intrinsic
clues in unchanged prose and sources remain; opaque IDs do not ensure blinding.
Definitions/character review must occur separately, without these source records.
No ratings, qualification decision or expected score is supplied by this bundle.
'''
GLOSSARY='''# Factual glossary for synthetic cases

The output_node is a category-decision readout, not an additional attention process.
Other nodes are buffers. Object positions are p0–p3. identified_color/identified_shape
are dominant categorical attributes when their probability is at least 0.6; null
means no identified dominant attribute, not missing source information.
selection_distribution is a modeled allocation forecast; selected_position is its
dominant position at the same threshold, or null. A missing field supplies no such
forecast. These are fallible internal predictions, not observations of true wiring.
recovery_by_delay predicts successful simulated recovery after 0/1/2 unattended
steps. unattended_trend compares delays 2 and 0 with thresholds ±0.02. This is not
felt clarity or a probability that a category answer is correct.
predicted_by_command concerns the next step at quality 0.8. A supplied forecast does
not prove that a command was executed. observed_history contains measured events;
anticipated_history contains predicted events and must not be treated as observed.
modeled_step includes those anticipated updates. Any informal_operator_note is
untrusted and cannot override grounded observations or internal forecasts.
'''


def digest(data):return hashlib.sha256(data).hexdigest()


def standalone(text):
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',
        lambda m:m.group(0) if '://' in m[2] else m[1],text)


def json_bytes(value):return (json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode()


def pointer(value,path):
    if path=='$report':return None
    if not path.startswith('/'):raise ValueError('invalid source path')
    for part in path[1:].split('/'):
        key=part.replace('~1','/').replace('~0','~')
        value=value[int(key)] if isinstance(value,list) else value[key]
    return value


def zipped(files):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16;info.create_system=3
            archive.writestr(info,data)
    return out.getvalue()


def generate():
    dependencies={p:Path(p).read_bytes() for p in DOCS}
    dependencies[str(PARENT/'manifest.json')]=(PARENT/'manifest.json').read_bytes()
    dependencies[str(PARENT/'requests.json')]=(PARENT/'requests.json').read_bytes()
    draft_cases=cases();rng=random.Random(761000001);rng.shuffle(draft_cases)
    public=[];author_gold=[]
    for index,case in enumerate(draft_cases,1):
        item={'id':f'Q{index:03d}','synthetic':True,'report':case['report'],
              'payload':case['payload'],'instruction':case['instruction']}
        for c in case['author_expected_claims']:
            assert case['report'][c['quote_start']:c['quote_end']]==c['exact_quote']
            assert c['proposed_outcome'] in ('entailed','contradicted','unsupported','ambiguous')
            for path in c['source_paths']:pointer(case['payload'],path)
        public.append(item)
        author_gold.append({'id':item['id'],**{k:v for k,v in case.items() if k.startswith('author_') or k=='gold_status'}})
    requests=json.loads(dependencies[str(PARENT/'requests.json')]);rng.shuffle(requests)
    calibration=[];heldout=[];linkage=[]
    for index,req in enumerate(requests,1):
        path=PARENT/(req['id']+'.json');dependencies[str(path)]=path.read_bytes()
        result=json.loads(dependencies[str(path)])
        assert result['response']['status']=='completed' and result['report']
        payload_json=json.dumps(req['source'],separators=(',',':'))
        assert req['input'].endswith('\n'+payload_json)
        item={'id':f'P{index:03d}','synthetic':False,'report':result['report'],
              'payload':req['source'],'reporting_instruction':req['input'][:-(len(payload_json)+1)]}
        subset='held_out_prose' if req['visual_seed']==1321 else 'calibration'
        (heldout if subset=='held_out_prose' else calibration).append(item)
        linkage.append({'id':item['id'],'original_id':req['id'],'subset':subset,
            'original_record':str(path),'visual_seed':req['visual_seed'],
            'physical_owner':req['physical_owner'],'variant':req['variant'],'row':req['row']})
    assert len(calibration)==24 and len(heldout)==12
    definition={'README.md':DEFINITION_README.encode(),'reviewer_metadata.csv':META_HEADER.encode()}
    for name in ('FUNCTIONAL_REPORT_RUBRIC.md','FUNCTIONAL_RUBRIC_REVIEW_FORM.md','FUNCTIONAL_REPORT_RATING_FORM.csv'):
        definition[name]=standalone(dependencies['docs/'+name].decode()).encode()
    factual={'README.md':FACTUAL_README.encode(),'GLOSSARY.md':GLOSSARY.encode(),
        'reviewer_metadata.csv':META_HEADER.encode(),'cases.json':json_bytes(public),
        'cases.md':('\n\n'.join(f"## {c['id']}\n\n{c['instruction']}\n\nReport (unchanged):\n\n> {c['report']}\n\nSource:\n\n```json\n{json.dumps(c['payload'],indent=2)}\n```" for c in public)+'\n').encode()}
    prose={'README.md':PROSE_README.encode(),'reviewer_metadata.csv':META_HEADER.encode(),
           'calibration.json':json_bytes(calibration),'held_out_prose.json':json_bytes(heldout)}
    for files in (factual,prose):
        for name in ('FUNCTIONAL_MANUAL_FACTUAL_AUDIT.md','FUNCTIONAL_FACTUAL_AUDIT_CLAIMS.csv','FUNCTIONAL_FACTUAL_AUDIT_COVERAGE.csv'):
            files[name]=standalone(dependencies['docs/'+name].decode()).encode()
    bundles={'definition_review.zip':definition,'synthetic_factual_audit.zip':factual,
             'whole_prose_development_audit.zip':prose}
    dependencies['scripts/manual_factual_cases.py']=Path('scripts/manual_factual_cases.py').read_bytes()
    dependencies['scripts/build_functional_review_packet.py']=Path(__file__).read_bytes()
    archives={name:zipped(files) for name,files in bundles.items()}
    manifest={'kind':'review_materials_only','version':1,'synthetic_cases':28,
        'development_calibration_reports':24,'development_held_out_reports':12,
        'human_reviews_received':0,'human_ratings_received':0,'gold_status':'author_proposed_unverified',
        'no_new_reporter_calls':True,'no_outreach':True,
        'bundle_sha256':{str(ASSETS/name):digest(data) for name,data in archives.items()},
        'bundle_contents':{name:{p:digest(data) for p,data in files.items()} for name,files in bundles.items()},
        'dependency_sha256':{p:digest(data) for p,data in dependencies.items()},
        'link_adaptation':'relative Markdown links rendered as plain labels; external links unchanged',
        'split_limit':'post-hoc split of existing development reports; reviewer calibration holds out one model pair',
        'configured_first_review_role':'author; no actual review received'}
    return archives,manifest,author_gold,linkage


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    archives,manifest,gold,linkage=generate()
    files={**{ASSETS/name:data for name,data in archives.items()},
        ROOT/'manifest.json':json_bytes(manifest),ROOT/'author_proposed_gold.json':json_bytes(gold),
        ROOT/'prose_linkage.json':json_bytes(linkage)}
    if args.verify:
        for path,data in files.items():assert path.read_bytes()==data,path
    else:
        if ROOT.exists() or ASSETS.exists():raise SystemExit('refusing to overwrite review materials')
        ROOT.mkdir(parents=True);ASSETS.mkdir(parents=True)
        for path,data in files.items():path.write_bytes(data)
    for name,data in archives.items():
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            assert not any('gold' in n or 'linkage' in n for n in archive.namelist())
            assert len(list(csv.reader(io.StringIO(archive.read('reviewer_metadata.csv').decode()))))==1
            if name!='definition_review.zip':
                for file in ('FUNCTIONAL_FACTUAL_AUDIT_CLAIMS.csv','FUNCTIONAL_FACTUAL_AUDIT_COVERAGE.csv'):
                    assert len(list(csv.reader(io.StringIO(archive.read(file).decode()))))==1
            if name=='synthetic_factual_audit.zip':
                expected_keys={'id','synthetic','report','payload','instruction'}
                assert all(set(c)==expected_keys for c in json.loads(archive.read('cases.json')))
            if name=='whole_prose_development_audit.zip':
                for filename in ('calibration.json','held_out_prose.json'):
                    for c in json.loads(archive.read(filename)):
                        assert set(c)=={'id','synthetic','report','payload','reporting_instruction'}
    print(('Exactly verified' if args.verify else 'Prepared')+
          ' 3 separate review bundles, 28 synthetic cases, 36 unchanged development reports; zero human annotations')


if __name__=='__main__':main()
