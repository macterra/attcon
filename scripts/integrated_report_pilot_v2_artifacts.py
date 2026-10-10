"""Reserve once and archive subagent prose for a fixed development budget."""
import argparse,hashlib,json,random
from pathlib import Path
from build_integrated_report_pilot_v1 import encoded
ROOT=Path('audits/integrated_report_pilot_v2')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,data):
    p=ROOT/name
    if p.exists():raise SystemExit('refusing overwrite '+name)
    p.write_bytes(encoded(data))
def reserve():
    assessment=json.loads((ROOT/'qualification_assessment.json').read_text());assert assessment['passed']
    requests=json.loads((ROOT/'requests.json').read_text());assert len(requests)==18
    save('attempts.json',{'status':'whole_batch_reserved_before_assignment','max_attempts':18,'no_retries':True,'ids':[r['id'] for r in requests],'request_sha256':sha(ROOT/'requests.json'),'qualification_sha256':sha(ROOT/'qualification_assessment.json'),'required_facts_sha256':sha(ROOT/'required_facts.json'),'Reporter_kind':'fresh separate-context AI subagent, external instrumentation; no Controller trial','code_sha256':sha(__file__)})
    print('Reserved all18 once; no generated reports yet')
def collect():
    attempts=json.loads((ROOT/'attempts.json').read_text());assert attempts['request_sha256']==sha(ROOT/'requests.json')
    rows=json.loads((ROOT/'reporter/reports.json').read_text());ids=set();result=[]
    for row in rows:
        assert row['id'] in attempts['ids'] and row['id'] not in ids;ids.add(row['id'])
        assert isinstance(row['report'],str);result.append({'id':row['id'],'status':'received' if row['report'].strip() else 'empty','word_count':len(row['report'].split()),'word_limit_exceeded':len(row['report'].split())>180})
    for rid in attempts['ids']:
        if rid not in ids:result.append({'id':rid,'status':'missing','word_count':0,'word_limit_exceeded':False})
    result.sort(key=lambda x:x['id']);save('collection.json',{'attempt_budget':18,'attempts':result,'report_sha256':sha(ROOT/'reporter/reports.json'),'Reporter_implementation_changed_from_historical_API':True,'secure_blinding':False,'qualification_scope':'bounded supportedness challenge, not open prose completeness','new_Controller_trials':0,'new_training':False})
    shuffled=rows.copy();random.Random(20261011).shuffle(shuffled)
    blind=[{'id':f'B{i+1:03d}','report':r['report']} for i,r in enumerate(shuffled)]
    save('blind_reports.json',blind);save('blind_linkage.json',[{'id':b['id'],'report_id':r['id']} for b,r in zip(blind,shuffled)])
    print('Archived',len(rows),'received reports; missing attempts retained; prepared text-only ratings')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('reserve','collect'),required=True);a=ap.parse_args()
    reserve() if a.stage=='reserve' else collect()
if __name__=='__main__':main()
