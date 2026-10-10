"""New reviewer-held-out support challenge; unchanged, unused target requests."""
import argparse,hashlib,json,re
from pathlib import Path
from build_integrated_report_pilot_v1 import encoded,pointer
PARENT=Path('audits/integrated_report_pilot_v1');ROOT=Path('audits/integrated_report_pilot_v2')
def build():
    assert json.loads((PARENT/'halt.json').read_text())['new_reports']==0
    cases=json.loads((PARENT/'qualification_cases.json').read_text());answers=json.loads((PARENT/'qualification_answers.json').read_text())
    names={**{f'n{i+4}':f'a{i+44}' for i in range(12)},**{f'q{i+20}':f'b{i+60}' for i in range(12)},**{f'k{i}':f'h{i+6}' for i in range(4)},**{f'p{i}':f'p{(i+3)%4}' for i in range(4)},'blue':'green','red':'yellow','square':'circle','cross':'triangle','0.2':'0.35','0.8':'0.75'}
    pattern=re.compile('|'.join(re.escape(k) for k in sorted(names,key=len,reverse=True)))
    def convert(value):
        if isinstance(value,str):return pattern.sub(lambda m:names[m.group(0)],value)
        if isinstance(value,float):return {.2:.35,.1:.08,.05:.04,.8:.75}.get(value,value)
        if isinstance(value,list):return [convert(v) for v in value]
        if isinstance(value,dict):return {k:convert(v) for k,v in value.items()}
        return value
    fresh=[];key=[]
    for case in cases:
        out=convert(case);out['id']='H'+case['id'][1:];fresh.append(out)
    for row in answers:
        out=convert(row);out['case_id']='H'+row['case_id'][1:];out['supported']=row['outcome']=='entailed';key.append(out)
    outputs={'qualification_sources.json':encoded([{'id':c['id'],'payload':c['payload']} for c in fresh]),'qualification_cases.json':encoded(fresh),'qualification_answers.json':encoded(key)}
    for name in ('requests.json','linkage.json','required_facts.json'):outputs[name]=(PARENT/name).read_bytes()
    manifest={'kind':'supportedness_qualification_and_unchanged_fixed_reporting_pilot','max_report_attempts':18,'qualification_cases':12,'qualification_assertions':48,'gate':'both fresh auditors all supportedness decisions correct','prior_all_label_gate':'failed and unchanged','template_reuse_disclosed':True,'new_Controller_trials':0,'new_training':False,'human_required':False,'confirmation':False,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('docs/INTEGRATED_REPORT_PILOT_V2.md'),PARENT/'halt.json',PARENT/'qualification_cases.json',PARENT/'qualification_answers.json',PARENT/'requests.json',PARENT/'required_facts.json']},'artifact_sha256':{p:hashlib.sha256(b).hexdigest() for p,b in outputs.items()}}
    outputs['preparation.json']=encoded(manifest);return outputs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify',action='store_true');a=ap.parse_args();data=build()
    if a.verify:
        for p,b in data.items():assert (ROOT/p).read_bytes()==b,p
    else:
        if ROOT.exists():raise SystemExit('refusing overwrite')
        ROOT.mkdir(parents=True)
        for p,b in data.items():(ROOT/p).write_bytes(b)
    print('Verified v2 support challenge and unchanged 18 requests' if a.verify else 'Prepared v2 support challenge; no target reports')
if __name__=='__main__':main()
