import json,sys,subprocess
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
BASE=Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_03')
CHUNK='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_03.json'
SHA='c5f5f6290de81b9cf87c770687b2999a7c70321d0dfdfa7e90012155827f5289'
c=w.bound_chunk(CHUNK,SHA)
def row(n):return c['records'][n]
def prop(n,uid,path,role,reason):
 u=next(u for u in row(n)['property_table'] if u['unit_id']==uid)
 return dict(unit_id=uid,value_path=path,value=w.resolve(u['value'],path),semantic_role=role,reason=reason)
def author(n,components,evidence,bindings,concerns,notes):
 r=row(n)
 assert len(components)==len(r['components']) and len(evidence)==len(r['evidence']) and len(bindings)==len(r['bindings'])
 cr=[dict(component_path=x['path'],original_status=x['recorded_status'],assessment=a,notes=b) for x,(a,b) in zip(r['components'],components)]
 er=[dict(evidence_path=x['path'],grounding=a,source_meaning=b,properties=p,notes=q) for x,(a,b,p,q) in zip(r['evidence'],evidence)]
 br=[]
 for x,(a,witness,own,reason) in zip(r['bindings'],bindings):
  br.append(dict(binding_path=x['path'],feature_witnesses=[dict(feature=f,evidence_paths=[r['evidence'][j]['path'] for j in ids]) for f,ids in witness],binding_evidence_paths=[r['evidence'][j]['path'] for j in own],assessment=a,notes=reason))
 out=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=r['reference_established'],component_reviews=cr,evidence_reviews=er,binding_reviews=br,additional_concerns=concerns,notes=notes)
 p=BASE/'work'/('authored_%02d.json'%n)
 with p.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
 subprocess.run([sys.executable,'scripts/attention_relational_template_review_workspace_v1.py','--chunk',CHUNK,'--chunk-sha256',SHA,'save-first-record','--id',r['id'],'--input',str(p),'--workspace',str(BASE/'work')],check=True)
