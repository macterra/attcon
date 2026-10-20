import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
P='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_14.json'
SHA='1e90a6d34227a09a71008057686ce3b9b84fca019230f3509aaa18047d25905f'
WORK='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_14/work'
d=w.bound_chunk(P,SHA)
def view(n,path):
 v=w.source_view(d,d['records'][n]['id'],path,0,10000)
 assert v['complete'];return v['value']
def report(n):
 t=view(n,['text']);print('REPORT',n,'chars',len(t),'complete',True);print(t)
 for k in ['id','unit_id','text_sha256','reference_established']:print(k,json.dumps(view(n,[k])))
def details(n):
 r=d['records'][n]
 if not view(n,['reference_established']):
  print('UNESTABLISHED DIAGNOSTICS',json.dumps(view(n,['reference_measurement'])));return
 for k in ['canonical_first_annotation','components','bindings','evidence']:print(k,json.dumps(view(n,[k])))
def props(n,indices):
 for i in indices:print('PROPERTY',i,json.dumps(view(n,['property_table',i])))
def save(n,cr,er,br,concerns,notes):
 r=d['records'][n];j={k:view(n,[k]) for k in ['id','unit_id','text_sha256','reference_established']}
 j.update(component_reviews=cr,evidence_reviews=er,binding_reviews=br,additional_concerns=concerns,notes=notes)
 raw=(json.dumps(j,ensure_ascii=True,indent=2)+'\n').encode();print(w.save_first_record(d,r['id'],WORK,raw))
def unestablished(n):
 r=d['records'][n];assert view(n,['reference_established']) is False
 save(n,[],[],[],[], 'unestablished; original reference_measurement diagnostics preserved exactly: '+json.dumps(view(n,['reference_measurement']),sort_keys=True))
def c(n,i,assessment,notes):
 s=view(n,['components',i]);return dict(component_path=s['path'],original_status=s['recorded_status'],assessment=assessment,notes=notes)
def p(n,i,path,role,reason):
 s=view(n,['property_table',i]);return dict(unit_id=s['unit_id'],value_path=path,value=w.resolve(s['value'],path),semantic_role=role,reason=reason)
def e(n,i,grounding,meaning,properties,notes):
 s=view(n,['evidence',i]);return dict(evidence_path=s['path'],grounding=grounding,source_meaning=meaning,properties=properties,notes=notes)
def b(n,i,witnesses,assessment,notes):
 s=view(n,['bindings',i]);byid={e['evidence_id']:e['path'] for e in d['records'][n]['evidence']}
 return dict(binding_path=s['path'],feature_witnesses=[dict(feature=f,evidence_paths=[view(n,['evidence',j])['path'] for j in js]) for f,js in witnesses],binding_evidence_paths=[byid[x] for x in s['own_evidence_ids']],assessment=assessment,notes=notes)
if __name__=='__main__':
 action=sys.argv[1];n=int(sys.argv[2])
 if action=='report':report(n)
 elif action=='details':details(n)
 elif action=='props':props(n,[int(v) for v in sys.argv[3:]])
 elif action=='unestablished':unestablished(n)
