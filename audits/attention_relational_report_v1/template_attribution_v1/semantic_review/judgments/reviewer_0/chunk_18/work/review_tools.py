import sys,json,pathlib,hashlib
sys.path.insert(0,str(pathlib.Path('scripts').resolve()))
import attention_relational_template_review_workspace_v1 as w
BASE=pathlib.Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review')
SHA='00f7c30a0463a31224f1d86edb5794dd76a4b87538d05fbddbb568b7c13d2ac1'
WORK=BASE/'judgments/reviewer_0/chunk_18/work'
def row(n):
 c=w.bound_chunk(BASE/'preparation/chunk_18.json',SHA); return c,c['records'][n]
def show(n,path,offset=0,limit=20):
 c,r=row(n); v=w.source_view(c,r['id'],path,offset,limit); print(json.dumps(v,ensure_ascii=False))
def save(n,comps,evs,binds,concerns,notes):
 c,r=row(n)
 cr=[dict(component_path=x['path'],original_status=x['recorded_status'],assessment=a,notes=t) for x,(a,t) in zip(r['components'],comps,strict=True)]
 er=[]
 for e,(g,s,refs,t) in zip(r['evidence'],evs,strict=True):
  ps=[]
  for uid,p,role,reason in refs:
   u=next(x for x in r['property_table'] if x['unit_id']==uid); value=w.resolve(u['value'],p)
   ps.append(dict(unit_id=uid,value_path=p,value=value,semantic_role=role,reason=reason))
  er.append(dict(evidence_path=e['path'],grounding=g,source_meaning=s,properties=ps,notes=t))
 br=[]
 for b,(a,features,t) in zip(r['bindings'],binds,strict=True):
  paths={e['evidence_id']:e['path'] for e in r['evidence']}
  br.append(dict(binding_path=b['path'],feature_witnesses=[dict(feature=f,evidence_paths=[paths[p] for p in ids]) for f,ids in features],binding_evidence_paths=[paths[p] for p in b['own_evidence_ids']],assessment=a,notes=t))
 out=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=r['reference_established'],component_reviews=cr,evidence_reviews=er,binding_reviews=br,additional_concerns=concerns,notes=notes)
 raw=(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode(); print(w.save_first_record(c,r['id'],WORK,raw))
def diagnostic_view(n):
 c,r=row(n)
 for p in [['reference_measurement','derivation','errors'],['reference_measurement','derivation','semantic_validation'],['components'],['evidence'],['bindings'],['review_state']]:show(n,p,0,100)
 ds=r['reference_measurement']['derivation']['quote_diagnostics']
 print('quote_diagnostic_count',len(ds))
 for k in range(len(ds)):
  v=w.source_view(c,r['id'],['reference_measurement','derivation','quote_diagnostics',k,'errors'],0,100)
  print(json.dumps({'diagnostic_index':k,'error_page':v}))
def save_unestablished(n,notes):
 c,r=row(n)
 d=r['reference_measurement']['derivation']
 save(n,[],[],[],['Original derivation diagnostics: '+json.dumps(d,ensure_ascii=False)],notes)
def compact_diagnostics(n):
 c,r=row(n)
 for p in [['reference_measurement','derivation','errors'],['reference_measurement','derivation','semantic_validation'],['evidence'],['review_state']]:show(n,p,0,100)
 ds=r['reference_measurement']['derivation']['quote_diagnostics']; print('quote_diagnostic_total',len(ds))
 for k in range(len(ds)):
  v=w.source_view(c,r['id'],['reference_measurement','derivation','quote_diagnostics',k,'errors'],0,100)
  print('diagnostic',k,'errors',json.dumps(v['value']),'total',v['total_items'],'complete',v['complete'])
