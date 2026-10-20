import json,sys,pathlib,hashlib
sys.path.insert(0,'scripts');import attention_relational_template_review_workspace_v1 as w
WORK=pathlib.Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_0/chunk_11/work')
c=w.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_11.json','861514d2ea78d2073085b44a0042c46c17bc60a0205e9640bb087b5067d1d6a2')
def begin(i):
 r=c['records'][i];assert r['reference_established'] is True
 assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
 assert r['canonical_first_annotation']['unit_id']==r['unit_id']
 return r,{k:r[k] for k in ['id','unit_id','text_sha256','reference_established']}
def prop(r,n,path,role,reason):
 u=r['property_table'][n];return dict(unit_id=u['unit_id'],value_path=path,value=w.resolve(u['value'],path),semantic_role=role,reason=reason)
def leaves(r,n,role,reason):
 out=[]
 def walk(v,p):
  if isinstance(v,dict):
   for k,t in v.items():walk(t,p+[k])
  else:out.append(prop(r,n,p,role,reason))
 walk(r['property_table'][n]['value'],[]);return out
def finish(i,r,x,cs,es,bs,concerns,notes):
 assert len(cs)==len(r['components']) and len(es)==len(r['evidence']) and len(bs)==len(r['bindings'])
 x['component_reviews']=[dict(component_path=p['path'],original_status=p['recorded_status'],assessment=j[0],notes=j[1]) for p,j in zip(r['components'],cs)]
 x['evidence_reviews']=[dict(evidence_path=e['path'],grounding=a,source_meaning=s,properties=p,notes=n) for e,(a,s,p,n) in zip(r['evidence'],es)]
 x['binding_reviews']=bs;x['additional_concerns']=concerns;x['notes']=notes
 p=WORK/('record'+str(i)+'_input.json');p.write_text(json.dumps(x,indent=2)+'\n');print(w.save_first_record(c,r['id'],WORK,p.read_bytes()))
