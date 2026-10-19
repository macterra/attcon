import json,sys
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as h
CHUNK='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_03.json'
SHA='c5f5f6290de81b9cf87c770687b2999a7c70321d0dfdfa7e90012155827f5289'
WORK='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_0/chunk_03/work'
c=h.bound_chunk(CHUNK,SHA)
def author(n,components,evidence,bindings,concerns,notes):
 r=c['records'][n]
 cr=[dict(component_path=x['path'],original_status=x['recorded_status'],assessment=a,notes=b) for x,(a,b) in zip(r['components'],components,strict=True)]
 er=[]
 for x,(g,s,refs,note) in zip(r['evidence'],evidence,strict=True):
  props=[]
  for uid,path,role,reason in refs:
   u=next(u for u in r['property_table'] if u['unit_id']==uid)
   props.append(dict(unit_id=uid,value_path=path,value=h.resolve(u['value'],path),semantic_role=role,reason=reason))
  er.append(dict(evidence_path=x['path'],grounding=g,source_meaning=s,properties=props,notes=note))
 br=[]
 for x,(assessment,witnesses,note) in zip(r['bindings'],bindings,strict=True):
  br.append(dict(binding_path=x['path'],feature_witnesses=[dict(feature=f,evidence_paths=[r['evidence'][i]['path'] for i in inds]) for f,inds in witnesses],binding_evidence_paths=[e['path'] for e in r['evidence'] if e['binding_id']==x['binding_id']],assessment=assessment,notes=note))
 value=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=r['reference_established'],component_reviews=cr,evidence_reviews=er,binding_reviews=br,additional_concerns=concerns,notes=notes)
 path=f'{WORK}/authored_{n:02d}.json'
 with open(path,'x') as f: json.dump(value,f,indent=2)
 print(h.save_first_record(c,r['id'],WORK,open(path,'rb').read()))
def retain_unestablished(n,inspection_note):
 r=c['records'][n]
 assert r['reference_established'] is False
 author(n,[],[],[],['Original unestablished annotation retained with empty semantic inventories; no salvage.'], 'unestablished. '+inspection_note+' Original reference_measurement diagnostics: '+json.dumps(r['reference_measurement'],sort_keys=True))
