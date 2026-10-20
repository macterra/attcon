from review_tools import *
def save(n, comps, evs, binds, concerns, notes):
 r=c['records'][n-1]
 cr=[dict(component_path=x['path'],original_status=x['recorded_status'],assessment=a,notes=t) for x,(a,t) in zip(r['components'],comps)]
 er=[]
 for e,(a,m,operands,t) in zip(r['evidence'],evs):
  ps=[]
  for uid,path,role,reason in operands:
   u=next(x for x in r['property_table'] if x['unit_id']==uid)
   ps.append(dict(unit_id=uid,value_path=path,value=w.resolve(u['value'],path),semantic_role=role,reason=reason))
  er.append(dict(evidence_path=e['path'],grounding=a,source_meaning=m,properties=ps,notes=t))
 br=[]
 for b,(a,features,t) in zip(r['bindings'],binds):
  br.append(dict(binding_path=b['path'],assessment=a,feature_witnesses=[dict(feature=f,evidence_paths=paths) for f,paths in features],binding_evidence_paths=[e['path'] for e in r['evidence'] if e['binding_id']==b['binding_id']],notes=t))
 assert len(comps)==len(r['components']) and len(evs)==len(r['evidence']) and len(binds)==len(r['bindings'])
 assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
 out=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=r['reference_established'],component_reviews=cr,evidence_reviews=er,binding_reviews=br,additional_concerns=concerns,notes=notes)
 f=work/('draft_%02d.json'%n);f.write_text(json.dumps(out,indent=2)+'\n');print(w.save_first_record(c,r['id'],work,f.read_bytes()))
