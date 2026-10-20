import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
import attention_relational_template_review_workspace_v1 as w
BASE=Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_19')
CHUNK='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_19.json'
SHA='d0e3176477f5885024db0d705318b4ab6b4b53df7ed2e712b4c714e72fdd6b52'
c=w.bound_chunk(CHUNK,SHA)
def row(n): return c['records'][n]
def view(n,path,offset=0,limit=20): return w.source_view(c,row(n)['id'],path,offset,limit)
def report(n):
 r=row(n); print(json.dumps({'index':n,'id':r['id'],'reference_established':r['reference_established'],'text_sha256_verified':hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']})); print(json.dumps(view(n,['text'],0,10000)))
def inventories(n):
 r=row(n)
 for p in [['components'],['canonical_first_annotation'],['bindings']]: print(json.dumps(view(n,p,0,20)))
 for off in range(0,len(r['evidence']),8):
  v=view(n,['evidence'],off,8)
  v['value']=[{'path':e['path'],'recorded_evidence':e['recorded_evidence'],'units':[x['unit_id'] for x in e['geometry']['intersections']]} for e in v['value']]
  print(json.dumps(v))
 if not r['reference_established']: print(json.dumps(view(n,['reference_measurement'],0,20)))
 ids={u['unit_id'] for e in r['evidence'] for u in e['geometry']['intersections']}
 for j,p in enumerate(r['property_table']):
  if p['unit_id'] in ids: print(json.dumps(view(n,['property_table',j],0,30)))
def property_ref(r,unit,path,role,reason):
 p=next(p for p in r['property_table'] if p['unit_id']==unit)
 return dict(unit_id=unit,value_path=path,value=w.resolve(p['value'],path),semantic_role=role,reason=reason)
def save(n,components,evidence,bindings,concerns,notes):
 r=row(n)
 # Inputs are authored semantic judgments; inventory construction copies only fixed fields.
 assert len(components)==len(r['components'])
 assert len(evidence)==len(r['evidence'])
 assert len(bindings)==len(r['bindings'])
 out=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=r['reference_established'],component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=concerns,notes=notes)
 for c0,(assessment,note) in zip(r['components'],components): out['component_reviews'].append(dict(component_path=c0['path'],original_status=c0['recorded_status'],assessment=assessment,notes=note))
 for e,(ground,meaning,props,note) in zip(r['evidence'],evidence): out['evidence_reviews'].append(dict(evidence_path=e['path'],grounding=ground,source_meaning=meaning,properties=[property_ref(r,*p) for p in props],notes=note))
 for b,(assessment,witnesses,note) in zip(r['bindings'],bindings):
  out['binding_reviews'].append(dict(binding_path=b['path'],feature_witnesses=[dict(feature=f,evidence_paths=paths) for f,paths in witnesses],binding_evidence_paths=[e['path'] for e in r['evidence'] if e['evidence_id'] in b['own_evidence_ids']],assessment=assessment,notes=note))
 raw=(json.dumps(out,indent=2,ensure_ascii=True,allow_nan=False)+'\n').encode()
 print(json.dumps(w.save_first_record(c,r['id'],BASE/'work',raw)))
if __name__=='__main__':
 {'report':report,'inventories':inventories}[sys.argv[1]](int(sys.argv[2]))
