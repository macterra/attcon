import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
import attention_relational_template_review_workspace_v1 as h
CHUNK='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_22.json'
SHA='45748ba13e82811ac3b495f913a9ef3e2f30824dc5303f5deea085b7f0316395'
WORK='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_0/chunk_22/work'
c=h.bound_chunk(CHUNK,SHA)
def view(i,path,offset=0,limit=20):
 print(json.dumps(h.source_view(c,c['records'][i]['id'],path,offset,limit)))
def base(i):
 r=c['records'][i];return {k:r[k] for k in ['id','unit_id','text_sha256','reference_established']}
def prop(i,unit,path,role,reason):
 t=next(x for x in c['records'][i]['property_table'] if x['unit_id']==unit)
 return dict(unit_id=unit,value_path=path,value=h.resolve(t['value'],path),semantic_role=role,reason=reason)
def comp(i,k,assessment,notes):
 x=c['records'][i]['components'][k];return dict(component_path=x['path'],original_status=x['recorded_status'],assessment=assessment,notes=notes)
def ev(i,k,grounding,source_meaning,properties,notes):
 return dict(evidence_path=c['records'][i]['evidence'][k]['path'],grounding=grounding,source_meaning=source_meaning,properties=properties,notes=notes)
def save(i,record):
 raw=json.dumps(record,ensure_ascii=False,indent=2).encode(); print(json.dumps(h.save_first_record(c,c['records'][i]['id'],WORK,raw)))
def save_unestablished(i,notes):
 r=c['records'][i];o=base(i);o.update(component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes=notes+' Original derivation diagnostics: '+json.dumps({k:r['reference_measurement']['derivation'][k] for k in ['errors','quote_diagnostics','valid']},ensure_ascii=False));save(i,o)
def inspect(i):
 r=c['records'][i]
 view(i,['canonical_first_annotation'],0,10)
 view(i,['bindings'],0,100)
 ids=[]
 for off in range(0,len(r['evidence']),5):
  z=h.source_view(c,r['id'],['evidence'],off,5)
  print('evidence_page',off,'total',z['total_items'],'next',z['next_offset'])
  for e in z['value']:
   units=[v['unit_id'] for v in e['geometry']['intersections']]
   print(json.dumps(dict(path=e['path'],unit_ids=units,geometry=e['geometry'])))
   ids.extend(units)
 for uid in dict.fromkeys(ids):
  j=next(k for k,p in enumerate(r['property_table']) if p['unit_id']==uid)
  view(i,['property_table',j],0,30)
def concise(i,path,offset=0,limit=20):
 z=h.source_view(c,c['records'][i]['id'],path,offset,limit)
 print(json.dumps(dict(id=z['id'],path=path,total=z['total_items'],offset=offset,next=z['next_offset'],value=z['value'])))
def detail(i):
 r=c['records'][i]
 concise(i,['canonical_first_annotation'],0,10)
 concise(i,['bindings'],0,100)
 ids=[]
 for off in range(0,len(r['evidence']),5):
  z=h.source_view(c,r['id'],['evidence'],off,5)
  print('evidence_page',off,'total',z['total_items'],'next',z['next_offset'])
  for e in z['value']:
   units=[v['unit_id'] for v in e['geometry']['intersections']]
   print(json.dumps(dict(path=e['path'],units=units)))
   ids.extend(units)
 for uid in dict.fromkeys(ids):
  j=next(k for k,p in enumerate(r['property_table']) if p['unit_id']==uid)
  z=h.source_view(c,r['id'],['property_table',j],0,30)
  p=z['value'];print(json.dumps({k:p[k] for k in ['unit_id','kind','context','value','source_resolution']}))
