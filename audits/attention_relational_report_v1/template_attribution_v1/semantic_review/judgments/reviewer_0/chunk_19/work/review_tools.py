import sys,json,pathlib
sys.path.insert(0,str(pathlib.Path.cwd()/'scripts'))
import attention_relational_template_review_workspace_v1 as w
CP='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_19.json'
SHA='d0e3176477f5885024db0d705318b4ab6b4b53df7ed2e712b4c714e72fdd6b52'
WD=pathlib.Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_0/chunk_19/work')
c=w.bound_chunk(CP,SHA)
def view(i,p,offset=0,limit=10000):
 return w.source_view(c,c['records'][i]['id'],p,offset,limit)
def report(i):
 r=c['records'][i]
 for p in [['id'],['unit_id'],['text_sha256'],['reference_established'],['text']]: print(json.dumps(view(i,p),separators=(',',':')))
def details(i):
 r=c['records'][i]
 for p in [['canonical_first_annotation'],['bindings']]: print(json.dumps(view(i,p),separators=(',',':')))
 for n in range(len(r['evidence'])):
  print(json.dumps(view(i,['evidence',n]),separators=(',',':')))
 used={x['unit_id'] for e in r['evidence'] for x in e['geometry']['intersections']}
 for n,u in enumerate(r['property_table']):
  if u['unit_id'] in used: print(json.dumps(view(i,['property_table',n]),separators=(',',':')))
 if not r['reference_established']: print(json.dumps(view(i,['reference_measurement']),separators=(',',':')))
def prop(i,uid,path,role,reason):
 u=next(u for u in c['records'][i]['property_table'] if u['unit_id']==uid)
 return dict(unit_id=uid,value_path=path,value=w.resolve(u['value'],path),semantic_role=role,reason=reason)
def save(i,components,evidence,bindings,concerns,notes):
 r=c['records'][i]
 out=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=r['reference_established'],component_reviews=components,evidence_reviews=evidence,binding_reviews=bindings,additional_concerns=concerns,notes=notes)
 raw=(json.dumps(out,indent=2)+'\n').encode();draft=WD/f'judgment_{i:02d}.json';draft.open('xb').write(raw)
 print(w.save_first_record(c,r['id'],WD,raw))
def cr(path,status,assessment,notes):return dict(component_path=path,original_status=status,assessment=assessment,notes=notes)
def er(path,grounding,meaning,properties,notes):return dict(evidence_path=path,grounding=grounding,source_meaning=meaning,properties=properties,notes=notes)
if __name__=='__main__':
 {'report':report,'details':details}[sys.argv[1]](int(sys.argv[2]))
