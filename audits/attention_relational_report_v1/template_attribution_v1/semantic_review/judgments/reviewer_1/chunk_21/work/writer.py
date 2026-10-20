import json,runpy,sys
from pathlib import Path
def save(n,components,evidence,bindings,concerns,notes):
 sys.argv=['session.py',str(n),'view','["reference_established"]'];s=runpy.run_path(str(Path(__file__).parent/'session.py'));r=s['r'];t={p['unit_id']:p for p in r['property_table']}
 def properties(spec):
  out=[]
  for u,paths,role,reason in spec:
   if paths=='leaves':
    paths=[]
    def walk(v,p):
     if type(v)==dict:
      for k,x in v.items():walk(x,p+[k])
     else:paths.append(p)
    walk(t[u]['value'],[])
   for path in paths:
    v=t[u]['value']
    for p in path:v=v[p]
    out.append(dict(unit_id=u,value_path=path,value=v,semantic_role=role,reason=reason))
  return out
 cr=[dict(component_path=x['path'],original_status=x['recorded_status'],assessment=a,notes=b) for x,(a,b) in zip(r['components'],components)]
 er=[dict(evidence_path=e['path'],grounding=g,source_meaning=m,properties=properties(ps),notes=note) for e,(g,m,ps,note) in zip(r['evidence'],evidence)]
 j=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=True,component_reviews=cr,evidence_reviews=er,binding_reviews=bindings,additional_concerns=concerns,notes=notes)
 print(json.dumps(s['w'].save_first_record(s['c'],s['i'],s['W'],(json.dumps(j,indent=2)+'\n').encode())))
