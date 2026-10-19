import json,sys
from pathlib import Path
sys.path.insert(0,'scripts');import attention_relational_template_review_workspace_v1 as h
w=Path(__file__).parent
c=h.bound_chunk(Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_00.json'),'1634476fd3bd669ab467dad2137fbe29b50b2a6064d3b412c13d003578ccc3f6')
def setup(n):return c['records'][n]
def prop(r,n,path,role,reason):
 u=r['property_table'][n];v=u['value']
 for p in path:v=v[p]
 return dict(unit_id=u['unit_id'],value_path=path,value=v,semantic_role=role,reason=reason)
def save(r,n,components,evidence,bindings,concerns,notes):
 # Mechanical serialization only; every semantic assessment, note, path inventory and operand is caller-supplied.
 assert len(components)==len(r['components']) and len(evidence)==len(r['evidence']) and len(bindings)==len(r['bindings'])
 cs=[dict(component_path=x['path'],original_status=x['recorded_status'],assessment=a,notes=s) for x,(a,s) in zip(r['components'],components)]
 es=[dict(evidence_path=x['path'],grounding=g,source_meaning=m,properties=p,notes=s) for x,(g,m,p,s) in zip(r['evidence'],evidence)]
 o={k:r[k] for k in ['id','unit_id','text_sha256','reference_established']};o.update(component_reviews=cs,evidence_reviews=es,binding_reviews=bindings,additional_concerns=concerns,notes=notes)
 p=w/('input_%02d.json'%n);p.open('x').write(json.dumps(o,indent=2)+'\n');print(h.save_first_record(c,r['id'],w,p.read_bytes()))
