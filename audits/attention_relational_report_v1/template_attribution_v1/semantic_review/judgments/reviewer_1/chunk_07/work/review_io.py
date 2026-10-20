import sys,json,pathlib
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as h
CHUNK='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_07.json'
SHA='75ae0b350529484b0168488f74a4d9270567afe75d8ec1c47cd93598dcf8527e'
W=pathlib.Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_07/work')
c=h.bound_chunk(CHUNK,SHA)
def prop(r,i,path,role,reason):
 u=r['property_table'][i]
 return dict(unit_id=u['unit_id'],value_path=path,value=h.resolve(u['value'],path),semantic_role=role,reason=reason)
def save(n,components,evidence,bindings,concerns,notes):
 r=c['records'][n]
 j={k:r[k] for k in ('id','unit_id','text_sha256','reference_established')}
 j.update(component_reviews=[dict(component_path=x['path'],original_status=x['recorded_status'],assessment=a,notes=b) for x,(a,b) in zip(r['components'],components)],evidence_reviews=[dict(evidence_path=r['evidence'][i]['path'],grounding=a,source_meaning=b,properties=p,notes=q) for i,a,b,p,q in evidence],binding_reviews=bindings,additional_concerns=concerns,notes=notes)
 p=W/f'input_{n:02}.json';p.write_text(json.dumps(j,indent=2)+'\n');print(h.save_first_record(c,r['id'],W,p.read_bytes()))
if __name__=='__main__':
 n=int(sys.argv[1]);r=c['records'][n]
 paths=json.loads(sys.argv[2])
 for p in paths:
  v=h.source_view(c,r['id'],p,0,10000)
  print(json.dumps(v,separators=(',',':')))
