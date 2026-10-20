import sys,json,pathlib,hashlib
sys.path.insert(0,str(pathlib.Path.cwd()/'scripts'))
import attention_relational_template_review_workspace_v1 as h
CHUNK='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_13.json'
SHA='cda7fcb6e29aa7076403202a25c76a764aab8ea322c91eaa90dc38fb06448bf0'
WORK=pathlib.Path(__file__).parent
c=h.bound_chunk(CHUNK,SHA)
def row(n):return c['records'][n]
def view(n,path,offset=0,limit=100):
 v=h.source_view(c,row(n)['id'],path,offset,limit)
 print(json.dumps(v,ensure_ascii=False))
def prop(n,unit,path,role,reason):
 t=next(x for x in row(n)['property_table'] if x['unit_id']==unit)
 return dict(unit_id=unit,value_path=path,value=h.resolve(t['value'],path),semantic_role=role,reason=reason)
def save(n,components,evidence,bindings,concerns,notes):
 r=row(n)
 out=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=r['reference_established'],component_reviews=components,evidence_reviews=evidence,binding_reviews=bindings,additional_concerns=concerns,notes=notes)
 raw=(json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode()
 print(json.dumps(h.save_first_record(c,r['id'],WORK,raw)))
def comp(path,status,assessment,notes):return dict(component_path=path,original_status=status,assessment=assessment,notes=notes)
def evid(path,grounding,meaning,properties,notes):return dict(evidence_path=path,grounding=grounding,source_meaning=meaning,properties=properties,notes=notes)
if __name__=='__main__':
 n=int(sys.argv[1]);path=json.loads(sys.argv[2]);view(n,path,int(sys.argv[3]) if len(sys.argv)>3 else 0,int(sys.argv[4]) if len(sys.argv)>4 else 100)
