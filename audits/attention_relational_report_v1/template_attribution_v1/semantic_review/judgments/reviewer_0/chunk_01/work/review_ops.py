import sys,json
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
CHUNK='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_01.json'
SHA='5e5797c6c195d8a7e569f94e3d6455a34e81423f1c2fc5feb7da7c8215fd84fb'
WORK='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_0/chunk_01/work'
c=w.bound_chunk(CHUNK,SHA)
def base(i):
 r=c['records'][i]; return {k:r[k] for k in ['id','unit_id','text_sha256','reference_established']}
def save(i,out):
 r=c['records'][i]; print(w.save_first_record(c,r['id'],WORK,json.dumps(out,ensure_ascii=False,indent=2).encode()+b'\n'))
def view(i,path,offset=0,limit=100): print(json.dumps(w.source_view(c,c['records'][i]['id'],path,offset,limit)))
def prop(i,unit,path,role,reason):
 u=next(u for u in c['records'][i]['property_table'] if u['unit_id']==unit)
 return dict(unit_id=unit,value_path=path,value=w.resolve(u['value'],path),semantic_role=role,reason=reason)
if __name__=='__main__':
 i=int(sys.argv[1]); view(i,['text'],0,10000)
 for key in ['id','unit_id','text_sha256','reference_established','review_state','components','bindings']: view(i,[key])
 if not c['records'][i]['reference_established']:
  for path in [['reference_measurement','derivation','errors'],['reference_measurement','derivation','semantic_validation'],['evidence']]: view(i,path)
