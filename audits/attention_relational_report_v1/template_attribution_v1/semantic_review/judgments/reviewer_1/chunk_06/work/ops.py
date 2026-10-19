import sys,json
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as h
CP='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_06.json'
CH='b8ee8f672053efcae8359b3688fbeb9d231d88f772241cf897ae50be7ec15708'
WORK=Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_06/work')
c=h.bound_chunk(CP,CH)
r=c['records'][int(sys.argv[2])-1]
def view(path,offset=0,limit=10000):
 v=h.source_view(c,r['id'],path,offset,limit)
 print('ID',r['id'],'PATH',json.dumps(path),'TOTAL',v['total_items'],'OFFSET',v['offset'],'END',v['page_end'],'COMPLETE',v['complete'],'NEXT',v['next_offset'])
 print(v['value'] if isinstance(v['value'],str) else json.dumps(v['value'],ensure_ascii=False))
if sys.argv[1]=='view':
 view(json.loads(sys.argv[3]),int(sys.argv[4]) if len(sys.argv)>4 else 0,int(sys.argv[5]) if len(sys.argv)>5 else 10000)
elif sys.argv[1]=='invalid':
 assert r['reference_established'] is False and r['review_state']=='unestablished'
 a=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=False,component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='unestablished; original annotation is not salvaged. Original derivation diagnostics: '+json.dumps(r['reference_measurement']['derivation'],ensure_ascii=False,sort_keys=True))
 print(h.save_first_record(c,r['id'],WORK,json.dumps(a,ensure_ascii=False,indent=2).encode()))
elif sys.argv[1]=='meta':
 for p in [['unit_id'],['text_sha256'],['reference_established'],['review_state'],['components'],['bindings']]:view(p)
elif sys.argv[1]=='save':
 print(h.save_first_record(c,r['id'],WORK,Path(sys.argv[3]).read_bytes()))
