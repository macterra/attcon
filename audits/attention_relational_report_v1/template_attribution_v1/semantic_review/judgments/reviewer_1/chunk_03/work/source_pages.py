import sys,json
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
C='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_03.json'
H='c5f5f6290de81b9cf87c770687b2999a7c70321d0dfdfa7e90012155827f5289'
c=w.bound_chunk(C,H)
n=int(sys.argv[1]);i=c['ordered_template_ids'][n];r=c['records'][n]
def v(p,offset=0,limit=20):
 x=w.source_view(c,i,p,offset,limit)
 print('PATH',json.dumps(p),'PAGE',offset,x['page_end'],'TOTAL',x['total_items'],'NEXT',x['next_offset'])
 print(json.dumps(x['value'],separators=(',',':')))
mode=sys.argv[2]
if mode=='text':
 for k in ['id','unit_id','text_sha256','reference_established']:v([k],limit=200)
 for off in range(0,len(r['text']),3000):v(['text'],off,3000)
elif mode=='annotation':
 for k in ['features','auxiliary','composites']:v(['canonical_first_annotation',k],limit=20)
 v(['bindings'],limit=5)
elif mode=='evidence':
 start=int(sys.argv[3]);stop=int(sys.argv[4])
 for j in range(start,min(stop,len(r['evidence']))):
  v(['evidence',j,'path'],limit=20);v(['evidence',j,'recorded_evidence'],limit=20);v(['evidence',j,'geometry'],limit=20)
elif mode=='units':
 for uid in sys.argv[3:]:
  idx=next(j for j,u in enumerate(r['property_table']) if u['unit_id']==uid)
  v(['property_table',idx],limit=20)
elif mode=='diagnostics':
 v(['reference_measurement','derivation'],limit=20)
 v(['canonical_first_annotation'],limit=10)
