import sys,json
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
CHUNK='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_17.json'
SHA='ba5eb6bf6688dc4786333182b76ea6a7b712b62a3327312edd3e8f8c75ecf010'
c=w.bound_chunk(CHUNK,SHA); n=int(sys.argv[1]);r=c['records'][n];i=r['id']
def v(p,offset=0,limit=10000):return w.source_view(c,i,p,offset,limit)
mode=sys.argv[2]
if mode=='text':
 print(json.dumps(v(['text'],int(sys.argv[3]) if len(sys.argv)>3 else 0,10000)))
elif mode=='details':
 for p in ['id','unit_id','text_sha256','reference_established','canonical_first_annotation','components','bindings']:
  z=v([p]);print(json.dumps(z))
 if not r['reference_established']:print(json.dumps(v(['reference_measurement'])))
elif mode=='evidence':
 off=int(sys.argv[3]);lim=int(sys.argv[4]);z=v(['evidence'],off,lim)
 print('evidence page',off,'count',z['total_items'],'next',z['next_offset'])
 for j,e in enumerate(z['value'],off):
  print(json.dumps({'index':j,'path':e['path'],'evidence':e['recorded_evidence'],'intersections':e['geometry']['intersections']}))
elif mode=='properties':
 indices=[int(x) for x in sys.argv[3:]]
 for ix in indices:
  print(json.dumps(v(['property_table',ix])))
elif mode=='operands':
 z=v(['evidence'],0,10000); ids=[]
 print('evidence_count',z['total_items'])
 for j,e in enumerate(z['value']):
  print('E',j,json.dumps({'path':e['path'],'recorded':e['recorded_evidence'],'geometry':e['geometry']}))
  ids.extend(x['unit_id'] for x in e['geometry']['intersections'])
 for uid in dict.fromkeys(ids):
  ix=next(j for j,x in enumerate(r['property_table']) if x['unit_id']==uid)
  print('P',ix,json.dumps(v(['property_table',ix])['value']))
