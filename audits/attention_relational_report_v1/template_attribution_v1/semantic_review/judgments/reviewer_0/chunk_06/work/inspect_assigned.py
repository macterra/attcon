import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as h
assert hashlib.sha256(Path('scripts/attention_relational_template_review_workspace_v1.py').read_bytes()).hexdigest()=='20834fcb9c47ff0376088d5800196cbbd42763348a10f329f96a689fad89060b'
c=h.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_06.json','b8ee8f672053efcae8359b3688fbeb9d231d88f772241cf897ae50be7ec15708')
r=c['records'][int(sys.argv[1])]
def view(p,offset=0,limit=20):
 v=h.source_view(c,r['id'],p,offset,limit)
 print(json.dumps({k:v[k] for k in ['source_path','total_items','offset','page_end','next_offset','value']},ensure_ascii=False))
print('RECORD',sys.argv[1],r['id'])
if sys.argv[2]=='text':
 for o in range(0,len(r['text']),4000):view(['text'],o,4000)
elif sys.argv[2]=='meta':
 for p in [['reference_established'],['components'],['bindings'],['evidence']]:view(p,0,12)
 if not r['reference_established']:
  for p in [['reference_measurement','derivation','errors'],['reference_measurement','derivation','semantic_validation']]:view(p)
else:view(json.loads(sys.argv[2]),int(sys.argv[3]) if len(sys.argv)>3 else 0,int(sys.argv[4]) if len(sys.argv)>4 else 10)
