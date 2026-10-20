import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path('scripts').resolve()))
import attention_relational_template_review_workspace_v1 as h
assert hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest()=='20834fcb9c47ff0376088d5800196cbbd42763348a10f329f96a689fad89060b'
x=h.bound_chunk(Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_12.json'),'ae80eb6032940985f0e44b2993e800da8f95bfe7ec26edff50e1d569c3a2026b')
r=x['records'][int(sys.argv[1])-1]
paths=json.loads(sys.argv[2])
for path in paths:
 v=h.source_view(x,r['id'],path,int(sys.argv[3]) if len(sys.argv)>3 else 0,10000); print(json.dumps({k:v[k] for k in ['id','source_path','total_items','value','page_end','next_offset']},ensure_ascii=False))
