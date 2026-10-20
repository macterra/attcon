import sys,json
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
c=w.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_15.json','c0d951e9d053c8952703280c254ff6c7619a920d2261a820975e40a10d106d45');i=c['ordered_template_ids'][int(sys.argv[1])]
def v(p):return w.source_view(c,i,p,0,10000)['value']
assert v(['reference_established']) is False
r={k:v([k]) for k in ['id','unit_id','text_sha256','reference_established']}
d=v(['reference_measurement','derivation'])
r.update(component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='Unestablished original. No semantic salvage; inventories remain empty. Original diagnostics: '+json.dumps({k:d[k] for k in ['errors','semantic_validation','valid']},ensure_ascii=False))
print(w.save_first_record(c,i,Path(__file__).parent,(json.dumps(r,ensure_ascii=False,indent=2)+'\n').encode()))
