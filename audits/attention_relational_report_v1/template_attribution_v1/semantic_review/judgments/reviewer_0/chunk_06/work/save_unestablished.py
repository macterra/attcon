import sys,json
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as h
c=h.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_06.json','b8ee8f672053efcae8359b3688fbeb9d231d88f772241cf897ae50be7ec15708')
i=int(sys.argv[1]);r=c['records'][i];assert r['reference_established'] is False
w=Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_0/chunk_06/work')
a=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=False,component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes=sys.argv[2]+' Original derivation diagnostics: '+json.dumps({'errors':r['reference_measurement']['derivation']['errors'],'semantic_validation':r['reference_measurement']['derivation']['semantic_validation'],'failed_quote_diagnostics':[d for d in r['reference_measurement']['derivation']['quote_diagnostics'] if d['errors']]},sort_keys=True))
p=w/f'authored_{i:02}.json'
with p.open('x') as f:f.write(json.dumps(a,indent=2)+'\n')
print(h.save_first_record(c,r['id'],w,p.read_bytes()))
