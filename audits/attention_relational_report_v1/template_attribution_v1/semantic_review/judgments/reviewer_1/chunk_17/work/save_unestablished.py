import sys,json
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
base=Path(__file__).parent
c=w.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_17.json','ba5eb6bf6688dc4786333182b76ea6a7b712b62a3327312edd3e8f8c75ecf010')
n=int(sys.argv[1]);r=c['records'][n];assert r['reference_established'] is False
assert r['components']==[] and r['evidence']==[] and r['bindings']==[] and r['canonical_first_annotation'] is None
out={k:r[k] for k in ['id','unit_id','text_sha256','reference_established']}
out.update(component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='unestablished; original diagnostics preserved without semantic salvage: '+json.dumps(r['reference_measurement'],sort_keys=True))
p=base/('input_'+str(n)+'.json')
with p.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(w.save_first_record(c,r['id'],base,p.read_bytes()))
