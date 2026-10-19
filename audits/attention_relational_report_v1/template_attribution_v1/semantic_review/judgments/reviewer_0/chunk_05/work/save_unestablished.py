import sys,json
from pathlib import Path
sys.path.insert(0,'scripts');import attention_relational_template_review_workspace_v1 as w
c=w.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_05.json','9821af4a85f4f4f5c5b4fd00d5516a46fcb0e0142af0b0629c8a5fda898b2bc9')
r=c['records'][int(sys.argv[1])]
assert r['reference_established'] is False
j={k:r[k] for k in ['id','unit_id','text_sha256','reference_established']}
j.update(component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='unestablished. '+sys.argv[2]+' Original diagnostics: '+json.dumps(r['reference_measurement']['derivation'],ensure_ascii=False))
p=Path(__file__).parent/('author_'+sys.argv[1]+'.json')
with p.open('x') as f:json.dump(j,f,ensure_ascii=False,indent=2)
print(json.dumps(w.save_first_record(c,r['id'],Path(__file__).parent,p.read_bytes())))
