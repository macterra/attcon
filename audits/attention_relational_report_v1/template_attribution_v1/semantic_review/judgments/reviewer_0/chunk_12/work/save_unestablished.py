import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path('scripts').resolve()))
import attention_relational_template_review_workspace_v1 as h
assert hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest()=='20834fcb9c47ff0376088d5800196cbbd42763348a10f329f96a689fad89060b'
x=h.bound_chunk(Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_12.json'),'ae80eb6032940985f0e44b2993e800da8f95bfe7ec26edff50e1d569c3a2026b')
r=x['records'][int(sys.argv[1])-1]
assert r['reference_established'] is False and r['review_state']=='unestablished'
assert r['components']==[] and r['evidence']==[] and r['bindings']==[] and r['canonical_first_annotation'] is None
out={k:r[k] for k in ['id','unit_id','text_sha256','reference_established']}
out.update(component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=['Original annotation is unestablished; no semantic labels are salvaged or inferred from available report text.'],notes='unestablished; original reference measurement and diagnostics preserved exactly: '+json.dumps(r['reference_measurement'],ensure_ascii=False,sort_keys=True))
raw=(json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode()
print(json.dumps(h.save_first_record(x,r['id'],Path(__file__).parent,raw)))
