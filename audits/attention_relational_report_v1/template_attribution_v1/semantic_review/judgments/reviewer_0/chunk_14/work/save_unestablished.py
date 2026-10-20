import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
import attention_relational_template_review_workspace_v1 as w
p=Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_14.json');sha='1e90a6d34227a09a71008057686ce3b9b84fca019230f3509aaa18047d25905f'
c=w.bound_chunk(p,sha);i=int(sys.argv[1]);r=c['records'][i]
assert r['reference_established'] is False
assert r['canonical_first_annotation'] is None
assert not r['components'] and not r['evidence'] and not r['bindings']
assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
o={k:r[k] for k in ['id','unit_id','text_sha256','reference_established']}
o.update(component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=['Original annotation unestablished; no semantic labels salvaged.'],notes='unestablished. Original diagnostics preserved exactly: '+json.dumps(r['reference_measurement']['derivation'],ensure_ascii=True,separators=(',',':')))
raw=(json.dumps(o,indent=2)+'\n').encode();base=Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_0/chunk_14/work');draft=base/('draft_record_'+str(i).zfill(2)+'.json')
with draft.open('xb') as f:f.write(raw)
print(json.dumps(w.save_first_record(c,r['id'],base,raw)))
