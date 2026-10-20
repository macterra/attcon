import sys,json
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
base=Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_17/work')
c=w.bound_chunk('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_17.json','ba5eb6bf6688dc4786333182b76ea6a7b712b62a3327312edd3e8f8c75ecf010')
n=int(sys.argv[1]);r=c['records'][n];spec=json.loads(Path(sys.argv[2]).read_text())
components=[]
assert len(spec['components'])==len(r['components'])
for x,s in zip(r['components'],spec['components']):
 assessment,notes=s
 components.append(dict(component_path=x['path'],original_status=x['recorded_status'],assessment=assessment,notes=notes))
evidence=[]
assert len(spec['evidence'])==len(r['evidence'])
for e,s in zip(r['evidence'],spec['evidence']):
 grounding,meaning,refs,notes=s
 props=[]
 for uid,path,role,reason in refs:
  unit=next(x for x in r['property_table'] if x['unit_id']==uid)
  props.append(dict(unit_id=uid,value_path=path,value=w.resolve(unit['value'],path),semantic_role=role,reason=reason))
 evidence.append(dict(evidence_path=e['path'],grounding=grounding,source_meaning=meaning,properties=props,notes=notes))
bindings=[]
assert len(spec['bindings'])==len(r['bindings'])
for b,s in zip(r['bindings'],spec['bindings']):
 assessment,witness_indices,notes=s
 witnesses=[dict(feature=f,evidence_paths=[r['evidence'][j]['path'] for j in js]) for f,js in witness_indices]
 bindings.append(dict(binding_path=b['path'],feature_witnesses=witnesses,binding_evidence_paths=[e['path'] for e in r['evidence'] if e['evidence_id'] in b['own_evidence_ids']],assessment=assessment,notes=notes))
out={k:r[k] for k in ['id','unit_id','text_sha256','reference_established']}
out.update(component_reviews=components,evidence_reviews=evidence,binding_reviews=bindings,additional_concerns=spec['concerns'],notes=spec['notes'])
p=base/('input_'+str(n)+'.json')
with p.open('x') as f:json.dump(out,f,indent=2,ensure_ascii=False);f.write('\n')
print(w.save_first_record(c,r['id'],base,p.read_bytes()))
