import sys,json,runpy
plan_path=sys.argv[2];num=sys.argv[1]
sys.argv=['review_local.py',num,'noop']
d=runpy.run_path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_22/work/review_local.py');r=d['r'];w=d['w'];p=json.load(open(plan_path));units={u['unit_id']:u for u in r['property_table']}
rec=dict(id=r['id'],unit_id=r['unit_id'],text_sha256=r['text_sha256'],reference_established=r['reference_established'],component_reviews=[],evidence_reviews=[],binding_reviews=p['bindings'],additional_concerns=p['concerns'],notes=p['notes'])
for c,a in zip(r['components'],p['components'],strict=True):rec['component_reviews'].append(dict(component_path=c['path'],original_status=c['recorded_status'],assessment=a[0],notes=a[1]))
for e,a in zip(r['evidence'],p['evidence'],strict=True):
 props=[]
 for uid,path,role,reason in a[2]:props.append(dict(unit_id=uid,value_path=path,value=w.resolve(units[uid]['value'],path),semantic_role=role,reason=reason))
 rec['evidence_reviews'].append(dict(evidence_path=e['path'],grounding=a[0],source_meaning=a[1],properties=props,notes=a[3]))
raw=(json.dumps(rec,ensure_ascii=True,indent=2)+'\n').encode();print(json.dumps(w.save_first_record(d['c'],r['id'],d['WORK'],raw)))
