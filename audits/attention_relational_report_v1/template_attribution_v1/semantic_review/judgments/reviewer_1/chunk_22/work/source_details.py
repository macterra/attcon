import sys,json,runpy
num=sys.argv[1];extra=[int(x) for x in sys.argv[2:]];sys.argv=['review_local.py',num,'noop'];d=runpy.run_path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_22/work/review_local.py');w,c,r=d['w'],d['c'],d['r']
for key in ['components','canonical_first_annotation','bindings']:
 print(key,json.dumps(w.source_view(c,r['id'],[key],0,100)['value']))
units=set(extra)
for n in range(len(r['evidence'])):
 e=w.source_view(c,r['id'],['evidence',n],0,100)['value'];print('E',n,json.dumps(e['path']),json.dumps(e['recorded_evidence']));units.update(int(i['unit_id'][1:]) for i in e['geometry']['intersections'])
for n in sorted(units):print('UNIT',json.dumps(w.source_view(c,r['id'],['property_table',n],0,100)['value']))
