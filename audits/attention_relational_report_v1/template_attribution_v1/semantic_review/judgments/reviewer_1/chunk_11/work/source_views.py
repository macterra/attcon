import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as w
BASE=Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_1/chunk_11')
CHUNK='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_11.json'
SHA='861514d2ea78d2073085b44a0042c46c17bc60a0205e9640bb087b5067d1d6a2'
c=w.bound_chunk(CHUNK,SHA)
i=int(sys.argv[1]);r=c['records'][i]
def get(p):
 v=w.source_view(c,r['id'],p,limit=10000);assert v['complete'];return v['value']
def emit(k,v):print(k,json.dumps(v,separators=(',',':')))
if sys.argv[2]=='text':
 emit('record',[i,r['id'],r['unit_id'],r['text_sha256']]);v=w.source_view(c,r['id'],['text'],offset=int(sys.argv[3]) if len(sys.argv)>3 else 0,limit=5000);emit('page',[v['offset'],v['total_items'],v['page_end'],v['next_offset']]);print(v['value']);assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
else:
 emit('established',get(['reference_established']))
 if not r['reference_established']:
  d=get(['reference_measurement','derivation']);emit('diagnostics',{k:d[k] for k in ['errors','semantic_validation','valid','text_sha256','unit_id']});emit('failed_quote_diagnostics',[q for q in d['quote_diagnostics'] if q['errors']]);emit('annotation',get(['canonical_first_annotation']));sys.exit()
 a=get(['canonical_first_annotation']);emit('component_status_rationale',[[group,key,{k:v for k,v in val.items() if k!='evidence' and k!='bindings'}] for group in ['features','auxiliary','composites'] for key,val in a[group].items()]);emit('original_bindings',a['composites'])
 emit('bindings',get(['bindings']))
 evid=get(['evidence']);emit('evidence_paths_overlaps',[[e['path'],e['recorded_evidence'],e['geometry']['intersections']] for e in evid])
 uids={x['unit_id'] for e in evid for x in e['geometry']['intersections']}
 contexts={tuple(sorted(p['context'].items())) for p in r['property_table'] if p['unit_id'] in uids and p['kind']=='origin'}
 for p in r['property_table']:
  if p['kind']=='role_peaks' and any(all(p['context'][k]==v for k,v in dict(ctx).items() if k!='role') for ctx in contexts):uids.add(p['unit_id'])
 for n,p in enumerate(r['property_table']):
  if p['unit_id'] in uids:
   emit('property_index',n)
   emit('property',{key:get(['property_table',n,key]) for key in ['unit_id','kind','context','value','packet_path','source_resolution']})
