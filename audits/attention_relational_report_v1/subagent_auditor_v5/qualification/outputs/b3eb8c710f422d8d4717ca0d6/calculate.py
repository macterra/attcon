import json,math,hashlib
from pathlib import Path
ROOT=Path('audits/attention_relational_report_v1/subagent_auditor_v5/qualification/outputs/b3eb8c710f422d8d4717ca0d6')
BATCH=Path('audits/attention_relational_report_v1/subagent_auditor_v5/qualification/preparation/public/b3eb8c710f422d8d4717ca0d6.json')
b=json.loads(BATCH.read_bytes()); result=[]
for req in b['requests']:
 p=json.loads(req['input'].rsplit('\n',1)[1]); packet=p['packet']; rows=[]
 for c in p['source_contexts']:
  row={'context_id':c['context_id'],'kind':c['kind']}
  for k in ('panel','view','command','node','role','origin','packet_path','metadata','operands'):
   if k in c:row[k]=c[k]
  if c['kind']=='role_distribution':
   v=packet
   for k in c['packet_path']:v=v[k]
   cells=[(i,j,x) for i,r in enumerate(v) for j,x in enumerate(r)];mx=max(x for i,j,x in cells)
   row['computed']={'dimensions':[len(v),len(v[0])],'peaks':[[i,j] for i,j,x in cells if x==mx], 'peak_probability':mx,'nonzero_cells':[[i,j,x] for i,j,x in cells if x!=0], 'entropy_nats':-sum(x*math.log(x) for i,j,x in cells if x>0),'expected_row':sum(i*x for i,j,x in cells),'expected_column':sum(j*x for i,j,x in cells)}
  rows.append(row)
 # Relations derive from the exact retained operands, using this request alone.
 byid={r['context_id']:r for r in rows}
 for r in rows:
  if r['kind']=='red_blue_peak_relation':
   a,z=[byid[k]['computed']['peaks'] for k in r['operands']]
   pairs=[(x,y) for x in a for y in z]
   r['computed']={'horizontal':sorted({('left_of' if x[1]<y[1] else 'right_of' if x[1]>y[1] else 'same_column') for x,y in pairs}), 'vertical':sorted({('above' if x[0]<y[0] else 'below' if x[0]>y[0] else 'same_row') for x,y in pairs}),'all_left':all(x[1]<y[1] for x,y in pairs),'all_above':all(x[0]<y[0] for x,y in pairs)}
 result.append({'request_id':req['id'],'unit_id':p['unit_id'],'packet_content_sha256':p['packet_content_sha256'],'independent_packet_sha256':hashlib.sha256(json.dumps(packet,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest(),'request_input_sha256':hashlib.sha256(req['input'].encode()).hexdigest(),'metadata':{k:v for k,v in packet.items() if k!='panels'},'contexts':rows})
(ROOT/'calculations.json').write_text(json.dumps(result,indent=2)+'\n')
for i,r in enumerate(result):
 print('\nPOSITION',i,r['request_id'],'packet',r['packet_content_sha256'])
 # All operands referenced in these reports, plus metadata.
 for c in r['contexts']:
  address=(c.get('panel'),c.get('view'),c.get('command'),c.get('node'))
  wanted=address in [('p0','current',None,'n0'),('p0','current',None,'n1'),('p0','current',None,'n2'),('p0','current',None,'n3'),('p1','current',None,'n0'),('p0','candidate','k0','n1'),('p1','candidate','k1','n0'),('p0','candidate','k2','n0')]
  if wanted and c['kind']!='red_blue_peak_relation':print(json.dumps(c,separators=(',',':')))
