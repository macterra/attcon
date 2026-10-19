import json,hashlib,math,re
from pathlib import Path
ROOT=Path('audits/attention_relational_report_v1/subagent_auditor_v5/qualification/outputs/ba31238fcb404e000717adfdb')
BATCH=Path('audits/attention_relational_report_v1/subagent_auditor_v5/qualification/preparation/public/ba31238fcb404e000717adfdb.json')
batch=json.loads(BATCH.read_bytes()); results=[]
for req in batch['requests']:
 q=json.loads(req['input'].rsplit('\n',1)[1]); records=[]
 for c in q['source_contexts']:
  if c['kind']!='role_distribution':continue
  d=q['packet']
  for x in c['packet_path']:d=d[x]
  maximum=max(v for row in d for v in row)
  peaks=[[r,col] for r,row in enumerate(d) for col,v in enumerate(row) if v==maximum]
  records.append(dict(context_id=c['context_id'],packet_path=c['packet_path'],panel=c['panel'],view=c['view'],command=c['command'],node=c['node'],role=c['role'],origin=c['origin'],distribution_sha256=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest(),peaks=peaks,maximum=maximum,entropy=-sum(v*math.log(v) for row in d for v in row if v),expected_row=sum(r*v for r,row in enumerate(d) for v in row),expected_column=sum(col*v for row in d for col,v in enumerate(row))))
 results.append(dict(request_id=req['id'],unit_id=q['unit_id'],packet_sha256=q['packet_content_sha256'],report_sha256=q['report_text_sha256'],records=records))
ROOT.joinpath('calculations.json').write_text(json.dumps(results,indent=2)+'\n')
for i,(req,res) in enumerate(zip(batch['requests'],results)):
 q=json.loads(req['input'].rsplit('\n',1)[1]); print('\nREQUEST',i,req['id'],q['unit_id'])
 print('SPANS:')
 for s in q['spans']:print(s['span_id'],s['text'] if len(s['text'])<1000 else s['text'][-550:])
 print('ORIGINS:')
 for c in q['source_contexts']:
  if c['kind']=='node_origin':print(c['context_id'],c['panel'],c['view'],c['command'],c['node'],c['origin'])
 mismatches=[]
 pattern=r'(p\d)/(current|candidate/(k\d))/(n\d)/(red|blue)=\{([^}]*)\}'
 for m in re.finditer(pattern,q['report_text']):
  panel,view,command,node,role,coords=m.groups(); command=command or None; view='current' if view=='current' else 'candidate'
  row=next(r for r in res['records'] if (r['panel'],r['view'],r['command'],r['node'],r['role'])==(panel,view,command,node,role))
  claimed=[list(map(int,x)) for x in re.findall(r'\((\d+),(\d+)\)',coords)]
  if sorted(claimed)!=sorted(row['peaks']):mismatches.append((m.group(),row['context_id'],row['peaks']))
 print('PEAK MISMATCHES',mismatches)
 print('red columns first three',[(r['node'],r['peaks']) for r in res['records'] if r['panel']=='p0' and r['view']=='current' and r['role']=='red'][:3])
