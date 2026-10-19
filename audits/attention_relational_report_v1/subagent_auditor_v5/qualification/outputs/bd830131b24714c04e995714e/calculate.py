import json,hashlib,math
from pathlib import Path
ROOT=Path('audits/attention_relational_report_v1/subagent_auditor_v5/qualification/outputs/bd830131b24714c04e995714e')
BATCH=Path('audits/attention_relational_report_v1/subagent_auditor_v5/qualification/preparation/public/bd830131b24714c04e995714e.json')
b=json.loads(BATCH.read_bytes()); results=[]
for i,request in enumerate(b['requests']):
 p=json.loads(request['input'].rsplit('\n',1)[1]); packet=p['packet']; rows=[]
 for panel in packet['panels']:
  for view,command,nodes in [('current',None,panel['current'])]+[('candidate',cmd,nodes) for cmd,nodes in panel['candidates'].items()]:
   for node in nodes:
    addr=dict(panel=panel['panel'],view=view,command=command,node=node['node'])
    contexts=[c for c in p['source_contexts'] if all(c.get(k)==v for k,v in addr.items()) and c['kind']!='record_metadata']
    roles={}
    for role,dist in node['distributions'].items():
     cells=[(r,c,v) for r,row in enumerate(dist) for c,v in enumerate(row)]; maximum=max(v for r,c,v in cells)
     roles[role]=dict(peak_probability=maximum,peaks=[[r,c] for r,c,v in cells if v==maximum],entropy_nats=-sum(v*math.log(v) for r,c,v in cells if v),nonzero=[[r,c,v] for r,c,v in cells if v],distribution=dist)
    pairs=[(r,bl) for r in roles['red']['peaks'] for bl in roles['blue']['peaks']]
    relation=dict(pairs=pairs,all_left=all(r[1]<bl[1] for r,bl in pairs),all_above=all(r[0]<bl[0] for r,bl in pairs),any_above=any(r[0]<bl[0] for r,bl in pairs),any_below=any(r[0]>bl[0] for r,bl in pairs))
    rows.append(dict(address=addr,origin=node['origin'],roles=roles,relation=relation,contexts=contexts))
 results.append(dict(position=i,request_id=request['id'],packet_content_sha256=p['packet_content_sha256'],packet_exact_serialization_sha256=hashlib.sha256(json.dumps(packet,sort_keys=True,separators=(',',':')).encode()).hexdigest(),metadata={k:v for k,v in packet.items() if k!='panels'},metadata_contexts=[c for c in p['source_contexts'] if c['kind']=='record_metadata'],spans=p['spans'],rows=rows))
(ROOT/'calculations.json').write_text(json.dumps(results,indent=2)+'\n')
for item in results:
 print('\nPOSITION',item['position'],'packet',item['packet_content_sha256']);print('SPANS',json.dumps(item['spans']))
 for row in item['rows']:
  a=row['address']; relevant=(a['view']=='current') or (item['position']==1 and a['panel']=='p1' and a['command']=='k1' and a['node']=='n0') or (item['position'] in [9,11] and a['panel']=='p0' and a['command']=='k0' and a['node']=='n1') or (item['position']==23 and a['panel']=='p0' and a['command']=='k2' and a['node']=='n0')
  if relevant:
   print(a,'origin',row['origin'],'ctx',[(c['context_id'],c['kind'],c.get('role')) for c in row['contexts']]);print('roles', {r:{k:v for k,v in d.items() if k!='distribution'} for r,d in row['roles'].items()},'relation',row['relation'])
