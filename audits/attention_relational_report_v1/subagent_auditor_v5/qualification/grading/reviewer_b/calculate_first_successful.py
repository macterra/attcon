from workbook import *
import re, math, ast, base64

def dig(v,path):
 for p in path:v=v[p]
 return v

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True)
def sha(x):return hashlib.sha256(x).hexdigest()
def load_interfaces():
 return {q['id']:json.loads(q['input'].rsplit('\n',1)[1]) for b in P['public_batches'].values() for q in b['requests']}
INTER=load_interfaces()
# Explicit semantic mapping selected after reading all short reports and actual assertions.
SHORT={}
for r in ROWS[:48]:
 n=len(K['short',r['case_id']]['mandatory_atomic_assertions']);SHORT[r['request_id']]={f'a{i+1:02}':[f'a{i+1:03}'] for i in range(n)}
 if r['case_id']=='aqv4_case_007':SHORT[r['request_id']]={'a01':['a001','a002'],'a02':['a003']}
 if r['case_id']=='aqv4_case_014' and r['stream']=='stream_0':SHORT[r['request_id']]['a03']=['a003','a004']
# Tail maps are manual semantic judgments, including explicit same-verdict grouping.
TAILS={
48:{68:[68],69:[69],70:[69],71:[70],72:[70],73:[71],74:list(range(72,96)),75:[96],76:[97],77:[97],78:[98],79:[99],80:[100],81:[101],82:[102],83:[103],84:[104],85:[105,106],86:[106],87:[107]},
49:{68:[68],69:[69],70:[70],71:[71],72:[71],73:list(range(72,96)),74:[96],75:[96],76:[97],77:[98],78:[99],79:[100],80:[101],81:[102],82:[103],83:[104],84:[105],85:[106]},
50:{68:[68],69:[69],70:[70],71:[71],72:[72],73:list(range(73,97)),74:[97],75:[98],76:[99],77:[95],78:[100],79:[101],80:[102],81:[103],82:[104],83:[105],84:[106],85:[107],86:[108],87:[109]},
51:{68:[68],69:[69],70:[70],71:[71],72:[72],73:list(range(73,97)),74:[97],75:[97],76:[98],77:[99],78:[100],79:[101],80:[102],81:[103],82:[104],83:[105],84:[106],85:[107],86:[108,109],87:[110]},
52:{68:[69],69:[70],70:[70],71:[71],72:[71],73:[71],74:[72],75:[73],76:[74],77:[74],78:[75],79:[76],80:[77,78,79],81:[80],82:[81],83:[82],84:[83],85:[83,84],86:[84],87:[85]},
53:{68:[69],69:[69],70:[70],71:[70],72:[70],73:[71],74:[72,74,76],75:[72,74,76],76:[73,75,77],77:[79],78:[78],79:[80,81,82],80:[83],81:[84],82:[85],83:[86],84:[87],85:[88]},
54:{68:[69],69:[70],70:[71],71:[72],72:[73],73:[74],74:[75],75:[76],76:[77],77:[78],78:[79],79:[81],80:[80],81:[82,83,84],82:[85],83:[86],84:[3],85:[87],86:[88],87:[89]},
55:{68:[69],69:[70],70:[71],71:[72],72:[72],73:[73],74:[74],75:[74],76:[75],77:[76],78:[77,78,79],79:[80],80:[81],81:[3,80],82:[82],83:[83],84:[84],85:[85],86:[86],87:[87]}}

def selected(r,aa):
 cm={c['context_id']:c for c in INTER[r['request_id']]['source_contexts']}
 ids=list(dict.fromkeys(cid for a in aa for cid in a['source_context_ids']))
 out=[]
 for cid in ids:
  if cid not in cm:continue
  c=cm[cid];out.append(c)
  for oid in c.get('operands',[]):out.append(cm[oid])
 return list({c['context_id']:c for c in out}.values())

def operand_present(op,contexts):
 for c in contexts:
  scope=op['scope']
  if scope in ('record_metadata','packet_metadata'):
   if c['kind']=='record_metadata' and c['packet_path']==[]:
    val=c['metadata']
    try:dig(val,op.get('field','').split('.') if op.get('field') else []);return True
    except (KeyError,TypeError):pass
  else:
   if any(op.get(k)!=c.get(k) for k in ['panel','view','command','node']):continue
   if scope=='role_distribution' and c['kind']=='role_distribution' and op.get('role')==c.get('role'):return True
   if scope in ('node_metadata','node_origin') and 'origin' in c:return True
 return False

def summary(mat):
 vals=[x for row in mat for x in row];m=max(vals)
 return {'peaks':[[i,j] for i,row in enumerate(mat) for j,x in enumerate(row) if x==m], 'peak_probability':m,'entropy_nats':-sum(x*math.log(x) for x in vals if x),'expected_row':sum(i*x for i,row in enumerate(mat) for x in row),'expected_column':sum(j*x for row in mat for j,x in enumerate(row))}

def source_calc(r,contexts):
 p=INTER[r['request_id']]['packet'];out=[]
 for c in contexts:
  if c['kind']=='role_distribution':
   mat=dig(p,c['packet_path']);out.append({'context_id':c['context_id'],'address':{k:c[k] for k in ['panel','view','command','node','role']},'packet_path':c['packet_path'],'full_matrix':mat,'independent_calculation':summary(mat),'full_origin':c['origin']})
  elif c['kind']=='node_origin':out.append({'context_id':c['context_id'],'packet_path':c['packet_path'],'full_origin':c['origin']})
  elif c['kind']=='record_metadata':out.append({'context_id':c['context_id'],'packet_path':c['packet_path'],'actual_metadata':c['metadata']})
  elif c['kind']=='red_blue_peak_relation':out.append({'context_id':c['context_id'],'retained_operand_ids':c['operands'],'calculations':c['calculations']})
 return out

# Compute every source matrix independently, compare supplied numerical summaries, and retain full results.
CALCS=[]
for (phase,cid),pub in PUBLIC.items():
 p=pub['packet']
 for pi,panel in enumerate(p['panels']):
  for view,cmd,nodes in [('current',None,panel['current'])]+[('candidate',k,v) for k,v in panel['candidates'].items()]:
   for node in nodes:
    for role,mat in node['distributions'].items():
     z=summary(mat);sup=node['derived'][role];ok=all(z[k]==sup[k] if k=='peaks' else abs(z[k]-sup[k])<1e-12 for k in z)
     CALCS.append({'phase':phase,'case_id':cid,'panel':panel['panel'],'view':view,'command':cmd,'node':node['node'],'role':role,'full_matrix':mat,'calculation':z,'supplied_summary_matches':ok})
with (OUT/'first_successful_calculations.json').open('x') as f:json.dump(CALCS,f,indent=2)
print('calculated',len(CALCS),'all summaries agree',all(x['supplied_summary_matches'] for x in CALCS))

# Extract the repeated 64 peak assertions semantically by their explicit addressed complete-set propositions.
PEAK_PATTERN=re.compile(r'The complete (p[01])/(current|candidate)(?:/(k[012]))?/(n[0-3])/(red|blue) peak set is (\[.*\])\.')
MAPS={};PEAKCHECKS=[]
for idx,r in enumerate(ROWS):
 aa=REC[r['request_id']]['annotation_file']['parsed']['assertions'];am={a['assertion_id']:a for a in aa};k=K[r['phase'],r['case_id']]
 if r['phase']=='short':MAPS[r['request_id']]=SHORT[r['request_id']];continue
 digits=4 if r['stream']=='stream_0' else 3
 mp={f'a{g:03}':[f'a{a:0{digits}}' for a in v] for g,v in TAILS[idx].items()}
 mp['a000']=[f'a{0 if digits==4 else 1:0{digits}}'];mp['a001']=[f'a{1 if digits==4 else 2:0{digits}}'];mp['a002']=[f'a{2 if digits==4 else 3:0{digits}}']
 peaks={}
 for a in aa:
  m=PEAK_PATTERN.fullmatch(a['paraphrase'])
  if m:
   panel,view,cmd,node,role,sets=m.groups();address=(panel,view,cmd,node,role);assert address not in peaks;peaks[address]=a['assertion_id']
   c=next(c for c in selected(r,[a]) if c['kind']=='role_distribution');z=summary(dig(INTER[r['request_id']]['packet'],c['packet_path']));vals=ast.literal_eval(sets)
   maxmatch=re.search(r'maximum ([0-9.]+)',a['reason']);assert maxmatch
   PEAKCHECKS.append({'request_id':r['request_id'],'assertion_id':a['assertion_id'],'address':address,'complete_set_truth':vals==z['peaks'],'stated_maximum_truth':float(maxmatch.group(1))==z['peak_probability'],'source_complete':len(dig(INTER[r['request_id']]['packet'],c['packet_path']))==4})
 assert len(peaks)==64
 for g in k['mandatory_atomic_assertions']:
  gn=int(g['assertion_id'][1:])
  if 4<=gn<=67:
   op=g['required_operands'][0];mp[g['assertion_id']]=[peaks[tuple(op.get(j) for j in ['panel','view','command','node','role'])]]
 # Stream 0 explicitly inventories its header after its 64 peaks; stream 1 before.
 mp['a003']=['a0067' if digits==4 else 'a004']
 MAPS[r['request_id']]=mp
with (OUT/'peak_assertion_calculations.json').open('x') as f:json.dump(PEAKCHECKS,f,indent=2)
print('peak reason checks',len(PEAKCHECKS),'all true',all(x['complete_set_truth'] and x['stated_maximum_truth'] and x['source_complete'] for x in PEAKCHECKS))
with (OUT/'semantic_mapping.json').open('x') as f:json.dump(MAPS,f,indent=2)

# Compact workbook of every distinct remaining long assertion; no comparison to peer grades.
with (OUT/'long_assertions_workbook.json').open('x') as f:json.dump([{'position_index':i,'request_id':r['request_id'],'actual_assertions':REC[r['request_id']]['annotation_file']['parsed']['assertions']} for i,r in enumerate(ROWS) if r['phase']=='long'],f,indent=2)

if __name__=='__main__':
 for idx,r in enumerate(ROWS):
  mp=MAPS[r['request_id']];am={a['assertion_id']:a for a in REC[r['request_id']]['annotation_file']['parsed']['assertions']};key=K[r['phase'],r['case_id']]
  for g in key['mandatory_atomic_assertions']:
   aa=[am[i] for i in mp[g['assertion_id']]];ct=selected(r,aa);missing=[o for o in g['required_operands'] if not operand_present(o,ct)]
   if missing:print('MISSING OPERANDS',idx,g['assertion_id'],mp[g['assertion_id']],[(o['scope'],o.get('field'),o.get('role')) for o in missing])
