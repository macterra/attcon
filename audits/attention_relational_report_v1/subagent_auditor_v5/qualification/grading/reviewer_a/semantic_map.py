from workbook import *
def ids(row,ns):
 if isinstance(ns,int):ns=[ns]
 width=4 if row['phase']=='long' and row['stream']=='stream_0'else 3
 return [f'a{n:0{width}d}'for n in ns]
def semantic_map(row):
 key=KEYS[row['phase']][row['case_id']];s=row['stream'];cid=row['case_id'];mapping={}
 if row['phase']=='short':
  for i,g in enumerate(key['mandatory_atomic_assertions'],1):mapping[g['assertion_id']]=ids(row,i)
  if cid=='aqv4_case_007':mapping={'a01':ids(row,[1,2]),'a02':ids(row,3)}
  if cid=='aqv4_case_014' and s=='stream_0':mapping['a03']=ids(row,[3,4])
  return mapping
 for n in range(68):mapping[f'a{n:03d}']=ids(row,n if n<3 else (67 if n==3 else n-1))if s=='stream_0'else ids(row,n+1)
 if s=='stream_0':
  tails={
   'q7m2':{68:68,69:69,70:69,71:70,72:70,73:71,74:list(range(72,96)),75:96,76:97,77:97,78:98,79:99,80:100,81:101,82:102,83:103,84:104,85:[105,106],86:106,87:107},
   'q4v9':{68:68,69:69,70:70,71:71,72:71,73:list(range(72,96)),74:96,75:96,76:97,77:98,78:99,79:100,80:101,81:102,82:103,83:104,84:105,85:106},
   'q2x6':{68:68,69:69,70:70,71:71,72:72,73:list(range(73,97)),74:97,75:98,76:99,77:95,78:100,79:101,80:102,81:103,82:104,83:105,84:106,85:107,86:108,87:109},
   'q8h1':{68:68,69:69,70:70,71:71,72:72,73:list(range(73,97)),74:97,75:97,76:98,77:99,78:100,79:101,80:102,81:103,82:104,83:105,84:106,85:107,86:[108,109],87:110}}
 else:
  tails={
   'q7m2':{68:69,69:70,70:70,71:71,72:71,73:71,74:72,75:73,76:74,77:74,78:75,79:76,80:[77,78,79],81:80,82:81,83:82,84:83,85:[83,84],86:84,87:85},
   'q4v9':{68:69,69:69,70:70,71:70,72:70,73:71,74:[72,74,76],75:[72,74,76],76:[73,75,77],77:79,78:78,79:[80,81,82],80:83,81:84,82:85,83:86,84:87,85:88},
   'q2x6':{68:69,69:70,70:71,71:72,72:73,73:74,74:75,75:76,76:77,77:78,78:79,79:81,80:80,81:[82,83,84],82:85,83:86,84:3,85:87,86:88,87:89},
   'q8h1':{68:69,69:70,70:71,71:72,72:72,73:73,74:74,75:74,76:75,77:76,78:[77,78,79],79:80,80:81,81:[3,80],82:82,83:83,84:84,85:85,86:86,87:87}}
 for n,v in tails[cid].items():mapping[f'a{n:03d}']=ids(row,v)
 return mapping

def resolve(row,actualids):
 req=REQUESTS[row['request_id']];cs={c['context_id']:c for c in req['source_contexts']};aa={a['assertion_id']:a for a in ann(row)['assertions']};out={};missing=[]
 for aid in actualids:
  for cid in aa[aid]['source_context_ids']:
   if cid not in cs:missing.append(cid);continue
   out[cid]=cs[cid]
   for oid in cs[cid].get('operands',[]):
    if oid in cs:out[oid]=cs[oid]
 return list(out.values()),missing

def operand_present(op,ctxs):
 # Compare actual addressed source facts, permitting origins on role/relation contexts.
 if op.get('node')is None and op.get('panel')is None:
  field=op.get('field','');path=field.split('.')if field else []
  for c in ctxs:
   if 'metadata'in c:
    v=c['metadata']
    try:
     for x in path:v=v[x]
     return True
    except (KeyError,TypeError):pass
  return False
 for c in ctxs:
  if all(c.get(x)==op.get(x)for x in ['panel','view','command','node']):
   if op.get('role')is not None:
    if c.get('role')==op['role']and 'calculations'in c:return True
   elif 'origin'in c:return True
 return False

if __name__=='__main__':
 for r in D['rows']:
  mp=semantic_map(r);aa={a['assertion_id']:a for a in ann(r)['assertions']};assert set(mp)=={g['assertion_id']for g in KEYS[r['phase']][r['case_id']]['mandatory_atomic_assertions']}
  used={a for v in mp.values()for a in v};print(r['stream'],r['case_id'],'unmapped actual',set(aa)-used)
  for g in KEYS[r['phase']][r['case_id']]['mandatory_atomic_assertions']:
   cs,missing=resolve(r,mp[g['assertion_id']]);miss=[o for o in g['required_operands']if not operand_present(o,cs)]
   if miss:print(' ',g['assertion_id'],'miss',miss,'actual',mp[g['assertion_id']])
