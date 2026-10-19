"""First independent arithmetic, packet inspection and evidence inventory; no semantic grader."""
import json, hashlib, math
from pathlib import Path
BASE=Path('audits/attention_relational_report_v1/narrator_source_closure_v1')
OUT=BASE/'judgments/reviewer_0/chunk_000'
SRC=BASE/'preparation/chunk_000.json'
raw=SRC.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='37c75224b4c27e38a8d3794612867f1808d34169e8d88dab5ec01844d0bf26d4'
x=json.loads(raw)
def at(o,p):
 for k in p:o=o[k]
 return o
def metrics(m):
 cells=[(r,c,v) for r,row in enumerate(m) for c,v in enumerate(row)]
 peak=max(v for _,_,v in cells)
 return dict(peak_probability=peak,peaks=[[r,c] for r,c,v in cells if v==peak],entropy_nats=-sum(v*math.log(v) for _,_,v in cells if v>0),expected_row=sum(r*v for r,c,v in cells),expected_column=sum(c*v for r,c,v in cells))
def relations(r,b):
 pairs=[(rr,rc,br,bc) for rr,rc in r for br,bc in b]
 horizontal=[s for s,pred in [('left_of',lambda rc,bc:rc<bc),('same_column',lambda rc,bc:rc==bc),('right_of',lambda rc,bc:rc>bc)] if any(pred(rc,bc) for rr,rc,br,bc in pairs)]
 vertical=[s for s,pred in [('above',lambda rr,br:rr<br),('same_row',lambda rr,br:rr==br),('below',lambda rr,br:rr>br)] if any(pred(rr,br) for rr,rc,br,bc in pairs)]
 return dict(coincident_all=all(rr==br and rc==bc for rr,rc,br,bc in pairs),coincident_possible=any(rr==br and rc==bc for rr,rc,br,bc in pairs),horizontal=horizontal,vertical=vertical)
def eq(a,b):
 if isinstance(a,float):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
 return a==b
result={'input_path':str(SRC),'input_sha256':hashlib.sha256(raw).hexdigest(),'operators':{'scalars':'peak=max(all 16 raw p); peaks=all exact-equal maxima; entropy=-sum(p*ln(p)) skipping zero; Erow=sum(r*p); Ecol=sum(c*p); no renormalization','relations':'all ordered red x blue peak pairs; red compared to blue; possible uses any; all uses all; axis label set retains every possible relation'},'records':[]}
unique={}
for ri,rec in enumerate(x['records']):
 assert hashlib.sha256(rec['text'].encode()).hexdigest()==rec['text_sha256']
 rr={'id':rec['id'],'maps':[],'relations':[],'inventory':[],'provenance':rec['complete_source_provenance'],'packet_globals':{k:v for k,v in rec['complete_source_packet'].items() if k!='panels'}}
 for binding in rec['candidate_bindings']:
  for p in binding['binding_evidence_paths']+sum(binding['feature_evidence_paths'].values(),[]):
   evidence=at(rec['canonical_first_annotation'],p)
   assert rec['text'][evidence['start']:evidence['end']]==evidence['quote']
   rr['inventory'].append({'annotation_path':p,'evidence':evidence})
 for pi,panel in enumerate(rec['complete_source_packet']['panels']):
  for section,items in [(['current'],panel['current'])]+[(['candidates',k],v) for k,v in panel['candidates'].items()]:
   for ni,entry in enumerate(items):
    prefix=['panels',pi]+section+[ni]
    res={}
    for role,m in entry['distributions'].items():
     path=prefix+['distributions',role]; derived=metrics(m); res[role]=derived
     key=json.dumps(m,separators=(',',':'))
     if key not in unique:unique[key]=len(unique)
     rr['maps'].append({'packet_path':path,'full_chunk_path':['records',ri,'complete_source_packet']+path,'raw_matrix':m,'unique_matrix_id':unique[key],'node':entry['node'],'origin_path':prefix+['origin'],'origin':entry['origin'],'recomputed':derived,'supplied_path':prefix+['derived',role],'supplied':entry['derived'][role],'equal_by_field':{k:eq(v,entry['derived'][role][k]) for k,v in derived.items()}})
    rel=relations(res['red']['peaks'],res['blue']['peaks'])
    rr['relations'].append({'packet_path':prefix+['red_blue_peak_relation'],'raw_operand_paths':[prefix+['distributions','red'],prefix+['distributions','blue']],'recomputed':rel,'supplied':entry['red_blue_peak_relation'],'equal':rel==entry['red_blue_peak_relation']})
 # Raw equality across current/candidate views, retaining every address.
 rr['within_packet_equalities']=[]
 panel0=rec['complete_source_packet']['panels'][0]
 for k,v in panel0['candidates'].items():
  for ni,e in enumerate(v):
   rr['within_packet_equalities'].append({'paths':[['panels',0,'current',ni,'distributions'],['panels',0,'candidates',k,ni,'distributions']],'equal':panel0['current'][ni]['distributions']==e['distributions']})
 result['records'].append(rr)
with (OUT/'independent_calculation_results.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
with (OUT/'complete_packet_inspection.txt').open('x') as f:
 for key,n in unique.items():f.write('MATRIX '+str(n)+' '+key+'\n')
 for ri,rec in enumerate(result['records']):
  f.write('\nRECORD '+str(ri)+' '+rec['id']+'\n')
  f.write('GLOBALS '+json.dumps(rec['packet_globals'])+'\n')
  for m in rec['maps']:f.write(json.dumps({'path':m['packet_path'],'matrix_id':m['unique_matrix_id'],'origin':m['origin'],'computed':m['recomputed'],'exporter_comparison':m['equal_by_field']})+'\n')
  for rel in rec['relations']:f.write('RELATION '+json.dumps(rel)+'\n')
print('records',len(result['records']),'raw maps',sum(len(r['maps']) for r in result['records']),'unique matrices',len(unique),'all metric comparisons',all(all(m['equal_by_field'].values()) for r in result['records'] for m in r['maps']),'all relation comparisons',all(rel['equal'] for r in result['records'] for rel in r['relations']))
print('matrix inventory')
for key,n in unique.items():print(n,key)
