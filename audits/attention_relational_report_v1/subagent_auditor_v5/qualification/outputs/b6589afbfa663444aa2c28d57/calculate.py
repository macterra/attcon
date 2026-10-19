import json,math,hashlib
from pathlib import Path
ROOT=Path('audits/attention_relational_report_v1/subagent_auditor_v5/qualification/outputs/b6589afbfa663444aa2c28d57')
BATCH=Path('audits/attention_relational_report_v1/subagent_auditor_v5/qualification/preparation/public/b6589afbfa663444aa2c28d57.json')
def calc(g):
 m=max(x for row in g for x in row)
 return dict(peaks=[[r,c] for r,row in enumerate(g) for c,x in enumerate(row) if x==m],peak_probability=m,entropy_nats=-sum(x*math.log(x) for row in g for x in row if x),expected_row=sum(r*x for r,row in enumerate(g) for x in row),expected_column=sum(c*x for row in g for c,x in enumerate(row)),mass=sum(sum(row) for row in g))
def address(c):
 return '/'.join(str(c.get(k)) for k in ('panel','view','command','node','role'))
def run():
 b=json.loads(BATCH.read_text()); results=[]
 for request in b['requests']:
  p=json.loads(request['input'].rsplit('\n',1)[1]); q=p['packet']; contexts={c['context_id']:c for c in p['source_contexts']}; rows=[]
  for c in p['source_contexts']:
   row=dict(context_id=c['context_id'],kind=c['kind'],address=address(c),packet_path=c.get('packet_path'),origin=c.get('origin'))
   if c['kind']=='role_distribution':
    v=q
    for k in c['packet_path']:v=v[k]
    row['calculated']=calc(v);row['supplied_calculation_matches']=all(row['calculated'][k]==v for k,v in c['calculations'].items())
   elif c['kind']=='red_blue_peak_relation':
    peaks=[]
    for op in c['operands']:
     v=q
     for k in contexts[op]['packet_path']:v=v[k]
     peaks.append(calc(v)['peaks'])
    pairs=[(r,bl) for r in peaks[0] for bl in peaks[1]]
    row['operands']=c['operands'];row['operand_addresses']=[address(contexts[x]) for x in c['operands']]
    row['calculated']=dict(horizontal=sorted({('left_of' if r[1]<bl[1] else 'right_of' if r[1]>bl[1] else 'same_column') for r,bl in pairs}),vertical=sorted({('above' if r[0]<bl[0] else 'below' if r[0]>bl[0] else 'same_row') for r,bl in pairs}),coincident_possible=any(r==bl for r,bl in pairs),coincident_all=all(r==bl for r,bl in pairs))
   elif c['kind']=='record_metadata':row['metadata']=c['metadata']
   rows.append(row)
  result=dict(request_id=request['id'],unit_id=p['unit_id'],packet_content_sha256=p['packet_content_sha256'],report_text_sha256=p['report_text_sha256'],decision_time=q['decision_time'],contexts=rows)
  (ROOT/(request['id']+'.calculations.json')).write_text(json.dumps(result,indent=2)+'\n');results.append(result)
 return results
if __name__=='__main__':
 r=run();print([(x['request_id'],x['packet_content_sha256'],len(x['contexts'])) for x in r])
