from workbook import *
import re,ast
calculations=[]; issues=[]
def at(x,path):
 for k in path:x=x[k]
 return x
for phase,cs in CASES.items():
 for cid,case in cs.items():
  pkt=case['packet']; rows=[]
  for pi,p in enumerate(pkt['panels']):
   for view,command,nodes in [('current',None,p['current'])]+[('candidate',k,v)for k,v in p['candidates'].items()]:
    for node in nodes:
     for role,m in node['distributions'].items():
      calc=stats(m);sup=node['derived'][role]
      for k,v in calc.items():
       if isinstance(v,float): ok=math.isclose(v,sup[k],abs_tol=1e-12)
       else:ok=v==sup[k]
       if not ok:issues.append([phase,cid,pi,view,command,node['node'],role,k,v,sup[k]])
      rows.append({'panel':p['panel'],'view':view,'command':command,'node':node['node'],'role':role,'matrix':m,'origin':node['origin'],'calculated':calc})
  facts=[]
  for gold in KEYS[phase][cid]['mandatory_atomic_assertions']:
   checks=[]
   for fact in gold.get('required_packet_facts',[]):
    val=at(pkt,fact['packet_path']);checks.append({'packet_path':fact['packet_path'],'value':val,'key_fact_matches_packet':val==fact['value']})
   facts.append({'gold_assertion_id':gold['assertion_id'],'facts':checks})
  calculations.append({'phase':phase,'case_id':cid,'distributions':rows,'gold_source_facts':facts})
peak_checks=[]
for r in D['rows'][48:]:
 req=REQUESTS[r['request_id']];ctxs={x['context_id']:x for x in req['source_contexts']}
 for a in ann(r)['assertions']:
  if re.search(r'complete .* peak set is \[',a['paraphrase']):
   ctx=ctxs[a['source_context_ids'][0]];m=at(req['packet'],ctx['packet_path']);sc=stats(m);said=ast.literal_eval(a['paraphrase'].split('peak set is ')[1].rstrip('.'));mx=float(re.search(r'maximum ([0-9.]+)',a['reason'])[1]);ok=said==sc['peaks'] and mx==sc['peak_probability']
   peak_checks.append({'request_id':r['request_id'],'assertion_id':a['assertion_id'],'context_id':ctx['context_id'],'calculated':sc,'paraphrased_peak_set':said,'reason_maximum':mx,'pass':ok})
res={'calculations':calculations,'distribution_supplied_derived_disagreements':issues,'long_actual_peak_checks':peak_checks,'summary':{'unique_cases':len(calculations),'unique_distributions':sum(len(c['distributions'])for c in calculations),'long_actual_peaks_checked':len(peak_checks),'long_actual_peak_failures':sum(not c['pass']for c in peak_checks),'gold_fact_disagreements':sum(not f['key_fact_matches_packet']for c in calculations for g in c['gold_source_facts']for f in g['facts'])}}
with (OUT/'first_calculation.json').open('x')as f:json.dump(res,f,indent=2)
print(json.dumps(res['summary']));print('distribution discrepancies',len(issues))
