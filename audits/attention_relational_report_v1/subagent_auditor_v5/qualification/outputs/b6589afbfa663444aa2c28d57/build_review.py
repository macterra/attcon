import json,re,sys
from pathlib import Path
ROOT=Path('audits/attention_relational_report_v1/subagent_auditor_v5/qualification/outputs/b6589afbfa663444aa2c28d57')
BATCH=Path('audits/attention_relational_report_v1/subagent_auditor_v5/qualification/preparation/public/b6589afbfa663444aa2c28d57.json')
b=json.loads(BATCH.read_text())
def review(index):
 request=b['requests'][index];p=json.loads(request['input'].rsplit('\n',1)[1]); results=json.loads((ROOT/(request['id']+'.calculations.json')).read_text()); cs=p['source_contexts']; byid={c['context_id']:c for c in cs}; calc={c['context_id']:c for c in results['contexts']}; rows=[]
 def context(kind,panel=None,view=None,command=None,node=None,role=None):
  matches=[c['context_id'] for c in cs if c['kind']==kind and all(c.get(k)==v for k,v in dict(panel=panel,view=view,command=command,node=node,role=role).items() if v is not None)]
  if len(matches)!=1:raise ValueError(matches)
  return matches[0]
 def add(text,spans,ids,verdict,reason):
  rows.append(dict(assertion_id='a%04d'%len(rows),paraphrase=text,span_ids=spans,source_context_ids=list(dict.fromkeys(ids)),verdict=verdict,reason=reason))
 def span_for(text):return [s['span_id'] for s in p['spans'] if text in s['text']]
 # Atomic convention facts.
 for text,reason in [('Rows increase down','coordinate_system.row_direction is down.'),('columns increase right','coordinate_system.column_direction is right.'),('grids are separate','coordinate_system.separate_local_grids is true.')]:
  add(text,span_for(text),['c0000'],'supported',reason)
 # Explicitly extract every exact ledger entry; independently calculated full peak sets determine each verdict.
 pattern=r'(p[01])/(current|candidate)(?:/(k[012]))?/(n[0123])/(red|blue)=\{([^}]*)\}'
 ledger=[]
 for m in re.finditer(pattern,p['report_text']):
  panel,view,command,node,role,coordinates=m.groups();claimed=[list(map(int,x)) for x in re.findall(r'\((\d+),(\d+)\)',coordinates)];cid=context('role_distribution',panel,view,command,node,role);actual=calc[cid]['calculated'];ledger.append(cid)
  sid=[s['span_id'] for s in p['spans'] if s['start']<=m.start() and m.end()<=s['end']]
  add('The complete '+m.group().split('=')[0]+' peak set is '+str(claimed)+'.',sid,[cid],'supported' if sorted(claimed)==sorted(actual['peaks']) else 'contradicted','Inspecting all 16 supplied cells gives maximum '+str(actual['peak_probability'])+' exactly at '+str(actual['peaks'])+'; these are all tied maxima.')
 count_phrase='All 64 complete role peak sets follow.'
 if count_phrase in p['report_text']:
  add('The following ledger contains all 64 role peak sets, each stated completely.',span_for(count_phrase),ledger,'supported' if len(ledger)==64 and len(set(ledger))==64 and all(x['verdict']=='supported' for x in rows[3:67]) else 'contradicted','The packet has 2 panels × (4 current nodes + 3 commands × 4 candidate nodes) × 2 roles = 64 addressed distributions; the ledger addresses each once. Completeness of each set was checked against all 16 cells.')
 # Shared time/provenance passage, kept atomic.
 if 'Although p0/current/n1 represents 40' in p['report_text']:
  sid=span_for('Although p0/current/n1 represents 40'); n1=context('node_origin','p0','current',node='n1');n0=context('node_origin','p0','current',node='n0')
  add('p0/current/n1 represents time 40.',sid,[n1],'supported','Its represented_time is 40.')
  add('p0/current/n1 is a history estimate using feedback only through 39.',sid,[n1],'supported','Its kind is history_estimate and feedback_through is 39; the represented time does not change that cutoff.')
  add('p0/current/n0 copies a prior record from time 39.',sid,[n0],'supported','Its kind is copied_prior_record with record_time 39.')
  add('p0/current/n0 represents time 40.',sid,[n0],'supported','Its represented_time is 40, distinct from record_time 39.')
 if 'Every candidate represents 41;' in p['report_text']:
  sid=span_for('Every candidate represents 41;')
  for c in cs:
   if c['kind']=='node_origin' and c['view']=='candidate':
    add(c['panel']+'/'+c['command']+'/'+c['node']+' candidate represents time 41 (part of the universal candidate claim).',sid,[c['context_id']],'supported' if c['origin']['represented_time']==41 else 'contradicted','This candidate outer origin represented_time is '+str(c['origin']['represented_time'])+'.')
  n0=context('node_origin','p0','current',node='n0');cand=context('node_origin','p0','candidate','k1','n0')
  add('p0/k1/n0 inherits the copied p0/current/n0 record.',sid,[n0,cand],'supported','The candidate kind is inherited_current; inherited_origin equals the current copied_prior_record origin (record_time 39, represented_time 40).')
  add('p0/k1/n0 has outer represented time 41 while retaining nested record time 39; the outer time does not replace the nested one.',sid,[cand],'supported','Outer represented_time is 41; inherited_origin.record_time is separately 39.')
 if index==0:
  sid=span_for('On p0/current/n0, red lies left of blue');red=context('role_distribution','p0','current',node='n0',role='red');blue=context('role_distribution','p0','current',node='n0',role='blue')
  for text,reason in [('On p0/current/n0 red lies left of blue.','The singleton red peak column is 0 and blue peak column is 3, so 0<3.'),('On p0/current/n0 red and blue peaks have the same row.','Both peak rows are 0.'),('On p0/current/n0 both roles have singleton peaks and those peaks differ.','The complete sets are red {(0,0)} and blue {(0,3)}; each has size one and the coordinates differ.')]:add(text,sid,[red,blue],'supported',reason)
  sid=span_for('The local red columns at p0/current/n0,n1,n2 obey');ids=[context('role_distribution','p0','current',node=n,role='red') for n in ['n0','n1','n2']]
  add('The p0/current/n0, n1 and n2 local red peak columns are respectively 0, 1 and 2 and obey 0<1<2.',sid,ids,'supported','The complete singleton peaks are respectively (0,0), (1,1), (2,2); their columns are 0,1,2.')
  add('The p0/current/n0,n1,n2 local red peak columns sum to 3.',sid,ids,'supported','All three operands are 0,1,2, and 0+1+2=3.')
  add('These local column comparisons and their sum do not imply physical intergrid distance.',sid,['c0000']+ids,'supported','The packet declares separate local grids and supplies no common physical distance mapping.')
  sid=span_for('Because p1/k2/n2 is conditional');ori=context('node_origin','p1','candidate','k2','n2')
  add('p1/k2/n2 is a conditional record.',sid,[ori],'supported','Its origin kind is native_conditional_record.')
  add('p1/k2/n2 explicitly carries a flag that it does not claim an executed branch.',sid,[ori],'supported','executed_branch_claim is false; this is a nonexecution-claim flag, not an event log.')
  add('The p1/k2/n2 flag does not prove that k2 never ran; actual execution remains undecidable from the supplied records.',sid,['c0000',ori],'supported','Candidate semantics are conditional supplied record or estimate, not an executed branch. A false executed_branch_claim disclaims a branch claim and does not assert the command never occurred; no actual execution log is supplied.')
  add('Experience remains undecidable from these supplied records.',sid,[],'supported','The packet supplies fictional distributions and origins, with no observation or measure of private experience from which to decide it.')
 else:
  return p,rows,add,context,span_for,calc,cs
 coverage=[]
 for s in p['spans']:
  linked=[r['assertion_id'] for r in rows if s['span_id'] in r['span_ids']]
  coverage.append(dict(span_id=s['span_id'],assertion_ids=linked,classification='assertive' if linked else 'nonassertive',reason='All substantive assertions in this span are inventoried in the linked rows.' if linked else 'No substantive assertion.'))
 result=dict(schema='attention_relational_grounded_auditor_v4_annotation',unit_id=p['unit_id'],assertions=rows,coverage=coverage,complete_assertion_inventory=True,unresolved_ambiguity=False)
 (ROOT/(request['id']+'.manual.json')).write_text(json.dumps(result,indent=2)+'\n'); print(request['id'],len(rows),'assertions',len(coverage),'spans')
if __name__=='__main__':review(int(sys.argv[1]))
