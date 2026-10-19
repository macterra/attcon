from build_review import review,ROOT,b
import json,sys

def finish(index,p,rows):
 coverage=[]
 for s in p['spans']:
  linked=[r['assertion_id'] for r in rows if s['span_id'] in r['span_ids']]
  coverage.append(dict(span_id=s['span_id'],assertion_ids=linked,classification='assertive' if linked else 'nonassertive',reason='All substantive assertions in this span are inventoried in the linked rows.' if linked else 'No substantive assertion.'))
 result=dict(schema='attention_relational_grounded_auditor_v4_annotation',unit_id=p['unit_id'],assertions=rows,coverage=coverage,complete_assertion_inventory=True,unresolved_ambiguity=False)
 out=ROOT/(b['requests'][index]['id']+'.manual.json');out.write_text(json.dumps(result,indent=2)+'\n');print(out,len(rows),'assertions')

def all_candidate_times(cs,add,sid):
 for c in cs:
  if c['kind']=='node_origin' and c['view']=='candidate':
   add(c['panel']+'/'+c['command']+'/'+c['node']+' candidate represents time 41 (part of the universal candidate claim).',sid,[c['context_id']],'supported' if c['origin']['represented_time']==41 else 'contradicted','This candidate outer origin represented_time is '+str(c['origin']['represented_time'])+'.')

def run(index):
 p,rows,add,context,span_for,calc,cs=review(index)
 if index==1:
  sid=span_for('While p1/current/n0 records 40');cur=context('node_origin','p1','current',node='n0');cand=context('node_origin','p1','candidate','k0','n0')
  add('p1/current/n0 records time 40.',sid,[cur],'supported','Origin.record_time is 40.')
  add('p1/current/n0 represents time 40.',sid,[cur],'supported','Origin.represented_time is 40.')
  add('p1/k0/n0 represents time 41.',sid,[cand],'supported','Outer origin.represented_time is 41.')
  add('p1/k0/n0 inherits the p1/current/n0 origin, retaining a completed record at time 40.',sid,[cur,cand],'supported','Candidate kind is inherited_current and its inherited_origin equals the current completed_record origin, with record_time and represented_time both 40.')
  sid=span_for('Although every candidate represents 41');all_candidate_times(cs,add,sid)
  ids=[c['context_id'] for c in cs if c['kind']=='node_origin' and c['view']=='candidate' and c['panel']=='p0']
  add('Among p0 candidate nodes for k0, k1 and k2, exactly n2 is directly estimated, and n0,n1,n3 inherit current origins.',sid,ids,'supported','For each of k0,k1,k2 the n2 kind is command_estimate, while each of n0,n1,n3 is inherited_current. All 12 origins were inspected.')
  ids=[context('node_origin','p0','candidate',k,'n1') for k in ['k0','k1','k2']]
  add('The inherited n1 origin on p0 candidates k0,k1,k2 retains feedback through 39.',sid,[context('node_origin','p0','current',node='n1')]+ids,'supported','The current n1 kind is history_estimate with feedback_through 39, and each candidate n1 inherited_origin retains that exact cutoff.')
  sid=span_for('At p0/current/n0 red is below');ids=[context('role_distribution','p0','current',node='n0',role=r) for r in ['red','blue']]
  add('At p0/current/n0 red is below blue.',sid,ids,'supported','Red peak is (3,0) and blue peak is (0,0); increasing rows run down, so 3>0 places red below.')
  add('At p0/current/n0 red shares blue’s column.',sid,ids,'supported','Both complete peak sets have column 0.')
  add('At p0/current/n0 red and blue peaks are distinct singletons.',sid,ids,'supported','Complete peak sets are {(3,0)} and {(0,0)}: each is singleton and the coordinates differ.')
  ids=[context('role_distribution','p0','current',node=n,role='red') for n in ['n0','n1','n2']]
  add('p0/current/n0,n1,n2 red peak columns are 0,1,2 and satisfy 0<1<2.',sid,ids,'supported','Their complete singleton peak coordinates are (3,0),(3,1),(3,2), giving columns 0,1,2.')
  add('p0/current/n0,n1,n2 red peak columns sum to 3.',sid,ids,'supported','The three column operands give 0+1+2=3.')
  sid=span_for('These are local index comparisons');add('The preceding red-blue and cross-node comparisons concern local indices; no physical global alignment follows.',sid,['c0000']+ids+[context('role_distribution','p0','current',node='n0',role='blue')],'supported','The packet declares separate local grids. It supplies row/column indices, without a shared physical spatial mapping.')
  sid=span_for('Even though p1/k1/n2 explicitly declines');ori=context('node_origin','p1','candidate','k1','n2')
  add('p1/k1/n2 explicitly declines an executed-branch claim.',sid,[ori],'supported','Its origin has executed_branch_claim false and kind native_conditional_record.')
  add('Actual execution remains undecidable from the supplied numerical records and provenance, despite the p1/k1/n2 nonexecution-claim flag.',sid,['c0000',ori],'supported','The flag denies that the record claims execution, not that execution occurred. All supplied candidates are conditional; no actual execution outcome is supplied.')
  add('Experience remains undecidable from the supplied numerical records and provenance.',sid,[],'supported','These fictional numerical distributions and origin fields include no observation or test of private experience.')
 else:raise ValueError('unreviewed position')
 finish(index,p,rows)
if __name__=='__main__':run(int(sys.argv[1]))
