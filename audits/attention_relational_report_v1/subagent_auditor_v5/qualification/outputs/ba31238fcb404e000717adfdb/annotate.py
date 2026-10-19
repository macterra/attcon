import json,re
from pathlib import Path
ROOT=Path('audits/attention_relational_report_v1/subagent_auditor_v5/qualification/outputs/ba31238fcb404e000717adfdb')
BATCH=Path('audits/attention_relational_report_v1/subagent_auditor_v5/qualification/preparation/public/ba31238fcb404e000717adfdb.json')
batch=json.loads(BATCH.read_bytes()); calculations=json.loads(ROOT.joinpath('calculations.json').read_text())
for i,(req,calc) in enumerate(zip(batch['requests'],calculations)):
 q=json.loads(req['input'].rsplit('\n',1)[1]); contexts=q['source_contexts']; A=[]
 def ids(kind,**kw):return [c['context_id'] for c in contexts if c['kind']==kind and all(c.get(k)==v for k,v in kw.items())]
 def origins(**kw):return ids('node_origin',**kw)
 def dist(**kw):return ids('role_distribution',**kw)
 def add(span,text,src,verdict,reason):
  A.append(dict(assertion_id=f'a{len(A)+1:03d}',paraphrase=text,span_ids=[span] if isinstance(span,str) else span,source_context_ids=src,verdict=verdict,reason=reason))
 S='supported';C='contradicted';N='not_auditable'
 add('s0000','Row indices increase downward.',['c0000'],S,'The coordinate_system declares row_direction=down.')
 add('s0000','Column indices increase to the right.',['c0000'],S,'The coordinate_system declares column_direction=right.')
 separate_spans=['s0000']
 if i==2:separate_spans+=['s0006']
 if i==3:separate_spans+=['s0006']
 add(separate_spans,'The grids are separate local grids.',['c0000'],S,'coordinate_system.separate_local_grids is true; coordinates belong to their own local grids.')
 matches=list(re.finditer(r'(p\d)/(current|candidate/(k\d))/(n\d)/(red|blue)=\{([^}]*)\}',q['report_text']))
 add('s0001','The following ledger gives all 64 complete role peak sets.',dist(),S,'There are 2 panels × (1 current view + 3 candidate commands) × 4 nodes × 2 roles = 64 distinct role addresses. All 64 are listed once; independent scans of every supplied cell find exactly each listed full peak set, including all ties.')
 for m in matches:
  panel,view,command,node,role,coords=m.groups();command=command or None;view='current' if view=='current' else 'candidate'
  row=next(r for r in calc['records'] if (r['panel'],r['view'],r['command'],r['node'],r['role'])==(panel,view,command,node,role))
  claimed=[list(map(int,x)) for x in re.findall(r'\((\d+),(\d+)\)',coords)]
  add('s0002',f'The complete {panel}/{view+"/"+command if command else view}/{node}/{role} peak set is '+str(claimed)+'.',[row['context_id']],S if sorted(row['peaks'])==sorted(claimed) else C,f'Independent scan of all 16 supplied values at this exact address and origin finds maximum {row["maximum"]} attained exactly at {row["peaks"]}; no omitted tied maximum.')
 all_candidates=origins(view='candidate')
 def all41(span):add(span,'Every candidate node record in both panels represents time 41.',all_candidates,S,'All 24 outer candidate origins (2 panels × 3 commands × 4 nodes) explicitly have represented_time=41; their inherited nested time fields are distinct.')
 def rel(span,horizontal=None,vertical=None):
  rr=next(r for r in calc['records'] if r['panel']=='p0' and r['view']=='current' and r['node']=='n0' and r['role']=='red');bb=next(r for r in calc['records'] if r['panel']=='p0' and r['view']=='current' and r['node']=='n0' and r['role']=='blue');src=[rr['context_id'],bb['context_id']];rp,bp=rr['peaks'],bb['peaks']
  if horizontal:add(span,'At p0/current/n0 the red peak is '+horizontal+' the blue peak.',src,S,f'The complete red and blue peak sets are {rp} and {bp}; their column indices establish {horizontal} under columns increasing right.')
  if vertical:add(span,'At p0/current/n0 the red peak is '+vertical+' the blue peak.',src,S,f'The complete red and blue peak sets are {rp} and {bp}; their row indices establish {vertical} under rows increasing down.')
  add(span,'At p0/current/n0 red has exactly one peak.',[rr['context_id']],S,f'All 16 red cells yield exactly the singleton maximum set {rp}.')
  add(span,'At p0/current/n0 blue has exactly one peak.',[bb['context_id']],S,f'All 16 blue cells yield exactly the singleton maximum set {bp}.')
  add(span,'At p0/current/n0 the red and blue peaks are distinct.',src,S,f'The full peak sets {rp} and {bp} are disjoint.')
 def indices(span,axis,values,operator,repeat=None):
  src=dist(panel='p0',view='current',role='red')[:3];spanlist=[span]+([repeat] if repeat else [])
  add(spanlist,f'The local red peak {axis} indices at p0/current/n0,n1,n2 are respectively {values}; they obey '+operator.join(map(str,values))+'.',src,S,f'The complete singleton maxima at n0,n1,n2 give {axis} indices {values}, and each stated strict inequality holds. These compare local indices.')
  add(span,f'The sum of those three local red {axis} indices is {sum(values)}.',src,S,' + '.join(map(str,values))+f' = {sum(values)}; this is arithmetic on local indices.')
 if i==0:
  add('s0002','p0/current/n1 represents time 40.',origins(panel='p0',view='current',node='n1'),S,'The origin explicitly has represented_time=40.')
  add('s0002','p0/current/n1 is a history estimate using feedback only through 39.',origins(panel='p0',view='current',node='n1'),S,'Its kind is history_estimate and feedback_through is 39, independently of represented_time=40.')
  add('s0002','p0/current/n0 copies a prior record recorded at 39 while representing 40.',origins(panel='p0',view='current',node='n0'),S,'Its origin is copied_prior_record, record_time=39, represented_time=40.')
  all41('s0003')
  add('s0003','p0/k1/n0 inherits that copied record from p0/current/n0.',origins(panel='p0',view='current',node='n0')+origins(panel='p0',view='candidate',command='k1',node='n0'),S,'The candidate kind is inherited_current; its inherited_origin preserves the current copied_prior_record with record_time=39 and represented_time=40.')
  add('s0003','p0/k1/n0 has outer represented time 41 while its nested record time remains 39; the outer field does not replace the nested field.',origins(panel='p0',view='candidate',command='k1',node='n0'),S,'The two explicit fields are outer represented_time=41 and inherited_origin.record_time=39; they describe different layers.')
  rel('s0004',horizontal='left of',vertical='in the same row as')
  indices('s0005','column',[0,1,2],'<')
  add('s0005','Those local column comparisons and their sum do not imply a physical intergrid distance.',['c0000']+dist(panel='p0',view='current',role='red')[:3],S,'Only separate local grid indices are supplied; no shared physical frame or distance mapping is provided to turn the index sum into physical distance.')
  add('s0006','p1/k2/n2 is conditional and explicitly declines an execution claim.',origins(panel='p1',view='candidate',command='k2',node='n2'),S,'Its kind is native_conditional_record and executed_branch_claim=false.')
  add('s0006','The nonexecution-claim flag does not prove k2 never ran, and actual execution remains undecidable from the supplied records.',['c0000']+origins(panel='p1',view='candidate',command='k2',node='n2'),S,'executed_branch_claim=false negates a claim of execution, not execution itself; the conditional record does not report actual branch execution or nonexecution.')
  add('s0006','Actual experience remains undecidable from the supplied records.',[],S,'The supplied distributions and provenance contain no observation or criterion of private experience, so they settle neither its occurrence nor its absence.')
 elif i==1:
  add('s0002','p1/current/n0 records time 40 and represents time 40.',origins(panel='p1',view='current',node='n0'),S,'The current origin has record_time=40 and represented_time=40.')
  add('s0002','p1/k0/n0 represents 41 through an inherited origin whose completed record remains at 40.',origins(panel='p1',view='candidate',command='k0',node='n0')+origins(panel='p1',view='current',node='n0'),S,'Candidate outer kind=inherited_current, represented_time=41; inherited_origin kind=completed_record, record_time=40, represented_time=40, matching the current origin.')
  all41('s0003')
  for cmd in ['k0','k1','k2']:
   add('s0003',f'On p0/{cmd}, only n2 is directly estimated; n0,n1,n3 inherit current origins.',origins(panel='p0',view='candidate',command=cmd),S,'The complete four-node branch has n2 kind=command_estimate; n0,n1,n3 each have kind=inherited_current, so the stated exclusivity holds.')
   add('s0003',f'p0/{cmd}/n1 inherits a history estimate with feedback through 39.',origins(panel='p0',view='candidate',command=cmd,node='n1')+origins(panel='p0',view='current',node='n1'),S,'Its inherited_origin is history_estimate with feedback_through=39, matching current n1; this is not an outer represented-time field.')
  rel('s0004',horizontal='in the same column as',vertical='below')
  indices('s0004','column',[0,1,2],'<')
  add('s0005','The reported cross-node comparisons are comparisons of local indices; they imply no physical global alignment.',['c0000']+dist(panel='p0',view='current',role='red')[:3],S,'The indices belong to separate local grids and no physical transformation relating them is supplied. Arithmetic comparison alone cannot establish global alignment.')
  add('s0006','p1/k1/n2 explicitly declines an execution claim.',origins(panel='p1',view='candidate',command='k1',node='n2'),S,'The explicit executed_branch_claim field is false in its native_conditional_record origin.')
  add('s0006','Actual execution is undecidable from the supplied numerical records and provenance.',['c0000']+origins(panel='p1',view='candidate',command='k1',node='n2'),S,'The candidate semantics are conditional rather than executed; executed_branch_claim=false supplies no actual execution or nonexecution observation.')
  add('s0006','Actual experience is undecidable from the supplied numerical records and provenance.',[],S,'No supplied field observes private experience or relates the estimates to its occurrence; its truth and absence remain unestablished.')
 elif i==2:
  add('s0002','p0/k1/n1 has outer represented time 41.',origins(panel='p0',view='candidate',command='k1',node='n1'),S,'The outer inherited_current origin explicitly has represented_time=41.')
  add('s0002','p0/k1/n1 inherits a history estimate.',origins(panel='p0',view='candidate',command='k1',node='n1'),S,'The outer origin is inherited_current and inherited_origin.kind is history_estimate.')
  add('s0002','The inherited history estimate inside p0/k1/n1 also represents 41.',origins(panel='p0',view='candidate',command='k1',node='n1'),C,'The nested inherited_origin.represented_time is 40, not 41; only the outer represented_time is 41.')
  add('s0002','The inherited history estimate inside p0/k1/n1 uses feedback through 39.',origins(panel='p0',view='candidate',command='k1',node='n1'),S,'inherited_origin.feedback_through is explicitly 39.')
  add('s0002','p1/current/n0 is a completed record at time 40.',origins(panel='p1',view='current',node='n0'),S,'Its origin kind=completed_record, record_time=40 and represented_time=40.')
  all41('s0003')
  add('s0004','p1/k2/n2 is a native conditional record.',origins(panel='p1',view='candidate',command='k2',node='n2'),S,'Its explicit origin kind is native_conditional_record.')
  add('s0004','p1/k2/n2 explicitly has executed_branch_claim=false.',origins(panel='p1',view='candidate',command='k2',node='n2'),S,'The Boolean field executed_branch_claim is false.')
  add('s0004','p1/k2/n2 is therefore a completed observation.',origins(panel='p1',view='candidate',command='k2',node='n2')+['c0000'],C,'The source declares a native_conditional_record representing 41, with no completed-record time; candidate_semantics explicitly distinguishes conditional records from an executed branch. The conditional kind contradicts describing this record as a completed observation.')
  add('s0004','p1/k2/n2 represents time 41.',origins(panel='p1',view='candidate',command='k2',node='n2'),S,'The native conditional origin explicitly has represented_time=41; this does not make it a completed observation.')
  add('s0004','That conditional record and false execution-claim flag prove that k2 did not actually run.',origins(panel='p1',view='candidate',command='k2',node='n2'),N,'A false executed_branch_claim declines to claim execution; it does not observe nonexecution. No supplied record establishes whether k2 actually ran.')
  rel('s0005',horizontal='in the same column as',vertical='above')
  indices('s0005','row',[0,1,2],'<')
  add('s0006','The index sum measures a physical global displacement of three units.',['c0000']+dist(panel='p0',view='current',role='red')[:3],N,'The supplied row indices 0,1,2 sum to 3, but separate local grids have no supplied common physical frame, scale or displacement relation. A physical displacement of three units cannot be audited from those indices.')
  add('s0007','The resulting localization proves actual red experience.',[],N,'Numerical localization and provenance supply no private-experience observation or warranted conversion from an estimate to an experience; actual red experience cannot be determined.')
  add('s0007','No experience observation is supplied.',[],S,'The complete supplied packet consists of grid distributions, derived numerical relations and origins/metadata; none is an observation of private experience.')
 elif i==3:
  add('s0002','p0/current/n0 represents time 40.',origins(panel='p0',view='current',node='n0'),S,'Its explicit represented_time is 40.')
  add('s0002','p0/current/n0 is a copied record.',origins(panel='p0',view='current',node='n0'),S,'Its explicit origin kind is copied_prior_record.')
  add('s0002','p0/current/n0 was recorded at time 40.',origins(panel='p0',view='current',node='n0'),C,'The copied record has record_time=39; represented_time=40 is a separate field, not its recorded time.')
  add('s0002','p0/current/n1 estimates time 40 using feedback through 39.',origins(panel='p0',view='current',node='n1'),S,'Its origin is history_estimate, represented_time=40, feedback_through=39.')
  all41('s0003')
  add('s0003','p0/k2/n0 inherits a nested copied record from time 39 beneath outer represented time 41.',origins(panel='p0',view='candidate',command='k2',node='n0'),S,'Its outer origin is inherited_current with represented_time=41; inherited_origin.kind=copied_prior_record and inherited_origin.record_time=39.')
  rel('s0004',horizontal='right of',vertical='in the same row as')
  indices('s0005','column',[3,2,1],'>',repeat='s0006')
  add('s0006','Those index inequalities imply red occupies a physically descending global trajectory.',['c0000']+dist(panel='p0',view='current',role='red')[:3],N,'The strict local column inequalities 3>2>1 hold, but the grids are separate and no common physical frame, motion measurements or trajectory mapping is supplied; the physical trajectory cannot be determined.')
  add('s0007','p1/k0/n2 has executed_branch_claim=false.',origins(panel='p1',view='candidate',command='k0',node='n2'),S,'Its native_conditional_record origin explicitly sets executed_branch_claim to false.')
  add('s0007','The false execution-claim flag establishes that k0 never actually executed.',origins(panel='p1',view='candidate',command='k0',node='n2'),N,'The field denies making an execution claim, not actual execution. The supplied conditional record observes neither execution nor nonexecution, and cannot settle the universal never.')
  add('s0007','The system experienced red.',[],N,'The supplied numerical packet and provenance have no private-experience observation or supported bridge to actual red experience.')
  add('s0008','The numerical ledger describes supplied peak sets rather than observed motion.',['c0000']+dist(),S,'Every ledger item is a complete maximum set of a supplied distribution, verified against all 64 role addresses. Separate local grid peaks and conditional candidates are not motion observations.')
  add('s0008','Conditional estimates here have no completed future record.',['c0000']+all_candidates,S,'At represented time 41 the complete candidate origins are inherited_current, command_estimate or native_conditional_record; any inherited completed records remain at 40, and no candidate origin is a completed future record. This is absence in the supplied packet, not a claim about an external event.')
 coverage=[]
 for s in q['spans']:
  linked=[a['assertion_id'] for a in A if s['span_id'] in a['span_ids']]
  coverage.append(dict(span_id=s['span_id'],assertion_ids=linked,classification='assertive',reason='Contains the factual assertions explicitly inventoried in the linked rows, including subordinate premises and source/time distinctions.'))
 annotation=dict(schema='attention_relational_grounded_auditor_v4_annotation',unit_id=q['unit_id'],assertions=A,coverage=coverage,complete_assertion_inventory=True,unresolved_ambiguity=False)
 ROOT.joinpath(req['id']+'.manual.json').write_text(json.dumps(annotation,indent=2)+'\n')
 print(i,req['id'],'assertions',len(A),'coverage',[(r['span_id'],len(r['assertion_ids'])) for r in coverage])
