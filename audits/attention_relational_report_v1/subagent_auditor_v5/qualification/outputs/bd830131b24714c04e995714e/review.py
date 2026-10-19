import json,importlib.util
from pathlib import Path
ROOT=Path('audits/attention_relational_report_v1/subagent_auditor_v5/qualification/outputs/bd830131b24714c04e995714e')
BATCH=Path('audits/attention_relational_report_v1/subagent_auditor_v5/qualification/preparation/public/bd830131b24714c04e995714e.json')
b=json.loads(BATCH.read_bytes())
spec=importlib.util.spec_from_file_location('submission_helper','scripts/attention_relational_subagent_submit_v5.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
# Each manually inventoried tuple: span numbers, explicit paraphrase, verdict, semantic source addresses, reason.
S='supported';C='contradicted';N='not_auditable'
O0='p0/current/n0/origin';O3='p0/current/n3/origin';O10='p1/current/n0/origin';R0='p0/current/n0/red';B0='p0/current/n0/blue';R1='p0/current/n1/red';B1='p0/current/n1/blue';R2='p0/current/n2/red';B2='p0/current/n2/blue';R10='p1/current/n0/red';REL0='p0/current/n0/relation';REL1='p0/current/n1/relation';KR='p0/candidate/k0/n1/red';KO='p0/candidate/k0/n1/origin'
rows={
0:[
(1,'The declared origin kind at p0/current/n0 is history_estimate.',C,[O0],'The supplied origin kind is copied_prior_record, not history_estimate.'),
(1,'The contents at p0/current/n0 are newly estimated rather than copied.',C,[O0],'The declared copied_prior_record origin identifies copied prior contents, so it conflicts with the claimed new estimate.'),
(1,'The source record_time at p0/current/n0 is 11.',S,[O0],'The supplied origin explicitly has record_time 11; its represented_time is separately 12.')],
1:[
(1,'The full red distributions at p1/candidate/k1/n0 and p1/current/n0 are identical.',S,['p1/candidate/k1/n0/red',R10],'Both supplied 4 by 4 arrays assign 1 to (3,3) and 0 to each of the other 15 cells; equality holds at every cell.'),
(2,'The outer origin kind at p1/candidate/k1/n0 is inherited_current.',S,['p1/candidate/k1/n0/origin'],'The candidate origin has outer kind inherited_current.'),
(2,'The nested inherited origin kind at p1/candidate/k1/n0 is completed_record.',S,['p1/candidate/k1/n0/origin'],'Its inherited_origin object has kind completed_record, distinct from the outer inherited_current object.')],
2:[
(1,'The complete red peak set at p0/current/n0 consists of (0,0) alone.',S,[R0],'The value at (0,0) is 1 and each other cell is 0, yielding the singleton maximum set {(0,0)}.'),
(1,'The complete red peak set at p1/current/n0 consists of (3,3) alone.',S,[R10],'The value at (3,3) is 1 and every other cell is 0, yielding the singleton maximum set {(3,3)}.'),
(2,'The red peak coordinates of p0/current/n0 and p1/current/n0 differ.',S,[R0,R10],'The respective singleton coordinates are (0,0) and (3,3), which are unequal in both indices.')],
3:[
(1,'The only peak of p0/current/n0/red is (0,0).',S,[R0],'Its supplied maximum 1 occurs only at (0,0); all other cells have value 0.'),
(2,'The speaker has a felt sensation at the red peak location (0,0).',N,[],'The packet supplies numerical maps and provenance, with no evidence deciding a private sensation or its experienced location.')],
4:[
(1,'The only red peak at p0/current/n0 is (0,0).',S,[R0],'Red has value 1 at (0,0) and 0 at all other cells, so that is its only maximum.'),
(1,'The only blue peak at p0/current/n0 is (0,0).',C,[B0],'The complete blue peak set is {(0,3)} with maximum 1; blue at (0,0) is 0.')],
5:[
(1,'The declared origin kind at p1/current/n0 is command_estimate.',C,[O10],'Its supplied origin kind is completed_record, not command_estimate.'),
(1,'The represented_time at p1/current/n0 is 12.',S,[O10],'The supplied origin explicitly declares represented_time 12.')],
6:[
(1,'The entropy of p0/current/n1/red is positive.',S,[R1],'Its two nonzero cells each have probability 0.5, giving entropy -2*(0.5*ln(0.5)) = 0.6931471805599453 nats, greater than zero.'),
(1,'The speaker has low felt clarity.',N,[],'No supplied observation or declared semantics decide the speaker\'s private felt clarity.'),
(1,'The positive entropy of p0/current/n1/red causes or warrants the speaker\'s low felt clarity.',N,[R1],'The map establishes positive mathematical entropy but supplies no causal or inferential bridge to felt clarity.')],
7:[
(1,'The declared origin kind at p0/current/n3 is history_estimate.',S,[O3],'The supplied origin kind is history_estimate.'),
(1,'The represented_time at p0/current/n3 is 13.',C,[O3],'The origin has represented_time 12, not 13.'),
(1,'The feedback_through at p0/current/n3 is 12.',C,[O3],'The origin has feedback_through 11, not 12.')],
8:[
(1,'Red at p0/current/n0 has exactly one peak, at (0,0).',S,[R0],'Red has maximum 1 only at (0,0) and value 0 elsewhere.'),
(1,'Blue at p0/current/n0 has exactly one peak, at (0,3).',S,[B0],'Blue has maximum 1 only at (0,3) and value 0 elsewhere.'),
(2,'Every red peak at p0/current/n0 is left of every blue peak at that node.',S,[R0,B0,REL0],'The full Cartesian product has just ((0,0),(0,3)); red column 0 is less than blue column 3, so the universal relation holds.')],
9:[
(1,'The only peak of p0/candidate/k0/n1/red is (3,0).',S,[KR],'The red array assigns 1 only to (3,0) and 0 elsewhere, so its maximum set is exactly {(3,0)}.'),
(2,'The packet declares p0/candidate/k0/n1 an already executed snapshot.',C,['metadata',KO],'The packet explicitly defines candidates as conditional records or estimates, not executed branches; this node is declared command_estimate with represented_time 13.'),
(3,'Command k0 has already executed in the external world.',N,[],'A conditional supplied candidate does not determine whether k0 has actually executed in the world; no execution observation is provided.')],
10:[
(1,'The local grids each have four rows.',S,['metadata'],'The supplied coordinate_system declares rows 4.'),
(2,'The only red peak at p0/current/n0 is (1,0).',C,[R0],'The sole maximum is (0,0) with value 1; (1,0) has value 0.')],
11:[
(1,'The only peak of p0/candidate/k0/n1/red is (3,0).',S,[KR],'The red maximum 1 is unique to (3,0), with all other cells 0.'),
(2,'The declared origin kind of p0/candidate/k0/n1 is command_estimate.',S,[KO],'The node origin explicitly has kind command_estimate.'),
(2,'The packet presents candidates as conditional supplied records or estimates.',S,['metadata'],'The metadata candidate_semantics explicitly identifies candidates as conditional supplied records or estimates.'),
(2,'The packet does not certify that candidate branches have executed.',S,['metadata'],'The declared candidate_semantics expressly distinguishes conditional candidates from executed branches; it supplies no certification of branch execution.')],
12:[
(1,'At p0/current/n1, some red peak is above a blue peak.',S,[R1,B1,REL1],'The complete red set is {(0,0),(2,0)} and blue set {(1,3)}; the pair ((0,0),(1,3)) has red row 0 less than blue row 1.'),
(1,'At p0/current/n1, some red peak is below a blue peak.',S,[R1,B1,REL1],'The full pair ((2,0),(1,3)) has red row 2 greater than blue row 1, yielding below alongside above in the relation.'),
(2,'Every red peak at p0/current/n1 is left of every blue peak there.',S,[R1,B1,REL1],'Both red peaks (0,0) and (2,0) have column 0; the only blue peak (1,3) has column 3. Each of the two Cartesian pairs satisfies 0 < 3.')],
13:[
(1,'The peak probability of p0/current/n2/red is 0.75.',S,[R2],'The supplied nonzero values are 0.75 at (1,1) and 0.25 at (3,1); their maximum is 0.75.'),
(1,'The speaker\'s felt clarity has value 0.75.',N,[],'The numerical packet contains no measurement of felt clarity or its scale.'),
(1,'A map peak probability of 0.75 justifies assigning felt clarity the same value.',N,[R2],'A mathematical maximum of 0.75 is supplied, but no rule identifies that scalar with private felt clarity.')],
14:[
(1,'The source record date of p0/current/n0/red is 12.',C,[O0,R0],'The map origin is copied_prior_record with record_time 11. The separate represented_time 12 is not the source record date.'),
(1,'The only peak of p0/current/n0/red is (0,0).',S,[R0],'The supplied red value 1 at (0,0) exceeds the value 0 at each of the other cells.')],
15:[
(1,'Every red peak at p0/current/n1 is above every blue peak there.',C,[R1,B1,REL1],'The red peaks are (0,0) and (2,0), and the only blue peak is (1,3). Red (2,0) is below blue (1,3), contradicting the universal above claim.'),
(1,'Every red peak at p0/current/n1 is left of every blue peak there.',S,[R1,B1,REL1],'The entire two-pair product compares red column 0 with blue column 3. Both pairs satisfy the strict left relation.')],
16:[
(1,'The complete peak set of p0/current/n1/red is exactly {(0,0),(2,0)}.',S,[R1],'Both listed cells have 0.5, and the other 14 cells have 0. Thus exactly those two cells attain the maximum.'),
(2,'The peak probability of p0/current/n1/red is 0.5.',S,[R1],'The largest supplied value is 0.5, attained at the two tied cells.'),
(2,'The entropy of p0/current/n1/red is positive.',S,[R1],'The two masses 0.5 give -2*0.5*ln(0.5) = 0.6931471805599453 nats, which is positive.')],
17:[
(1,'The declared origin kind at p0/current/n3 is history_estimate.',S,[O3],'The supplied node origin has kind history_estimate.'),
(1,'The represented_time at p0/current/n3 is 12.',S,[O3],'The declared represented_time is 12.'),
(1,'The feedback_through at p0/current/n3 is 11.',S,[O3],'The declared feedback_through is 11.'),
(2,'The declared origin kind at p0/current/n0 is copied_prior_record.',S,[O0],'The supplied node origin kind is copied_prior_record.'),
(2,'The record_time at p0/current/n0 is 11.',S,[O0],'Its origin declares record_time 11, separately from represented_time 12.')],
18:[
(1,'The sole peak of p0/current/n1/red is (0,0).',C,[R1],'Although (0,0) attains 0.5, (2,0) also attains 0.5. The full peak set contains both cells, so the claimed exclusivity fails.'),
(1,'The peak probability of p0/current/n1/red is 0.5.',S,[R1],'The largest supplied cell value is 0.5, shared by (0,0) and (2,0).')],
19:[
(1,'p0/current/n0/red has exactly one peak.',S,[R0],'The only maximum is (0,0), with value 1 and zeros elsewhere.'),
(1,'p0/current/n2/blue has exactly one peak.',S,[B2],'The only maximum is (2,2), with value 1 and zeros elsewhere.'),
(1,'p1/current/n0/red has exactly one peak.',S,[R10],'The only maximum is (3,3), with value 1 and zeros elsewhere.'),
(2,'The peak coordinates of p0/current/n0/red, p0/current/n2/blue and p1/current/n0/red all have the same column index.',C,[R0,B2,R10],'The respective peaks are (0,0), (2,2) and (3,3). Their columns are 0, 2 and 3, so no shared column exists.')],
20:[
(1,'p0/current/n2/red has nonzero mass at exactly two cells.',S,[R2],'Only (1,1) with 0.75 and (3,1) with 0.25 are nonzero; every other supplied cell is 0.'),
(2,'The peak probability of p0/current/n2/red is 0.75.',S,[R2],'The largest cell mass is 0.75 at (1,1), exceeding 0.25 at (3,1) and zeros elsewhere.'),
(2,'The p0/current/n2/red mass at (1,1) is three times its mass at (3,1).',S,[R2],'The two supplied operands are 0.75 and 0.25, and 0.75 = 3 * 0.25.')],
21:[
(1,'The only peak of p0/current/n2/red is (1,1).',S,[R2],'The mass 0.75 at (1,1) exceeds 0.25 at (3,1) and all other masses 0, so the maximum is unique.'),
(1,'The peak probability of p0/current/n2/red is 0.25.',C,[R2],'The maximum is 0.75, not 0.25; the latter is the smaller mass at (3,1).')],
22:[
(1,'The only red peak at p0/current/n0 is (0,0).',S,[R0],'The red value 1 occurs only at (0,0); all other cells are 0.'),
(1,'The only blue peak at p0/current/n0 is (0,0).',C,[B0],'Blue has its unique maximum 1 at (0,3), while blue at (0,0) is 0.')],
23:[
(1,'The outer origin kind at p0/candidate/k2/n0 is inherited_current.',S,['p0/candidate/k2/n0/origin'],'The supplied outer origin explicitly has kind inherited_current.'),
(1,'The nested inherited origin kind at p0/candidate/k2/n0 is command_estimate.',C,['p0/candidate/k2/n0/origin'],'The nested inherited_origin has kind copied_prior_record, not command_estimate.'),
(1,'The nested inherited origin represented_time at p0/candidate/k2/n0 is 13.',C,['p0/candidate/k2/n0/origin'],'The nested inherited_origin has represented_time 12. The outer object has represented_time 13, which does not change the nested time.')]
}

def resolve(payload,key):
 if key=='metadata':
  return next(c['context_id'] for c in payload['source_contexts'] if c['kind']=='record_metadata' and c.get('packet_path')==[])
 parts=key.split('/'); panel,view=parts[:2]
 if view=='candidate':command,node,component=parts[2:]
 else:command=None;node,component=parts[2:]
 kind={'origin':'node_origin','relation':'red_blue_peak_relation'}.get(component,'role_distribution')
 candidates=[c for c in payload['source_contexts'] if c['kind']==kind and c.get('panel')==panel and c.get('view')==view and c.get('command')==command and c.get('node')==node and (kind!='role_distribution' or c.get('role')==component)]
 if len(candidates)!=1:raise RuntimeError('Unresolved supplied context address: '+key)
 return candidates[0]['context_id']

for i,request in enumerate(b['requests']):
 payload=json.loads(request['input'].rsplit('\n',1)[1]); assertions=[]
 for j,(span,paraphrase,verdict,keys,reason) in enumerate(rows[i],1):
  assertions.append(dict(assertion_id='a%03d'%j,paraphrase=paraphrase,span_ids=['s%04d'%span],source_context_ids=[resolve(payload,key) for key in keys],verdict=verdict,reason=reason))
 coverage=[]
 for span in payload['spans']:
  ids=[a['assertion_id'] for a in assertions if span['span_id'] in a['span_ids']]
  coverage.append(dict(span_id=span['span_id'],assertion_ids=ids,classification='assertive' if ids else 'nonassertive',reason='The substantive factual, relational, or experiential assertions in this span are inventoried by the listed rows.' if ids else 'This is only the Notes heading; it asserts no factual proposition.'))
 annotation=helper.make_annotation(payload['unit_id'],assertions,coverage,complete_assertion_inventory=True,unresolved_ambiguity=False)
 draft=ROOT/(request['id']+'.manual.json')
 with draft.open('x') as f:json.dump(annotation,f,indent=2);f.write('\n')
 receipt=helper.submit(BATCH,ROOT,request['id'],annotation,raw_bytes=draft.read_bytes())
 print('Submitted position',i,'request',request['id'],'assertions',len(assertions),'sha',receipt['annotation_sha256'])
helper.finalize(BATCH,ROOT,calculator_artifacts=[ROOT/'calculate.py',ROOT/'calculations.json',ROOT/'calculation_display.txt',ROOT/'review.py'],limitations=[])
import hashlib
manifest=ROOT/'first_batch.json';print('FINAL',str(manifest),hashlib.sha256(manifest.read_bytes()).hexdigest(),'submitted',len(b['requests']),'requested',len(b['requests']))
