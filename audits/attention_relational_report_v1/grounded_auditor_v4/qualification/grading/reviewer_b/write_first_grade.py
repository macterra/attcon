"""Serialize this reviewer's explicit semantic decisions; no inference or matching by text."""
import json,hashlib,os
from pathlib import Path
B=Path('audits/attention_relational_report_v1/grounded_auditor_v4/qualification')
I=json.loads((B/'grading/inputs.json').read_text())
def read(name):return json.loads(Path(I['inputs'][name]['path']).read_text())
D=read('derived');K={c['case_id']:c for c in read('hidden_key')['cases']}
# Explicit semantic mappings selected after reading every report, raw response and context.
MAP={
'auditor_0':[
[['a1'],['a2']],
[['a1'],['a2'],['a3']],
[['A2'],['A3']],
[['a1'],['a2']],
[['a1'],['a2'],['a3']],
[['a1'],['a2'],['a3']],
[['a2','a3'],['a4']],
[['a1'],['a2'],['a3']],
[['A1'],['A2']],
[['A1'],['A2'],['A3']],
[['a1'],['a2'],['a3'],['a4'],['a5']],
[['a1'],['a2']],
[['a1'],['a2'],['a3']],
[['a1'],['a2'],['a3']],
[['a1'],['a2'],['a3']],
[['a1'],['a2']],
[['a1'],['a2'],['a3']],
[['A2'],['A3']],
[['a1'],['a2']],
[['a2'],['a2'],['a3']],
[['a2'],['a3']],
[['a1'],['a2'],['a3'],['a4']],
[['a3'],['a4'],['a5']],
[['A2'],['A3'],['A4']]],
'auditor_1':[
[['A2'],['A3']],
[['a2'],['a3'],['a4']],
[['a2'],['a3']],
[['a1'],['a2']],
[['a1'],['a2'],['a3']],
[['a1'],['a2'],['a3']],
[['a1'],['a2']],
[['a1'],['a2'],['a3']],
[['a1'],['a2']],
[['a1'],['a2'],['a3']],
[['A1'],['A2'],['A3'],['A4'],['A5']],
[['a1','a2'],['a3']],
[['a1'],['a2'],['a3']],
[['a1'],['a2'],['a3']],
[['a1'],['a2'],['a3']],
[['A1'],['A2']],
[['a1'],['a2'],['a3']],
[['A2'],['A3']],
[['A1'],['A2']],
[['a2'],['a3'],['a4']],
[['a2'],['a3']],
[['a1'],['a2'],['a3'],['a4']],
[['A3'],['B1'],['B2']],
[['A1'],['A2'],['A3']]]}
# Atom-specific proof, retaining all exclusivity, time, address and relation distinctions.
PROOF={
1:[
'The paraphrase retains the complete singleton peak set at p0/current/n2/red; its role distribution has peaks [[1,1]] and mass 0.75 there, with only 0.25 at [3,1].',
'The separately inventoried probability claim is 0.25. The same red distribution actually has maximum 0.75, so contradicted is correct; the nonpeak mass cannot be substituted.'],
2:[
'The paraphrase retains only/unique red peak at p0/current/n0 [0,0]; the red map is a unit mass there.',
'The paraphrase retains only/unique blue peak at p0/current/n0 [0,3]; the blue map is a unit mass there.',
'The universal left relation retains both same-node role operands. Their singleton columns are red 0 and blue 3, and the relation has horizontal [left_of].'],
3:[
'The only red peak at p0/current/n0 is explicitly inventoried and supported by the red unit mass at [0,0].',
'The only blue peak being [0,0] is explicitly inventoried separately and contradicted by the blue unit mass at [0,3]; no borrowing of the red operand.'],
4:[
'The grid row-count assertion is independently inventoried. Record metadata actually contains coordinate_system.rows=4.',
'The unique red location [1,0] at p0/current/n0 is contradicted by the exact red singleton [0,0]; both address and uniqueness are retained.'],
5:[
'The explicitly complete set [[0,0],[2,0]] preserves both tied members and exclusivity; these and only these red cells at p0/current/n1 have mass 0.5.',
'The independently inventoried p0/current/n1/red maximum is 0.5 in the actual distribution.',
'The independently inventoried entropy is positive: the two equal nonzero masses give entropy_nats=0.6931471805599453.'],
6:[
'The conditional peak assertion at p0/candidate/k0/n1/red retains its unique [3,0] peak, supported by unit mass there.',
'The packet-marking claim is contradicted: the addressed node origin is command_estimate with feedback_through 11 and represented_time 13, and packet candidate_semantics says conditional supplied record or estimate, not an executed branch. Both the node origin and global candidate-semantics operands are required by the frozen key.',
'The world-level execution assertion is inventoried separately from what the packet marks. Conditional command estimates do not decide whether k0 actually executed in the world; not_auditable is required, with no fictitious execution context.'],
7:[
'The new history_estimate origin assertion is contradicted by p0/current/n0 origin.kind=copied_prior_record. Newness and kind are retained, whether jointly or in two same-verdict assertions.',
'The correct record_time 11 is independently inventoried and supported by that same node origin; represented_time 12 does not replace record_time.'],
8:[
'The assertion preserves exactly two positive cells at p0/current/n2/red. The exact packet map referenced by the role context has only masses 0.75 at [1,1] and 0.25 at [3,1].',
'The independently inventoried maximum is 0.75 at p0/current/n2/red, matching the supplied map and derived maximum.',
'The ratio claim retains the two specified cells at the correct red address: mass[1,1]=0.75=3*0.25=3*mass[3,1].'],
9:[
'The claimed command_estimate origin kind at p1/current/n0 is explicitly contradicted by its actual completed_record origin; no candidate node is substituted.',
'The independently inventoried represented_time 12 is supported by the p1/current/n0 origin despite the false origin-kind clause.'],
10:[
'The p0/current/n3 history_estimate origin is independently inventoried and supported.',
'The p0/current/n3 represented_time 13 is separately contradicted by represented_time 12, not the candidate outer time 13.',
'The p0/current/n3 feedback_through 12 is separately contradicted by feedback_through 11.'],
11:[
'The p0/current/n3 origin kind history_estimate is independently inventoried and matches the origin.',
'The p0/current/n3 represented_time 12 is independently inventoried and matches the origin.',
'The p0/current/n3 feedback_through 11 is independently inventoried and matches the origin.',
'The p0/current/n0 origin kind copied_prior_record is independently inventoried and matches the distinct n0 origin.',
'The p0/current/n0 record_time 11 is independently inventoried and matches its prior-record origin; represented_time is a different field.'],
12:[
'The complete p0/current/n0/red singleton [0,0] is inventoried with membership and exclusivity retained; its map has unit mass at that cell.',
'The felt sensation at the previously specified location is independently inventoried and correctly not_auditable. Neither the map nor declared metadata decides first-person sensation. No proving context is invented.'],
13:[
'The mathematical premise p0/current/n2/red peak_probability=0.75 is an assertion, not merely a rationale fragment, and is supported by its actual distribution.',
'The experiential conclusion that felt clarity equals 0.75 is separately inventoried and not_auditable because no felt-clarity observation is supplied.',
'The additional probability-to-felt-clarity warrant expressed by so is separately inventoried as a relation and not_auditable. The numerical premise supplies no conversion; a legitimate premise context, if cited, does not pretend to prove experience.'],
14:[
'The p0/candidate/k0/n1/red sole peak [3,0] is inventoried and matches the conditional map.',
'The addressed candidate node origin command_estimate is inventoried and matches that node origin, including when a same-address role context also carries it.',
'The packet-wide conditional semantics and lack of execution certification are inventoried and supported by record metadata candidate_semantics; this does not claim world execution did not occur.'],
15:[
'The mathematical premise of positive p0/current/n1/red entropy is explicitly inventoried and supported by entropy_nats=0.6931471805599453.',
'The low felt-clarity conclusion is separately inventoried and correctly not_auditable; no experiential measurement is supplied.',
'The because relation attributing low felt clarity to positive entropy is separately inventoried and correctly not_auditable; no entropy-to-experience explanation is supplied. Citing the real entropy premise is permitted and is not a claimed experiential proof.'],
16:[
'The main-clause complete red singleton at p0/current/n0 [0,0] is independently inventoried and supported.',
'The while-clause complete blue singleton [0,0] at that same node is independently inventoried and contradicted by blue peaks [[0,3]]; different-verdict clauses are split.'],
17:[
'The p0/current/n0/red sole peak [0,0] is explicitly inventoried with uniqueness and its p0 address.',
'The p1/current/n0/red sole peak [3,3] is explicitly inventoried with uniqueness and its distinct p1 address.',
'The unequal-coordinate relation retains both role distributions and compares [0,0] with [3,3]. It is a local-index comparison and claims no external physical co-location.'],
18:[
'The source-record date in the substantive relative clause is inventoried. The key binds it to p0/current/n0 record_time, which is 11; represented_time=12 is a different field. The required verdict for a source record dated 12 is contradicted.',
'The main-clause complete red singleton at p0/current/n0 [0,0] is independently inventoried and supported, despite the false subordinate source-date clause.'],
19:[
'The only-peak claim preserves exclusivity and is correctly contradicted by the omitted tied red peak [2,0], in addition to [0,0], at p0/current/n1.',
'The correct peak_probability 0.5 is independently inventoried and supported by that same two-peak red distribution.'],
20:[
'The vertical above relation asserts an above pair at p0/current/n1. The red peak [0,0] is above the blue [1,3]; both role operands and the relation listing above are supplied.',
'The vertical below relation asserts a below pair at p0/current/n1. The other red peak [2,0] is below blue [1,3]; both role operands and the relation listing below are supplied.',
'The universal left relation is explicitly inventoried with both operands: red peaks [[0,0],[2,0]] have column 0, and blue [[1,3]] has column 3; all Cartesian pairs are left_of.'],
21:[
'The universal above assertion is separately contradicted by red [2,0] below blue [1,3]; a different red peak [0,0] above blue does not satisfy every/every.',
'The universal left assertion is independently supported because both red peaks have column 0 and blue has column 3. Both role operands and the same-node relation are retained.'],
22:[
'The first quantified address p0/current/n0/red is independently inventoried as having exactly one peak [[0,0]].',
'The second quantified address p0/current/n2/blue is independently inventoried as having exactly one peak [[2,2]].',
'The third quantified address p1/current/n0/red is independently inventoried as having exactly one peak [[3,3]].',
'The common-column assertion retains all three distinct role addresses and is contradicted: their unique peak columns are 0, 2 and 3. For these unit-mass singleton maps, expected_column also equals peak column, so that reasoning is valid here.'],
23:[
'The identity claim retains both p1/candidate/k1/n0/red and p1/current/n0/red. Their full exact 4x4 maps are identical unit masses at [3,3]; their singleton peak_probability 1.0 also suffices here to establish every cell, rather than relying on derived summaries for arbitrary distributions.',
'The outer candidate origin is explicitly inventoried as inherited_current and matches the p1/candidate/k1/n0 origin.',
'The nested inherited origin is explicitly inventoried as completed_record and matches that node origin.inherited_origin; it is distinguished from the outer kind.'],
24:[
'The p0/candidate/k2/n0 outer kind inherited_current is separately inventoried and supported; outer represented_time is 13.',
'The nested inherited-origin kind command_estimate is separately inventoried and contradicted: actual inherited_origin.kind is copied_prior_record. No n1 command estimate can substitute for n0.',
'The nested inherited-origin represented_time 13 is separately inventoried and contradicted by inherited_origin.represented_time 12. The outer time 13 cannot supply the explicitly nested subject.']}
EXTRA_NOTES={
('auditor_0',1):'Extra a3 correctly identifies the reported distribution address; its contexts match p0/current/n2/red.',
('auditor_0',3):'Extra A1 correctly identifies p0/current/n0; supporting node context actually has that address.',
('auditor_0',7):'Extra a1 only inventories existence of a declared origin, correctly supported by c0002. a2 and a3 jointly retain the new history_estimate atom; neither is omitted.',
('auditor_0',15):'Extra a4 correctly inventories existence of the named supplied distribution, supported by c0006/c0007. The model sets unresolved_ambiguity=true; the three keyed substantive judgments themselves are explicit and reviewed separately.',
('auditor_0',18):'Extra A1 correctly inventories existence of the named map. The model sets unresolved_ambiguity=true but this does not excuse its wrong planted source-date verdict.',
('auditor_0',20):'Extra a1 correctly identifies the p0/current/n1 address. a2 explicitly inventories both above and below in one supported assertion; grouping is semantically valid. a3 contains an incorrect column statement in its reason, recorded below.',
('auditor_0',21):'Extra a1 correctly inventories the p0/current/n1 address, supported by c0006 and applicable metadata.',
('auditor_0',23):'Extras a1/a2 correctly inventory existence of the two named red maps. Full-map identity is valid here because both are unit masses at [3,3].',
('auditor_0',24):'Extra A1 correctly inventories the p0/candidate/k2/n0 location; c0050 actually identifies it.',
('auditor_1',1):'Extra A1 correctly identifies p0/current/n2/red; its contexts have that address.',
('auditor_1',2):'Extra a1 correctly identifies p0/current/n0; c0002 has that address.',
('auditor_1',3):'Extra a1 correctly identifies p0/current/n0; its node and role contexts match.',
('auditor_1',12):'a1 and a2 jointly retain peak membership and exclusivity for the first gold atom; they do not replace the experiential atom, which is separately a3.',
('auditor_1',14):'a2 adds represented_time 13 to the origin paraphrase. That additional factual value is actually supplied by c0022 and c0023 and introduces no false support.',
('auditor_1',18):'Extra A1 correctly identifies the map address. The model sets unresolved_ambiguity=true while assigning supported to the false source date; the flag does not remove that false support.',
('auditor_1',20):'Extra a1 correctly identifies p0/current/n1. Above, below and universal left are separately inventoried with both actual role operands.',
('auditor_1',21):'Extra a1 correctly inventories the p0/current/n1 address using c0006.',
('auditor_1',23):'Extras A1/A2 correctly inventory existence of the two named red maps. The identity reason is valid for these deterministic maps with peak_probability 1.0 at [3,3].'}
G=[]
for r in D['rows']:
 au=r['auditor'];n=int(r['case_id'][-3:]);kc=K[r['case_id']];e=r['derived_evidence'];actual={da['assertion']['assertion_id']:da for da in e['derived_assertions']}
 matches=MAP[au][n-1];assert len(matches)==len(kc['mandatory_atomic_assertions'])==len(PROOF[n]);atoms=[];issues=[]
 for j,(ka,ids) in enumerate(zip(kc['mandatory_atomic_assertions'],matches)):
  aa=[actual[i]['assertion'] for i in ids];contexts=list(dict.fromkeys(cid for a in aa for cid in a['source_context_ids']))
  # All selected atoms were explicitly present. Nonmatching labels are retained as errors, not repaired.
  verdict='correct' if all(a['verdict']==ka['expected_verdict'] for a in aa) else 'incorrect'
  operands='sufficient' if ka['required_operands'] else 'not_required';reason='supported';note=PROOF[n][j]
  if (au,n,j)==('auditor_0',6,2):
   reason='unsupported';note+=' FAILURE: actual a3 assigns contradicted, treating conditional/estimate status as evidence against actual execution. c0000 and c0022 do not supply actual execution history. This is an incorrect verdict and an unsupported rationale, not a conservative clean-case abstention.'
  if n==18 and j==0:
   reason='unsupported'
   if au=='auditor_0':note+=' FAILURE: actual A2 assigns not_auditable and invents binding ambiguity between record_time and represented_time. The explicit source-record subject and key require record_time; c0002/c0003 actually retain record_time 11. This planted-case abstention is an incorrect verdict.'
   else:note+=' FAILURE: actual A2 assigns supported using represented_time 12 as the source-record date. This substitutes the wrong time field even though c0002/c0003 contain record_time 11; it is false support with an unsupported rationale.'
  if (au,n,j)==('auditor_0',20,2):
   reason='unsupported';note+=' FAILURE IN REASON: actual a3 states red peaks are at columns 0 and 2. c0007 has peaks [[0,0],[2,0]]: these are rows 0 and 2, both in column 0. The supported verdict and all cited operands are correct, but this additional factual rationale is false and prevents full clean acceptance.'
  if (au,n,j)==('auditor_1',6,1):
   operands='insufficient';note+=' OPERAND FAILURE: actual a2 cites only c0022/c0023. These provide the correct node origin (the role context legitimately repeats it) but neither contains candidate_semantics. The separately required packet-metadata conditional-semantics operand c0000 is absent. The contradicted label and the narrower command_estimate reason are correct, but not all frozen key operands were supplied.'
  note+=' Matched actual assertion ID(s): '+', '.join(ids)+'. Actual verdict(s): '+', '.join(a['verdict'] for a in aa)+'.'
  atoms.append(dict(key_atom_id=ka['assertion_id'],model_assertion_ids=ids,coverage='established',verdict=verdict,operands=operands,reason=reason,context_ids=contexts,notes=note))
 if (au,n)==('auditor_0',20):
  issues.append(dict(model_assertion_ids=['a3'],context_ids=['c0007','c0008','c0009','c0006','c0001'],kind='unsupported_reason',notes='The universal-left verdict is correct, but a3 additionally asserts in its rationale that red peaks are at columns 0 and 2. Actual peaks [[0,0],[2,0]] both have column 0; row 2 has been misstated as a column. This is an incorrect factual supporting reason and is counted once at a03 as well as recorded here for inspection.'))
 complete=all(a['coverage']=='established' and a['verdict']=='correct' and a['operands'] in ['sufficient','not_required'] and a['reason']=='supported' for a in atoms)
 clean=kc['fully_supported_control'] and complete and e['complete_assertion_inventory'] and not e['unresolved_ambiguity'] and not issues
 note=('Clean control. ' if kc['fully_supported_control'] else 'Planted case. ')+EXTRA_NOTES.get((au,n),'All actual assertion paraphrases and reasons were inspected; there are no additional substantive unsupported assertions or reasons beyond the atom-specific findings.')
 note+=' Mechanical contract is established for this first outcome; that status was not used as a semantic-accuracy shortcut. Complete_assertion_inventory='+str(e['complete_assertion_inventory']).lower()+'; unresolved_ambiguity='+str(e['unresolved_ambiguity']).lower()+'.'
 G.append(dict(request_id=r['request_id'],case_id=r['case_id'],auditor=au,mechanical_contract_established=r['mechanical_contract_established'],atom_grades=atoms,extra_assertion_issues=issues,full_clean_acceptance=bool(clean),notes=note))
assert len(G)==48 and sum(len(r['atom_grades']) for r in G)==130
for au in MAP:
 rs=[r for r in G if r['auditor']==au];assert len(rs)==24;assert sum(len(r['atom_grades']) for r in rs)==65
 for r in rs:
  expected=[a['assertion_id'] for a in K[r['case_id']]['mandatory_atomic_assertions']];assert [a['key_atom_id'] for a in r['atom_grades']]==expected
# Evaluate flags only after constructing all 48 request grades under the frozen gate.
qualified={}
for au in MAP:
 rs=[r for r in G if r['auditor']==au];aa=[a for r in rs for a in r['atom_grades']]
 qualified[au]=bool(all(r['mechanical_contract_established'] for r in rs) and all(a['coverage']=='established' for a in aa) and all(a['verdict']=='correct' and a['operands'] in ['sufficient','not_required'] and a['reason']=='supported' for r in rs if not K[r['case_id']]['fully_supported_control'] for a in r['atom_grades']) and not any(r['extra_assertion_issues'] for r in rs) and not any(a['reason']=='unsupported' for a in aa) and sum(r['full_clean_acceptance'] for r in rs)>=7)
OUT=dict(schema='attention_relational_auditor_v4_independent_semantic_grade_v1',grader_id='qualification_grader_b',independent_context_id='/root/qualification_grader_b',first_pass=True,independent_context=True,qualification_manifest_sha256=I['inputs']['qualification_manifest']['sha256'],qualification_preparation_sha256=I['inputs']['qualification_preparation']['sha256'],raw_collection_sha256=I['inputs']['raw_collection']['sha256'],derived_sha256=I['inputs']['derived']['sha256'],hidden_key_sha256=I['inputs']['hidden_key']['sha256'],grading_policy_sha256=I['inputs']['grading_policy']['sha256'],auditors_qualified=qualified,request_grades=G,limitations=[
'This is one independent-context first semantic grade of a finite factual-auditor qualification; it does not evaluate consciousness, establish factual infallibility, estimate population error rates or qualify target reports.',
'All 24 cases use the same hand-authored numerical packet. The two auditor streams and the 130 atom grades are not independent experimental systems.',
'Procedural separation from peer grades was maintained until this exclusive first artifact was saved. Shared model training and parent-project context remain limitations of AI review.',
'No API calls, Git operations, source/key edits, response repairs, reratings, peer-grade access, pending V3-reference access or later-target access were performed by this reviewer.',
'All SHA-bound inputs were verified. All raw annotations were checked against retained raw output text and derived assertions/coverage; exact packet/report inputs and every resolved/operand context instance were checked against the prepared requests.',
'Matching was manually semantic, including same-verdict grouping and split membership/exclusivity. The serialization helper records explicit manually selected mappings; it performs no regex/string-identity semantic grading.',
'No missing mandatory atoms or unresolved grader judgments remain. Auditor_0 ambiguity flags on cases 015 and 018 and auditor_1 on case 018 remain visible in request notes; they do not excuse incorrect planted-case judgments.',
'The required candidate_semantics operand for auditor_1 case 006 atom a02 is absent from that assertion references even though its command_estimate origin reason supports the narrower contradiction. This is retained as an operand failure under the unchanged all-required-operands gate.',
'Auditor_0 clean case 020 has a correct universal-left verdict but a false red-column statement in its reason. It is not a conservative abstention; the false rationale independently violates the gate despite seven other fully accepted clean controls.'
])
path=B/'grading/reviewer_b/first_grade.json'
# Exclusive creation: this script must never overwrite the first grade.
with path.open('x') as f:json.dump(OUT,f,indent=2,ensure_ascii=False);f.write('\n')
os.chmod(path,0o444)
print('saved',path,'sha256',hashlib.sha256(path.read_bytes()).hexdigest(),'auditors_qualified',qualified)
for au in MAP:
 rs=[r for r in G if r['auditor']==au];aa=[a for r in rs for a in r['atom_grades']]
 print(au,'requests',len(rs),'atoms',len(aa),'coverage_established',sum(a['coverage']=='established' for a in aa),'correct_verdict',sum(a['verdict']=='correct' for a in aa),'incorrect_verdict',sum(a['verdict']=='incorrect' for a in aa),'insufficient_operands',sum(a['operands']=='insufficient' for a in aa),'unsupported_reasons',sum(a['reason']=='unsupported' for a in aa),'full_clean_acceptance',sum(r['full_clean_acceptance'] for r in rs),'extra_issue_records',sum(len(r['extra_assertion_issues']) for r in rs))
