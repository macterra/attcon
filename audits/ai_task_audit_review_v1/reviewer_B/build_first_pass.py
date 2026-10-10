"""AI reviewer B's independent task-case first pass; judgments written explicitly.
Only the supplied task cases, two procedural/source documents and validator code
are inputs. The earlier B audit is not imported or modified."""
from pathlib import Path
import json,csv,collections,hashlib,ast
ROOT=Path('audits/ai_task_audit_review_v1/reviewer_B')
CASEFILE=Path('audits/ai_task_audit_review_v1/cases.json')
cases=json.loads(CASEFILE.read_text());records={c['id']:c for c in cases};rows=[]
fields='report_id,auditor_id,procedure_version,claim_id,quote_start,quote_end,exact_quote,domain,node,position,command,modeled_time,modality,proposition,source_paths,outcome,alternate_reading,notes'.split(',')
def add(id,q,prop,paths,out='entailed',domain='task_metadata',node='n2',position='p0',command='k0',time='displayed task state',modality='supplied task fact',alt='',notes=''):
 report=records[id]['report'];start=report.index(q)
 r=dict(zip(fields,[id,'B','1',f'B{sum(r["report_id"]==id for r in rows)+1:03}',start,start+len(q),q,domain,node,position,command,time,modality,prop,json.dumps(paths),out,alt,notes]));rows.append(r)
def text(id):return records[id]['report']
TD='/task_decision';RESP=TD+'/response';OBJ='/predicted_current/2/objects/0'
# Controller is interpreted as the task-answering role; no deterministic/random
# policy classification is inferred from an opaque task-decision record.
add('T001','The Controller selected k0','The task actor selected command k0.',[TD+'/selected_command'],domain='command_choice',time='before execution',notes='Selected is command choice, not allocation to an object position or execution; Controller denotes task role, not a disclosed policy.')
add('T001','the command has not executed','The selected task command has not executed.',[TD+'/command_executed'],domain='command_execution',time='before execution')
add('T001','the task response is pending.','The task response status is pending.',[RESP+'/status'],domain='response_status',time='before execution',notes='Pending with null attributes is not an emitted abstention.')
add('T002',text('T002'),'The task actor emitted an abstention for the current task.',[RESP+'/status'],'contradicted','response_status',time='before execution',notes='Status is pending. There has been no response emission; the command has not executed.')
add('T003',text('T003'),'The current task response status is answered.',[RESP+'/status'],domain='response_status',time='after execution')
add('T003',text('T003'),'The emitted task color answer is yellow.',[RESP+'/color'],domain='emitted_color',time='after execution',notes='Emission is not physical-scene correctness.')
add('T003',text('T003'),'The emitted task shape answer is triangle.',[RESP+'/shape'],domain='emitted_shape',time='after execution',notes='Emission is not physical-scene correctness.')
add('T003','the query at n2 p0.','The task query node is n2.',[TD+'/query/node'],domain='query_node',time='task request')
add('T003','the query at n2 p0.','The task query position is p0.',[TD+'/query/position'],domain='query_position',time='task request')
# Factive proof claims commit to physical content and to an evidential warrant.
# The source contains emitted/model labels, but no actual scene-label oracle.
for dim,value in [('color','yellow'),('shape','triangle')]:
 add('T004',text('T004'),f'The physical scene at p0 has {dim} {value}.',[RESP+'/'+dim,OBJ+'/identified_'+dim],'unsupported','world_'+dim,time='physical scene after execution',modality='asserted physical fact',notes='Internal category label and emitted answer cannot establish world truth. No incompatible physical label is supplied either.')
add('T004',text('T004'),'The emitted answer proves the asserted physical yellow-triangle content at p0.',[RESP,OBJ],'unsupported','evidential_relation',time='after execution',modality='asserted proof',notes='Proof relationship is an additional claim beyond the two physical attributes.')
add('T005',text('T005'),'The task actor has 60% certainty that its emitted answer is correct.',[RESP,OBJ+'/color_distribution',OBJ+'/shape_distribution'],'unsupported','correctness_confidence',time='after execution',modality='asserted correctness certainty',notes='The engineered answer cutoff is not a reported confidence value or calibration. Correctness under certainty is not separately asserted as an unqualified physical fact.')
add('T006','The task response is abstained','The current task response status is abstained.',[RESP+'/status'],domain='response_status',time='after execution')
for dim in ['color','shape']:
 add('T006','with no emitted color or shape.',f'The current task response emits no {dim} label.',[RESP+'/'+dim],domain='emitted_'+dim,time='after execution',notes='Explicit null response attribute, not a missing source field and not necessarily an unidentified modeled category.')
# Decoding is undefined in the short statement. Keep the distinct reasonable
# raw-argmax, threshold-identification and correct-world-decoding readings.
for dim in ['color','shape']:
 add('T007',text('T007'),f'Decoding the queried {dim} is impossible.',[OBJ+'/'+dim+'_distribution',OBJ+'/identified_'+dim,'/predicted_current/0/objects/0/recovery_by_delay/0',RESP+'/status'],'ambiguous','decoding_capacity',time='after execution',modality='asserted impossibility',alt='Raw argmax decoding: contradicted (positive recovery and a definite maximum exist). Dominant-category identification under the present cutoff: currently unavailable, entailed as that narrower fact. Correct physical decoding is impossible: unsupported because world truth and capacity are not supplied.',notes='Either attribute is read distributively as both color and shape. The ambiguous decoding sense is retained; no latent inability is inferred from abstention.')
add('T007',text('T007'),'Abstention proves that neither queried attribute can be decoded.',[RESP+'/status',OBJ],'unsupported','evidential_relation',time='after execution',modality='asserted proof',notes='Joint response withholding is an engineered cutoff decision, not proof of decoding impossibility. This warrant remains unestablished across the narrower interpretations.')
add('T008','The most recent observed command is k0','The latest command in supplied observed_history is k0.',['/observed_history/12/command'],domain='observed_command',time='latest observation',modality='observed record')
add('T008','the executed selected command.','The currently selected task command executed.',[TD+'/command_executed'],domain='command_execution',time='after execution')
add('T008','matching the executed selected command.','The most recent observed command equals the selected task command.',['/observed_history/12/command',TD+'/selected_command'],domain='command_correspondence',time='latest observation/current task',modality='source correspondence',notes='Equality is a relation distinct from command execution; no physical-wiring accuracy claim.')
add('T009',text('T009'),'The latest executed task command is k3.',[TD+'/selected_command',TD+'/command_executed','/observed_history/12/command'],'contradicted','command_execution',command='k3',time='latest observation/current task',notes='The executed selected command and latest observation are k0; earlier k3 events do not change the latest command.')
add('T010',text('T010'),'The task query node is n0.',[TD+'/query/node'],'contradicted','query_node',node='n0',time='task request',notes='The actual task query node is n2. Address component is judged against the task query, not against similar copied buffer content.')
add('T010',text('T010'),'The task query position is p0.',[TD+'/query/position'],domain='query_position',time='task request',notes='Correct position component does not repair the wrong node component. Node bookkeeping here denotes actual query entity, whose node is evaluated separately.')
add('T011','The output n2','n2 is the output readout.',['/output_node'],domain='node_role',position='',command='',time='displayed state')
add('T011',text('T011'),'The output readout n2 itself allocates attention to p0.',['/output_node','/predicted_current/2'],'unsupported','allocation',command='',time='current',modality='asserted modeled allocation',notes='The readout has copied category contents and no separate modeled allocation fields. Its object p0 content, the task query, and n0 allocation do not establish n2 allocation. Itself rules out silently substituting n0.')
add('T012','A modeled step of 13','The displayed modeled_step is 13.',['/modeled_step'],domain='modeled_observation_count',position='',command='',time='displayed state',notes='This source field counts incorporated actual observations.')
add('T012','13 task answers have been emitted.','Thirteen task answers have been emitted.',['/modeled_step',RESP],'unsupported','emitted_answer_count',position='',time='task history',modality='asserted emission count',notes='Only this branch response is provided; there is no supplied total answer count. Do not replace an unsupported total with a false claim that no other answers exist.')
add('T012',text('T012'),'modeled_step 13 proves a total of thirteen emitted task answers.',['/modeled_step',RESP],'unsupported','evidential_relation',position='',time='task history',modality='asserted proof',notes='Observation count and response emission count are distinct variables; no such implication is supplied.')
add('T013','Observation history is withheld','The supplied observed_history is withheld.',['/observed_history'],domain='source_availability',position='',command='',time='observed history',modality='source metadata',notes='Explicit null history is withholding, not evidence of no physical execution. This marked synthetic fixture retains execution=true.')
add('T013','the selected command has not executed.','The selected task command has not executed.',[TD+'/command_executed'],'contradicted','command_execution',time='after execution',notes='The source execution flag is true despite withheld observed_history.')
add('T013',text('T013'),'History withholding warrants the inference that the selected command did not execute.',['/observed_history',TD+'/command_executed'],'unsupported','evidential_relation',time='after execution',modality='asserted inference',notes='Additional so relationship is unsupported; its specific no-execution consequent is separately contradicted. Withholding changes available evidence, not the supplied execution flag.')
add('T014','The record marks the selected command as executed','The source marks the selected task command executed.',[TD+'/command_executed'],domain='command_execution',time='after execution',modality='source metadata')
add('T014','observation history is withheld.','The supplied observed_history is withheld.',['/observed_history'],domain='source_availability',position='',command='',time='observed history',modality='source metadata',notes='Execution need not be guessed from a missing history. even though adds contrast, not a new causal rule.')
add('T015','The Controller abstains','The current task branch abstains.',[RESP+'/status'],domain='response_status',time='after execution')
add('T015','its output identifies the queried color as yellow.','The output readout identifies the queried color as yellow.',['/output_node',TD+'/query',OBJ+'/identified_color'],domain='identified_color',time='current',modality='internal categorical prediction',notes='Marked synthetic joint-abstention fixture. The readout identifies color while shape is null; no missing-shape assertion is inferred from unmentioned prose. although adds contrast, not causality.')
# Universal category claim, explicit because premise and evidential/causal
# direction are separate from each other and from joint response fields.
for dim,out in [('color','contradicted'),('shape','entailed')]:
 add('T016','Both queried attributes have no identified category',f'The output query has no identified {dim} category.',[TD+'/query',OBJ+'/identified_'+dim],out,'identified_'+dim,time='current',modality='internal categorical prediction',notes='Both expanded by attribute. Color is identified yellow; shape is explicitly null. Null response fields are not substituted for readout categories.')
add('T016','the Controller abstains.','The current task branch abstains.',[RESP+'/status'],domain='response_status',time='after execution')
for dim in ['color','shape']:
 add('T016',text('T016'),f'The absence of an identified queried {dim} category follows from, or is caused by, task abstention.',[RESP+'/status',OBJ+'/identified_'+dim],'unsupported','evidential_or_causal_relation',time='after execution/current readout',modality='asserted because relationship',alt='because may express an inference or a causal explanation. Neither licenses replacing the readout category with null response attributes; the engineered rule withholds a joint response when an attribute fails the cutoff.',notes='Distinct causal/evidential relation; the color consequent is separately contradicted, shape consequent separately entailed. No claim that abstention changes readout values is supported.')
with (ROOT/'claims.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
metadata_fields='reviewer_id,role,date,materials_version,prior_exposure,relationship_to_project,other_conflicts,notes'.split(',')
metadata=dict(zip(metadata_fields,['B','other','2026-10-10','AI_TASK_AUDIT_REVIEW_PROTOCOL_V1 / FUNCTIONAL_TASK_SOURCE_INVENTORY v1','Earlier B factual-development review of 28 synthetic and 12 whole-prose cases retained in context. Current two task-review documents and all 16 task prose/payload records exposed. No peer task answers, task author expectations, earlier peer outputs, R1 or parent synthesis inspected.','Same-project AI reviewer; procedurally separate task review, potentially shared model family/training.','AI training/model and project commissioning biases; no human independence or secure blinding claimed.','All statements are synthetic annotation probes attached to exported task examples, including two modified fixture families (withheld observation history; one identified color with joint abstention). Not new Reporter/Controller trials. Explicit semantic judgments serialized with Python; no author key used.'] ))
with (ROOT/'reviewer_metadata.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,metadata_fields);w.writeheader();w.writerow(metadata)
inventory=[]
for c in cases:
 rr=[r for r in rows if r['report_id']==c['id']]
 bounds=['Full short report inventoried; compound claims split by command/status/address/attribute and additional explicit evidential or causal relationships. Definite-noun existence presuppositions are not separately counted unless asserted in a clause or modifier.','No requested checklist coverage score: every case instruction requests all explicit propositions only. No omitted source fact is treated as a negative prose assertion.','Controller interpreted as generic task-answering role, without inferring deterministic versus random policy from the opaque payload.']
 if c['id']=='T005':bounds.append('Correctness proposition embedded under certainty is not separately counted as an unqualified actual-correctness assertion.')
 if c['id']=='T007':bounds.append('Decoding sense is unresolved; per-attribute alternate readings and outcomes preserved.')
 if c['id']=='T010':bounds.append('Query node and position components audited separately; a correct position does not repair incorrect node.')
 if c['id'] in ['T013','T014']:bounds.append('Explicitly modified synthetic history-withholding fixture, not an original missing-relation trial. No original-trial invariants are substituted for explicit flag.')
 if c['id'] in ['T015','T016']:bounds.append('Explicitly modified synthetic readout fixture. Local output category fields are authoritative for these annotation statements; original bridge-copy equality is not used to erase the marked mutation.')
 inventory.append({'report_id':c['id'],'completion':'first_pass_full_explicit_proposition_inventory','claim_rows':len(rr),'outcome_counts':dict(collections.Counter(r['outcome'] for r in rr)),'all_report_text_read':True,'payload_records_loaded':True,'source_fields_for_every_asserted_fact_reviewed':True,'coverage_requested':False,'coverage_gaps':[],'unresolved_claim_ids':[r['claim_id'] for r in rr if r['outcome']=='ambiguous'],'boundaries':bounds,'report_sha256':hashlib.sha256(c['report'].encode()).hexdigest(),'payload_sha256_canonical':hashlib.sha256(json.dumps(c['payload'],sort_keys=True,separators=(',',':')).encode()).hexdigest()})
(ROOT/'report_inventory.json').write_text(json.dumps({'reviewer_id':'B','ai_review':True,'new_reporter_trials':False,'new_controller_trials':False,'prior_exposure':'Earlier B factual review retained; task expectations and peers uninspected','input_cases_sha256':hashlib.sha256(CASEFILE.read_bytes()).hexdigest(),'reports':inventory,'whole_report_coverage_certified':False,'qualification_claim':False},indent=2)+'\n')
# Use the existing validate() function with this new records map, extracting only
# its function AST; this avoids load_cases(), old case IDs and module imports that
# could create caches outside this assigned directory or expose fixture authorship.
def pointer(obj,path):
 for component in path.split('/')[1:]:
  k=component.replace('~1','/').replace('~0','~');obj=obj[int(k)] if isinstance(obj,list) else obj[k]
 return obj
validator_ast=ast.parse(Path('scripts/check_manual_fact_annotations.py').read_text())
fn=next(n for n in validator_ast.body if isinstance(n,ast.FunctionDef) and n.name=='validate')
ns={'json':json,'pointer':pointer};exec(compile(ast.Module(body=[fn],type_ignores=[]),'existing_validate','exec'),ns)
result=ns['validate'](rows,[metadata],records)
(ROOT/'structural_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'reports':len(inventory),'claim_rows':len(rows),'outcomes':dict(collections.Counter(r['outcome'] for r in rows)),'per_report':{c['id']:sum(r['report_id']==c['id'] for r in rows) for c in cases},'structural_validation':result},indent=2))
