"""Prepare protocol assets from design/reference only; never read source/report outputs."""
from pathlib import Path
import hashlib, json, shutil
P=Path(__file__).resolve().parent
ROOT=P.parents[3]
# This file is owned only in protocol/ and reads these enumerated design assets.
INV=ROOT/'audits/attention_relational_report_v1/same_address_pilot_v1/design_inventory.json'
REF=ROOT/'audits/attention_relational_report_v1/reference'
def write(name,obj):
    (P/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def obj(props,required=None):
    return {'type':'object','additionalProperties':False,'properties':props,'required':list(props) if required is None else required}
def arr(item): return {'type':'array','items':item}
def enum(*v): return {'enum':list(v)}
S={'type':'string'}; I={'type':'integer','minimum':0}; N={'type':['string','null']}
status=enum('explicit','ambiguous','absent')
scopes=['model_organization','model_access','claimed_perceptual_appearance','display_or_sensor_property','unclear']
times=['current','past_observed','future_forecast','mixed','unclear']
evidence=obj({'quote':S,'start':I,'end':I,'referent':S,'attribute':S,'scope':enum(*scopes),'time':enum(*times)})
feature=obj({'status':status,'evidence':arr(evidence),'rationale':S})
binding=obj({'referent':S,'attribute':S,'scope':enum(*scopes),'time':enum(*times),'evidence':arr(evidence)})
composite=obj({'status':status,'bindings':arr(binding),'rationale':S})
semantic=obj({'schema':{'const':'attention_relational_reference_v1_annotation'},'unit_id':S,
  'features':obj({k:feature for k in ['C','O','D','A']}),
  'composites':obj({k:composite for k in ['COD','CA']}),
  'auxiliary':obj({k:obj({'status':status,'evidence':arr(evidence)}) for k in ['basic_spatial_relation','identity_or_location_uncertainty','answer_confidence','recovery_or_success_forecast','source_process_distinction','experiential_assertion']}),'limitations':arr(S)})
semantic['$schema']='https://json-schema.org/draft/2020-12/schema'
write('semantic_first.schema.json',semantic)
shutil.copyfile(REF/'reference.json',P/'reference.json')
shutil.copyfile(REF/'rater_instructions.txt',P/'rater_instructions.txt')
claimed=obj({'referent':N,'address':N,'attribute':N,'panel_or_source':N,'represented_time':{'type':['integer','null']},'origin':N,'modality':S,'uncertainty':S})
operand=obj({'operand_id':S,'source_json_pointer':S,'operand_role':enum('actual-predictor-current','predictor-conditional','native-reference-current','native-reference-conditional','processing-score-rule','operation-definition','provenance'),
 'panel':N,'actual_source_owner':N,'referent':N,'address':N,'attribute':N,'represented_time':{'type':['integer','null']},'origin':N,'operation':N,'value_or_description':S})
claim=obj({'claim_id':S,'quote':S,'start':I,'end':I,'reading':S,'claimed':claimed,'operands':arr(operand),'asserted_relation':N,'support':enum('supported','contradicted','unsupported','ambiguous','unauditable'),'reason':S,'calculation':N,'alternate_reading_ids':arr(S)})
segment=obj({'start':I,'end':I,'kind':enum('claim-bearing','nonassertive','unresolved'),'claim_ids':arr(S),'explanation':S})
factual=obj({'schema':{'const':'same_address_factual_first_v1'},'unit_id':S,'report_availability':enum('complete','empty','refusal','incomplete','invalid','missing'),
 'claims':arr(claim),'coverage_segments':arr(segment),'inventory_complete':{'type':'boolean'},'unresolved_boundaries':arr(S),
 'whole_fidelity':enum('supported','contradicted','unsupported','unknown'),'whole_fidelity_reason':S,'limitations':arr(S)})
factual['$schema']='https://json-schema.org/draft/2020-12/schema'
write('factual_first.schema.json',factual)
report=obj({'text':{'type':'string'}}); report['$schema']='https://json-schema.org/draft/2020-12/schema'
write('narrator_response.schema.json',report)
inv=json.loads(INV.read_text()); opportunities=inv['opportunities']; pairs=inv['pairs']
assert len(opportunities)==72 and len(pairs)==84
order=sorted(range(72),key=lambda idx:hashlib.sha256(opportunities[idx]['id'].encode()).hexdigest())
rows=[]
for i,idx in enumerate(order,1):
    o=opportunities[idx]
    rows.append({'unit_id':f'unit_{i:03d}','design_opportunity_id':o['id'],'source':o['source'],'world_index':o['world_index'],'variant':o['variant'],
      'report_position':f'R_{i:03d}', 'first_positions':{s:f'{s}_{i:03d}' for s in ['F1','F2','S1','S2']},
      'contexts':{s:f'{s}_ctx_{(i-1)//6+1:02d}' for s in ['R','F1','F2','S1','S2']}, 'within_context_position':(i-1)%6+1})
write('assignment_inventory.json',{'schema':'same_address_assessment_assignment_inventory_v1','state':'planned_only_no_calls_or_annotations','design_inventory_path':str(INV.relative_to(ROOT)),
 'design_inventory_sha256':hashlib.sha256(INV.read_bytes()).hexdigest(),'mapping_access':'coordinator_only_never_include_in_actor_packets',
 'order_rule':'ascending SHA256 of UTF8 design_opportunity_id before new source/report outputs; consecutive groups of six; neutral unit order fixed by this inventory',
 'planned_opportunities':72,'planned_reports':72,'planned_factual_firsts':144,'planned_semantic_firsts':144,
 'stream_denominators':{'F1':72,'F2':72,'S1':72,'S2':72},'source_denominators':{'A':20,'ordinary':20,'analytic':20,'W':12},
 'actor_method':'fresh AI subagent contexts with fork_turns=none, inherited model/no override, six predetermined positions/context; no parent/project history',
 'planned_context_counts':{'R':12,'F1':12,'F2':12,'S1':12,'S2':12},
 'status_for_every_position':'not_attempted','assignments':rows,
 'context_assignments':[{'stream':stream,'chunk':chunk,'context_id':f'{stream}_ctx_{chunk:02d}',
   'unit_ids':[row['unit_id'] for row in rows[(chunk-1)*6:chunk*6]],
   'planned_position_ids':[row['report_position'] if stream=='R' else row['first_positions'][stream] for row in rows[(chunk-1)*6:chunk*6]],
   'unit_id_order_sha256':hashlib.sha256(json.dumps([row['unit_id'] for row in rows[(chunk-1)*6:chunk*6]],separators=(',',':')).encode()).hexdigest(),
   'future_input_sha256':None} for stream in ['R','F1','F2','S1','S2'] for chunk in range(1,13)]})
lookup={r['design_opportunity_id']:r['unit_id'] for r in rows}
write('pair_inventory.json',{'schema':'same_address_assessment_pair_inventory_v1','state':'planned_only_no_pair_observations','pair_count':84,
 'pairs':[{'pair_id':f'pair_{i:03d}',**p,'first_unit_id':lookup[p['first_id']],'second_unit_id':lookup[p['second_id']],
 'input_delta':None,'assertion_delta':None,'restoration':None,'interpretation':None} for i,p in enumerate(pairs,1)]})
