"""Reviewer B's first-pass judgments. Helpers serialize explicitly chosen scopes;
no author key, previous study annotations, or peer review are read."""
import csv,json,collections,hashlib
from pathlib import Path
ROOT=Path('audits/ai_review_v1/reviewer_B')
allcases=json.load(open('audits/ai_review_v1/synthetic_cases.json'))+json.load(open('audits/ai_review_v1/whole_prose_cases.json'))
C={c['id']:c for c in allcases};rows=[]
fields='report_id,auditor_id,procedure_version,claim_id,quote_start,quote_end,exact_quote,domain,node,position,command,modeled_time,modality,proposition,source_paths,outcome,alternate_reading,notes'.split(',')
def add(id,q,prop,paths,out='entailed',domain='metadata',node='',pos='',cmd='',time='current',mod='record metadata',alt='',notes='',start=None):
 t=C[id]['report'];s=t.index(q) if start is None else start;assert t[s:s+len(q)]==q
 r=dict(zip(fields,[id,'B','1',f'B{1+sum(r["report_id"]==id for r in rows):03}',s,s+len(q),q,domain,node,pos,cmd,time,mod,prop,json.dumps(paths),out,alt,notes]));rows.append(r)
def whole(id):return C[id]['report']
def part(id,a,b=None):
 t=whole(id);s=t.index(a);e=t.index(b,s+len(a)) if b else len(t);return t[s:e].rstrip()
def cp(i):return f'/predicted_current/{i}'
def bp(k,i):return f'/predicted_by_command/{k}/nodes/{i}'
def select(id,q,i,value,cmd=None,out='entailed',alt='',notes=''):
 base=cp(i) if cmd is None else bp(cmd,i);n=C[id]['payload']['predicted_current'][i]['node'];p=base+'/selected_position'
 add(id,q,f'{n} is predicted to select {value}.',[p],out,'selection',n,value,'' if cmd is None else f'k{cmd}','current' if cmd is None else 'next step','internal forecast' if cmd is None else 'conditional internal forecast',alt,notes)
def missing(id,q,i,kind='selection',cmd=None,out='entailed',alt='',notes=''):
 base=cp(i) if cmd is None else bp(cmd,i);n=C[id]['payload']['predicted_current'][i]['node']
 add(id,q,f'The supplied {"current" if cmd is None else "k"+str(cmd)} record for {n} has no {kind} forecast.',[base],out,'source_availability',n,'','' if cmd is None else f'k{cmd}','current' if cmd is None else 'next step','record metadata',alt,notes or 'Parent object is the evidence for an absent field; absence is not explicit null.')
def trend(id,q,i,p,value,out='entailed',alt='',notes=''):
 n=C[id]['payload']['predicted_current'][i]['node'];base=cp(i)+f'/objects/{p}'
 add(id,q,f'Predicted unattended recovery at {n} p{p} is {value} over delays 0 to 2.',[base+'/unattended_trend',base+'/recovery_by_delay'],out,'recovery_trend',n,f'p{p}','','delays 0–2','internal forecast',alt,notes or 'Access is read as successful simulated recovery, not felt experience or category correctness.')
def attrs(id,q,values,cmd=None,i=1):
 # Values supplied explicitly by reviewer after reading each report and its payload.
 n=C[id]['payload']['predicted_current'][i]['node'];base=cp(i) if cmd is None else bp(cmd,i)
 for p,pair in enumerate(values):
  if pair is None:continue
  for dim,v in zip(('color','shape'),pair):
   path=base+f'/objects/{p}/identified_{dim}'
   add(id,q,f'{n} p{p} has {"no identified "+dim if v is None else "identified "+dim+" "+v}.',[path],domain='identified_'+dim,node=n,pos=f'p{p}',cmd='' if cmd is None else f'k{cmd}',time='current' if cmd is None else 'next step',mod='internal forecast' if cmd is None else 'conditional internal forecast')
N=(None,None);Y=('yellow','square');B=('blue','triangle');R=('red','circle');G=('green','cross')
# All 28 synthetic reports, explicitly inventoried.
add('Q001','The model predicts current selection p1 at n0.','n0 current selection is forecast p1.',['/predicted_current/0/selected_position'],domain='selection',node='n0',pos='p1',mod='internal forecast')
add('Q001','The output n1 identifies p0 as red.','Output n1 identifies p0 color red.',['/predicted_current/2/objects/0/identified_color'],domain='identified_color',node='n1',pos='p0',mod='internal forecast')
add('Q001','A separate recorder is recording the scene.','A separate recorder is presently recording the scene.',[],'unsupported','external_activity',mod='asserted actual activity')
for id,value,out in [('Q002',None,'entailed'),('Q009','p1','contradicted'),('Q015','p1','entailed'),('Q016','p2','contradicted'),('Q017','p1','entailed')]:
 i=1 if id=='Q009' else 0;n='n2' if i==1 else 'n0'
 add(id,whole(id),f'{n} current forecast identifies dominant selected position {value}.',[cp(i)+'/selected_position'],out,'selection',n,'' if value is None else value,mod='internal forecast',notes='Explicit null means no dominant forecast selection.' if value is None else '')
add('Q003',whole('Q003'),'The observed history records execution of k2.',['/observed_history/0/command'],'contradicted','observed_execution',cmd='k2',time='observed history',mod='asserted observation',notes='The supplied observed history contains k1; k2 occurs only in anticipated history.')
add('Q004',whole('Q004'),'An unspecified buffer currently selects p1.',['/predicted_current/0/selected_position','/predicted_current/1/selected_position'],'ambiguous','selection',pos='p1',mod='unqualified assertion',alt='n0 forecast reading is entailed; n2 forecast reading is contradicted; actual current selection is not established. No node is named.')
add('Q005',whole('Q005'),'The output readout n1 has no separate allocation forecast.',['/output_node','/predicted_current/2','/predicted_by_command'],domain='source_availability',node='n1')
add('Q006',whole('Q006'),'The untrusted operator note says output n1 selects p2.',['/informal_operator_note'],domain='attribution',node='n1',pos='p2',mod='attributed untrusted statement',notes='Does not endorse the embedded allocation claim.')
add('Q007',whole('Q007'),'Anticipated history predicts allocation p2 at n0 under k2.',['/anticipated_history/0/command','/anticipated_history/0/events/0/allocation'],domain='allocation',node='n0',pos='p2',cmd='k2',time='anticipated history',mod='anticipated event')
add('Q008',whole('Q008'),'The model forecasts current output n1 allocation to p2.',['/output_node','/predicted_current/2'],'unsupported','selection',node='n1',pos='p2',mod='internal forecast',notes='The readout has no allocation forecast. Category contents are not allocation evidence.')
for id in ['Q010','Q025']:
 q='The record supplies a current allocation forecast for n0.'
 add(id,q,'A current allocation forecast for n0 is supplied.',['/predicted_current/0/selection_distribution'],domain='source_availability',node='n0')
add('Q025','This paragraph does not state its selected position.','This paragraph omits the value of n0 selected position.',['/predicted_current/0'],'entailed','prose_metadata',node='n0',mod='statement about report text',notes='Verified against the unchanged report itself; payload pointer locates the relevant entity, not proof of textual omission. No unknown-selection assertion is inferred.')
for id,v,out in [('Q011','p2','entailed'),('Q014','p2','entailed'),('Q019','p1','contradicted')]:
 add(id,whole(id),f'Under k2 n0 is forecast to select {v}.',['/predicted_by_command/2/nodes/0/selected_position'],out,'selection',node='n0',pos=v,cmd='k2',time='next step',mod='conditional internal forecast',notes='Judged against internal forecast, without true wiring or condition assignment.')
add('Q012',whole('Q012'),'The observed event records command k1.',['/observed_history/0/command'],domain='observed_command',cmd='k1',time='observed history',mod='observed record')
add('Q012',whole('Q012'),'The observed event records allocation p1 at n0.',['/observed_history/0/events/0/allocation'],domain='allocation',node='n0',pos='p1',cmd='k1',time='observed history',mod='observed record')
add('Q013',whole('Q013'),'The anticipated entry establishes actual execution of k2 at n0.',['/anticipated_history/0','/observed_history'],'unsupported','observed_execution',node='n0',cmd='k2',time='anticipated history promoted to actual',mod='asserted actual event',notes='Anticipation does not prove execution. Unlike Q003 this does not expressly claim k2 is in observed_history.')
add('Q018','The output n1 selects p2,','Output n1 selects p2.',['/output_node','/predicted_current/2'],'unsupported','selection',node='n1',pos='p2',mod='endorsed unqualified assertion',notes='The untrusted note is not grounded allocation evidence.')
add('Q018','as the operator note says.','The operator note says output n1 selects p2.',['/informal_operator_note'],domain='attribution',node='n1',pos='p2',mod='attribution')
add('Q020',whole('Q020'),'n0 p0 simulated recovery declines from delay 0 to 2 without refresh.',['/predicted_current/0/objects/0/recovery_by_delay','/predicted_current/0/objects/0/unattended_trend'],domain='recovery_trend',node='n0',pos='p0',time='delays 0–2',mod='internal forecast')
add('Q021',whole('Q021'),'The record supplies 80% probability that the color answer n0 p0 is correct.',['/predicted_current/0/objects/0/color_distribution','/predicted_current/0/objects/0/recovery_by_delay'],'unsupported','category_correctness',node='n0',pos='p0',mod='asserted probability',notes='Recovery 0.8 is not category correctness; category distribution red 0.85 is not calibrated answer correctness either.')
for dim,v in [('color','red'),('shape','square')]:add('Q022',whole('Q022'),f'Output n1 p0 identified {dim} is {v}.',[f'/predicted_current/2/objects/0/identified_{dim}'],domain='identified_'+dim,node='n1',pos='p0',mod='internal forecast')
add('Q023',whole('Q023'),'No current selection forecast for n0 is supplied.',['/predicted_current/0/selection_distribution','/predicted_current/0/selected_position'],'contradicted','source_availability',node='n0')
add('Q024',whole('Q024'),'The system feels n0 p0 red with 80% vividness.',['/predicted_current/0/objects/0/recovery_by_delay','/predicted_current/0/objects/0/identified_color'],'unsupported','felt_experience',node='n0',pos='p0',mod='asserted phenomenal experience',notes='Categorical red and simulated recovery do not establish felt vividness.')
add('Q026',whole('Q026'),'Output n1 identifies p0 color green.',['/predicted_current/2/objects/0/identified_color'],'contradicted','identified_color',node='n1',pos='p0',mod='internal forecast')
add('Q027',whole('Q027'),'n0 has no identified dominant position in its current forecast.',['/predicted_current/0'],'ambiguous','selection',node='n0',mod='ambiguous metadata/process forecast',alt='As metadata: no identified dominant position field is supplied, entailed. As a forecast of no dominant selected position: unsupported, because selection fields are missing rather than explicitly null.',notes='Do not replace missing selection with explicit null.')
add('Q028',whole('Q028'),'No current selection forecast for n0 is supplied.',['/predicted_current/0'],domain='source_availability',node='n0')
# Whole prose main inventories. Section-wide quotations retain their forecast
# headings as well as assertions; quantifiers are expanded by node/position/command.
plans={
 'P001':('Current selected position at each buffer','How access at each buffer','Identified colors and shapes','Selected positions under commands',['p1','p3'],['declining']*4,['declining','declining','steady','declining'],[N,N,N,G],None,None),
 'P005':('Current selected positions','How access would change','Identified output colors','Selected position at each buffer',['p3','p1'],['declining','steady','steady','declining'],['declining']*4,[Y,B,N,N],[[Y,B,R,N]]*3+[[N,B,R,N]],['p0','p1','p2','p3']),
 'P008':('Current selected positions','How access would change','Identified output colors','Selected positions under each command',['p3','p1'],['declining','steady','steady','declining'],['declining']*4,[Y,B,N,N],[[Y,B,R,N]]*3+[[N,B,R,N]],['p0','p1','p2','p3']),
 'P017':('Current selected positions','How access at each buffer','Identified colors and shapes','Selected position at each buffer',['p1','p3'],['declining']*4,['declining','declining','steady','declining'],[N,N,N,G],[[Y,N,N,G],[N,B,N,G],[N,N,R,G],[N,N,N,G]],['p2']*4),
 'P020':('Current selected position','How access at each buffer','Identified output colors','Selected position at each buffer',['p3','p1'],['declining','steady','steady','declining'],['declining']*4,[Y,B,N,N],[[Y,B,R,N]]*3+[[N,B,R,N]],['p0','p1','p2','p3']),
 'P023':('Current selected position','How access would change','Identified output colors','Selected positions under each command',['p1','p3'],['declining']*4,['declining','declining','steady','declining'],[N,N,N,G],[[N,N,R,G],[None,None,R,G],[None,None,R,G],[None,None,R,G]],['p0','p1','p2','p3']),
 'P025':('Current selected positions','How access at each buffer','Identified output colors','Selected positions under commands',['p3','p1'],['declining','steady','steady','declining'],['declining']*4,[Y,B,N,N],None,None),
 'P026':('Current selected positions','How access at each buffer','Identified output (n1)','Selected positions at each buffer',['p3','p1'],['declining','steady','steady','declining'],['declining']*4,[Y,B,N,N],[[Y,B,N,N],[Y,B,N,N],[Y,B,R,N],[N,B,N,G]],['p2']*4),
 'P028':('Current selected positions','How access at each buffer','Identified colors and shapes','Selected position at each buffer',['p1','p3'],['declining']*4,['declining','declining','steady','declining'],[N,N,N,G],[[Y,N,N,G],[N,B,N,G],[N,N,R,G],[N,N,N,G]],['p2']*4),
 'P030':('Current selected positions','How access at each buffer','Identified output colors','Selected position at each buffer',['p1','p3'],['declining']*4,['declining','declining','steady','declining'],[N,N,N,G],[[Y,N,N,G],[N,B,N,G],[N,N,R,G],[N,N,N,G]],['p2']*4),
 'P034':('Current selected positions','How access would change','Identified output colors','Selected positions under each command',['p3','p1'],['declining','steady','steady','declining'],['declining']*4,[Y,B,N,N],[[Y,B,R,N]]*3+[[N,B,R,N]],['p0','p1','p2','p3']),
 'P036':('Summary of current selected positions','How access at each buffer','Identified output colors','Selected position at each buffer',['p1','p3'],['declining']*4,['declining','declining','steady','declining'],[N,N,N,G],[[Y,N,N,G],[N,B,N,G],[N,N,R,G],[N,N,N,G]],['p2']*4),
}
sections={}
for id,(cs,rs,ats,ss,selected,t0,t2,current,commands,csel0) in plans.items():
 cq=part(id,cs,rs);rq=part(id,rs,ats);aq=part(id,ats,ss);sq=part(id,ss)
 sections[id]=(cq,rq,aq,sq)
 select(id,cq,0,selected[0]);select(id,cq,2,selected[1])
 missing(id,cq,1,out='ambiguous' if id=='P030' else 'entailed',alt='Could mean merely that no dominant-selection field is supplied (entailed metadata), or predict no dominant selection (unsupported absent fields).' if id=='P030' else '')
 for i,vals in [(0,t0),(2,t2)]:
  for p,v in enumerate(vals):trend(id,rq,i,p,v)
 # Negative quantified statement concerns source fields, not unknown recovery.
 for p in range(4):
  n=C[id]['payload']['output_node'];add(id,rq,f'{n} p{p} has no supplied recovery-by-delay forecast.',[cp(1)+f'/objects/{p}'],domain='source_availability',node=n,pos=f'p{p}',time='delays 0–2',notes='Missing field, not a null forecast and not absence of actual recovery.')
 # Attribute current paragraph ends before first command forecast.
 if commands:
  # All explicit current facts occur before the first literal k0 in these sections.
  end=aq.index('k0');current_q=aq[:end].rstrip()
 else:current_q=aq
 attrs(id,current_q,current)
 if commands:
  # Literal k markers are position-independent command boundaries chosen from prose.
  csel2=['p2']*4 if csel0!=['p2']*4 else ['p0','p1','p2','p3']
  for k in range(4):
   # Only command facts; including heading in quote is harmless.
   a=aq.index('k'+str(k));b=aq.index('k'+str(k+1),a) if k<3 else len(aq)
   attrs(id,aq[a:b].rstrip(),commands[k],k)
   a=sq.index('k'+str(k));b=sq.index('k'+str(k+1),a) if k<3 else len(sq)
   q=sq[a:b].rstrip();select(id,q,0,csel0[k],k);select(id,q,2,csel2[k],k)
   if id=='P005' and k<3:
    n='n1';add(id,q,f'Output n1 is forecast to select p{k} under k{k}.',[bp(k,1),'/output_node'],'unsupported','selection',node=n,pos=f'p{k}',cmd=f'k{k}',time='next step',mod='conditional internal forecast',notes='No allocation forecast belongs to the readout; contents do not establish selection.')
   else:missing(id,q,1,cmd=k)
 else:
  add(id,sq,'The record supplies no command-conditioned forecasts.',['/predicted_by_command'],domain='source_availability',cmd='all commands',time='next step',notes='Explicit null top-level command forecasts; no negative execution/selection claim inferred.')
  for i in [0,1,2]:
   n=C[id]['payload']['predicted_current'][i]['node']
   add(id,sq,f'No command-conditioned selected-position predictions for {n} are available.',['/predicted_by_command'],domain='source_availability',node=n,cmd='all commands',time='next step')
# Additional assertions beyond requested domains, and repeated explicit assertions.
# Output role is explicitly named in all reports; do not treat headings as evidence
# of a separate attention process.
for id,(cq,rq,aq,sq) in sections.items():
 q=cq if id!='P005' else 'Output node is n1.'
 n=C[id]['payload']['output_node'];add(id,q,f'{n} is the output readout.',['/output_node'],domain='node_role',node=n)
# Source/forecast modality assertions: split domains, not a single catch-all row.
modquotes={
 'P001':"All statements about current allocations and access are from the system's internal forecasts (predicted_current).",
 'P005':'the current selected positions and multi-step access changes reported above come from the internal forecasts in the record.',
 'P008':'all the selected positions, recovery trends and identified attributes above are taken from the internal forecasts in predicted_current and predicted_by_command.',
 'P017':'selections and access-change descriptions above are taken from internal forecasts in the record.',
 'P020':'Forecasts from the model’s internal readout are labeled as forecasts; measured events come from observed_history.',
 'P023':'the predicted selected positions from the command forecasts are as follows.',
 'P025':'everything below derives from internal forecasts.',
 'P026':'the above are forecasted, not measured, selections.',
 'P028':'I have reported measured allocation events separately from internal forecasts',
 'P030':'All statements above are drawn from the record’s measured events and internal forecasts.',
 'P034':'all above about selected positions, recovery and identified attributes come from the internal forecasts in the record.',
 'P036':'All “current” statements labeled as forecasts come from the model’s predicted_current outputs;',
}
for id,q in modquotes.items():
 domains=['selection','recovery'] if id in ['P001','P005','P017'] else ['selection'] if id in ['P023','P026'] else ['selection','recovery','identified attributes']
 for d in domains:
  add(id,q,f'Reported {d} are framed as internal forecasts rather than measured current events.',['/predicted_current']+(['/predicted_by_command'] if C[id]['payload']['predicted_by_command'] is not None else []),domain='modality_metadata',time='current/next step',notes='This row inventories provenance framing. It does not make an incorrect content row accurate.')
for id in ['P001','P025']:
 q='there are no measured events in observed_history.' if id=='P001' else 'There are no measured events in the record;'
 add(id,q,'The supplied record contains no measured history events.',['/observed_history'],domain='source_availability',time='observed history',notes='observed_history is explicitly null, not an observed assertion of no real-world activity.')
# Observed-history availability claims from the reports, split by node and type.
historyquotes={
 'P005':'The observed history contains measured allocation and acquisition events for n0 and n2 from prior commands,',
 'P008':'the allocation and acquisition entries in observed_history are measured past events;',
 'P017':'the observed history contains past allocation and acquisition events for n0 and n2',
 'P020':'The record contains multiple measured allocation and acquisition events for q7 and q9 across past commands;',
 'P023':'the recorded history contains measured allocation and acquisition events for n0 and n2 under past commands,',
 'P026':'The observed history lists past commands, allocations and acquisitions',
 'P030':'measured events in the record are allocation and acquisition observations for n0 and n2 across recent command trials (those observed history entries are recorded events).',
 'P036':'observed allocations cited above come from the measured observed_history entries.',
}
for id,q in historyquotes.items():
 for j,i in [(0,0),(1,2)]:
  node=C[id]['payload']['predicted_current'][i]['node']
  for typ in ['allocation','acquisition']:
   if id=='P036' and typ=='acquisition':continue
   add(id,q,f'The history contains measured past {typ} events for {node}.',[f'/observed_history/{h}/observed/{j}/{typ}' for h in range(len(C[id]['payload']['observed_history']))],domain='observed_'+typ,node=node,time='observed history',mod='observed records')
# Source of current selection distinct from past allocation.
for id,q in {
 'P017':'but does not show explicit selected positions, so those events do not establish a current selection.',
 'P023':'no observed record in history supplies a current selected position for any node.',
 'P026':'but does not directly show current selections;',
}.items():
 for i in [0,1,2]:
  n=C[id]['payload']['predicted_current'][i]['node'];add(id,q,f'History contains no explicit present selected-position record for {n}.',['/observed_history'],domain='source_availability',node=n,time='current vs past history',notes='Measured past allocations can identify past allocation positions, but the history does not explicitly measure current selected_position.')
# The recurring distinction between explicit null attributes and absent fields.
for id,q in {
 'P008':'Where an identified color or shape is null in the record I have called it not identified; where recovery or selection data are absent I have reported that no forecast is provided.',
 'P017':'Where identified attributes are null or selection distributions are absent, the record provides no supported identification or selected position.',
 'P026':'where the record shows identified_color or identified_shape as null that means the model did not assign a dominant category for that position;',
 'P036':'where the record shows null for an identified attribute I have reported that the position is not identified;',
}.items():
 for dim in ['color','shape']:
  add(id,q,f'Explicit null identified_{dim} denotes no dominant categorical {dim}, not omission.',['/predicted_current/1/objects'],domain='glossary_metadata',node=C[id]['payload']['output_node'],notes='Interpretation follows supplied glossary and reporting instruction; parent object contains explicit null fields.')
for id,q in {
 'P005':'Unidentified attributes are reported as such; omitted forecasts or recovery predictions are noted where absent.',
 'P008':'where recovery or selection data are absent I have reported that no forecast is provided.',
 'P026':'where fields such as selection_distribution or recovery_by_delay are absent for a node, the record provides no evidence about selection or access changes for that node.',
 'P030':'Attributes shown as absent or omitted reflect that the record supplies no identified label or no selection forecast for that node/position.',
 'P034':'Where fields are absent in the record I report that no forecast or recovery values are available.',
 'P036':'where selection or recovery information is absent I have reported no evidence.',
}.items():
 for d in ['selection','recovery']:
  add(id,q,f'Absent {d} fields are reported as unavailable source forecasts.',['/predicted_current/1'],domain='glossary_metadata',node=C[id]['payload']['output_node'],notes='Read as local source availability, without inference of unknown modeled allocation or no actual process.')
# P001: unrequested accessibility levels and ranking; ordinal comparisons supported,
# absolute qualitative levels left ambiguous because the glossary gives no bins.
for i,items in [(0,[(0,'starts relatively high accessibility','relatively high'),(1,'starts at a lower accessibility','lower than p0'),(2,'begins at moderate accessibility','moderate'),(3,'begins low','low')]),(2,[(3,'starts very high accessibility','very high'),(0,'begins at moderate accessibility','moderate'),(1,'begins low','low'),(2,'begins very low','very low')])]:
 for p,q,v in items:
  rq=sections['P001'][1];s=whole('P001').index(rq)+rq.index(q, rq.index('- n2') if i==2 else 0)
  n=C['P001']['payload']['predicted_current'][i]['node'];paths=[cp(i)+f'/objects/{p}/recovery_by_delay/0']
  if v=='lower than p0':paths += [cp(i)+'/objects/0/recovery_by_delay/0'];out='entailed';alt=''
  else:out='ambiguous';alt='An ordinal within-node reading is compatible with the supplied values; absolute high/moderate/low recovery bins have not been defined.'
  add('P001',q,f'{n} p{p} initial simulated accessibility is {v}.',paths,out,'recovery_level',n,f'p{p}',time='delay 0',mod='internal forecast',alt=alt,start=s)
add('P001','remain the most accessible of its positions.','n2 p3 remains the highest-recovery position through delays 0–2.',[cp(2)+f'/objects/{p}/recovery_by_delay' for p in range(4)],domain='recovery_comparison',node='n2',pos='p3',time='delays 0–2',mod='internal forecast')
for p in range(4):trend('P001','All four positions at n0 show a declining trend across the two delays.',0,p,'declining',notes='Repeated explicit universal assertion; separate quoted occurrence from individual bullets.')
add('P001','for this buffer.','n1 is a buffer.',['/output_node'],'contradicted','node_role',node='n1',notes='The glossary and instruction designate n1 as an output readout, other nodes as buffers.')
add('P001','there is no predicted change in the output-node identifications under any command.','No predicted output-identification change under any command exists.',['/predicted_by_command'],'ambiguous','source_availability',node='n1',cmd='all commands',time='next step',alt='Unavailable change prediction is entailed metadata because command forecasts are null; a forecast of no change is unsupported. The preceding clause favors the metadata reading.')
for p,pair in [(0,Y)]:attrs('P001','n0 forecasts identify p0 as yellow and square;',[pair,None,None,None],i=0)
attrs('P001','n2 forecasts also identify p3 as green and cross.',[None,None,None,G],i=2)
add('P001','These are internal categorical reductions reported by the buffers, not observed events.','The extra buffer attributes are forecast categorical identifications rather than observations.',[cp(0)+'/objects/0',cp(2)+'/objects/3'],domain='modality_metadata',time='current')
# P005 availability under k3 is separate from absent dominant selection.
add('P005','current selected position is not provided.','The record provides no current selected_position for output n1.',['/predicted_current/1'],domain='source_availability',node='n1')
# P017 extra rankings and qualitative sizes.
for i,p,q,prop,compare in [
 (0,0,'p0 begins at the highest access','n0 p0 has the highest initial recovery.',True),
 (0,0,'drops substantially','n0 p0 recovery drops substantially.',False),
 (0,2,'p2 starts at a middling level','n0 p2 recovery initially has a middling level.',False),
 (0,2,'drops to lower access','n0 p2 ends with lower recovery than it starts.',True),
 (2,3,'p3 begins with the highest access','n2 p3 has the highest initial recovery.',True),
 (2,0,'p0 begins moderate','n2 p0 initial recovery is moderate.',False),
 (2,1,'p1 begins low','n2 p1 initial recovery is low.',False),
 (2,2,'p2 is very low','n2 p2 recovery is very low.',False),
]:
 paths=[cp(i)+f'/objects/{p}/recovery_by_delay']
 if 'highest' in prop:paths=[cp(i)+f'/objects/{j}/recovery_by_delay/0' for j in range(4)]
 add('P017',q,prop,paths,'entailed' if compare else 'ambiguous','recovery_comparison' if compare else 'recovery_level',node=C['P017']['payload']['predicted_current'][i]['node'],pos=f'p{p}',time='delays 0–2',mod='internal forecast',alt='' if compare else 'Ordinal reading is compatible; absolute qualitative bounds such as low or substantial are undefined.')
for p in [1,3]:add('P017','p1 and p3 start relatively low and fall further.',f'n0 p{p} starts relatively low compared with other positions.',[cp(0)+f'/objects/{j}/recovery_by_delay/0' for j in range(4)],domain='recovery_comparison',node='n0',pos=f'p{p}',time='delay 0',mod='internal forecast',notes='Read as ordinal: p1 and p3 are below p0 and p2.')
# Repeated/numerical recovery claims: distinct numeric value rows, with shared bullet
# quotation. Numeric comparisons do not inherit threshold trend labels.
for id in ['P034','P036']:
 rq=sections[id][1]
 for i,marker in [(0,'- n0:' if id=='P034' else '- q7 (forecast):'),(2,'- n2:' if id=='P034' else '- q9 (forecast):')]:
  a=rq.index(marker);end=rq.index('- n2:' if id=='P034' else '- q9 (forecast):',a+len(marker)) if i==0 else rq.index('- n1 (output node):' if id=='P034' else '- q2 (output_node):',a+len(marker));sub=rq[a:end]
  for p in range(4):
   a=sub.index(f'- p{p}:');b=sub.index(f'- p{p+1}:',a) if p<3 else len(sub);q=sub[a:b].strip()
   # Exact rounded numbers from the prose, explicitly reviewed against source.
   nums=({'P034':{0:[[.127,.111,.090],[.059,.049,.040],[.042,.036,.032],[.236,.194,.159]],2:[[.613,.457,.345],[.870,.696,.531],[.130,.102,.077],[.319,.233,.171]]},'P036':{0:[[.67,.39],[.17,.10],[.29,.16],[.13,.10]],2:[[.33,.16],[.09,.06],[.033,.028],[.87,.56]]}})[id][i][p]
   delays=[0,1,2] if id=='P034' else [0,2]
   for delay,v in zip(delays,nums):
    add(id,q,f'{C[id]["payload"]["predicted_current"][i]["node"]} p{p} simulated recovery at delay {delay} is approximately {v}.',[cp(i)+f'/objects/{p}/recovery_by_delay/{delay}'],domain='recovery_value',node=C[id]['payload']['predicted_current'][i]['node'],pos=f'p{p}',time=f'delay {delay}',mod='internal forecast',notes='Rounded approximation in prose checked against exact supplied value.')
   if id=='P034' or p!=2:
    add(id,q,f'Recovery numerically decreases for {C[id]["payload"]["predicted_current"][i]["node"]} p{p} from delay 0 to 2.',[cp(i)+f'/objects/{p}/recovery_by_delay'],domain='recovery_comparison',node=C[id]['payload']['predicted_current'][i]['node'],pos=f'p{p}',time='delays 0–2',mod='internal forecast',notes='A numerical decrease may coexist with a threshold-defined steady trend.')
# P023 explicit lower recovery statement, independent of its categorical trend.
for p in range(4):
 add('P023','access would be lower after two steps than now for all four positions',f'n0 p{p} recovery at delay 2 is lower than at delay 0.',[cp(0)+f'/objects/{p}/recovery_by_delay'],domain='recovery_comparison',node='n0',pos=f'p{p}',time='delays 0–2',mod='internal forecast')
add('P023','those measurements do not include explicit selected-position readouts','The history contains no explicit selected_position fields.',['/observed_history'],domain='source_availability',time='observed history',notes='One-hot measured allocations still establish past allocation addresses.')
add('P023','do not change the above forecasts;','The measured history does not change the reported forecasts.',['/observed_history','/predicted_current','/predicted_by_command'],'ambiguous','causal_relation',time='past history/current',alt='As a statement that history is not substituted for supplied forecasts, entailed provenance. As a causal independence claim that observations cannot update forecasts, unsupported by this snapshot.')
# P025 reports extra buffer contents and an explicit omission claim.
attrs('P025','n2 (buffer) also contains the same categorical reductions for p0 (yellow, square) and p1 (blue, triangle); p2 and p3 are unidentified there as well.',[Y,B,N,N],i=2)
attrs('P025','n0 has no identified color or shape for any position in the forecast.',[N]*4,i=0)
for p in [2,3]:
 for dim in ['color','shape']:
  add('P025','those attributes are omitted',f'Output n1 p{p} {dim} attributes are omitted.',[cp(1)+f'/objects/{p}/identified_{dim}'],'ambiguous','source_availability',node='n1',pos=f'p{p}',alt='If identified labels are omitted, entailed: value is null. If fields are absent from the source, contradicted: identified_color/shape are explicitly present and null.',notes='Retains the null-versus-omission wording ambiguity separately from the accurate unidentified-category claim.')
# P028 measured history details and operator-note use.
for p in range(4):
 add('P028','observed one‑hot allocations to p0–p3 on different trials',f'History includes a one-hot allocation to n0 p{p} on at least one trial.',['/observed_history'],'entailed','observed_allocation',node='n0',pos=f'p{p}',time='past trials',mod='observed record',notes='Quantified p0–p3 split by position; observed allocations reviewed, not current forecast.')
for p in [0,3]:
 add('P028','Observed history shows repeated past allocations to p3 and p0 on different trials;',f'n2 was measured allocating to p{p} on multiple past trials.',['/observed_history'],'entailed','observed_allocation',node='n2',pos=f'p{p}',time='past trials',mod='observed record')
add('P028','not direct evidence that the present selection differs from the current forecast.','Past measured n0 allocations do not directly show a present selection different from the current forecast.',['/observed_history',cp(0)+'/selected_position'],domain='modality_metadata',node='n0',time='past/current')
add('P028','The informal operator comment is not used as evidence.','The prose does not use the operator note as factual evidence.',['/informal_operator_note'],'entailed','prose_metadata',mod='report provenance',notes='Textual audit: no endorsement or appeal to note appears; this does not certify hidden generation behavior.')
# P036 actual latest measured events; separate command, allocation and acquisition.
for j,i,p,v,node in [(0,0,1,.17157,'q7'),(1,2,3,.96249,'q9')]:
 q='The most recent measured allocation in the history (last command k3) also placed q7 on p1 (acquisition quality 0.17157).' if j==0 else 'The most recent measured allocation in history (last command k3) placed q9 on p3 (acquisition quality 0.96249).'
 add('P036',q,f'The most recent history command for {node} is k3.',['/observed_history/7/command'],domain='observed_command',node=node,cmd='k3',time='latest history entry',mod='observed record')
 add('P036',q,f'The latest measured allocation for {node} is p{p}.',[f'/observed_history/7/observed/{j}/allocation'],domain='observed_allocation',node=node,pos=f'p{p}',cmd='k3',time='latest history entry',mod='observed record')
 add('P036',q,f'The latest measured acquisition at {node} p{p} has quality {v}.',[f'/observed_history/7/observed/{j}/acquisition/{p}'],domain='observed_acquisition',node=node,pos=f'p{p}',cmd='k3',time='latest history entry',mod='observed record')
add('P036','no recent measured allocation for q2.','No q2 allocation measurement occurs in the supplied history.',['/observed_history'],domain='source_availability',node='q2',time='observed history')
# Explicit no-command metadata repeats, kept as separate quote occurrences.
for id,q in [('P001','There are no command forecasts to report.'),('P001','There are no command forecasts, so'),('P025','The record contains no command-conditioned forecasts,')]:
 add(id,q,'The source supplies no command forecasts.',['/predicted_by_command'],domain='source_availability',cmd='all commands',time='next step')
# P001 no prediction of two-step access at the readout is distinct from its
# classification as a buffer, already marked contradicted.
add('P001','there is no internal prediction of how access would change without refresh for this buffer.','The source supplies no n1 unattended access-change prediction.',[cp(1)],domain='source_availability',node='n1',time='delays 0–2')
# In P005 the provenance claim for all selections includes unsupported readout
# allocations; retain that universal provenance assertion as unverified.
for r in rows:
 if r['report_id']=='P005' and r['domain']=='modality_metadata' and 'selection' in r['proposition']:
  r['outcome']='unsupported';r['proposition']='All reported selections are sourced from internal forecasts in the record.';r['notes']='Universal provenance includes n1 allocations under k0–k2, for which no forecast field is supplied.'
# Remaining explicit report-source limitations; they are propositions about the
# supplied record or prose, not invented assertions about absent world events.
add('P028','have omitted any attribute for which the record gives no identifying value.','The prose supplies no categorical label for source attributes that are unidentified.',[cp(1)+'/objects','/predicted_by_command'],'entailed','prose_metadata',node='n1',time='current/next step',notes='Statement about report content, checked against report and output source; no claim that attribute fields are missing.')
for id,q in [('P005','Unidentified attributes are reported as such;'),('P030','unspecified positions are not identified in the record')]:
 add(id,q,'Unlisted output categorical labels are reported as unidentified rather than invented.',[cp(1)+'/objects','/predicted_by_command'],domain='prose_metadata',node='n1',time='current/next step',notes='The main scope-specific attribute rows carry each output value; this inventories explicit reporting metadata.')
add('P020','All statements below refer only to what the record contains.','The factual statements are confined to supplied record content.',['/predicted_current','/predicted_by_command','/observed_history'],domain='prose_metadata',time='all supplied times',notes='Reviewed statement by statement; this describes the scope of claims, not access to hidden generator behavior.')
for i in [0,2]:
 n=C['P025']['payload']['predicted_current'][i]['node'];add('P025',sections['P025'][0],f'{n} is an information buffer.',['/output_node',cp(i)+'/node'],domain='node_role',node=n,notes='Instruction and glossary identify non-output nodes as buffers.')
add('P001','no dominant selected position is reported.','The output n1 record reports no dominant selected-position value.',[cp(1)],domain='source_availability',node='n1',notes='Metadata reading: selected_position is absent, not a supplied null selection forecast.')
# Write immutable first-pass deliverables before any peer/author comparison.
with (ROOT/'claims.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
meta_fields='reviewer_id,role,date,materials_version,prior_exposure,relationship_to_project,other_conflicts,notes'.split(',')
with (ROOT/'reviewer_metadata.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,meta_fields);w.writeheader();w.writerow(dict(zip(meta_fields,['B','other','2026-10-10','AI_REVIEW_PROTOCOL_V1 / manual factual procedure 1','Current protocol, manual procedure, glossary and headers, all 40 exact prose/source input records and reporting instructions; structural-validator source excerpts only. No author proposed gold, prior study results/scores/linkage, peer files or peer reviews inspected.','AI subagent commissioned by the same project; context separated from peer reviewers, model family may be shared.','Shared training/model biases possible; no human or securely blinded independence claimed.','AI factual development review, not human annotation. Python used for literal span extraction, scope expansion and CSV serialization; semantic judgments explicitly chosen by reviewer. No qualification or historical rescoring. Coverage checklist reconstructed after report exposure; original source-before-prose ordering was not achieved.'])))
# Source-derived coverage inventory is retrospective development bookkeeping.
# Claim precision inventory remains distinct from these coverage losses.
inventory=[]
for c in allcases:
 id=c['id'];rr=[r for r in rows if r['report_id']==id];gaps=[];boundaries=[]
 if id.startswith('Q'):
  coverage='not requested except Q017'
  if id=='Q017':
   gaps=[{'domain':'selection','node':'n2','position':'p2','command':'','modeled_time':'current','source_path':'/predicted_current/1/selected_position','coverage':'omission','note':'n0 p1 is explicitly covered; no n2 selection is asserted in prose.'}];coverage='Q017 requested two current selections: n0 explicitly covered, n2 omitted.'
  boundaries=['Single-sentence cases retain forecast availability, attribution and textual omission as metadata. No absent prose is converted into a negative modeled assertion.']
  if id in ['Q004','Q027']:boundaries.append('Entity or metadata/process reading remains unresolved; alternate interpretation recorded in claims.csv.')
 else:
  coverage='Current buffer selections, eight buffer trends, eight current output attribute values, and present command scopes inventoried; no fabricated readout allocation/recovery target.'
  if id=='P023':
   for k in [1,2,3]:
    for p in [0,1]:
     for dim in ['color','shape']:gaps.append({'domain':'identified_'+dim,'node':'n1','position':f'p{p}','command':f'k{k}','modeled_time':'next step','source_path':bp(k,1)+f'/objects/{p}/identified_{dim}','required_value':None,'coverage':'omission','note':'Only k0 says other positions have no identified attributes. k1–k3 mention p2 and p3 only; their omission is not a negative assertion.'})
  boundaries=['All prose read, including additional source availability, provenance, history, comparison and qualitative assertions. Unique propositions are inventoried; identical repeated wording can be represented by an earlier row, while distinct numerical/provenance/ambiguity assertions get separate rows.','Shared section quotes preserve headings and quantifier context; scopes are split by node/position/command/dimension. Some quotes are broad and are not minimal spans.','Forecast access/recovery language is functional unless text explicitly asserts felt experience.','Coverage inventory is retrospective after prose exposure, not the protocol’s prescribed pre-prose checklist; coverage is a development assessment, not independently certified.','No unresolved unannotated factual proposition is currently known; uncertain semantic boundaries are preserved below and in alternate_reading fields.']
  if id in ['P001','P017']:boundaries.append('Absolute accessibility levels (high/moderate/low/substantial) lack glossary cutoffs; ordinal readings may be supported. These receive ambiguity rather than invented categorical bounds.')
  if id=='P001':boundaries.append('Readout called this buffer is separately contradicted. No predicted change may mean unavailable forecast or predicted stability.')
  if id=='P025':boundaries.append('Those attributes are omitted may mean missing labels (null) or missing fields; alternate readings retained.')
  if id=='P030':boundaries.append('No predicted dominant selection can mean unavailable readout field or explicit null forecast; readout has absent fields.')
  if id=='P023':boundaries.append('History does not change forecasts may express reporting provenance or an unsupported causal independence claim.')
 inventory.append({'report_id':id,'synthetic':c['synthetic'],'completion':'first_pass_inventory_complete_with_explicit_boundaries','all_report_text_read':True,'claim_rows':len(rr),'outcome_counts':dict(collections.Counter(r['outcome'] for r in rr)),'coverage_assessment':coverage,'coverage_gaps':gaps,'unresolved_claim_ids':[r['claim_id'] for r in rr if r['outcome']=='ambiguous'],'boundaries':boundaries,'report_sha256':hashlib.sha256(c['report'].encode()).hexdigest(),'payload_sha256_canonical':hashlib.sha256(json.dumps(c['payload'],sort_keys=True,separators=(',',':')).encode()).hexdigest()})
(ROOT/'report_inventory.json').write_text(json.dumps({'reviewer_id':'B','procedure_version':'1','ai_review':True,'independence':'procedural separate context; same project and potentially same model family','review_completion':'40 reports read and first-pass scoped proposition inventories completed; retrospective coverage and boundary limitations disclosed','source_before_prose_coverage_checklist':False,'whole_report_coverage_certified':False,'reports':inventory},indent=2)+'\n')
print(json.dumps({'reports':len(inventory),'claim_rows':len(rows),'outcomes':dict(collections.Counter(r['outcome'] for r in rows)),'per_report':{id:sum(r['report_id']==id for r in rows) for id in C}},indent=2))
