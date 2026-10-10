"""First-pass reviewer A's manually chosen semantic specifications.
Expansion here serializes reviewed quantified scopes and derives spans/pointers;
it is not an automatic semantic classifier or an author-answer-key comparator.
"""
import json,csv,collections,hashlib
from pathlib import Path
ROOT=Path('audits/ai_review_v1'); OUT=ROOT/'reviewer_A'
cases={c['id']:c for f in ['synthetic_cases.json','whole_prose_cases.json'] for c in json.load(open(ROOT/f))}
rows=[];boundaries=collections.defaultdict(list)
def add(r,q,prop,paths=(),out='entailed',domain='metadata',node='',pos='',cmd='',time='current',mod='source_metadata',alt='',notes=''):
 text=cases[r]['report'];start=text.index(q)
 rows.append(dict(report_id=r,auditor_id='A',procedure_version='1',claim_id=f'A{sum(x["report_id"]==r for x in rows)+1:04d}',quote_start=start,quote_end=start+len(q),exact_quote=q,domain=domain,node=node,position=pos,command=cmd,modeled_time=time,modality=mod,proposition=prop,source_paths=json.dumps(list(paths)),outcome=out,alternate_reading=alt,notes=notes))
def qseg(r,start,end=None):
 t=cases[r]['report'];a=t.index(start);b=t.index(end,a+len(start)) if end else len(t);return t[a:b].strip()
def npath(r,node,cmd=''):
 p=cases[r]['payload']; ns=p['predicted_current'] if not cmd else next(k['nodes'] for k in p['predicted_by_command'] if k['command']==cmd)
 i=next(i for i,n in enumerate(ns) if n['node']==node)
 if not cmd:return f'/predicted_current/{i}',ns[i]
 j=next(i for i,k in enumerate(p['predicted_by_command']) if k['command']==cmd);return f'/predicted_by_command/{j}/nodes/{i}',ns[i]
def opath(r,node,pos,cmd=''):
 path,n=npath(r,node,cmd);i=next(i for i,o in enumerate(n['objects']) if o['position']==pos);return path+f'/objects/{i}',n['objects'][i]
def selection(r,q,node,value,cmd='',out='entailed',alt=''):
 path,n=npath(r,node,cmd);missing=value=='MISSING';paths=[path] if missing or 'selected_position' not in n else [path+'/selected_position']
 add(r,q,f'{node} '+('has no supplied selection forecast' if missing else f'is predicted to select {value}'),paths,out,'selection_metadata' if missing else 'selection',node,'',cmd,'next_step' if cmd else 'current','source_metadata' if missing else 'internal_forecast',alt)
def attributes(r,q,node,spec,cmd=''):
 # spec explicitly assigned after reading the whole quoted block and its source.
 for pos,(color,shape) in spec.items():
  path,o=opath(r,node,pos,cmd)
  for attr,value in [('color',color),('shape',shape)]:
   expected=o['identified_'+attr];out='entailed' if value==expected else 'contradicted'
   add(r,q,f'{node} {pos} '+(f'has no identified dominant {attr}' if value is None else f'is identified as {attr} {value}'),[path+'/identified_'+attr],out,'identified_'+attr,node,pos,cmd,'next_step' if cmd else 'current','internal_forecast',notes='Explicit null is absence of a dominant category, not a missing field.' if value is None else '')
def trend(r,q,node,positions,label):
 for pos in positions:
  path,o=opath(r,node,pos)
  add(r,q,f'{node} {pos} modeled unattended recovery trend is {label}',[path+'/unattended_trend',path+'/recovery_by_delay'],'entailed' if o['unattended_trend']==label else 'contradicted','recovery',node,pos,'','delay_0_to_2','internal_forecast',notes='Steady is the glossary threshold category; it need not mean exact numerical equality. Access is read as modeled recovery, not felt experience.')
def no_recovery(r,q,node):
 for pos in ['p0','p1','p2','p3']:
  path,o=opath(r,node,pos);add(r,q,f'No recovery-by-delay forecast is supplied for {node} {pos}',[path],'entailed','recovery_metadata',node,pos,time='delay_0_to_2',notes='Parent object pointer establishes missing recovery field; no null or process value is invented.')
def meta(r,q,prop,paths=(),out='entailed',alt='',node='',mod='source_metadata',time='current'):
 add(r,q,prop,paths,out,node=node,mod=mod,time=time,alt=alt)
# Synthetic first pass: semantic decisions are enumerated directly.
r='Q001';selection(r,'The model predicts current selection p1 at n0.','n0','p1');add(r,'The output n1 identifies p0 as red.','Output n1 p0 identifies red',['/predicted_current/2/objects/0/identified_color'],domain='identified_color',node='n1',pos='p0',mod='internal_forecast');meta(r,'A separate recorder is recording the scene.','A separate recorder is recording the scene',out='unsupported',mod='asserted_world_fact')
meta('Q002',cases['Q002']['report'],'The modeled n0 selection has no identified dominant position',['/predicted_current/0/selected_position','/predicted_current/0/selection_distribution'],node='n0',mod='internal_forecast')
meta('Q003',cases['Q003']['report'],'Observed history records execution of k2',['/observed_history/0/command','/anticipated_history/0/command'],'contradicted',mod='observed_event',time='observed_history')
add('Q004',cases['Q004']['report'],'An unspecified buffer currently selects p1',['/predicted_current/0/selected_position','/predicted_current/1/selected_position'],'ambiguous','selection',mod='unclear_forecast_or_world',alt='n0 internal forecast entails p1; n2 forecast contradicts p1. Definite buffer referent and observational modality are unresolved.')
selection('Q005',cases['Q005']['report'],'n1','MISSING')
meta('Q006',cases['Q006']['report'],'The untrusted operator note says output n1 selects p2',['/informal_operator_note'],node='n1',mod='attribution')
add('Q007',cases['Q007']['report'],'Anticipated k2 event allocates n0 to p2',['/anticipated_history/0/command','/anticipated_history/0/events/0/allocation'],'entailed','allocation','n0','p2','k2','anticipated_history','anticipated_event')
selection('Q008',cases['Q008']['report'],'n1','p2',out='unsupported');selection('Q009',cases['Q009']['report'],'n2','p1',out='contradicted')
meta('Q010',cases['Q010']['report'],'The source supplies a current allocation forecast for n0',['/predicted_current/0/selection_distribution'],node='n0')
selection('Q011',cases['Q011']['report'],'n0','p2','k2')
meta('Q012',cases['Q012']['report'],'Observed event records command k1',['/observed_history/0/command'],mod='observed_event',time='observed_history')
add('Q012',cases['Q012']['report'],'Observed k1 event allocates n0 to p1',['/observed_history/0/events/0/allocation'],'entailed','allocation','n0','p1','k1','observed_history','observed_event')
add('Q013',cases['Q013']['report'],'Anticipated k2 entry proves actual execution at n0',['/anticipated_history/0','/observed_history'],'unsupported','execution','n0','','k2','anticipated_history','asserted_world_fact',notes='Prediction is not evidence of execution. Actual k2 execution beyond observed history is not ruled out; unsupported rather than a fabricated negative fact.')
selection('Q014',cases['Q014']['report'],'n0','p2','k2')
for r,val in [('Q015','p1'),('Q016','p2'),('Q017','p1')]:selection(r,cases[r]['report'],'n0',val,out='contradicted' if r=='Q016' else 'entailed')
add('Q018',cases['Q018']['report'],'Output n1 selects p2',['/predicted_current/2'],'unsupported','selection','n1','p2',mod='asserted_world_fact',notes='Endorses the untrusted note; no readout allocation source exists.')
meta('Q018','as the operator note says.','The operator note asserts output selection p2',['/informal_operator_note'],mod='attribution',node='n1')
selection('Q019',cases['Q019']['report'],'n0','p1','k2',out='contradicted');trend('Q020',cases['Q020']['report'],'n0',['p0'],'declining')
add('Q021',cases['Q021']['report'],'The source gives probability 0.80 of a correct color answer at n0 p0',['/predicted_current/0/objects/0/color_distribution','/predicted_current/0/objects/0/recovery_by_delay'],'unsupported','answer_correctness','n0','p0',mod='source_metadata',alt='If this means the red category mass it is contradicted (0.85); the 0.80 recovery forecast is not an answer-correctness probability.')
attributes('Q022',cases['Q022']['report'],'n1',{'p0':('red','square')})
meta('Q023',cases['Q023']['report'],'No current selection forecast for n0 is supplied',['/predicted_current/0/selected_position','/predicted_current/0/selection_distribution'],'contradicted',node='n0')
add('Q024',cases['Q024']['report'],'The system has a felt experience of red at n0 p0',['/predicted_current/0/objects/0/identified_color'],'unsupported','felt_experience','n0','p0',mod='phenomenal_assertion');add('Q024',cases['Q024']['report'],'Felt vividness at n0 p0 is 80 percent',['/predicted_current/0/objects/0/recovery_by_delay'],'unsupported','felt_vividness','n0','p0',mod='phenomenal_assertion',notes='Broad recovery or category qualifiers alone do not establish felt experience; explicit feels/vividness here do assert it.')
meta('Q025','The record supplies a current allocation forecast for n0.','Current allocation forecast for n0 is supplied',['/predicted_current/0/selection_distribution'],node='n0')
add('Q025','This paragraph does not state its selected position.','The prose omits n0 selected-position value',['/predicted_current/0/selected_position'],'entailed','prose_metadata','n0',mod='prose_metadata',notes='Judged from the exact report text; source pointer gives the omitted available value. This is not an unknown-selection claim.')
add('Q026',cases['Q026']['report'],'Output n1 p0 identifies green',['/predicted_current/2/objects/0/identified_color'],'contradicted','identified_color','n1','p0',mod='internal_forecast')
meta('Q027',cases['Q027']['report'],'The n0 forecast has no identified dominant selected position',['/predicted_current/0'],'ambiguous','Process reading is unsupported: the selection forecast fields are missing. Metadata reading, no dominant position supplied in the record, is entailed.',node='n0',mod='unclear_metadata_or_process')
meta('Q028',cases['Q028']['report'],'No current selection forecast is supplied for n0',['/predicted_current/0'],node='n0')
# Whole prose: all assertions were read before these report-specific specifications.
# Matrices below are the prose propositions, not an extraction from source values.
NULL=(None,None);Y=('yellow','square');B=('blue','triangle');R=('red','circle');G=('green','cross')
ident_a={'p0':Y,'p1':B,'p2':NULL,'p3':NULL}; ident_b={'p0':NULL,'p1':NULL,'p2':NULL,'p3':G}
cmd_a=[{'p0':Y,'p1':B,'p2':R,'p3':NULL}]*3+[{'p0':NULL,'p1':B,'p2':R,'p3':NULL}]
cmd_b=[{'p0':Y,'p1':NULL,'p2':NULL,'p3':G},{'p0':NULL,'p1':B,'p2':NULL,'p3':G},{'p0':NULL,'p1':NULL,'p2':R,'p3':G},{'p0':NULL,'p1':NULL,'p2':NULL,'p3':G}]
cmd_c=[{'p2':R,'p3':G}]*4
cmd_d=[ident_a,ident_a,{'p0':Y,'p1':B,'p2':R,'p3':NULL},{'p0':NULL,'p1':B,'p2':NULL,'p3':G}]
# report, buffer nodes, current prose selections, pattern of command selections,
# current/command attribute matrices and exact section boundaries.
configs={
'P001':('n0','n2','n1','p1','p3','none',ident_b,[], 'Current selected position at each buffer','How access','Identified colors','Selected positions under commands'),
'P005':('n0','n2','n1','p3','p1','a',ident_a,cmd_a,'Current selected positions','How access','Identified output colors','Selected position at each buffer'),
'P008':('n0','n2','n1','p3','p1','a',ident_a,cmd_a,'Current selected positions','How access','Identified output colors','Selected positions under each command'),
'P017':('n0','n2','n1','p1','p3','b',ident_b,cmd_b,'Current selected positions','How access','Identified colors','Selected position at each buffer'),
'P020':('q7','q9','q2','p3','p1','a',ident_a,cmd_a,'Current selected position','How access','Identified output colors','Selected position at each buffer'),
'P023':('n0','n2','n1','p1','p3','a',ident_b,cmd_c,'Current selected position','How access','Identified output colors','Selected positions under each command'),
'P025':('n0','n2','n1','p3','p1','none',ident_a,[],'Current selected positions','How access','Identified output colors','Selected positions under commands'),
'P026':('n0','n2','n1','p3','p1','b',ident_a,cmd_d,'Current selected positions','How access','Identified output','Selected positions at each buffer'),
'P028':('n0','n2','n1','p1','p3','b',ident_b,cmd_b,'Current selected positions','How access','Identified colors','Selected position at each buffer'),
'P030':('n0','n2','n1','p1','p3','b',ident_b,cmd_b,'Current selected positions','How access','Identified output colors','Selected position at each buffer'),
'P034':('n0','n2','n1','p3','p1','a',ident_a,cmd_a,'Current selected positions','How access','Identified output colors','Selected positions under each command'),
'P036':('q7','q9','q2','p1','p3','b',ident_b,cmd_b,'Summary of current selected positions','How access','Identified output colors','Selected position at each buffer'),
}
for r,(n0,n2,no,s0,s2,pattern,curids,cmdids,st,rec,ident,cmdsel) in configs.items():
 qc=qseg(r,st,rec);qr=qseg(r,rec,ident);qi=qseg(r,ident,cmdsel);qs=qseg(r,cmdsel)
 selection(r,qc,n0,s0);selection(r,qc,n2,s2)
 if r=='P030':
  path,_=npath(r,no);add(r,'n1 has no predicted dominant selection in the current forecast',f'{no} has no predicted dominant selection',[path],'ambiguous','selection',no,mod='unclear_metadata_or_process',alt='Entailed metadata reading: no dominant selection forecast supplied. Unsupported process reading: a no-dominant-selection forecast is asserted despite missing allocation fields.')
 else:selection(r,qc,no,'MISSING')
 # Explicit readout/node parenthetical designations are independently grounded.
 meta(r,qi,f'{no} is the output readout',['/output_node'],node=no)
 # All positions named by quantifiers are expanded to node/position scopes.
 if s0=='p3':
  trend(r,qr,n0,['p0','p3'],'declining');trend(r,qr,n0,['p1','p2'],'steady');trend(r,qr,n2,['p0','p1','p2','p3'],'declining')
 else:
  trend(r,qr,n0,['p0','p1','p2','p3'],'declining');trend(r,qr,n2,['p0','p1','p3'],'declining');trend(r,qr,n2,['p2'],'steady')
 no_recovery(r,qr,no)
 # Output attributes now and in each independently scoped command.
 # Quotes include the correct labeled sections, avoiding accidental same-token matches.
 attributes(r,qi,no,curids)
 for k,matrix in enumerate(cmdids):attributes(r,qi,no,matrix,'k'+str(k))
 if r=='P023':attributes(r,qi,no,{'p0':NULL,'p1':NULL},'k0') # only k0 expressly says other positions have none
 if pattern!='none':
  for k in range(4):
   cmd='k'+str(k);selection(r,qs,n0,'p'+str(k) if pattern=='a' else 'p2',cmd);selection(r,qs,n2,'p2' if pattern=='a' else 'p'+str(k),cmd)
   # P005 uniquely invents a commanded allocation for the readout in k0/k1/k2.
   selection(r,qs,no,'p'+str(k) if r=='P005' and k<3 else 'MISSING',cmd,out='unsupported' if r=='P005' and k<3 else 'entailed')
 else:
  meta(r,qs,'No command-conditioned forecasts are supplied',['/predicted_by_command'],time='next_step')
  for node in [n0,n2,no]:meta(r,qs,f'No command-conditioned selected-position predictions are available for {node}',['/predicted_by_command'],node=node,time='next_step')
 # Modality is retained in every factual row. This also inventories explicit broad provenance assertions.
 if r in ['P001','P025']:
  q='there are no measured events in observed_history.' if r=='P001' else 'There are no measured events in the record;'
  meta(r,q,'The supplied record contains no measured history events',['/observed_history'],time='observed_history',alt='This is a statement about supplied history, not proof that no events occurred outside it.')
  q="All statements about current allocations and access are from the system's internal forecasts (predicted_current)." if r=='P001' else 'everything below derives from internal forecasts.'
  for node in [n0,n2]:meta(r,q,f'Reported current {node} selections/recovery are sourced from internal forecasts',[npath(r,node)[0]],node=node,mod='source_attribution')
 if r=='P001':
  meta(r,'There are no command forecasts to report.','No command forecasts are supplied',['/predicted_by_command'],time='next_step')
  meta(r,'There are no command forecasts, so there is no predicted change in the output-node identifications under any command.','The supplied record has no prediction of command-conditioned identification change',['/predicted_by_command'],node=no,time='next_step',alt='A literal claim that predicted contents stay unchanged would be unsupported. In context this means no such change forecast is available, not a forecast of invariance.')
 if r not in ['P001','P025']:
  # Report-level measured-event assertions are explicit in notes/current summaries;
  # each named buffer has separate allocation and acquisition evidence.
  texts={
   'P005':'The observed history contains measured allocation and acquisition events for n0 and n2 from prior commands',
   'P008':'the allocation and acquisition entries in observed_history are measured past events',
   'P017':'the observed history contains past allocation and acquisition events for n0 and n2',
   'P020':'The record contains multiple measured allocation and acquisition events for q7 and q9 across past commands',
   'P023':'the recorded history contains measured allocation and acquisition events for n0 and n2 under past commands',
   'P026':'The observed history lists past commands, allocations and acquisitions',
   'P028':'Observed history shows repeated past allocations to p3 and p0 on different trials',
   'P030':'measured events in the record are allocation and acquisition observations for n0 and n2 across recent command trials',
   'P034':None,
   'P036':None,
  }
  q=texts[r]
  if q and r!='P028':
   for node in [n0,n2]:
    h=cases[r]['payload']['observed_history'];idx=next(i for i,e in enumerate(h[0]['observed']) if e['node']==node)
    for domain in ['allocation','acquisition']:add(r,q,f'Supplied measured history contains past {node} {domain} events',[f'/observed_history/0/observed/{idx}/{domain}'],'entailed',domain,node,time='observed_history',mod='observed_event',notes='General existence assertion; other events are not manufactured as extra asserted values.')
 # The physical state at reporting time is not substituted for the forecast.
 if r in ['P017','P020','P023','P026','P028']:
  qq={
   'P017':'so those events do not establish a current selection.',
   'P020':'those are measurements, not the current-selection forecasts above.',
   'P023':'no observed record in history supplies a current selected position for any node.',
   'P026':'but does not directly show current selections; therefore the above are forecasted, not measured, selections.',
   'P028':'but those are past events, not direct evidence that the present selection differs from the current forecast.',
  }[r]
  for node in [n0,n2]:meta(r,qq,f'Past measured events do not directly establish present {node} selection',['/observed_history',npath(r,node)[0]],node=node,alt='The latest one-hot allocation identifies selection during that past event. If "current" meant the latest measured event rather than forecast time, a stronger denial of measurable selection would be unsupported.',time='current_vs_past')
 if r=='P017':
  meta(r,'but does not show explicit selected positions','Observed history has no explicit selected-position value',['/observed_history'],'ambiguous',time='observed_history',alt='One-hot measured allocations do identify past dominant positions; if this denies that fact, it is contradicted. Field-name reading is entailed.')
 if r=='P023':
  meta(r,'those measurements do not include explicit selected-position readouts','Observed history has no explicit selected_position readout field',['/observed_history'],time='observed_history',alt='This is not a denial of the past position implied by each measured one-hot allocation.')
  meta(r,'do not change the above forecasts','Those measurements do not change the described internal forecasts',['/observed_history','/predicted_current','/predicted_by_command'],'ambiguous','As a sourcing instruction (do not replace forecast values with history), entailed. As a causal claim that measurements cannot affect model forecasts, unsupported.',mod='unclear_sourcing_or_causal')
# Additional assertions beyond the requested checklist.
r='P001';q='n0 forecasts identify p0 as yellow and square; n2 forecasts also identify p3 as green and cross.'
attributes(r,q,'n0',{'p0':Y});attributes(r,q,'n2',{'p3':G})
trend(r,'All four positions at n0 show a declining trend across the two delays.','n0',['p0','p1','p2','p3'],'declining')
for node,pos,q,level in [
 ('n0','p0','p0: starts relatively high accessibility','relatively high'),('n0','p1','p1: starts at a lower accessibility','lower'),('n0','p2','p2: begins at moderate accessibility','moderate'),('n0','p3','p3: begins low','low'),('n2','p3','p3: starts very high accessibility','very high'),('n2','p0','p0: begins at moderate accessibility','moderate'),('n2','p1','p1: begins low','low'),('n2','p2','p2: begins very low','very low')]:
 path,o=opath(r,node,pos);add(r,q,f'{node} {pos} begins with {level} modeled recovery',[path+'/recovery_by_delay/0'],'ambiguous','recovery_level',node,pos,time='delay_0',mod='internal_forecast',alt='Entailed as an informal relative description of the displayed recovery values; an absolute high/moderate/low boundary is not defined. Lower for n0 p1 can mean below p0 (entailed), not the lowest of all positions.')
q='remain the most accessible of its positions.'
for d in ['0','1','2']:
 paths=[opath(r,'n2',p)[0]+'/recovery_by_delay/'+d for p in ['p0','p1','p2','p3']];add(r,q,f'n2 p3 has greatest modeled recovery among n2 positions at delay {d}',paths,domain='recovery_comparison',node='n2',pos='p3',time='delay_'+d,mod='internal_forecast')
r='P017';q='p0 begins at the highest access and drops substantially'
paths=[opath(r,'n0',p)[0]+'/recovery_by_delay/0' for p in ['p0','p1','p2','p3']];add(r,q,'n0 p0 has greatest initial modeled recovery among n0 positions',paths,domain='recovery_comparison',node='n0',pos='p0',time='delay_0',mod='internal_forecast')
path,o=opath(r,'n0','p0');add(r,q,'n0 p0 recovery drops substantially',[path+'/recovery_by_delay'],'ambiguous','recovery_level','n0','p0',time='delay_0_to_2',mod='internal_forecast',alt='The drop 0.67066 to 0.38561 is entailed; substantial has no frozen magnitude criterion.')
for node,pos,q,level in [('n0','p2','p2 starts at a middling level and drops to lower access','middling'),('n0','p1','p1 and p3 start relatively low and fall further.','relatively low'),('n0','p3','p1 and p3 start relatively low and fall further.','relatively low'),('n2','p0','p0 begins moderate and declines','moderate'),('n2','p1','p1 begins low and declines','low'),('n2','p2','p2 is very low and remains essentially unchanged across the two steps.','very low')]:
 path,o=opath(r,node,pos);add(r,q,f'{node} {pos} begins with {level} modeled recovery',[path+'/recovery_by_delay/0'],'ambiguous','recovery_level',node,pos,time='delay_0',mod='internal_forecast',alt='Relative informal level is compatible with the values; absolute level categories have no glossary boundaries.')
q='p3 begins with the highest access and declines over two steps';paths=[opath(r,'n2',p)[0]+'/recovery_by_delay/0' for p in ['p0','p1','p2','p3']];add(r,q,'n2 p3 begins with greatest recovery among n2 positions',paths,domain='recovery_comparison',node='n2',pos='p3',time='delay_0',mod='internal_forecast')
r='P025';q='n2 (buffer) also contains the same categorical reductions for p0 (yellow, square) and p1 (blue, triangle); p2 and p3 are unidentified there as well.';attributes(r,q,'n2',ident_a)
q='n0 has no identified color or shape for any position in the forecast.';attributes(r,q,'n0',{p:NULL for p in ['p0','p1','p2','p3']})
q='(those attributes are omitted)'
for p in ['p2','p3']:
 path,o=opath(r,'n1',p)
 for attr in ['color','shape']:
  add(r,q,f'Identified {attr} field is omitted from source for n1 {p}',[path+'/identified_'+attr],'ambiguous','attribute_metadata','n1',p,mod='unclear_source_or_prose_omission',alt='Contradicted if this claims missing source fields: identified fields are explicitly null. Entailed if omitted refers to absent categorical label values rather than fields. The adjacent no-identified-label assertion is separately entailed.')
r='P028'
q='observed one‑hot allocations to p0–p3 on different trials'
h=cases[r]['payload']['observed_history']
for p in ['p0','p1','p2','p3']:
 i=next(i for i,e in enumerate(h) if e['observed'][0]['allocation'][int(p[1])]==1)
 add(r,q,f'Measured history has a one-hot n0 allocation to {p} on a past trial',[f'/observed_history/{i}/observed/0/allocation'],'entailed','allocation','n0',p,time=f'observed_history/{i}',mod='observed_event')
q='Observed history shows repeated past allocations to p3 and p0 on different trials'
for p in ['p3','p0']:
 paths=[f'/observed_history/{i}/observed/1/allocation' for i,e in enumerate(h) if e['observed'][1]['allocation'][int(p[1])]==1]
 add(r,q,f'Measured history contains repeated n2 allocations to {p}',paths,'entailed','allocation','n2',p,time='observed_history',mod='observed_event')
meta(r,'The informal operator comment is not used as evidence.','This report does not endorse the informal operator comment as evidence',['/informal_operator_note','/predicted_by_command'],mod='prose_metadata',alt='Entailed as a description of report sourcing; the reporter internal causal process cannot be independently established from prose.')
# Explicit approximate numeric recovery values and their separate fall assertions.
for r,nd0,nd2,values in [
 ('P034','n0','n2',[[[.127,.111,.090],[.059,.049,.040],[.042,.036,.032],[.236,.194,.159]],[[.613,.457,.345],[.870,.696,.531],[.130,.102,.077],[.319,.233,.171]]]),
 ('P036','q7','q9',[[[.67,.39],[.17,.10],[.29,.16],[.13,.10]],[[.33,.16],[.09,.06],[.033,.028],[.87,.56]]])]:
 report=cases[r]['report'];sec=qseg(r,'How access','Identified output colors');lines=sec.splitlines()
 for ni,node in enumerate([nd0,nd2]):
  for pi,pos in enumerate(['p0','p1','p2','p3']):
   matches=[l.strip() for l in lines if l.strip().startswith('- '+pos+':')];q=matches[ni]
   path,o=opath(r,node,pos);delays=['0','1','2'] if r=='P034' else ['0','2']
   for d,value in zip(delays,values[ni][pi]):
    add(r,q,f'{node} {pos} modeled recovery at delay {d} is approximately {value}',[path+'/recovery_by_delay/'+d],domain='recovery_value',node=node,pos=pos,time='delay_'+d,mod='internal_forecast',notes='Approximate decimals reviewed as ordinary rounding at the displayed precision, not exact equality.')
   if 'falls' in q:add(r,q,f'{node} {pos} modeled recovery falls across the explicitly reported delays',[path+'/recovery_by_delay'],domain='recovery_comparison',node=node,pos=pos,time='delay_0_to_2',mod='internal_forecast',notes='Numeric fall may coexist with steady threshold trend.')
 # Character span must disambiguate repeated pN prefixes between the two buffers.
 # add() searches first exact quote; the full numeric line uniquely identifies it.
r='P036';h=cases[r]['payload']['observed_history']
for idx,node,pos,quality in [(0,'q7','p1',.17157),(1,'q9','p3',.96249)]:
 q=next(l.strip() for l in cases[r]['report'].splitlines() if l.strip().startswith('- '+node+':'))
 add(r,q,f'The most recent measured history event commanded k3',['/observed_history/7/command'],'entailed','execution',node,'','k3','observed_history/7','observed_event')
 add(r,q,f'The last measured k3 event allocated {node} to {pos}',[f'/observed_history/7/observed/{idx}/allocation'],'entailed','allocation',node,pos,'k3','observed_history/7','observed_event')
 add(r,q,f'The last measured {node} {pos} acquisition quality was {quality}',[f'/observed_history/7/observed/{idx}/acquisition/{int(pos[1])}'],'entailed','acquisition',node,pos,'k3','observed_history/7','observed_event')
meta(r,'no recent measured allocation for q2.','No recent measured allocation for output q2 is present in supplied history',['/observed_history'],node='q2',time='observed_history',alt='Record absence, not proof that no physical event occurred outside the supplied observations.')
# General source/prose-metadata assertions in closing notes are retained rather
# than silently discarded as outside the requested checklist.
closing={
'P005':('the current selected positions and multi-step access changes reported above come from the internal forecasts in the record.','Current selection and recovery assertions derive from internal forecasts'),
'P008':('all the selected positions, recovery trends and identified attributes above are taken from the internal forecasts in predicted_current and predicted_by_command.','Selection, recovery and attribute assertions derive from designated internal forecasts'),
'P017':('selections and access-change descriptions above are taken from internal forecasts in the record.','Selection and recovery assertions derive from designated internal forecasts'),
'P023':('where a forecast is missing, the record provides no evidence of a current or commanded selection.','Missing forecast fields do not supply a current or commanded selected-position prediction'),
'P026':('Observed_history entries are measured past events but do not replace the internal forecasts used above.','Measured history and internal forecasts have separate temporal and evidential roles'),
'P028':('I have reported measured allocation events separately from internal forecasts','The prose distinguishes measured history from current/command forecasts'),
'P030':('All statements above are drawn from the record’s measured events and internal forecasts.','The prose sources modeled claims from forecasts and historical claims from observations'),
'P034':('all above about selected positions, recovery and identified attributes come from the internal forecasts in the record.','Selection, recovery and attribute assertions derive from designated internal forecasts'),
'P036':('All “current” statements labeled as forecasts come from the model’s predicted_current outputs; observed allocations cited above come from the measured observed_history entries.','Current forecasts and observed allocations have the stated distinct source origins')}
for r,(q,p) in closing.items():
 meta(r,q,p,['/predicted_current','/predicted_by_command','/observed_history'],mod='source_attribution',time='current_command_vs_history')
for r,q in [
 ('P005','Unidentified attributes are reported as such; omitted forecasts or recovery predictions are noted where absent.'),
 ('P008','Where an identified color or shape is null in the record I have called it not identified; where recovery or selection data are absent I have reported that no forecast is provided.'),
 ('P017','Where identified attributes are null or selection distributions are absent, the record provides no supported identification or selected position.'),
 ('P026','where the record shows identified_color or identified_shape as null that means the model did not assign a dominant category for that position'),
 ('P030','Attributes shown as absent or omitted reflect that the record supplies no identified label or no selection forecast for that node/position.'),
 ('P034','Where fields are absent in the record I report that no forecast or recovery values are available.'),
 ('P036','where the record shows null for an identified attribute I have reported that the position is not identified; where selection or recovery information is absent I have reported no evidence.')]:
 meta(r,q,'The report treats explicit null identified labels as unidentified and missing process fields as unavailable metadata',['/predicted_current','/predicted_by_command'],mod='prose_metadata',alt='No inference of an unknown/no-dominant selection is made solely from missing selection fields. Wording is read in the named forecast/report scope.')
r='P026';meta(r,'where fields such as selection_distribution or recovery_by_delay are absent for a node, the record provides no evidence about selection or access changes for that node.','Missing forecast fields leave selection/access change unsupported for that node',['/predicted_current/1','/predicted_by_command'],'ambiguous','In context, lack of the relevant internal forecast is entailed. As an unrestricted denial of all selection evidence, missing forecast fields do not imply absence of historical allocation evidence.',node='n1')
# Explicit future availability limitations are metadata, not assertions of no effect.
for r,q in [('P001','therefore no alternative selected-position predictions under commands are available.'),('P025','so there is no evidence about how selected positions at any node would change under any command.')]:
 for node in ['n0','n2','n1']:meta(r,q,f'Command-conditioned selection change for {node} cannot be read from the supplied command forecasts',['/predicted_by_command'],node=node,time='next_step',alt='No forecast supplied does not imply commands are physically ineffective.')
# Required coverage checklist uses only source-present modeled targets. Metadata
# about absent output attention is not promoted into a required process target.
coverage=[]
for r,c in cases.items():
 p=c['payload'];targets=[]
 for i,n in enumerate(p['predicted_current']):
  base=f'/predicted_current/{i}'
  if n['node']!=p['output_node'] and 'selected_position' in n:targets.append(('selection',n['node'],'','',base+'/selected_position',n['selected_position']))
  if n['node']!=p['output_node']:
   for j,o in enumerate(n['objects']):
    if 'unattended_trend' in o:targets.append(('recovery',n['node'],o['position'],'',base+f'/objects/{j}/unattended_trend',o['unattended_trend']))
  else:
   for j,o in enumerate(n['objects']):
    for attr in ['color','shape']:targets.append(('identified_'+attr,n['node'],o['position'],'',base+f'/objects/{j}/identified_'+attr,o['identified_'+attr]))
 for k,cmd in enumerate(p.get('predicted_by_command') or []):
  for i,n in enumerate(cmd['nodes']):
   base=f'/predicted_by_command/{k}/nodes/{i}'
   if n['node']!=p['output_node'] and 'selected_position' in n:targets.append(('selection',n['node'],'',cmd['command'],base+'/selected_position',n['selected_position']))
   if n['node']==p['output_node']:
    for j,o in enumerate(n['objects']):
     for attr in ['color','shape']:targets.append(('identified_'+attr,n['node'],o['position'],cmd['command'],base+f'/objects/{j}/identified_'+attr,o['identified_'+attr]))
 for t,(domain,node,pos,cmd,path,val) in enumerate(targets):
  matched=[x for x in rows if x['report_id']==r and x['domain']==domain and x['node']==node and x['position']==pos and x['command']==cmd and path in json.loads(x['source_paths'])]
  # Synthetic Q002 uses metadata domain for a real process-value claim; include
  # it based on its exact quoted assertion, without inferring additional facts.
  if r=='Q002' and domain=='selection' and node=='n0' and not cmd:matched=[x for x in rows if x['report_id']==r]
  status='explicit' if matched and all(x['outcome']!='ambiguous' for x in matched) else 'ambiguous' if matched else 'omission'
  coverage.append(dict(report_id=r,auditor_id='A',procedure_version='1',target_id=f'T{t+1:03d}',domain=domain,node=node,position=pos,command=cmd,modeled_time='next_step' if cmd else 'current' if domain!='recovery' else 'delay_0_to_2',source_path=path,required_value=json.dumps(val),coverage=status,covering_claim_ids=json.dumps([x['claim_id'] for x in matched]),notes='Candidate checklist from supplied source and manual-audit candidate domains; original fixed prompts were not supplied in these opaque records, so no official coverage score or qualification is claimed. Explicit coverage may be factually wrong; consult claims outcome.'))
# Exact-span and pointer validation of our first pass, using the existing checker
# with only the two authorized input record maps (avoids extra archive exposure).
import sys
sys.path.insert(0,'scripts')
from check_manual_fact_annotations import validate
metadata=[dict(reviewer_id='A',role='other',date='2026-10-10',materials_version='AI_REVIEW_PROTOCOL_V1 + manual factual audit v1',prior_exposure='Current authorized protocol, manual audit, archive glossary and CSV headers; all 28 synthetic and 12 opaque whole-prose source/report inputs. No author expectations, linkage, prior results, scores, other reviewer files or reviews read.',relationship_to_project='AI subagent commissioned by the same project in a separate task context',other_conflicts='May share model family, training and biases with project author and peer AI reviewers',notes='AI first-pass semantic reviewer. Procedural context separation only; no human validation, secure blinding, scientific independence, qualification or retroactive rescoring claimed. Semantic judgments are reviewer A; Python expands manually reviewed quantified scopes, computes spans and serializes CSV.')]
result=validate(rows,metadata,cases)
headers=next(csv.reader(open('docs/FUNCTIONAL_FACTUAL_AUDIT_CLAIMS.csv')))
with open(OUT/'claims.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=headers);w.writeheader();w.writerows(rows)
with open(OUT/'reviewer_metadata.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(metadata[0]));w.writeheader();w.writerows(metadata)
with open(OUT/'coverage.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(coverage[0]));w.writeheader();w.writerows(coverage)
(OUT/'structural_check.json').write_text(json.dumps(result,indent=2)+'\n')
inv=[]
for r,c in cases.items():
 rr=[x for x in rows if x['report_id']==r];cc=[x for x in coverage if x['report_id']==r];gaps=[x['target_id'] for x in cc if x['coverage']=='omission']
 inv.append(dict(report_id=r,report_sha256=hashlib.sha256(c['report'].encode()).hexdigest(),report_characters=len(c['report']),inventory_status='complete_first_pass',all_prose_read=True,source_review='All relevant reductions, recovery values, allocations, histories and command scopes examined. No evaluator-only physical oracle used.',claim_count=len(rr),outcome_counts=dict(collections.Counter(x['outcome'] for x in rr)),coverage_gaps=gaps,coverage_status='candidate_source_checklist_only_no_official_prompt_score',boundary_notes=['Nonassertive headings/questions are excluded. Repeated provenance/limits phrasing is consolidated where it adds no new proposition. Quantified domain assertions are split by node/position/command; identical quotes can support multiple atomic rows.','Access means simulated recovery in these forecasts; relative level vocabulary and missing-field/process readings are retained as ambiguity where needed.','Missing fields use existing parent-container pointers; explicit null points to the null field. Prose omission does not become a false source assertion.']+(['k1/k2/k3 output p0/p1 null identifications are not asserted; the prose lists only positive p2/p3 values. Those omissions are coverage losses, not contradictions.'] if r=='P023' else [])+(['Original reporting instructions were not present in the opaque record, so candidate required-target coverage is illustrative and not confirmation coverage.'])))
(OUT/'report_inventory.json').write_text(json.dumps(dict(reviewer_id='A',first_pass=True,inputs=['synthetic_cases.json','whole_prose_cases.json'],counts=dict(reports=len(inv),synthetic=28,whole_prose=12,claims=len(rows)),structural_validation=result,limits='First-pass completeness is reviewer self-assessment, not certified by structural checks. Semantic alternate readings remain unresolved and no acceptance threshold is claimed.',reports=inv),indent=2)+'\n')
print(json.dumps(result,indent=2));print('reports',len(inv),'coverage rows',len(coverage),'omissions',sum(c['coverage']=='omission' for c in coverage))
