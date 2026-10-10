"""Prepare fixed synthetic qualification and unchanged exported pilot sources."""
import argparse,copy,gzip,hashlib,json,random
from pathlib import Path
ROOT=Path('audits/integrated_report_pilot_v1')
SOURCE=Path('audits/functional_task_interface_v1/records.json.gz')
INSTRUCTION='''Describe the supplied internal record in ordinary prose, at most 180 words.
The output_node supplies the Controller's category decisions; other nodes have modeled selection and recovery. Predictions are fallible model states, not physical truth. An identified attribute has maximum category probability at least 0.6; explicit null means no identified category. Recovery is predicted availability, not answer correctness or felt clarity. predicted_by_command contains unexecuted next-state forecasts. observed_history contains actual measured command/allocation/acquisition events, and null means withheld. The task query, chosen command, execution flag and pending/answered/abstained response are separate facts. Joint abstention does not imply both attributes unknown. No experiential vocabulary is required.
For the queried position, describe current color and shape identification at each node, present selection and current/delayed recovery where supplied, the query and task status (including chosen/executed command and emitted attributes), and the output content available under each modeled command. When relation information is withheld, say so rather than guessing command effects. Numbers may be used when useful. Describe only supported information.'''

def encoded(x):return (json.dumps(x,indent=2)+'\n').encode()
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build():
    cases=[];gold=[];checks=[]
    for i in range(12):
        n='n'+str(i+4);out='q'+str(i+20);cmd='k'+str(i%4);pos='p'+str(i%4)
        pattern=i%6
        color='blue' if i<6 else 'red';shape='square' if i<6 else 'cross'
        status='pending' if pattern==0 else ('abstained' if pattern in (2,4) else 'answered')
        obj={'position':pos,'identified_color':None if pattern==4 else color,
             'identified_shape':None if pattern in (2,4) else shape}
        source={'output_node':out,'modeled_step':12 if pattern==0 else 13,
            'observed_history':None if pattern==3 else [{'command':cmd}],
            'predicted_current':[{'node':n,'selected_position':pos,'objects':[dict(obj,recovery_by_delay={'0':.2,'1':.1,'2':.05})]},
                {'node':out,'objects':[copy.deepcopy(obj)]}],
            'predicted_by_command':[{'command':cmd,'nodes':[{'node':n,'objects':[dict(obj,recovery_by_delay={'0':.8})]}, {'node':out,'objects':[copy.deepcopy(obj)]}]}],
            'task_decision':{'query':{'node':out,'position':pos},'selected_command':cmd,'command_executed':status!='pending',
                'response':{'status':status,'color':color if status=='answered' else None,'shape':shape if status=='answered' else None}}}
        if pattern==0:
            items=[(f'The chosen command is {cmd}.','entailed',['/task_decision/selected_command']),
                ('The task response is abstained.','contradicted',['/task_decision/response/status']),
                (f'The output currently identifies {color} at {pos}.','entailed',['/predicted_current/1/objects/0/identified_color']),
                (f'The physical scene certainly contains {color} at {pos}.','unsupported',[])]
        elif pattern==1:
            items=[(f'The latest observed command is {cmd}.','entailed',['/observed_history/0/command']),
                (f'The supplied next-state forecast for {cmd} guarantees the next command will be executed.','unsupported',[]),
                (f'The output {out} has a supplied current allocation distribution.','contradicted',['/predicted_current/1']),
                (f'Current recovery at {n} {pos} is modeled as 0.2.','entailed',['/predicted_current/0/objects/0/recovery_by_delay/0'])]
        elif pattern==2:
            items=[('The task response is abstained.','entailed',['/task_decision/response/status']),
                (f'The output has an identified {shape} at {pos}.','contradicted',['/predicted_current/1/objects/0/identified_shape']),
                (f'The output retains an identified {color} at {pos}.','entailed',['/predicted_current/1/objects/0/identified_color']),
                ('The Controller is 80% confident its answer is correct.','unsupported',[])]
        elif pattern==3:
            items=[('The selected command is marked executed.','entailed',['/task_decision/command_executed']),
                ('The selected command has not executed.','contradicted',['/task_decision/command_executed']),
                ('Observation history is withheld.','entailed',['/observed_history']),
                ('The recovery value establishes felt clarity.','unsupported',[])]
        elif pattern==4:
            items=[(f'A next-state forecast for {cmd} is supplied.','entailed',['/predicted_by_command/0/command']),
                (f'No current position is identified as selected at {n}.','contradicted',['/predicted_current/0/selected_position']),
                (f'The output has no identified shape at {pos}.','entailed',['/predicted_current/1/objects/0/identified_shape']),
                ('The modeled step count proves that 13 answers have been emitted.','unsupported',[])]
        else:
            items=[(f'Current recovery at {n} {pos} is modeled as 0.2.','entailed',['/predicted_current/0/objects/0/recovery_by_delay/0']),
                (f'The next-state recovery forecast for {cmd} at {n} {pos} is 0.8.','entailed',['/predicted_by_command/0/nodes/0/objects/0/recovery_by_delay/0']),
                (f'Current recovery at {n} {pos} equals its next-state recovery forecast.','contradicted',['/predicted_current/0/objects/0/recovery_by_delay/0','/predicted_by_command/0/nodes/0/objects/0/recovery_by_delay/0']),
                ('The command forecasts establish conscious experience.','unsupported',[])]
        cid=f'Q{i+1:03d}';sentences=[{'id':f'S{j+1}','text':v[0]} for j,v in enumerate(items)]
        cases.append({'id':cid,'synthetic':True,'sentences':sentences,'report':' '.join(v[0] for v in items),'payload':source})
        for j,(text,outcome,paths) in enumerate(items):gold.append({'case_id':cid,'sentence_id':f'S{j+1}','quote':text,'outcome':outcome,'source_paths':paths})
        checks.append({'id':cid,'payload':source})
    records=json.loads(gzip.decompress(SOURCE.read_bytes()))
    variants=('before_execution','after_execution','attenuated_readout','restored_readout','remapped','missing_relation')
    chosen=[r for r in records if r['seed']==2011 and r['row']==3 and r['position']==0 and r['policy']=='model_guided' and r['condition'] in (0,1,-1) and r['variant'] in variants]
    assert len(chosen)==18
    random.Random(20261010).shuffle(chosen)
    requests=[];linkage=[];coverage=[]
    for i,r in enumerate(chosen):
        rid=f'R{i+1:03d}';source=r['payload'];query=source['task_decision']['query']['position'];qindex=int(query[1:])
        requests.append({'id':rid,'instruction':INSTRUCTION,'payload':source})
        linkage.append({'id':rid,**{k:v for k,v in r.items() if k!='payload'}})
        # Source-first required facts; stored before any target prose is read.
        paths=['/task_decision/query/node','/task_decision/query/position','/task_decision/selected_command','/task_decision/command_executed','/task_decision/response/status','/task_decision/response/color','/task_decision/response/shape']
        for ni,node in enumerate(source['predicted_current']):
            for attr in ('identified_color','identified_shape'):paths.append(f'/predicted_current/{ni}/objects/{qindex}/{attr}')
            if node['node']!=source['output_node']:
                paths.append(f'/predicted_current/{ni}/selected_position')
                for delay in ('0','1','2'):paths.append(f'/predicted_current/{ni}/objects/{qindex}/recovery_by_delay/{delay}')
        future=source['predicted_by_command']
        if future is None:paths+=['/predicted_by_command','/observed_history']
        else:
            for ci,trial in enumerate(future):
                ni=next(j for j,n in enumerate(trial['nodes']) if n['node']==source['output_node'])
                for attr in ('identified_color','identified_shape'):paths.append(f'/predicted_by_command/{ci}/nodes/{ni}/objects/{qindex}/{attr}')
        coverage.append({'id':rid,'required_facts':[{'id':f'F{j+1:02d}','source_path':p,'value':pointer(source,p)} for j,p in enumerate(paths)]})
    outputs={'qualification_sources.json':encoded(checks),'qualification_cases.json':encoded(cases),'qualification_answers.json':encoded(gold),'requests.json':encoded(requests),'linkage.json':encoded(linkage),'required_facts.json':encoded(coverage)}
    manifest={'kind':'fixed_budget_integrated_task_development','date':'2026-10-10','qualification_cases':12,'qualification_assertions':48,'qualification_requires_both_auditors_all_correct':True,'max_report_attempts':18,'new_Controller_trials':0,'new_training':False,'human_required':False,'source_sha256':{str(p):digest(p) for p in [SOURCE,Path(__file__),Path('docs/INTEGRATED_REPORT_PILOT_V1.md')]},'artifact_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in outputs.items()},'confirmation':False,'model_checkpoints':1,'episodes':3}
    outputs['preparation.json']=encoded(manifest)
    return outputs

def pointer(obj,path):
    for part in path.split('/')[1:]:obj=obj[int(part)] if isinstance(obj,list) else obj[part]
    return obj

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify',action='store_true');a=ap.parse_args();data=build()
    if a.verify:
        for p,b in data.items():assert (ROOT/p).read_bytes()==b,p
    else:
        if ROOT.exists():raise SystemExit('refusing overwrite')
        ROOT.mkdir(parents=True)
        for p,b in data.items():(ROOT/p).write_bytes(b)
    print('Verified fixed qualification and 18 unchanged task requests' if a.verify else 'Prepared fixed qualification and 18 task requests; no target reports')
if __name__=='__main__':main()
