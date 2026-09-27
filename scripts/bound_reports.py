#!/usr/bin/env python3
"""Generate free prose from the complete object-linked attention model."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import torch
from openai import AsyncOpenAI
from attcon.nl_report import load_dotenv
from attcon.bound_content import BoundState, VisualEncoder, bind, scenes, full_record
from attcon.predictive_attention import PredictiveAttention, Forecast, simulate

GLOSSARY = '''You report for a visual-attention system from its current internal scene model.
The record contains two views, A and B, with four spatial locations in each.
Color and shape distributions describe the system's remembered visual contents;
a flat distribution provides no identified color or shape. Selection probabilities
model current attention. Recoverability values predict successful access to that
object's information now and after one and two steps without resampling.
Next-selection-by-command gives the predicted selection of that location for each
possible redirection command. A view is under this system's control to the extent
that alternative commands change its predicted allocation. These are fallible
internal representations, not a statement of what is physically present.
Missing fields provide no evidence. Describe only what the supplied model supports.
Answer in ordinary prose, not JSON. No particular experiential vocabulary is required.'''


def make_conditions(base):
    attention=base.attention
    binding=base.replace_binding(base.binding.roll(1,-2))
    return {'model':base,'content':base.replace_visual(base.visual.roll(1,-2)),
        'binding':binding,
        'allocation':base.replace_attention(attention.intervene('allocation',attention.allocation.roll(1,-1))),
        'access':base.replace_attention(attention.intervene('access',attention.access.flip(-1))),
        'effects':base.replace_attention(attention.intervene('effects',attention.effects.flip(-2))),
        'restored':binding.replace_binding(base.binding),
        'visual_only':base,'attention_only':base,
        'shuffled':BoundState(Forecast(*(x.roll(-1,0) for x in (attention.allocation,attention.access,attention.effects))),
                             base.visual.roll(-1,0),base.binding.roll(-1,0),base.remembered.roll(-1,0))}


def prepare(config,path,root):
    root.mkdir(parents=True,exist_ok=True)
    request_path=root/'requests.json'
    if request_path.exists():return json.loads(request_path.read_text())
    torch.set_num_threads(2);records=[];tensors={};checkpoints={}
    for pair in config['pairs']:
        visual_path=Path(f"audits/bound_content/{pair['visual_folder']}/seed{pair['visual_seed']}.pt")
        attention_path=Path(f"audits/predictive_attention/confirmation_v1/seed{pair['attention_seed']}.pt")
        encoder=VisualEncoder();encoder.load_state_dict(torch.load(visual_path,weights_only=True)['state_dict']);encoder.eval()
        model=PredictiveAttention();model.load_state_dict(torch.load(attention_path,weights_only=True)['state_dict']);model.eval()
        for checkpoint in (visual_path,attention_path):checkpoints[str(checkpoint)]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        e=simulate(pair['process_seed'],pair['count'],12);patches,colors,shapes=scenes(pair['scene_seed'],pair['count'])
        with torch.no_grad():pred,_=model(e.observations)
        step=config['step'];f=Forecast(pred.allocation[:,step],pred.access[:,step],pred.effects[:,step])
        base=bind(encoder,patches,f,e.allocation[:,:step+1])
        conditions=make_conditions(base)
        tensors[pair['visual_seed']]={'patches':patches,'colors':colors,'shapes':shapes,'observations':e.observations,
            'allocations':e.allocation,'visual':base.visual,'binding':base.binding,'remembered':base.remembered,
            'allocation':f.allocation,'access':f.access,'effects':f.effects}
        for i in range(pair['count']):
            for condition,state in conditions.items():
                record=full_record(state,i)
                if condition in ('visual_only','attention_only'):
                    for view in record:
                        for obj in view['objects']:
                            keys=('selection_probability','recoverability_now_then_one_then_two_steps','next_selection_by_command') if condition=='visual_only' else ('color_distribution','shape_distribution')
                            for key in keys:obj[key]=None
                if config.get('command_major',False):
                    for view in record:
                        entries=view['objects']
                        view['command_predictions']={command:{obj['location']:obj['next_selection_by_command'][command] for obj in entries}
                            for command in ('upper','right','lower','left')} if entries[0]['next_selection_by_command'] is not None else None
                        for obj in entries:del obj['next_selection_by_command']
                glossary=GLOSSARY
                if config.get('command_major',False):
                    glossary=glossary.replace('Next-selection-by-command gives the predicted selection of that location for each\npossible redirection command.', 'Command_predictions is grouped by command: each command maps every location to its\npredicted next-selection probability. Compare these rows to assess controllability.')
                if config.get('derived_indexes',False):
                    for view in record:
                        entries=view['objects']
                        def winner(values):
                            if any(v is None for v in values.values()):return None
                            m=max(values.values());best=[k for k,v in values.items() if v==m]
                            return best[0] if len(best)==1 else None
                        view['derived_indexes']={
                            'most_selected_location':winner({o['location']:o['selection_probability'] for o in entries}),
                            'most_recoverable_location_now':winner({o['location']:None if o['recoverability_now_then_one_then_two_steps'] is None else o['recoverability_now_then_one_then_two_steps'][0] for o in entries}),
                            'next_selected_location_by_command':None if view['command_predictions'] is None else {c:winner(row) for c,row in view['command_predictions'].items()}}
                    glossary+='\nDerived indexes are exact argmax reductions of the full distributions. Current selection, identity certainty, and recoverability are distinct; next selection alone does not specify future recoverability.'
                prompt=glossary+f"\nUse at most {config['word_limit']} words.\n"+config.get('prose_instruction','')+'\n'+config['question']+'\n'+json.dumps(record)
                records.append({'id':f"{pair['visual_seed']}_{i}_{condition}",'seed':pair['visual_seed'],'episode':i,'condition':condition,
                                'source':record,'input':prompt})
    assert len(records)==config['max_attempts']
    request_path.write_text(json.dumps(records,indent=2)+'\n');torch.save(tensors,root/'source_states.pt')
    source_paths=['scripts/bound_reports.py','src/attcon/bound_content.py',str(path)]
    # Keep exact sources as well as hashes, so later versions cannot erase provenance.
    sources={s:Path(s).read_text() for s in source_paths}
    (root/'source_code.json').write_text(json.dumps(sources,indent=2)+'\n')
    (root/'manifest.json').write_text(json.dumps({'config':config,'checkpoint_sha256':checkpoints,
        'source_sha256':{s:hashlib.sha256(v.encode()).hexdigest() for s,v in sources.items()},
        'requests_sha256':hashlib.sha256(request_path.read_bytes()).hexdigest(),
        'states_sha256':hashlib.sha256((root/'source_states.pt').read_bytes()).hexdigest()},indent=2)+'\n')
    return records


async def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/bound_content/language_pilot_v1.json');args=p.parse_args()
    path=Path(args.config);config=json.loads(path.read_text());root=Path('audits/bound_content')/config['name']
    requests=prepare(config,path,root);load_dotenv();client=AsyncOpenAI(max_retries=0,timeout=120.);semaphore=asyncio.Semaphore(8)
    async def run(req):
        path=root/(req['id']+'.json')
        if path.exists():return
        async with semaphore:
            path.write_text(json.dumps({'id':req['id'],'status':'attempt_reserved'})+'\n')
            try:
                response=await client.responses.create(model=config['model'],input=req['input'],max_output_tokens=config['max_output_tokens'],
                    reasoning={'effort':config.get('reasoning','low')},text={'verbosity':'low'})
                result={'id':req['id'],'status':'received','response':response.model_dump(mode='json'),'report':response.output_text}
            except Exception as exc:
                result={'id':req['id'],'status':'error','error_type':type(exc).__name__,'http_status':getattr(exc,'status_code',None)}
            path.write_text(json.dumps(result,indent=2)+'\n');print(req['id'],result['status'],flush=True)
    await asyncio.gather(*(run(r) for r in requests));await client.close()


if __name__=='__main__':asyncio.run(main())
