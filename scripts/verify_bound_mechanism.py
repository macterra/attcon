#!/usr/bin/env python3
"""Offline re-evaluation of all saved visual/binding checkpoints and frozen gates."""
import hashlib
import json
from pathlib import Path
import torch
from attcon.bound_content import VisualEncoder, scenes, bind
from attcon.predictive_attention import PredictiveAttention, Forecast, simulate

torch.set_num_threads(2);root=Path('audits/bound_content');results={}
for family,seeds in [('pilot_v1',(1101,)),('confirmation_v1',(1201,1211,1221)),('confirmation_v2',(1301,1311,1321))]:
    results[family]={}
    for seed in seeds:
        record=json.loads((root/family/f'seed{seed}.json').read_text());checkpoint=root/family/f'seed{seed}.pt'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==record['checkpoint_sha256']
        for source,digest in record['source_sha256'].items():assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==digest
        encoder=VisualEncoder();encoder.load_state_dict(torch.load(checkpoint,weights_only=True)['state_dict']);encoder.eval()
        attention_path=Path(f"audits/predictive_attention/confirmation_v1/seed{record['attention_seed']}.pt")
        assert hashlib.sha256(attention_path.read_bytes()).hexdigest()==record['attention_checkpoint_sha256']
        model=PredictiveAttention();model.load_state_dict(torch.load(attention_path,weights_only=True)['state_dict']);model.eval()
        with torch.no_grad():
            patches,colors,shapes=scenes(seed+100000000,512);v=encoder(patches)
            metrics={'color_accuracy':(v[...,:4].argmax(-1)==colors).double().mean().item(),
                     'shape_accuracy':(v[...,4:].argmax(-1)==shapes).double().mean().item(),
                     'joint_accuracy':((v[...,:4].argmax(-1)==colors)&(v[...,4:].argmax(-1)==shapes)).double().mean().item()}
            e=simulate(seed+200000000,512,12);pred,_=model(e.observations)
            f=Forecast(pred.allocation[:,7],pred.access[:,7],pred.effects[:,7]);state=bind(encoder,patches,f,e.allocation[:,:8])
            row=torch.arange(512);view=e.controlled;slot=row%4;eligible=state.known[row,view,slot]
            color,shape=colors[row,view,slot],shapes[row,view,slot]
            command=state.content_policy(view,color,shape);altered=state.replace_binding(state.binding.roll(1,-2));changed=altered.content_policy(view,color,shape)
            restored=altered.replace_binding(state.binding)
            metrics.update({'query_count':512,'observed_query_count':int(eligible.sum()),
                'observed_content_query_hit':(command[eligible]==slot[eligible]).double().mean().item(),
                'binding_rotation_following':(changed[eligible]==(slot[eligible]+1)%4).double().mean().item(),
                'binding_command_change':(changed[eligible]!=command[eligible]).double().mean().item(),
                'restoration_exact':torch.equal(restored.content_policy(view,color,shape),command),
                'unseen_entries_uniform':bool(torch.all(state.visual[~state.remembered]==.25)),
                'binding_preserves_V_and_A':torch.equal(altered.visual,state.visual) and torch.equal(altered.attention.effects,state.attention.effects)})
        assert metrics==record['metrics'],(family,seed)
        gates={'visual_joint':metrics['joint_accuracy']>=.995,'content_query':metrics['observed_content_query_hit']>=.99,
               'binding_following':metrics['binding_rotation_following']>=.99,'restoration':metrics['restoration_exact'],
               'unseen_uniform':metrics['unseen_entries_uniform'],'preservation':metrics['binding_preserves_V_and_A']}
        results[family][seed]={'gates':gates,'all_gates_pass':all(gates.values()),'exact_replay':True}
(root/'mechanism_verification.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
