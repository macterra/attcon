#!/usr/bin/env python3
"""Train visual content recognition; retain all development/confirmation records."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from attcon.bound_content import VisualEncoder, scenes, bind
from attcon.predictive_attention import Forecast, PredictiveAttention, simulate


def main():
    p = argparse.ArgumentParser(); p.add_argument('--seed', type=int, default=1101); p.add_argument('--attention-seed', type=int, default=901)
    p.add_argument('--updates', type=int, default=600); p.add_argument('--output', default='audits/bound_content/pilot_v1'); args=p.parse_args()
    root=Path(args.output); root.mkdir(parents=True, exist_ok=True); result_path=root/f'seed{args.seed}.json'
    if result_path.exists():raise SystemExit('refusing to overwrite retained results')
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    encoder=VisualEncoder(); optimizer=torch.optim.Adam(encoder.parameters(),lr=.002)
    history=[]
    for update in range(args.updates):
        patches,colors,shapes=scenes(args.seed*100000+update,16)
        prediction=encoder(patches)
        loss=F.nll_loss(prediction[...,:4].clamp_min(1e-8).log().flatten(0,2),colors.flatten())+F.nll_loss(prediction[...,4:].clamp_min(1e-8).log().flatten(0,2),shapes.flatten())
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (update+1)%200==0:
            entry={'update':update+1,'loss':float(loss.detach())};history.append(entry);print(entry,flush=True)
    encoder.eval()
    with torch.no_grad():
        patches,colors,shapes=scenes(args.seed+100000000,512)
        v=encoder(patches)
        metrics={'color_accuracy':(v[...,:4].argmax(-1)==colors).double().mean().item(),
                 'shape_accuracy':(v[...,4:].argmax(-1)==shapes).double().mean().item(),
                 'joint_accuracy':((v[...,:4].argmax(-1)==colors)&(v[...,4:].argmax(-1)==shapes)).double().mean().item()}
        checkpoint=Path(f'audits/predictive_attention/confirmation_v1/seed{args.attention_seed}.pt')
        attention=PredictiveAttention();attention.load_state_dict(torch.load(checkpoint,weights_only=True)['state_dict']);attention.eval()
        e=simulate(args.seed+200000000,512,12);f,_=attention(e.observations)
        state=bind(encoder,patches,Forecast(f.allocation[:,7],f.access[:,7],f.effects[:,7]),e.allocation[:,:8])
        rows=torch.arange(512);view=e.controlled;slot=rows%4
        eligible=state.known[rows,view,slot]
        color,shape=colors[rows,view,slot],shapes[rows,view,slot]
        command=state.content_policy(view,color,shape)
        altered=state.replace_binding(state.binding.roll(1,-2));changed=altered.content_policy(view,color,shape)
        restored=altered.replace_binding(state.binding)
        metrics.update({'query_count':512,'observed_query_count':int(eligible.sum()),
            'observed_content_query_hit':(command[eligible]==slot[eligible]).double().mean().item(),
            'binding_rotation_following':(changed[eligible]==(slot[eligible]+1)%4).double().mean().item(),
            'binding_command_change':(changed[eligible]!=command[eligible]).double().mean().item(),
            'restoration_exact':torch.equal(restored.content_policy(view,color,shape),command),
            'unseen_entries_uniform':bool(torch.all(state.visual[~state.remembered]==.25)),
            'binding_preserves_V_and_A':torch.equal(altered.visual,state.visual) and torch.equal(altered.attention.effects,state.attention.effects)})
    path=root/f'seed{args.seed}.pt';torch.save({'state_dict':encoder.state_dict(),'seed':args.seed,'updates':args.updates},path)
    sources=['src/attcon/bound_content.py','scripts/train_bound_content.py']
    result={'stage':str(root),'seed':args.seed,'attention_seed':args.attention_seed,'updates':args.updates,'history':history,'metrics':metrics,
            'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'attention_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'source_sha256':{s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources}}
    result_path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':main()
