"""Post-hoc examiner probes of fixed models; no native Controller or reporter."""
import torch
from .functional_controls import advance_world, effect_table, last_forecast, control_mask, true_control_mask, counterfactual_recovery, physical_recovery

POLICIES=('random','agree','contradict')
WINDOWS=(1,2,4,8)


def probe_commands(replay,direction,old_channel,random_commands,offsets,policy):
    if policy not in POLICIES or old_channel not in (0,1):raise ValueError('invalid probe')
    phase=replay.clone();commands=[]
    for t in range(random_commands.shape[1]):
        phase=(phase+direction)%4
        automatic=phase[:,old_channel]
        if policy=='random':command=random_commands[:,t]
        elif policy=='agree':command=automatic
        else:command=(automatic+offsets[:,t])%4
        commands.append(command)
    return torch.stack(commands,1)


@torch.no_grad()
def probe(model,current,hidden,replay,direction,recovery,commands,new_control,windows=WINDOWS):
    if new_control not in (-1,0,1) or max(windows)>commands.shape[1]:raise ValueError('invalid continuation')
    controlled=torch.full((len(commands),),new_control,dtype=torch.long)
    phase,q=replay.clone(),recovery.clone();state=current.clone();h=hidden.clone()
    observations=[];records=[];snapshots={}
    for t in range(1,max(windows)+1):
        phase,q,x,_=advance_world(phase,direction,q,controlled,commands[:,t-1])
        seq,h=model(x[:,None],h);state=last_forecast(seq);observations.append(x)
        if t not in windows:continue
        target_effects=effect_table((phase+direction)%4,controlled)
        target_q=physical_recovery(q,target_effects)
        predicted=counterfactual_recovery(model,state,h)
        mask=control_mask(state);correct=(mask==true_control_mask(controlled)).all(-1)
        records.append({'additional_observations':t,
            'control_accuracy':float(correct.double().mean()),
            'effect_accuracy':float((state.effects.argmax(-1)==target_effects.argmax(-1)).double().mean()),
            'prospective_recovery_mae':float((predicted-target_q).abs().double().mean()),
            'incorrect_control_episodes':int((~correct).sum())})
        snapshots[t]={'modeled_allocation':state.allocation.clone(),'modeled_access':state.access.clone(),
            'modeled_effects':state.effects.clone(),'hidden':h.clone(),
            'control_mask':mask.clone(),'control_correct':correct.clone(),
            'predicted_recovery':predicted.clone(),'target_effects':target_effects.clone(),
            'target_recovery':target_q.clone(),'physical_current_recovery':q.clone(),
            'physical_replay':phase.clone()}
    return records,{'commands':commands.clone(),'observations':torch.stack(observations,1),
        'initial_allocation':current.allocation.clone(),'initial_access':current.access.clone(),
        'initial_effects':current.effects.clone(),'initial_hidden':hidden.clone(),
        'initial_physical_recovery':recovery.clone(),'initial_replay':replay.clone(),
        'direction':direction.clone(),'windows':snapshots}
