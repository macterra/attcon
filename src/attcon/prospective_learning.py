"""Matched fitted-control learning and reward/communication evaluation."""
import copy
import torch
from torch.nn import functional as F
from attcon.prospective import ProspectiveAgent, states_for, budget_for, action_values, rollout, analytic_policy, COSTS, CONDITIONS
from attcon.inspection import context_return_interval
from attcon.regulation import weight_fingerprint


def summary(result, data):
    answered = ~result['declined']
    return {'return':result['return'].mean().item(),'accuracy':result['correct'].mean().item(),
        'mean_inspections':result['inspections'].float().mean().item(),'answer_coverage':answered.float().mean().item(),
        'selective_accuracy':result['correct'][answered].mean().item() if answered.any() else None,
        'initial_sensor_a_rate':(result['initial_action']==7).float().mean().item(),
        'initial_sensor_b_rate':(result['initial_action']==8).float().mean().item(),
        'conditions':{name:{'return':result['return'][data.condition==i].mean().item(),
            'accuracy':result['correct'][data.condition==i].mean().item(),
            'mean_inspections':result['inspections'][data.condition==i].float().mean().item(),
            'decline_rate':result['declined'][data.condition==i].float().mean().item()}
            for i,name in enumerate(CONDITIONS) if (data.condition==i).any()}}


def train_fitted(data, validation, seed, architecture, family, epochs=80):
    torch.manual_seed(seed)
    agent=ProspectiveAgent(architecture,family)
    initial=weight_fingerprint(agent)
    target=copy.deepcopy(agent).eval().requires_grad_(False)
    optimizer=torch.optim.Adam(agent.parameters(),lr=.003)
    order_rng=torch.Generator().manual_seed(seed+10)
    delay_rng=torch.Generator().manual_seed(seed+11)
    losses,candidates,best,best_score,updates=[],[],None,None,0
    for epoch in range(1,epochs+1):
        total=0.
        for indices in torch.randperm(len(data),generator=order_rng).split(512):
            batch=data.subset(indices); delay=2*int(torch.randint(2,(),generator=delay_rng))
            with torch.no_grad():
                future=states_for(target,batch,delay)
                best_next=[action_values(target,future[node],batch,node)[1].max(-1).values for node in (1,2)]
            states=states_for(agent,batch,delay)
            choices=[]; predictions=[]
            for node,state in enumerate(states):
                logits,inspect=agent.values(state,batch,budget_for(batch.task,node))
                choices.append(F.cross_entropy(logits,batch.value))
                predictions.append(inspect)
            if batch.task=='serial':
                inspection=sum(F.mse_loss(predictions[node][:,0],-batch.cost+best_next[node]) for node in (0,1))/2
            else:
                inspection=F.mse_loss(predictions[0],torch.stack([-batch.cost+v for v in best_next],1))
            loss=sum(choices)/3+inspection
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(),5);optimizer.step()
            updates+=1;total+=loss.item()*len(batch)
        target.load_state_dict(agent.state_dict());losses.append(total/len(data))
        if epoch in (40,60,80) or epoch==epochs:
            metrics=summary(rollout(agent,validation),validation)
            candidates.append({'epoch':epoch,'validation':metrics})
            if best_score is None or metrics['return']>best_score:
                best,best_score=(copy.deepcopy(agent.state_dict()),epoch),metrics['return']
            print(f'{data.task} {architecture} {seed} {family} epoch {epoch}: {metrics["return"]:.4f}',flush=True)
    agent.load_state_dict(best[0]);agent.eval().requires_grad_(False)
    return agent,{'initial_sha256':initial,'selected_sha256':weight_fingerprint(agent),'selected_epoch':best[1],
        'updates':updates,'parameters':sum(p.numel() for p in agent.parameters()),'losses':losses,'candidates':candidates}


@torch.no_grad()
def calibration(agent,data,delay=1):
    root=states_for(agent,data,delay)[0]
    probabilities=agent.answer(root).softmax(-1)
    confidence,choice=probabilities.max(-1)
    correct=(choice==data.value).float()
    ece=0.
    for i in range(10):
        mask=(confidence>=i/10)&(confidence<(i+1)/10 if i<9 else confidence<=1)
        if mask.any():ece+=mask.float().mean().item()*abs(confidence[mask].mean().item()-correct[mask].mean().item())
    return {'brier_correctness':(confidence-correct).square().mean().item(),'ece_10_bins':ece,'accuracy':correct.mean().item(),'mean_confidence':confidence.mean().item()}


@torch.no_grad()
def evaluate(models,data,seed,delay=1,native=False):
    results={name:rollout(model,data,delay,native=native) for name,model in models.items()}
    fixed={name:rollout(models['state'],data,delay,forced=count,native=native) for name,count in (('never',0),('first',1),('second',2))}
    analytic=analytic_policy(data,native)
    if data.task=='serial':
        final_accuracy=fixed['second']['correct']
    else:
        final_accuracy=torch.where(data.quality>.7,fixed['first']['correct'],fixed['second']['correct'])
    costs={}
    for cost in COSTS:
        mask=data.cost==cost;subset=data.subset(mask)
        metrics={name:summary({key:value[mask] for key,value in result.items() if key!='visited'},subset) for name,result in {**results,**fixed,'analytic':analytic}.items()}
        intervals={name:context_return_interval((results['state']['return']-result['return'])[mask],data.group[mask],seed+13000) for name,result in {**results,**fixed}.items() if name!='state'}
        gates={'fresh_accuracy':metrics['state']['conditions']['fresh']['accuracy']>=.85,
            'forced_final_accuracy':final_accuracy[mask].mean().item()>=.85,
            'gain_over_fair_cue':intervals['cue']['mean']>=.02,'positive_fair_bound':intervals['cue']['low']>0}
        gates.update({'gain_over_'+name:intervals[name]['mean']>=.02 for name in fixed})
        costs[str(cost)]={'policies':metrics,'forced_final_accuracy':final_accuracy[mask].mean().item(),
            'analytic_expected_return':analytic['expected_return'][mask].mean().item(),
            'state_minus_comparator_intervals':intervals,'gates':gates,'all_gates_pass':all(gates.values())}
    return {'costs':costs,'calibration':{name:calibration(model,data,delay) for name,model in models.items()},
        'all_gates_pass':all(v['all_gates_pass'] for v in costs.values()),
        'boundary':'Fair cue head has exact retention of an observed quality cue. Blind confidence is diagnostic. Intervals are pointwise context bootstraps, conditional on trained systems.'}


def load_models(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    models={}
    for family,weights in saved['models'].items():
        model=ProspectiveAgent(saved['architecture'],family)
        model.load_state_dict(weights);model.eval().requires_grad_(False);models[family]=model
    return saved,models
