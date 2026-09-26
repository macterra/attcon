"""Post-training prospective reports and answer-preserving causal diagnostics."""
import torch
from torch import nn
from torch.nn import functional as F
from attcon.prospective import root_events, rollout, action_values, budget_for
from attcon.regulation import Readout
from attcon.history_reporting import pad_features
from attcon.regulation_interventions import projection_transplant


def labels_for(data):
    quality = (data.quality > .7).long()
    content = torch.where(data.condition == 0, data.value, torch.full_like(data.value,6))
    return 7 * quality + content


@torch.no_grad()
def report_features(agent, state, data):
    logits = agent.answer(state)
    probability = logits.softmax(-1)
    entropy = -(probability * logits.log_softmax(-1)).sum(-1) / torch.log(torch.tensor(6.))
    return {'state':pad_features(state,128),'action':pad_features(logits,128),
        'cue':pad_features(torch.stack((probability.max(-1).values,entropy,data.quality),1),128)}


def scores(predicted, labels):
    quality = (predicted // 7 == labels // 7).float().mean().item()
    verified = labels % 7 != 6
    known = (predicted[verified] % 7 == labels[verified] % 7).float().mean().item() if verified.any() else None
    unknown = (predicted[~verified] % 7 == 6).float().mean().item() if (~verified).any() else None
    balanced = (known + unknown) / 2 if known is not None and unknown is not None else None
    return {'quality_accuracy':quality,'verified_value_accuracy':known,'unverified_accuracy':unknown,
        'balanced_source_accuracy':balanced,'mean_quality_source_accuracy':(quality+balanced)/2 if balanced is not None else None,
        'joint_accuracy':(predicted==labels).float().mean().item()}


def fit_report(features, labels, validation_features, validation_labels, seed, steps=200):
    best,best_score,candidates=None,None,[]
    for l2 in (0.,.001):
        torch.manual_seed(seed)
        model=Readout(features,classes=14,hidden=32)
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        for _ in range(steps):
            penalty=sum(layer.weight.square().sum() for layer in model.network if isinstance(layer,nn.Linear))
            loss=F.cross_entropy(model(features),labels)+.5*l2*penalty
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        model.eval().requires_grad_(False)
        with torch.no_grad():metrics=scores(model(validation_features).argmax(-1),validation_labels)
        candidate={'l2':l2,'validation':metrics};candidates.append(candidate)
        score=(min(metrics['quality_accuracy'],metrics['balanced_source_accuracy']),metrics['mean_quality_source_accuracy'])
        if best_score is None or score>best_score:best,best_score=(model,candidate),score
    return best[0],{'selected':best[1],'candidates':candidates}


def quality_pairs(data):
    maps=[{},{}]
    for index,(group,value,condition,cost,quality) in enumerate(zip(data.group.tolist(),data.value.tolist(),data.condition.tolist(),data.cost.tolist(),data.quality.tolist())):
        key=(group,value,condition,cost);target=maps[int(quality>.7)]
        if key in target:raise ValueError('duplicate quality counterpart')
        target[key]=index
    if not maps[0] or maps[0].keys()!=maps[1].keys():raise ValueError('unpaired quality histories')
    keys=sorted(maps[0])
    return tuple(torch.tensor([mapping[key] for key in keys]) for mapping in maps)


def quality_directions(agent, fit_states, fit_data, seed):
    low,high=quality_pairs(fit_data)
    difference=(fit_states[high].double()-fit_states[low].double()).mean(0)
    weight=agent.answer.weight.detach().double()
    _,singular,vh=torch.linalg.svd(weight,full_matrices=True)
    rank=int((singular>max(weight.shape)*torch.finfo(weight.dtype).eps*singular.max()).sum())
    basis=vh[:rank].T
    def project(value):return value-basis@(basis.T@value)
    quality=project(difference)
    random=project(torch.randn(weight.shape[1],dtype=torch.float64,generator=torch.Generator().manual_seed(seed)))
    norms={'quality':quality.norm().item(),'random':random.norm().item()}
    if min(norms.values())<1e-8:raise ValueError('degenerate quality-null direction')
    return {'quality':(quality/quality.norm()).float(),'random':(random/random.norm()).float()},{'answer_rank':rank,'norms':norms}


@torch.no_grad()
def causal_audit(agent, models, fit_states, fit_data, states, data, seed):
    try:directions,metadata=quality_directions(agent,fit_states,fit_data,seed)
    except ValueError as error:return {'valid':False,'reason':str(error)}
    low,high=quality_pairs(data)
    mask=data.condition[high]!=0;low,high=low[mask],high[mask]
    recipients,donors=states[high],states[low]
    batch,donor_batch=data.subset(high),data.subset(low)
    before=rollout(agent,batch,initial_state=recipients)
    donor_run=rollout(agent,donor_batch,initial_state=donors)
    features=report_features(agent,recipients,batch)
    baseline={name:model(features[name]).argmax(-1) for name,model in models.items()}
    reference=projection_transplant(recipients,donors,directions['quality'])
    results={}
    for kind,direction in directions.items():
        delta=reference if kind=='quality' else projection_transplant(recipients,donors,direction,reference)
        changed=recipients+delta
        residual=(agent.answer(changed)-agent.answer(recipients)).abs().max().item()
        after=rollout(agent,batch,initial_state=changed)
        modified=report_features(agent,changed,batch)
        predicted={name:model(modified[name]).argmax(-1) for name,model in models.items()}
        restored=rollout(agent,batch,initial_state=changed-delta)
        restored_features=report_features(agent,changed-delta,batch)
        restoration=all(torch.equal(before[name],restored[name]) for name in ('answer','initial_action','inspections','return','visited')) and all(torch.equal(baseline[name],model(restored_features[name]).argmax(-1)) for name,model in models.items())
        invariant={name:torch.equal(baseline[name],predicted[name]) for name in ('action','cue')}
        initial_changed=before['initial_action']!=after['initial_action']
        trajectory_changed=(before['inspections']!=after['inspections'])|initial_changed
        report_changed=baseline['state']//7!=predicted['state']//7
        donor_differs=(before['initial_action']!=donor_run['initial_action'])|(before['inspections']!=donor_run['inspections'])
        results[kind]={'valid':residual<=1e-5 and restoration and all(invariant.values()),
            'max_logit_residual':residual,'restoration_valid':restoration,'comparator_report_invariance':invariant,
            'mean_delta_norm':delta.norm(dim=1).mean().item(),'quality_report_changed_rate':report_changed.float().mean().item(),
            'quality_report_donor_rate':(predicted['state']//7==0).float().mean().item(),
            'initial_policy_switch_rate':initial_changed.float().mean().item(),'inspection_trajectory_switch_rate':trajectory_changed.float().mean().item(),
            'joint_report_trajectory_change_rate':(report_changed&trajectory_changed).float().mean().item(),
            'donor_policy_difference_fraction':donor_differs.float().mean().item(),
            'return_change':(after['return']-before['return']).mean().item()}
    return {'valid':all(v['valid'] for v in results.values()),'fit':metadata,'recipient_count':len(batch),
        'baseline_quality_accuracy':(baseline['state']//7==1).float().mean().item(),
        'transplants':results,'boundary':'Synthetic quality-state transplants leave the actual sensor unchanged. Return effects quantify consequences, not correctness of an introspective report. Serial trajectories may change after the first observation even when initial inspect/answer choices do not.'}
