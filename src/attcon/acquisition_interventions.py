"""State perturbations with intact environmental rewards and future observations."""
import torch
from attcon.active_inspection import initial_events, rollout
from attcon.acquisition_learning import summarize_rollout
from attcon.regulation_interventions import projection_transplant


def condition_pairs(data):
    maps = [{}, {}]
    for i, (group, value, condition, cost) in enumerate(zip(data.group.tolist(), data.value.tolist(), data.condition.tolist(), data.cost.tolist())):
        if condition < 2:
            key = (group, value, cost)
            if key in maps[condition]:
                raise ValueError('duplicate paired condition')
            maps[condition][key] = i
    if not maps[0] or maps[0].keys() != maps[1].keys():
        raise ValueError('unpaired fresh/stale conditions')
    keys = sorted(maps[0])
    return tuple(torch.tensor([mapping[key] for key in keys]) for mapping in maps)


def fitted_directions(model, fit, seed):
    with torch.no_grad():
        states = model.advance(initial_events(fit))
        fresh, stale = condition_pairs(fit)
        difference = (states[fresh].double() - states[stale].double()).mean(0)
    weight = model.answer.weight.detach().double()
    _, singular, vh = torch.linalg.svd(weight, full_matrices=True)
    rank = int((singular > max(weight.shape) * torch.finfo(weight.dtype).eps * singular.max()).sum())
    basis = vh[:rank].T
    def project(value):
        return value - basis @ (basis.T @ value)
    access = project(difference)
    random = project(torch.randn(weight.shape[1], dtype=torch.float64, generator=torch.Generator().manual_seed(seed)))
    norms = {'availability': access.norm().item(), 'random': random.norm().item()}
    if min(norms.values()) < 1e-8:
        raise ValueError('degenerate null-space direction')
    return {'availability': (access / access.norm()).float(), 'random': (random / random.norm()).float()}, {'answer_rank': rank, 'direction_norms': norms}


@torch.no_grad()
def intervene(model, fit, data, seed):
    states = model.advance(initial_events(data))
    intact = rollout(model, data, initial_state=states)
    reset = rollout(model, data, initial_state=torch.zeros_like(states))
    restored = rollout(model, data, initial_state=states.clone())
    restoration_valid = all(torch.equal(intact[name], restored[name]) for name in ('return', 'answer', 'inspections', 'visited'))
    # Cost slices are necessary because a pooled response can hide policy reversals.
    def summary(result, batch):
        value = summarize_rollout(result, batch)
        value['conditions'] = {name: item for i, (name, item) in enumerate(value['conditions'].items()) if (batch.condition == i).any()}
        return value
    def metrics(result, batch):
        return {'all': summary(result, batch), 'costs': {str(cost): summary({key: value[batch.cost == cost] for key, value in result.items() if key != 'visited'}, batch.subset(batch.cost == cost)) for cost in (0.1, 0.25, 0.4)}}
    directions, metadata = fitted_directions(model, fit, seed)
    fresh, stale = condition_pairs(data)
    recipients, donors = states[fresh], states[stale]
    batch = data.subset(fresh)
    baseline = rollout(model, batch, initial_state=recipients)
    before_logits, before_inspect = model.values(recipients, batch.cost, 2)
    before_buy = before_inspect > before_logits.softmax(-1).max(-1).values
    reference = projection_transplant(recipients, donors, directions['availability'])
    changes = {}
    for kind, direction in directions.items():
        delta = reference if kind == 'availability' else projection_transplant(recipients, donors, direction, reference)
        changed = recipients + delta
        logits, inspect = model.values(changed, batch.cost, 2)
        after_buy = inspect > logits.softmax(-1).max(-1).values
        logit_residual = (logits - before_logits).abs().max().item()
        result = rollout(model, batch, initial_state=changed)
        recovered = rollout(model, batch, initial_state=changed - delta)
        recovered_valid = all(torch.equal(baseline[name], recovered[name]) for name in ('return', 'answer', 'inspections', 'visited'))
        changes[kind] = {'valid': logit_residual <= 1e-5 and recovered_valid, 'max_initial_logit_residual': logit_residual,
            'mean_delta_norm': delta.norm(dim=1).mean().item(), 'restoration_valid': recovered_valid,
            'answer_to_inspect': ((~before_buy) & after_buy).float().mean().item(),
            'inspect_to_answer': (before_buy & (~after_buy)).float().mean().item(),
            'initial_switch_rate': (before_buy != after_buy).float().mean().item(),
            'return_change': (result['return'] - baseline['return']).mean().item(), 'results': metrics(result, batch)}
    return {'intact': metrics(intact, data), 'reset': metrics(reset, data), 'reset_return_change': (reset['return'] - intact['return']).mean().item(),
        'restoration_valid': restoration_valid, 'choice_null_fit': metadata, 'fresh_baseline': metrics(baseline, batch),
        'transplants': changes, 'all_contrasts_valid': restoration_valid and all(value['valid'] for value in changes.values())}
