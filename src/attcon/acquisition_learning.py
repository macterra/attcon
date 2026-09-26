"""Full-information fitted control from environmental rewards and transitions."""
import copy

import torch
from torch.nn import functional as F

from attcon.active_inspection import AcquisitionAgent, COSTS, CONDITIONS, all_states, rollout, analytic_policy
from attcon.inspection import context_return_interval
from attcon.regulation import weight_fingerprint


@torch.no_grad()
def summarize_rollout(result, data):
    return {'return': result['return'].mean().item(), 'accuracy': result['correct'].mean().item(),
        'mean_inspections': result['inspections'].float().mean().item(),
        'conditions': {name: {'return': result['return'][data.condition == i].mean().item(),
            'accuracy': result['correct'][data.condition == i].mean().item(),
            'mean_inspections': result['inspections'][data.condition == i].float().mean().item()}
            for i, name in enumerate(CONDITIONS)}}


def train_controller(data, validation, family, seed, epochs=100):
    torch.manual_seed(seed)
    model = AcquisitionAgent(family)
    initial = weight_fingerprint(model)
    target = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    order_rng = torch.Generator().manual_seed(seed + 10)
    delay_rng = torch.Generator().manual_seed(seed + 11)
    candidates, losses, best, best_score, updates = [], [], None, None, 0
    for epoch in range(1, epochs + 1):
        total = 0.0
        for indices in torch.randperm(len(data), generator=order_rng).split(256):
            batch = data.subset(indices)
            delay = 2 * int(torch.randint(2, (), generator=delay_rng))
            with torch.no_grad():
                future = all_states(target, batch, delay)
                continuation = []
                for stage in (1, 2):
                    logits, inspection = target.values(future[stage], batch.cost, 2 - stage)
                    value = logits.softmax(-1).max(-1).values
                    continuation.append(value if stage == 2 else torch.maximum(value, inspection))
            states = all_states(model, batch, delay)
            choices, inspections = [], []
            for stage in (0, 1, 2):
                logits, inspect = model.values(states[stage], batch.cost, 2 - stage)
                choices.append(F.cross_entropy(logits, batch.value))
                if stage < 2:
                    inspections.append(F.mse_loss(inspect, -batch.cost + continuation[stage]))
            loss = sum(choices) / 3 + sum(inspections) / 2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += loss.item() * len(batch)
            updates += 1
        target.load_state_dict(model.state_dict())
        losses.append(total / len(data))
        if epoch in (60, 80, 100) or epoch == epochs:
            measured = summarize_rollout(rollout(model, validation), validation)
            candidates.append({'epoch': epoch, 'validation': measured})
            if best_score is None or measured['return'] > best_score:
                best = (copy.deepcopy(model.state_dict()), epoch)
                best_score = measured['return']
            print(f'{seed} {family} epoch {epoch}: validation return {measured["return"]:.4f}', flush=True)
    model.load_state_dict(best[0])
    model.eval().requires_grad_(False)
    return model, {'initial_sha256': initial, 'selected_sha256': weight_fingerprint(model),
        'updates': updates, 'epochs': epochs, 'selected_epoch': best[1], 'candidates': candidates, 'losses': losses,
        'parameters': sum(p.numel() for p in model.parameters())}


@torch.no_grad()
def evaluate_controllers(models, data, seed, delay=1, reliability=0.75):
    results = {name: rollout(model, data, delay) for name, model in models.items()}
    fixed = {name: rollout(models['state'], data, delay, forced=count) for name, count in (('never', 0), ('once', 1), ('twice', 2))}
    analytic = analytic_policy(data, reliability)
    costs = {}
    for cost in COSTS:
        mask = data.cost == cost
        subset = data.subset(mask)
        def select(result):
            return {key: value[mask] for key, value in result.items() if key != 'visited'}
        metrics = {name: summarize_rollout(select(result), subset) for name, result in {**results, **fixed, 'analytic': analytic}.items()}
        intervals = {name: context_return_interval((results['state']['return'] - result['return'])[mask], data.group[mask], seed + 8000) for name, result in {**results, **fixed}.items() if name != 'state'}
        gates = {f'gain_over_{name}': interval['mean'] >= .02 for name, interval in intervals.items()}
        gates.update({f'positive_bound_over_{name}': intervals[name]['low'] > 0 for name in ('action', 'confidence')})
        gates['fresh_accuracy'] = metrics['state']['conditions']['fresh']['accuracy'] >= .90
        gates['forced_verification_accuracy'] = metrics['twice']['accuracy'] >= .90
        costs[str(cost)] = {'policies': metrics, 'analytic_expected_return': analytic['expected_return'][mask].mean().item(),
            'state_minus_comparator_intervals': intervals, 'gates': gates, 'all_gates_pass': all(gates.values())}
    return {'costs': costs, 'all_gates_pass': all(value['all_gates_pass'] for value in costs.values()),
        'boundary': 'Closed-loop evaluation of controllers learned by full-information counterfactual replay. Pointwise context-bootstrap intervals are conditional on trained models. Analytic comparator knows sensor probabilities, not realized correctness.'}


def load_controllers(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    models = {}
    for family, weights in saved['models'].items():
        model = AcquisitionAgent(family)
        model.load_state_dict(weights)
        model.eval().requires_grad_(False)
        models[family] = model
    return saved, models
