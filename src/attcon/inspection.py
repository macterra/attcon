"""One-step inspection policies fitted only to environmental correctness rewards."""

import torch
from attcon.history_reporting import pad_features
from attcon.regulation import WIDTH, confidence_features, fit_readout

COSTS = (0.2, 0.4, 0.6)


def policy_features(agent, states):
    with torch.no_grad():
        logits = agent.choice(states)
        return {'state': pad_features(states, WIDTH), 'action': pad_features(logits, WIDTH), 'confidence': confidence_features(logits)}


def immediate_rewards(agent, states, values):
    with torch.no_grad():
        return (agent.choice(states).argmax(-1) == values).float()


def inspection_returns(probability, reward, cost):
    inspect = probability < 1 - cost
    return torch.where(inspect, torch.full_like(reward, 1 - cost), reward), inspect


def select_policy(features, reward, validation_features, validation_reward, *, seed, steps=300):
    """No access/report labels or test inputs enter training or selection."""
    best, best_score, candidates = None, None, []
    for l2 in (0.0, 0.001):
        model = fit_readout(features, reward, seed=seed, l2=l2, steps=steps, reward=True)
        with torch.no_grad():
            probability = model(validation_features)[:, 0].sigmoid()
        returns = {str(cost): inspection_returns(probability, validation_reward, cost)[0].mean().item() for cost in COSTS}
        score = sum(returns.values()) / len(COSTS)
        candidate = {'l2': l2, 'validation_returns': returns, 'selection_score': score}
        candidates.append(candidate)
        if best_score is None or score > best_score:
            best, best_score = (model, candidate), score
    return best[0], {'selected': best[1], 'candidates': candidates}


def context_return_interval(differences, groups, seed, draws=2000):
    """Resample complete contexts, preserving paired statuses and value variants."""
    unique = groups.unique(sorted=True)
    scores = torch.stack([differences[groups == group].mean() for group in unique])
    sample = torch.randint(len(scores), (draws, len(scores)), generator=torch.Generator().manual_seed(seed))
    means = scores[sample].mean(1)
    low, high = means.quantile(torch.tensor([0.025, 0.975])).tolist()
    return {'mean': scores.mean().item(), 'low': low, 'high': high, 'contexts': len(scores), 'draws': draws}


def evaluate_policies(models, features, reward, data, seed, null_models=()):
    with torch.no_grad():
        probabilities = {name: model(features[name])[:, 0].sigmoid() for name, model in models.items()}
        null_probabilities = [model(features['state'])[:, 0].sigmoid() for model in null_models]
    result = {'brier_scores': {name: (probability - reward).square().mean().item() for name, probability in probabilities.items()}, 'costs': {}}
    for cost in COSTS:
        returns, policies = {}, {}
        for name, probability in probabilities.items():
            returns[name], inspect = inspection_returns(probability, reward, cost)
            policies[name] = {'return': returns[name].mean().item(), 'inspection_rate': inspect.float().mean().item(),
                'seen_inspection_rate': inspect[data.seen].float().mean().item(),
                'unseen_inspection_rate': inspect[~data.seen].float().mean().item()}
        returns['never_inspect'] = reward
        returns['always_inspect'] = torch.full_like(reward, 1 - cost)
        null_returns = [inspection_returns(probability, reward, cost)[0].mean().item() for probability in null_probabilities]
        null_p95 = torch.tensor(null_returns).quantile(0.95).item() if null_returns else None
        intervals = {name: context_return_interval(returns['state'] - value, data.group, seed + 6100) for name, value in returns.items() if name != 'state'}
        gates = {f'gain_over_{name}': intervals[name]['mean'] >= 0.02 for name in ('action', 'confidence')}
        gates.update({f'positive_bound_over_{name}': intervals[name]['low'] > 0 for name in ('action', 'confidence')})
        gates.update({f'beats_{name}': intervals[name]['mean'] > 0 for name in ('never_inspect', 'always_inspect')})
        if null_p95 is not None:
            gates['gain_over_null_p95'] = policies['state']['return'] - null_p95 >= 0.05
        result['costs'][str(cost)] = {'policies': policies,
            'fixed_returns': {name: returns[name].mean().item() for name in ('never_inspect', 'always_inspect')},
            'hindsight_oracle_return': torch.maximum(reward, returns['always_inspect']).mean().item(),
            'state_minus_baseline_intervals': intervals, 'null_returns': null_returns, 'null_p95': null_p95,
            'gates': gates, 'all_gates_pass': all(gates.values())}
    return result


@torch.no_grad()
def policy_switches(models, agent, states, changed):
    before_features, after_features = policy_features(agent, states), policy_features(agent, changed)
    result = {}
    for name, model in models.items():
        before = model(before_features[name])[:, 0].sigmoid()
        after = model(after_features[name])[:, 0].sigmoid()
        result[name] = {'max_feature_residual': (after_features[name] - before_features[name]).abs().max().item(),
            'max_probability_residual': (after - before).abs().max().item(), 'costs': {}}
        for cost in COSTS:
            initial, final = before < 1 - cost, after < 1 - cost
            result[name]['costs'][str(cost)] = {
                'answer_to_inspect': ((~initial) & final).float().mean().item(),
                'inspect_to_answer': (initial & (~final)).float().mean().item(),
                'switch_rate': (initial != final).float().mean().item()}
    return result
