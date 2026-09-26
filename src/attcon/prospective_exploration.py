"""Epsilon-greedy learning from experienced rewards, with a native decline action."""
import copy
import torch
from torch.nn import functional as F
from attcon.prospective import ProspectiveAgent, states_for, action_values, rollout
from attcon.prospective_learning import summary
from attcon.regulation import weight_fingerprint


def environmental_rewards(actions, data):
    """The simulator alone consults hidden answers; the learner receives rewards."""
    correct = (actions == data.value).float()
    return torch.where(actions >= 7, -data.cost, torch.where(actions == 6, torch.full_like(data.cost, .30), correct))


def experienced_loss(logits, values, actions, rewards, next_values, active):
    """No hidden target values, access labels, or unchosen-action rewards accepted."""
    answer_mask = active & (actions < 6)
    inspection_mask = active & (actions >= 7)
    loss = logits.sum() * 0
    count = int(answer_mask.sum() + inspection_mask.sum())
    if answer_mask.any():
        probability = logits.softmax(-1)[answer_mask].gather(1, actions[answer_mask, None])[:, 0]
        loss = loss + F.binary_cross_entropy(probability.clamp(1e-6, 1-1e-6), rewards[answer_mask], reduction='sum')
    if inspection_mask.any():
        predicted = values[inspection_mask].gather(1, actions[inspection_mask, None])[:, 0]
        loss = loss + F.mse_loss(predicted, rewards[inspection_mask] + next_values[inspection_mask], reduction='sum')
    return loss, count


def exploratory_actions(values, epsilon, rng):
    allowed = values.detach() > -1e8
    random = torch.multinomial(allowed.float(), 1, generator=rng)[:, 0]
    explore = torch.rand(len(values), generator=rng) < epsilon
    return torch.where(explore, random, values.detach().argmax(-1))


def train_exploration(data, validation, seed, family, updates=600):
    torch.manual_seed(seed)
    agent = ProspectiveAgent('gru', family)
    initial = weight_fingerprint(agent)
    target = copy.deepcopy(agent).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(agent.parameters(), lr=.003)
    batch_rng = torch.Generator().manual_seed(seed + 14000)
    action_rng = torch.Generator().manual_seed(seed + 14001)
    delay_rng = torch.Generator().manual_seed(seed + 14002)
    best, best_score, candidates, losses = None, None, [], []
    experience = {'answers':0, 'declines':0, 'inspections':0, 'correct_answers':0}
    for update in range(1, updates + 1):
        indices = torch.randint(len(data), (512,), generator=batch_rng)
        batch = data.subset(indices)
        delay = 2 * int(torch.randint(2, (), generator=delay_rng))
        states = states_for(agent, batch, delay)
        with torch.no_grad():
            target_states = states_for(target, batch, delay)
            continuation = [action_values(target, state, batch, node, native=True)[1].max(-1).values for node, state in enumerate(target_states)]
        active = [torch.ones(len(batch),dtype=torch.bool), torch.zeros(len(batch),dtype=torch.bool), torch.zeros(len(batch),dtype=torch.bool)]
        epsilon = .4 - .3 * (update - 1) / max(updates - 1, 1)
        total_loss, total_count = states[0].sum() * 0, 0
        for node, state in enumerate(states):
            logits, values = action_values(agent, state, batch, node, native=True)
            actions = exploratory_actions(values, epsilon, action_rng)
            # Only rewards/continuations for active, selected actions enter the loss.
            rewards = environmental_rewards(actions, batch)
            next_value = torch.zeros_like(batch.cost)
            if node == 0:
                if batch.task == 'serial':
                    active[1] = active[0] & (actions == 7)
                    next_value = continuation[1]
                else:
                    active[1] = active[0] & (actions == 7)
                    active[2] = active[0] & (actions == 8)
                    next_value = torch.where(actions == 7, continuation[1], continuation[2])
            elif node == 1 and batch.task == 'serial':
                active[2] = active[1] & (actions == 7)
                next_value = continuation[2]
            loss, count = experienced_loss(logits, values, actions, rewards, next_value, active[node])
            total_loss, total_count = total_loss + loss, total_count + count
            experience['answers'] += int((active[node] & (actions < 6)).sum())
            experience['declines'] += int((active[node] & (actions == 6)).sum())
            experience['inspections'] += int((active[node] & (actions >= 7)).sum())
            experience['correct_answers'] += int((active[node] & (actions < 6) & (rewards == 1)).sum())
        loss = total_loss / max(total_count, 1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 5)
        optimizer.step()
        losses.append(loss.item())
        if update % 10 == 0:
            target.load_state_dict(agent.state_dict())
        if update in (200,400,600) or update == updates:
            metrics = summary(rollout(agent, validation, native=True), validation)
            candidates.append({'update':update,'validation':metrics})
            if best_score is None or metrics['return'] > best_score:
                best, best_score = (copy.deepcopy(agent.state_dict()), update), metrics['return']
            print(f'exploration {data.task} {seed} {family} {update}: {metrics["return"]:.4f}',flush=True)
    agent.load_state_dict(best[0]); agent.eval().requires_grad_(False)
    return agent, {'initial_sha256':initial,'selected_sha256':weight_fingerprint(agent),
        'selected_update':best[1],'updates':updates,'episodes_sampled':updates*512,
        'parameters':sum(p.numel() for p in agent.parameters()),'experience':experience,
        'losses':losses,'candidates':candidates}
