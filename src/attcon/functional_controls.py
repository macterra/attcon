"""Three-way functional access and matched open-loop/feedback diagnostics."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .predictive_attention import DECAY, Forecast

CONDITIONS = (0, 1, -1)
WINDOWS = (1, 2, 4, 8, 16)


@dataclass
class ControlEpisode:
    observations: torch.Tensor
    allocation: torch.Tensor
    access: torch.Tensor
    next_allocation: torch.Tensor
    controlled: torch.Tensor
    commands: torch.Tensor
    replay: torch.Tensor
    direction: torch.Tensor


def effect_table(next_replay, controlled):
    count = len(controlled)
    effects = F.one_hot(next_replay[:, None].expand(-1, 4, -1), 4).float().clone()
    rows = torch.arange(count)[controlled >= 0]
    owners = controlled[controlled >= 0]
    for command in range(4):
        effects[rows, command, owners] = F.one_hot(torch.full((len(rows),), command), 4).float()
    return effects


def observation(allocation, command, quality):
    return torch.cat((allocation.flatten(1), (quality * allocation).flatten(1),
                      F.one_hot(command, 4).float()), -1)


def advance_world(replay, direction, recovery, controlled, command, quality=.8):
    """Pure physical transition: no modeled forecasts or hidden state are consulted."""
    next_replay = (replay + direction) % 4
    slots = next_replay.clone()
    rows = torch.arange(len(command))[controlled >= 0]
    slots[rows, controlled[controlled >= 0]] = command[rows]
    allocation = F.one_hot(slots, 4).float()
    next_recovery = recovery * DECAY * (1-allocation) + quality * allocation
    return next_replay, next_recovery, observation(allocation, command, quality), allocation


def simulate_controls(seed, count=192, steps=16, condition=None):
    if condition not in (None, *CONDITIONS):
        raise ValueError('unknown control condition')
    if count <= 0 or steps <= 0 or (condition is None and count % 3):
        raise ValueError('mixed batches require positive count divisible by three')
    g = torch.Generator().manual_seed(seed)
    if condition is None:
        controlled = torch.tensor(CONDITIONS).repeat(count // 3)
        controlled = controlled[torch.randperm(count, generator=g)]
    else:
        controlled = torch.full((count,), condition, dtype=torch.long)
    replay = torch.randint(4, (count, 2), generator=g)
    direction = torch.randint(2, (count, 2), generator=g) * 2 - 1
    commands = torch.randint(4, (count, steps), generator=g)
    qualities = .1 + .9 * torch.rand(count, steps, 2, 1, generator=g)
    recovery = torch.zeros(count, 2, 4)
    xs, allocations, accesses, futures = [], [], [], []
    for t in range(steps):
        replay, recovery, x, a = advance_world(replay, direction, recovery, controlled,
                                               commands[:, t], qualities[:, t])
        xs.append(x); allocations.append(a)
        accesses.append(torch.stack((recovery, recovery*DECAY, recovery*DECAY**2), 1))
        futures.append(effect_table((replay+direction) % 4, controlled))
    return ControlEpisode(torch.stack(xs, 1), torch.stack(allocations, 1),
        torch.stack(accesses, 1), torch.stack(futures, 1), controlled, commands, replay, direction)


def last_forecast(sequence):
    return Forecast(sequence.allocation[:, -1], sequence.access[:, -1], sequence.effects[:, -1])


def control_mask(forecast):
    # An ideal command-controlled one-hot table has TV spread 1.5, an independent one 0.
    return forecast.controllability() > .75


def true_control_mask(controlled):
    return torch.stack((controlled == 0, controlled == 1), -1)


@torch.no_grad()
def advance_model(model, current, hidden, command, quality=.8):
    rows = torch.arange(len(command))
    predicted_allocation = current.effects[rows, command]
    x = observation(predicted_allocation, command, quality)
    seq, next_hidden = model(x[:, None], hidden.clone())
    return last_forecast(seq), next_hidden, x


@torch.no_grad()
def counterfactual_recovery(model, current, hidden, quality=.8):
    predictions = []
    for command in range(4):
        commands = torch.full((len(current.allocation),), command, dtype=torch.long)
        updated, _, _ = advance_model(model, current, hidden, commands, quality)
        predictions.append(updated.access[:, 0])
    return torch.stack(predictions, 1)


def physical_recovery(recovery, effects, quality=.8):
    return recovery[:, None] * DECAY * (1-effects) + quality*effects


def assess_static(model, episode):
    with torch.no_grad():
        sequence, hidden = model(episode.observations)
        current = last_forecast(sequence)
        predicted = counterfactual_recovery(model, current, hidden)
    target = physical_recovery(episode.access[:, -1, 0], episode.next_allocation[:, -1])
    tail = Forecast(sequence.allocation[:, 7:], sequence.access[:, 7:], sequence.effects[:, 7:])
    truth = true_control_mask(episode.controlled)[:, None]
    metrics = {
        'allocation_accuracy': float((tail.allocation.argmax(-1)==episode.allocation[:, 7:].argmax(-1)).double().mean()),
        'effect_accuracy': float((tail.effects.argmax(-1)==episode.next_allocation[:, 7:].argmax(-1)).double().mean()),
        'three_way_control_accuracy': float((control_mask(tail)==truth).all(-1).double().mean()),
        'recovery_mae': float((tail.access-episode.access[:, 7:]).abs().double().mean()),
        'counterfactual_recovery_mae': float((predicted-target).abs().double().mean()),
    }
    gates = {k: v >= .99 if k.endswith('accuracy') else v <= .04 for k,v in metrics.items()}
    if bool((episode.controlled == -1).all()):
        metrics['decoupled_effect_spread'] = float(tail.controllability().double().mean())
        metrics['decoupled_access_spread'] = float((predicted-predicted.mean(1, keepdim=True)).abs().double().mean())
        gates['decoupled_effect_spread'] = metrics['decoupled_effect_spread'] <= .10
        gates['decoupled_access_spread'] = metrics['decoupled_access_spread'] <= .02
    altered = current.intervene('effects', current.effects.flip(-2))
    hidden_before = hidden.clone()
    changed = counterfactual_recovery(model, altered, hidden)
    restored = counterfactual_recovery(model, altered.intervene('effects', current.effects), hidden)
    intervention = {
        'allocation_preserved': torch.equal(altered.allocation, current.allocation),
        'access_preserved': torch.equal(altered.access, current.access),
        'hidden_preserved': torch.equal(hidden, hidden_before),
        'restoration_exact': torch.equal(restored, predicted),
        'control_mask_swapped_exact': torch.equal(control_mask(altered), control_mask(current).flip(-1)),
        'counterfactual_change_mae': float((changed-predicted).abs().double().mean()),
    }
    trace = {'observations': episode.observations, 'allocation': episode.allocation,
        'access': episode.access, 'physical_effects': episode.next_allocation,
        'commands': episode.commands, 'replay': episode.replay, 'direction': episode.direction,
        'current_allocation': current.allocation, 'current_access': current.access,
        'current_effects': current.effects, 'hidden': hidden,
        'predicted_recovery': predicted, 'physical_recovery': target,
        'changed_recovery': changed, 'restored_recovery': restored}
    return {'metrics':metrics, 'gates':gates, 'all_gates_pass':all(gates.values()),
            'intervention':intervention}, trace


@torch.no_grad()
def matched_revision(model, episode, new_condition, commands, windows=WINDOWS):
    """Both routes are probed against one physical next-step target at each window.

    The no-feedback route advances its clock with predicted allocations. The feedback
    route advances with actual observations. Neither receives the physical owner ID.
    """
    if new_condition not in CONDITIONS or max(windows)>commands.shape[1]:
        raise ValueError('invalid adaptation condition or windows')
    seq, initial_hidden = model(episode.observations)
    initial = last_forecast(seq)
    baseline, updated = initial.clone(), initial.clone()
    baseline_hidden, updated_hidden = initial_hidden.clone(), initial_hidden.clone()
    replay, recovery = episode.replay.clone(), episode.access[:, -1, 0].clone()
    controlled = torch.full_like(episode.controlled, new_condition)
    observed, anticipated, records, traces = [], [], [], {}
    for t in range(1, max(windows)+1):
        command = commands[:, t-1]
        replay, recovery, x, _ = advance_world(replay, episode.direction, recovery, controlled, command)
        baseline, baseline_hidden, hypothetical = advance_model(model, baseline, baseline_hidden, command)
        next_seq, updated_hidden = model(x[:, None], updated_hidden)
        updated = last_forecast(next_seq)
        observed.append(x); anticipated.append(hypothetical)
        if t not in windows:
            continue
        target_effects = effect_table((replay+episode.direction) % 4, controlled)
        target_recovery = physical_recovery(recovery, target_effects)
        before = counterfactual_recovery(model, baseline, baseline_hidden)
        after = counterfactual_recovery(model, updated, updated_hidden)
        before_error = (before-target_recovery).abs().double().mean((1,2,3))
        after_error = (after-target_recovery).abs().double().mean((1,2,3))
        metrics = {
            'no_feedback_effect_accuracy': float((baseline.effects.argmax(-1)==target_effects.argmax(-1)).double().mean()),
            'feedback_effect_accuracy': float((updated.effects.argmax(-1)==target_effects.argmax(-1)).double().mean()),
            'feedback_control_accuracy': float((control_mask(updated)==true_control_mask(controlled)).all(-1).double().mean()),
            'no_feedback_prior_control_accuracy': float((control_mask(baseline)==true_control_mask(episode.controlled)).all(-1).double().mean()),
            'no_feedback_recovery_mae': float(before_error.mean()),
            'feedback_recovery_mae': float(after_error.mean()),
            'feedback_current_recovery_mae': float((updated.access[:, 0]-recovery).abs().double().mean()),
            'paired_episode_improvement': float((after_error<before_error).double().mean()),
        }
        records.append({'observations':t, 'metrics':metrics})
        traces[t] = {'target_effects':target_effects, 'target_recovery':target_recovery,
            'physical_current_recovery':recovery.clone(), 'physical_replay':replay.clone(),
            'no_feedback_effects':baseline.effects, 'feedback_effects':updated.effects,
            'no_feedback_recovery':before, 'feedback_recovery':after,
            'feedback_current_recovery':updated.access[:, 0]}
    final = records[-1]['metrics']
    gates = {'feedback_effect_accuracy':final['feedback_effect_accuracy']>=.99,
        'feedback_control_accuracy':final['feedback_control_accuracy']>=.99,
        'feedback_recovery_mae':final['feedback_recovery_mae']<=.04,
        'feedback_current_recovery_mae':final['feedback_current_recovery_mae']<=.04,
        'no_feedback_prior_control_accuracy':final['no_feedback_prior_control_accuracy']>=.99}
    if bool((episode.controlled==new_condition).all()):
        gates['no_change_effect_accuracy']=final['no_feedback_effect_accuracy']>=.99
    else:
        gates['paired_episode_improvement']=final['paired_episode_improvement']>=.95
    trace = {'commands':commands, 'observed':torch.stack(observed,1),
        'anticipated':torch.stack(anticipated,1), 'initial_allocation':initial.allocation,
        'initial_access':initial.access, 'initial_effects':initial.effects,
        'initial_hidden':initial_hidden, 'initial_physical_recovery':episode.access[:, -1, 0].clone(),
        'initial_replay':episode.replay.clone(), 'windows':traces}
    return {'old_condition':int(episode.controlled[0]), 'new_condition':new_condition,
        'windows':records, 'final_gates':gates, 'all_gates_pass':all(gates.values())}, trace
