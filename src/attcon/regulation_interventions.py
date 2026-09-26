"""Choice-preserving diagnostics; synthetic lesions have no access ground truth."""

import torch

from attcon.regulation import WIDTH, StateReporter
from attcon.report_calibration import ActionScoreReporter, EntropyReporter
from attcon.report_erasure import erasure_delta, loss_report_metrics


def paired_indices(data):
    """Align histories by context and hidden value without relying on row order."""
    seen, unseen = {}, {}
    for index, (group, value, available) in enumerate(zip(data.group.tolist(), data.value.tolist(), data.seen.tolist())):
        target = seen if available else unseen
        key = (group, value)
        if key in target:
            raise ValueError("duplicate context/value/status")
        target[key] = index
    if not seen or seen.keys() != unseen.keys():
        raise ValueError("unpaired histories")
    keys = sorted(seen)
    return torch.tensor([seen[key] for key in keys]), torch.tensor([unseen[key] for key in keys])


def choice_null_directions(choice_weight, fit_states, fit_data, seed):
    weight = choice_weight.detach().double()
    _, singular, vh = torch.linalg.svd(weight, full_matrices=True)
    tolerance = max(weight.shape) * torch.finfo(weight.dtype).eps * singular.max()
    rank = int((singular > tolerance).sum())
    row_space = vh[:rank].T
    def project(value):
        return value - row_space @ (row_space.T @ value)
    seen, unseen = paired_indices(fit_data)
    difference = (fit_states[seen].double() - fit_states[unseen].double()).mean(0)
    access = project(difference)
    random = project(torch.randn(weight.shape[1], generator=torch.Generator().manual_seed(seed), dtype=torch.float64))
    norms = {"access": access.norm().item(), "random": random.norm().item(), "unprojected_access": difference.norm().item()}
    if min(norms['access'], norms['random']) < 1e-8:
        raise ValueError("degenerate choice-null direction")
    return {"access": (access / access.norm()).float(), "random": (random / random.norm()).float()}, {"choice_rank": rank, "direction_norms": norms}


def projection_transplant(recipient, donor, direction, reference_delta=None):
    delta = (((donor - recipient) @ direction)[:, None]) * direction
    if reference_delta is not None:
        norm = delta.norm(dim=1, keepdim=True)
        required = reference_delta.norm(dim=1, keepdim=True)
        if ((norm < 1e-12) & (required > 1e-8)).any():
            raise ValueError("degenerate matched-norm control")
        delta = delta * required / norm.clamp_min(1e-12)
    return delta


def reporters_for(saved, agent, readouts):
    return {"state": StateReporter(readouts['state']),
            "action": ActionScoreReporter(agent.choice, readouts['action_logits'], WIDTH),
            "entropy": EntropyReporter(agent.choice, saved['selections']['entropy']['selected']['threshold'])}


@torch.no_grad()
def report_interventions(saved, agent, config, readouts, fit_states, fit_data, states, data, seed):
    reporters = reporters_for(saved, agent, readouts)
    seen, unseen = paired_indices(data)
    recipients, donors, target = states[seen], states[unseen], data.value[seen]
    before_choice = agent.choice(recipients).argmax(-1)
    before = {name: reporter(recipients).argmax(-1) for name, reporter in reporters.items()}
    shared = before_choice == target
    for value in before.values():
        shared &= value == target
    cohorts = {'all_seen': torch.ones(len(seen), dtype=torch.bool), 'shared_correct': shared}
    def measure(changed):
        choices = agent.choice(changed).argmax(-1)
        return {name: {cohort: loss_report_metrics(choices, reporter(changed).argmax(-1), target, config.values, mask)
                      for cohort, mask in cohorts.items()} for name, reporter in reporters.items()}
    erasures = {}
    for kind in ('content', 'random', 'permuted'):
        erasures[kind] = {}
        for strength in (0.0, 0.25, 0.5, 1.0):
            delta = erasure_delta(recipients, saved['seen_state_mean'], saved[kind + '_basis'], strength,
                                  None if kind == 'content' else saved['content_basis'])
            erasures[kind][str(strength)] = {'mean_delta_norm': delta.norm(dim=1).mean().item(), 'reports': measure(recipients + delta)}
    directions, metadata = choice_null_directions(agent.choice.weight, fit_states, fit_data, seed)
    reference = projection_transplant(recipients, donors, directions['access'])
    transplants = {}
    for kind, direction in directions.items():
        delta = reference if kind == 'access' else projection_transplant(recipients, donors, direction, reference)
        changed = recipients + delta
        residual = (agent.choice(changed) - agent.choice(recipients)).abs().max().item()
        after = {name: reporter(changed).argmax(-1) for name, reporter in reporters.items()}
        restored = changed - delta
        invariant = {name: bool(torch.equal(before[name], after[name])) for name in ('action', 'entropy')}
        transplants[kind] = {'valid': residual <= 1e-5 and all(invariant.values()),
            'max_logit_residual': residual, 'report_invariance': invariant,
            'mean_delta_norm': delta.norm(dim=1).mean().item(),
            'max_restoration_state_error': (restored - recipients).abs().max().item(),
            'restored_report_invariance': {name: bool(torch.equal(before[name], reporter(restored).argmax(-1))) for name, reporter in reporters.items()},
            'state_report_changed_rate': (after['state'] != before['state']).float().mean().item(),
            'reports': measure(changed)}
    return {'cohort_counts': {name: int(mask.sum()) for name, mask in cohorts.items()}, 'baseline': measure(recipients),
            'erasures': erasures, 'choice_null_fit': metadata, 'choice_null_transplants': transplants,
            'access_minus_random_state_report_change': transplants['access']['state_report_changed_rate'] - transplants['random']['state_report_changed_rate']}
