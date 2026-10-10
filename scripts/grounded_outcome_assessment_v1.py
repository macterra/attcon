"""Qualification of task-outcome forecasts and a separate frozen Controller.

Binary scores average actual Bernoulli squared errors across repeat outcomes;
they do not score a fractional repeat mean as if it were a single observation.
Resampling units are worlds, with every snapshot, command and noise repetition
kept together. None of these measurements is an appearance judgment.
"""
import torch

SNAPSHOTS = (0, 2, 4, 8)
BOOTSTRAPS = 256


def _values(data, forecasts):
    sensory = data['sensory'].detach().cpu()
    history = data['history'].detach().cpu()
    if sensory.ndim != 4 or sensory.shape[1:] != (4, 4, 66):
        raise ValueError('sensory must contain snapshots [0,2,4,8]: [N,4,4,66]')
    n = len(sensory)
    if not n or history.shape != (n, 4, 4, 4):
        raise ValueError('empty data or invalid history shape')
    current = data['current_labels'].detach().cpu().double()
    refresh = data['refresh_labels'].detach().cpu().double()
    delivery = data['delivery_labels'].detach().cpu()
    after = data['after_labels'].detach().cpu().double()
    if current.shape != (n, 4, 4) or refresh.ndim != 4 or refresh.shape[:2] != (n, 4) or refresh.shape[-1] != 4 or not refresh.shape[2]:
        raise ValueError('invalid current or refresh label shape')
    r = refresh.shape[2]
    if delivery.shape != (n, 4, r, 4) or after.shape != (n, 4, r, 4, 4):
        raise ValueError('invalid delivery or after label shape')
    if delivery.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8) or not ((delivery >= 0) & (delivery < 4)).all():
        raise ValueError('delivery labels must be integer locations 0..3')
    for labels in (current, refresh, after):
        if not ((labels == 0) | (labels == 1)).all():
            raise ValueError('correctness labels must be actual binary outcomes')
    if not ((sensory[..., 65] == 0) | (sensory[..., 65] == 1)).all():
        raise ValueError('invalid observed flags')
    if not ((history == 0) | (history == 1)).all() or not (history.sum(-1) <= 1).all():
        raise ValueError('history rows must be zero or delivery one-hot')
    p = {key: forecasts[key].detach().cpu().double() for key in ('current', 'refresh', 'delivery', 'after')}
    for key, shape in (('current', (n, 4, 4)), ('refresh', (n, 4, 4)), ('delivery', (n, 4, 4, 4)), ('after', (n, 4, 4, 4))):
        if p[key].shape != shape or not torch.isfinite(p[key]).all() or not ((p[key] >= 0) & (p[key] <= 1)).all():
            raise ValueError(f'invalid {key} probabilities')
    if not torch.allclose(p['delivery'].sum(-1), torch.ones(n, 4, 4, dtype=torch.float64), atol=1e-7, rtol=0):
        raise ValueError('delivery distributions must sum to one')
    return sensory, history, current, refresh, delivery.long(), after, p


def _interval(values):
    if not len(values) or not torch.isfinite(values).all():
        return None
    return torch.quantile(values.double(), torch.tensor([.025, .975], dtype=torch.float64)).tolist()


def _binary(probability, observed_mean, baseline, mask=None, repeats=1, indices=None):
    if mask is None:
        mask = torch.ones_like(probability, dtype=torch.bool)
    if mask.shape != probability.shape or observed_mean.shape != probability.shape:
        raise ValueError('binary metric shape mismatch')
    count = int(mask.sum())
    result = {'prediction_count': count, 'outcome_count': count*repeats,
              'world_count': int(mask.flatten(1).any(-1).sum())}
    if not count:
        return {**result, 'mean_probability': None, 'accuracy': None,
                'signed_residual': None, 'brier': None, 'ece': None,
                'constant_brier': None, 'brier_difference': None,
                'brier_difference_95': None, 'reliability': []}
    residual = probability-observed_mean
    # E[(p-Y)^2] = p^2-2pE[Y]+E[Y] for binary Y.
    errors = probability.square()-2*probability*observed_mean+observed_mean
    constant_errors = baseline.square()-2*baseline*observed_mean+observed_mean
    bins = torch.bucketize(probability.contiguous(), torch.arange(1, 10, dtype=torch.float64)/10, right=True)
    ece, reliability = 0., []
    for b in range(10):
        keep = mask & (bins == b)
        amount = int(keep.sum())
        predicted = float(probability[keep].mean()) if amount else None
        observed = float(observed_mean[keep].mean()) if amount else None
        if amount:
            ece += amount/count*abs(predicted-observed)
        reliability.append({'bin': b, 'prediction_count': amount,
                            'mean_probability': predicted, 'accuracy': observed})
    result.update(mean_probability=float(probability[mask].mean()), accuracy=float(observed_mean[mask].mean()),
                  signed_residual=float(residual[mask].mean()), brier=float(errors[mask].mean()), ece=ece,
                  constant_brier=float(constant_errors[mask].mean()),
                  brier_difference=float((errors-constant_errors)[mask].mean()), reliability=reliability)
    result['brier_difference_95'] = None
    if indices is not None:
        totals = torch.where(mask, errors-constant_errors, 0).flatten(1).sum(-1)
        counts = mask.flatten(1).sum(-1)
        denominator = counts[indices].sum(-1)
        if bool((denominator > 0).all()):
            result['brier_difference_95'] = _interval(totals[indices].sum(-1)/denominator)
    return result


def _delivery(probability, actual, baseline, mask=None, indices=None):
    n, snapshots, repeats, commands = actual.shape
    labels = torch.nn.functional.one_hot(actual, 4).double().mean(2)
    if mask is None:
        mask = torch.ones(n, snapshots, commands, dtype=torch.bool)
    count = int(mask.sum())
    result = {'prediction_count': count, 'outcome_count': count*repeats,
              'world_count': int(mask.flatten(1).any(-1).sum())}
    if not count:
        return {**result, 'brier': None, 'constant_brier': None, 'brier_difference': None,
                'brier_difference_95': None, 'nll': None, 'constant_nll': None,
                'top1': None, 'top_confidence_ece': None, 'top_confidence': None}
    errors = (probability.square()-2*probability*labels+labels).sum(-1)
    constant_errors = (baseline.square()-2*baseline*labels+labels).sum(-1)
    # Clamping applies only to scoring logarithms, not the archived forecasts.
    nll = -(labels*probability.clamp_min(1e-15).log()).sum(-1)
    constant_nll = -(labels*baseline.clamp_min(1e-15).log()).sum(-1)
    confidence, choice = probability.max(-1)
    correct = labels.gather(-1, choice[..., None])[..., 0]
    top = _binary(confidence, correct, torch.tensor(.25, dtype=torch.float64), mask, repeats)
    result.update(brier=float(errors[mask].mean()), constant_brier=float(constant_errors[mask].mean()),
                  brier_difference=float((errors-constant_errors)[mask].mean()),
                  nll=float(nll[mask].mean()), constant_nll=float(constant_nll[mask].mean()),
                  top1=float(correct[mask].mean()), top_confidence=float(confidence[mask].mean()),
                  top_confidence_ece=top['ece'], top_confidence_reliability=top['reliability'])
    result['brier_difference_95'] = None
    if indices is not None:
        totals = torch.where(mask, errors-constant_errors, 0).flatten(1).sum(-1)
        counts = mask.flatten(1).sum(-1)
        denominator = counts[indices].sum(-1)
        if bool((denominator > 0).all()):
            result['brier_difference_95'] = _interval(totals[indices].sum(-1)/denominator)
    return result


def _qualified(metric):
    return metric['prediction_count'] > 0 and metric['ece'] <= .10 and abs(metric['signed_residual']) <= .075


def assess(calibration_data, evaluation_data, calibration_forecasts, evaluation_forecasts, seed, kind):
    """Score frozen forecasts; constants use only calibration outcomes."""
    cal = _values(calibration_data, calibration_forecasts)
    sensory, history, current, refresh, delivery, after, p = _values(evaluation_data, evaluation_forecasts)
    baseline = {'current': cal[2].mean(), 'refresh': cal[3].mean(), 'after': cal[5].mean(),
                'delivery': torch.nn.functional.one_hot(cal[4], 4).double().mean((0, 1, 2))}
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(sensory), (BOOTSTRAPS, len(sensory)), generator=generator)
    seen = sensory[..., 65].bool()
    known = history.sum(-1) > 0
    observed = {'current': current, 'refresh': refresh.mean(2), 'after': after.mean(2)}
    repeat_counts = {'current': 1, 'refresh': refresh.shape[2], 'after': after.shape[2]}
    pooled = {key: _binary(p[key], observed[key], baseline[key], repeats=repeat_counts[key], indices=indices)
              for key in ('current', 'refresh', 'after')}
    strata = {}
    for key in ('current', 'refresh'):
        strata[key] = {name: _binary(p[key], observed[key], baseline[key], mask, repeat_counts[key], indices)
                       for name, mask in (('observed', seen), ('unobserved', ~seen))}
    strata['after'] = {}
    for seen_name, seen_mask in (('observed', seen), ('unobserved', ~seen)):
        for known_name, known_mask in (('known', known), ('unknown', ~known)):
            mask = seen_mask.unsqueeze(-2) & known_mask.unsqueeze(-1)
            strata['after'][f'{seen_name}_{known_name}'] = _binary(p['after'], observed['after'], baseline['after'], mask, after.shape[2], indices)
    pooled['delivery'] = _delivery(p['delivery'], delivery, baseline['delivery'], indices=indices)
    strata['delivery'] = {name: _delivery(p['delivery'], delivery, baseline['delivery'], mask, indices)
                         for name, mask in (('known', known), ('unknown', ~known))}
    null_mask = (history.sum((-1, -2)) == 0).unsqueeze(-1).expand_as(known)
    strata['delivery']['zero_history_null'] = _delivery(p['delivery'], delivery, baseline['delivery'], null_mask, indices)
    delivery_observed = torch.nn.functional.one_hot(delivery, 4).double().mean(2)
    cells = {}
    for name, row_mask in (('known', known), ('unknown', ~known)):
        cells[name] = {}
        for command in range(4):
            for location in range(4):
                cells[name][f'command{command}_location{location}'] = _binary(
                    p['delivery'][..., command, location], delivery_observed[..., command, location],
                    baseline['delivery'][command, location], row_mask[..., command], delivery.shape[2], indices)
    gates = {}
    if kind == 'structured':
        for key in ('current', 'refresh', 'after'):
            for name, metric in strata[key].items():
                gates[f'{key}_{name}_calibration'] = _qualified(metric)
            gates[f'{key}_pooled_brier'] = pooled[key]['brier'] <= pooled[key]['constant_brier']
        for name in ('known', 'unknown'):
            metric = strata['delivery'][name]
            gates[f'delivery_{name}_top_confidence_ece'] = metric['prediction_count'] > 0 and metric['top_confidence_ece'] <= .10
            for cell, metric in cells[name].items():
                gates[f'delivery_{name}_{cell}_calibration'] = _qualified(metric)
        gates['delivery_known_top1'] = strata['delivery']['known']['prediction_count'] > 0 and strata['delivery']['known']['top1'] >= .95
        gates['delivery_pooled_brier'] = pooled['delivery']['brier'] <= pooled['delivery']['constant_brier']
        gates['delivery_pooled_nll'] = pooled['delivery']['nll'] <= pooled['delivery']['constant_nll']
        null = strata['delivery']['zero_history_null']
        gates['zero_history_gross_chance'] = null['prediction_count'] > 0 and abs(null['top1']-.25) <= .075
    return {'kind': kind, 'snapshots': list(SNAPSHOTS), 'world_count': len(sensory),
            'repeat_count': refresh.shape[2], 'pooled': pooled, 'strata': strata,
            'delivery_probability_cells': cells,
            'calibration_constants': {key: value.tolist() for key, value in baseline.items()},
            'gates': gates, 'engineering_gates_pass': all(gates.values()) if gates else None,
            'bootstrap_count': BOOTSTRAPS, 'bootstrap_seed': seed,
            'top_confidence_ece_scope': 'reliability of predicted top location only; per-command/location binary reliability reported separately',
            'goal_achieved': False, 'appearance_judgments_used': False}


def _paired_summary(original, comparison, selection, indices):
    count = int(selection.sum())
    result = {'world_count': count, 'original_success': None, 'comparison_success': None,
              'difference': None, 'difference_95': None}
    if count:
        a, b = original.mean(-1), comparison.mean(-1)
        result.update(original_success=float(a[selection].mean()), comparison_success=float(b[selection].mean()),
                      difference=float((a-b)[selection].mean()), difference_95=_interval((a-b)[indices].mean(-1)))
    return result


def control(data, forecasts, seed, kind='structured'):
    """Choose a query without labels, then execute a fixed argmax Controller.

    Returns summary and an archive of choices, actual repeated outcomes and
    world resampling indices. Only the explicitly selected structured primary
    has intervention gates; its composition must match exactly to qualify.
    """
    sensory, history, current, refresh, delivery, after, p = _values(data, forecasts)
    n, step = len(sensory), 2
    generator = torch.Generator().manual_seed(seed)
    unseen = ~sensory[:, step, :, 65].bool()
    queries = torch.empty(n, dtype=torch.long)
    for world in range(n):
        choices = torch.where(unseen[world])[0]
        if not len(choices):
            choices = torch.arange(4)
        queries[world] = choices[torch.randint(len(choices), (), generator=generator)]
    subset = unseen.gather(1, queries[:, None])[:, 0]
    query_probabilities = p['after'][:, step].gather(-1, queries[:, None, None].expand(n, 4, 1))[..., 0]
    actions = query_probabilities.argmax(-1)
    rows = torch.arange(n)
    original = after[rows, step, :, actions, queries]
    all_command_outcomes = after[:, step].gather(-1, queries[:, None, None, None].expand(n, after.shape[2], 4, 1))[..., 0]
    random = all_command_outcomes.mean(-1)
    all_indices = torch.randint(n, (BOOTSTRAPS, n), generator=generator)
    subset_rows = rows[subset]
    subset_indices = (subset_rows[torch.randint(len(subset_rows), (BOOTSTRAPS, len(subset_rows)), generator=generator)]
                      if len(subset_rows) else torch.empty(BOOTSTRAPS, 0, dtype=torch.long))
    gain = {'all_worlds': _paired_summary(original, random, torch.ones(n, dtype=torch.bool), all_indices),
            'unseen_query_subset': _paired_summary(original, random, subset, subset_indices)}
    archive = {'queries': queries, 'actions': actions, 'unseen_query_subset': subset,
               'query_action_probabilities': query_probabilities, 'original_scores': original,
               'random_scores': random, 'all_world_bootstrap_indices': all_indices,
               'unseen_query_bootstrap_indices': subset_indices}
    result = {'snapshot': SNAPSHOTS[step], 'world_count': n, 'gain_vs_uniform_random_command': gain,
              'bootstrap_count': BOOTSTRAPS, 'seed': seed, 'gates': {},
              'query_rule': 'uniform among unseen locations; if none, uniform among all four',
              'ties_retained': True, 'goal_achieved': False}
    result['kind'] = kind
    structured = kind == 'structured'
    if structured:
        composed = p['delivery']*p['refresh'].unsqueeze(-2)+(1-p['delivery'])*p['current'].unsqueeze(-2)
        corrupted_delivery = p['delivery'].roll(1, -2)
        corrupted_after = corrupted_delivery*p['refresh'].unsqueeze(-2)+(1-corrupted_delivery)*p['current'].unsqueeze(-2)
        restored_delivery = corrupted_delivery.roll(-1, -2)
        restored_after = restored_delivery*p['refresh'].unsqueeze(-2)+(1-restored_delivery)*p['current'].unsqueeze(-2)
        corrupted_query = corrupted_after[:, step].gather(-1, queries[:, None, None].expand(n, 4, 1))[..., 0]
        corrupted_actions = corrupted_query.argmax(-1)
        corrupted = after[rows, step, :, corrupted_actions, queries]
        sorted_probabilities = query_probabilities.sort(-1, descending=True).values
        unique = sorted_probabilities[:, 0]-sorted_probabilities[:, 1] > 1e-6
        rotation = corrupted_actions[unique] == (actions[unique]+1)%4
        restoration = torch.equal(p['delivery'], restored_delivery) and torch.equal(p['after'], restored_after)
        corruption = {'all_worlds': _paired_summary(original, corrupted, torch.ones(n, dtype=torch.bool), all_indices),
                      'unseen_query_subset': _paired_summary(original, corrupted, subset, subset_indices)}
        result['route_intervention'] = {'operation': 'rotate represented command rows +1, keep sensory/current/refresh fixed',
            'original_composition_exact': torch.equal(p['after'], composed), 'restoration_exact': restoration,
            'unique_winner_worlds': int(unique.sum()), 'unique_winner_actions_rotate': bool(rotation.all()),
            'all_world_actions_changed': int((actions != corrupted_actions).sum()),
            'original_minus_corrupted_success': corruption}
        subset_gain = gain['unseen_query_subset']['difference_95']
        subset_corruption = corruption['unseen_query_subset']['difference_95']
        result['gates'] = {'unseen_query_controller_gain': subset_gain is not None and subset_gain[0] > 0,
                           'unseen_query_route_corruption_loss': subset_corruption is not None and subset_corruption[0] > 0,
                           'composition_exact': torch.equal(p['after'], composed), 'restoration_exact': restoration,
                           'unique_winner_actions_rotate': bool(rotation.all())}
        archive.update(corrupted_actions=corrupted_actions, corrupted_scores=corrupted,
                       corrupted_delivery=corrupted_delivery, corrupted_after=corrupted_after,
                       restored_delivery=restored_delivery, restored_after=restored_after,
                       unique_winner=unique)
    result['engineering_gates_pass'] = all(result['gates'].values()) if result['gates'] else None
    return result, archive
