"""Independent archived-data reconstruction; no production assessment imports.

Run only after the registered single attempt has completed. This audit rebuilds
worlds, pixels, fixed sensory decoding, evidence histories, executed labels,
frozen network forecasts, calibration fits, proper scores, Controller choices,
world bootstrap intervals and gates. It does not refit the outcome networks;
fixed-budget refitting belongs to the separately recorded exact replay.
"""
import gzip
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path('audits/grounded_attention_outcomes_v1')
OUT = ROOT/'results_review'
SEEDS = (1301, 1311, 1321)
KINDS = ('structured', 'joint', 'sensory_only', 'history_only')
TIMES = (0, 2, 4, 8)
LEVELS = ((38.2, 38.2), (39., 37.3), (39.5, 36.5), (40., 36.2), (40.2, 36.), (41.1, 35.8))
CHECKS = 0
MAX_ERROR = 0.


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    with gzip.open(path, 'rb') as stream:
        return torch.load(stream, weights_only=True)


def archive(seed, name, split=None):
    prefix = ROOT/f'model-{seed}'
    if split is not None:
        prefix /= f'split-{split}'
    return load(prefix/(name+'.pt.gz'))


def check(actual, expected, path, tolerance=2e-12):
    global CHECKS, MAX_ERROR
    CHECKS += 1
    if torch.is_tensor(actual) or torch.is_tensor(expected):
        assert torch.is_tensor(actual) and torch.is_tensor(expected), path
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, path
        if actual.is_floating_point():
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), path
            error = float((actual-expected).abs().max()) if actual.numel() else 0.
            MAX_ERROR = max(MAX_ERROR, error)
            assert error <= tolerance, (path, error)
        else:
            assert torch.equal(actual, expected), path
    elif isinstance(actual, dict):
        assert isinstance(expected, dict) and actual.keys() == expected.keys(), path
        for key in actual:
            check(actual[key], expected[key], f'{path}/{key}', tolerance)
    elif isinstance(actual, (list, tuple)):
        assert isinstance(expected, (list, tuple)) and len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)):
            check(a, b, f'{path}/{i}', tolerance)
    elif isinstance(actual, float):
        error = abs(actual-expected)
        MAX_ERROR = max(MAX_ERROR, error)
        assert error <= tolerance, (path, error)
    else:
        assert actual == expected, (path, actual, expected)


def rng(seed):
    return torch.Generator().manual_seed(seed)


def world(seed):
    g = rng(seed)
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing='ij')
    cx = 7.5+2*torch.rand(4, 1, 1, generator=g)-1
    cy = 7.5+2*torch.rand(4, 1, 1, generator=g)-1
    radius = 3.7+1.5*torch.rand(4, 1, 1, generator=g)
    dx, dy = xx[None]-cx, yy[None]-cy
    shape = torch.randint(2, (4, 1, 1), generator=g)
    mask = torch.where(shape == 0, dx.square()+dy.square() <= radius.square(),
                       (dx.abs() <= radius) & (dy.abs() <= radius))
    phase = 2*torch.pi*torch.rand(4, 1, 1, generator=g)
    labels = torch.randint(2, (4,), generator=rng(seed+3000000))
    mapping_rng = rng(seed+2000000)
    contrast = torch.randint(6, (4,), generator=mapping_rng)
    mapping = torch.randperm(4, generator=mapping_rng)
    return dict(labels=labels, contrasts=contrast, mask=mask, phase=phase, command_mapping=mapping)


def pixels(worlds, locations, noise):
    rows = torch.arange(len(locations)).reshape(len(locations), *([1]*(locations.ndim-1)))
    labels = worlds['labels'][rows, locations]
    phase = worlds['phase'][rows, locations]
    mask = worlds['mask'][rows, locations]
    contrast = worlds['contrasts'][rows, locations]
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing='ij')
    axis = torch.where(labels[..., None, None] == 0, yy, xx)
    texture = torch.sin(axis*torch.pi/2+phase) >= 0
    levels = (torch.tensor(LEVELS)/50)[contrast]
    target = torch.where(texture, levels[..., 0, None, None], levels[..., 1, None, None])
    image = torch.where(mask, target, torch.tensor(35.5/50))
    return (image[..., None, :, :].expand(*locations.shape, 3, 16, 16)+noise).clamp(0, 1)


def linear(x, state, name):
    return F.linear(x, state[name+'.weight'], state[name+'.bias'])


def decode(seed, image):
    state = torch.load(Path(f'audits/bound_content/confirmation_v2/seed{seed}.pt'), weights_only=True)['state_dict']
    adapter = load(Path(f'audits/sensory_task_adapter_v1/states-{seed}.pt.gz'))['instruments']['visual_hidden']['readout']
    parts_h, parts_z = [], []
    for x in image.reshape(-1, 3, 16, 16).split(256):
        x = F.max_pool2d(F.relu(F.conv2d(x, state['net.0.weight'], state['net.0.bias'], padding=1)), 2)
        x = F.max_pool2d(F.relu(F.conv2d(x, state['net.3.weight'], state['net.3.bias'], padding=1)), 2)
        h = F.relu(linear(x.flatten(1), state, 'net.7'))
        z = linear(F.relu(linear((h-adapter['mean'])/adapter['scale'], adapter['state_dict'], '0')), adapter['state_dict'], '2')[:, 0]
        parts_h.append(h); parts_z.append(z)
    shape = image.shape[:-3]
    return torch.cat(parts_h).reshape(*shape, 64), torch.cat(parts_z).reshape(shape)


def split_data(seed, split):
    data = archive(seed, 'dataset', split)
    actual = archive(seed, 'world-truth', split)
    n, repeat = len(data['sensory']), data['refresh_labels'].shape[2]
    count = (512, 256, 512)[split]
    assert n == count and repeat == (2, 2, 8)[split]
    base = seed*1000+split*100000
    worlds = [world(960000000+base+i) for i in range(n)]
    reconstructed = {key: torch.stack([w[key] for w in worlds]) for key in worlds[0]}
    check(reconstructed, actual, f'{seed}/{split}/world')
    commands = torch.randint(4, (n, 8), generator=rng(970000000+base))
    delivered = reconstructed['command_mapping'].gather(1, commands)
    noises = .015*torch.randn(n, 8, 3, 16, 16, generator=rng(980000000+base))
    check(noises, archive(seed, 'history-noise', split), f'{seed}/{split}/history-noise', 0.)
    images = pixels(reconstructed, delivered, noises)
    check(images, archive(seed, 'history-pixels', split), f'{seed}/{split}/history-pixels', 0.)
    hidden, logits = decode(seed, images)
    hist = dict(commands=commands, delivered=delivered, hidden=hidden, logits=logits)
    check(hist, archive(seed, 'history-decoded', split), f'{seed}/{split}/history-decoded', 0.)
    memory = torch.zeros(n, 4, 66)
    routes = torch.zeros(n, 4, 4)
    rows = torch.arange(n)
    all_s, all_h, all_c, all_r, all_d, all_a = [], [], [], [], [], []
    for time in range(9):
        if time in TIMES:
            k = TIMES.index(time)
            all_s.append(memory.clone()); all_h.append(routes.clone())
            correct = (memory[..., 64] > 0).long() == reconstructed['labels']
            all_c.append(correct)
            branch_loc = reconstructed['command_mapping'][:, None].expand(n, repeat, 4).clone()
            noise = .015*torch.randn(n, repeat, 4, 3, 16, 16, generator=rng(990000000+base+k*1000))
            check(noise, archive(seed, f'branch-{time}-noise', split), f'{seed}/{split}/{time}/noise', 0.)
            image = pixels(reconstructed, branch_loc, noise)
            check(image, archive(seed, f'branch-{time}-pixels', split), f'{seed}/{split}/{time}/pixels', 0.)
            h, z = decode(seed, image)
            success = (z > 0).long() == reconstructed['labels'][rows[:, None, None], branch_loc]
            # Independent explicit address lookup, rather than production scatter.
            fresh = torch.stack([success.gather(2, (branch_loc == loc).long().argmax(-1)[..., None])[..., 0]
                                 for loc in range(4)], -1)
            after = correct[:, None, None].expand(n, repeat, 4, 4).clone()
            for command in range(4):
                after[rows[:, None], torch.arange(repeat)[None], command, branch_loc[..., command]] = success[..., command]
            branch = dict(hidden=h, logits=z, delivered=branch_loc, refresh_correct=fresh, after_correct=after)
            check(branch, archive(seed, f'branch-{time}-decoded', split), f'{seed}/{split}/{time}/decoded', 0.)
            all_r.append(fresh); all_d.append(branch_loc); all_a.append(after)
        if time < 8:
            loc, command = delivered[:, time], commands[:, time]
            memory[rows, loc, :64] = hidden[:, time]
            memory[rows, loc, 64] = logits[:, time]
            memory[rows, loc, 65] = 1
            routes[rows, command] = 0
            routes[rows, command, loc] = 1
    expected = dict(sensory=torch.stack(all_s, 1), history=torch.stack(all_h, 1),
        current_labels=torch.stack(all_c, 1), refresh_labels=torch.stack(all_r, 1),
        delivery_labels=torch.stack(all_d, 1), after_labels=torch.stack(all_a, 1), snapshots=list(TIMES))
    check(expected, data, f'{seed}/{split}/dataset', 0.)
    return data, actual


def masked(data, kind):
    sensory, history = data['sensory'].float().clone(), data['history'].float().clone()
    if kind == 'sensory_only': history.zero_()
    if kind == 'history_only': sensory[..., :65] = 0
    return sensory, history


def normalization(x):
    flat = x.reshape(-1, x.shape[-1])
    mean = flat.mean(0)
    scale = (flat-mean).square().mean(0).sqrt()
    return dict(mean=mean, scale=torch.where(scale > 1e-6, scale, torch.ones_like(scale)))


def raw(readout, data):
    sensory, history = masked(data, readout['kind'])
    state, norm = readout['state_dict'], readout['normalization']
    shape = sensory.shape[:2]
    if readout['kind'] == 'joint':
        x = torch.cat((sensory.flatten(-2), history.flatten(-2)), -1)
        x = (x-norm['joint']['mean'])/norm['joint']['scale']
        h = F.relu(linear(x, state, 'joint.0'))
        z = linear(h, state, 'joint.2')
        return dict(delivery_logits=z[..., :16].reshape(*shape, 4, 4), current_logits=z[..., 16:20],
            refresh_logits=z[..., 20:24], after_logits=z[..., 24:].reshape(*shape, 4, 4), joint_hidden=h)
    x = (sensory-norm['sensory']['mean'])/norm['sensory']['scale']
    risk_h = F.relu(linear(x, state, 'risk.0'))
    risk = linear(risk_h, state, 'risk.2')
    y = history.flatten(-2)
    y = (y-norm['history']['mean'])/norm['history']['scale']
    route_h = F.relu(linear(y, state, 'route.0'))
    route = linear(route_h, state, 'route.2').reshape(*shape, 4, 4)
    return dict(delivery_logits=route, current_logits=risk[..., 0], refresh_logits=risk[..., 1],
        risk_hidden=risk_h, route_hidden=route_h)


def signed_map(scores, targets):
    scores, targets = scores.flatten().double(), targets.flatten().double()
    mean = scores.mean()
    scale = (scores-mean).square().mean().sqrt()
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    design = torch.stack((torch.ones_like(scores), (scores-mean)/scale), 1)
    beta = torch.zeros(2, dtype=torch.float64)
    penalty = .001*torch.eye(2, dtype=torch.float64)
    for _ in range(100):
        p = (design@beta).sigmoid()
        gradient = design.T@(p-targets)/len(design)+penalty@beta
        curvature = design.T@((p*(1-p))[:, None]*design)/len(design)+penalty
        beta -= torch.linalg.solve(curvature, gradient)
    gradient_max = (design.T@((design@beta).sigmoid()-targets)/len(design)+penalty@beta).abs().max()
    assert float(gradient_max) <= 1e-8
    objective = (F.softplus(design@beta)-targets*(design@beta)).mean()+.001/2*beta.square().sum()
    return dict(mean=mean, scale=scale, beta=beta, baseline=targets.mean(), gradient_max=gradient_max,
        objective=objective, count=len(scores), steps=100)


def calibrators(readout, data):
    z = raw(readout, data)
    target = data['delivery_labels']
    losses = [F.cross_entropy((z['delivery_logits'].double()/t).unsqueeze(2).expand(*target.shape, 4).reshape(-1, 4),
                              target.reshape(-1)) for t in (.5, 1., 2., 4.)]
    index = min(range(4), key=lambda i: float(losses[i]))
    result = dict(delivery_temperature=(.5, 1., 2., 4.)[index], delivery_temperature_candidates=[.5, 1., 2., 4.],
        delivery_temperature_losses=torch.stack(losses), current=signed_map(z['current_logits'], data['current_labels']),
        refresh=signed_map(z['refresh_logits'], data['refresh_labels'].double().mean(2)))
    if readout['kind'] == 'joint': result['after'] = signed_map(z['after_logits'], data['after_labels'].double().mean(2))
    return result


def forecast(readout, data):
    z = raw(readout, data)
    mapping = readout['calibration']
    def binary(name):
        m = mapping[name]
        return (m['beta'][0]+m['beta'][1]*(z[name+'_logits'].double()-m['mean'])/m['scale']).sigmoid()
    delivery = (z['delivery_logits'].double()/mapping['delivery_temperature']).softmax(-1)
    current, refresh = binary('current'), binary('refresh')
    after = binary('after') if readout['kind'] == 'joint' else delivery*refresh.unsqueeze(-2)+(1-delivery)*current.unsqueeze(-2)
    return dict(delivery=delivery, current=current, refresh=refresh, after=after, **z)


def interval(values):
    return torch.quantile(values.double(), torch.tensor([.025, .975], dtype=torch.float64)).tolist()


def binary(p, y, base, mask=None, repeats=1, indices=None):
    mask = torch.ones_like(p, dtype=torch.bool) if mask is None else mask
    count = int(mask.sum())
    result = dict(prediction_count=count, outcome_count=count*repeats, world_count=int(mask.flatten(1).any(-1).sum()))
    if not count:
        return dict(**result, mean_probability=None, accuracy=None, signed_residual=None, brier=None, ece=None,
                    constant_brier=None, brier_difference=None, brier_difference_95=None, reliability=[])
    # Mean of true Bernoulli squared errors, not squared error of a repeat mean.
    err = p.square()-2*p*y+y
    null = base.square()-2*base*y+y
    bins = torch.bucketize(p.contiguous(), torch.arange(1, 10, dtype=torch.float64)/10, right=True)
    ece, reliability = 0., []
    for k in range(10):
        keep = mask & (bins == k)
        amount = int(keep.sum())
        pp = float(p[keep].mean()) if amount else None
        yy = float(y[keep].mean()) if amount else None
        if amount: ece += amount/count*abs(pp-yy)
        reliability.append(dict(bin=k, prediction_count=amount, mean_probability=pp, accuracy=yy))
    ci = None
    if indices is not None:
        totals = torch.where(mask, err-null, 0).flatten(1).sum(-1)
        counts = mask.flatten(1).sum(-1)
        denominator = counts[indices].sum(-1)
        if (denominator > 0).all(): ci = interval(totals[indices].sum(-1)/denominator)
    return dict(**result, mean_probability=float(p[mask].mean()), accuracy=float(y[mask].mean()),
        signed_residual=float((p-y)[mask].mean()), brier=float(err[mask].mean()), ece=ece,
        constant_brier=float(null[mask].mean()), brier_difference=float((err-null)[mask].mean()),
        brier_difference_95=ci, reliability=reliability)


def route_metric(p, labels, base, mask=None, indices=None):
    n, s, r, c = labels.shape
    actual = F.one_hot(labels, 4).double().mean(2)
    mask = torch.ones(n, s, c, dtype=torch.bool) if mask is None else mask
    count = int(mask.sum())
    result = dict(prediction_count=count, outcome_count=count*r, world_count=int(mask.flatten(1).any(-1).sum()))
    if not count:
        return dict(**result, brier=None, constant_brier=None, brier_difference=None, brier_difference_95=None,
                    nll=None, constant_nll=None, top1=None, top_confidence_ece=None, top_confidence=None)
    err = (p.square()-2*p*actual+actual).sum(-1)
    null = (base.square()-2*base*actual+actual).sum(-1)
    confidence, choice = p.max(-1)
    hit = actual.gather(-1, choice[..., None])[..., 0]
    top = binary(confidence, hit, torch.tensor(.25, dtype=torch.float64), mask, r)
    ci = None
    if indices is not None:
        totals = torch.where(mask, err-null, 0).flatten(1).sum(-1)
        counts = mask.flatten(1).sum(-1)
        denominator = counts[indices].sum(-1)
        if (denominator > 0).all(): ci = interval(totals[indices].sum(-1)/denominator)
    return dict(**result, brier=float(err[mask].mean()), constant_brier=float(null[mask].mean()),
        brier_difference=float((err-null)[mask].mean()), brier_difference_95=ci,
        nll=float(-(actual*p.clamp_min(1e-15).log()).sum(-1)[mask].mean()),
        constant_nll=float(-(actual*base.clamp_min(1e-15).log()).sum(-1)[mask].mean()),
        top1=float(hit[mask].mean()), top_confidence=float(confidence[mask].mean()),
        top_confidence_ece=top['ece'], top_confidence_reliability=top['reliability'])


def qualified(metric):
    return metric['prediction_count'] > 0 and metric['ece'] <= .10 and abs(metric['signed_residual']) <= .075


def assess(cal, data, p, seed, kind):
    base = dict(current=cal['current_labels'].double().mean(), refresh=cal['refresh_labels'].double().mean(),
        after=cal['after_labels'].double().mean(), delivery=F.one_hot(cal['delivery_labels'], 4).double().mean((0, 1, 2)))
    n, r = len(data['sensory']), data['refresh_labels'].shape[2]
    indices = torch.randint(n, (256, n), generator=rng(seed))
    seen, known = data['sensory'][..., 65].bool(), data['history'].sum(-1) > 0
    y = dict(current=data['current_labels'].double(), refresh=data['refresh_labels'].double().mean(2),
             after=data['after_labels'].double().mean(2))
    pooled = {k: binary(p[k], y[k], base[k], repeats=1 if k == 'current' else r, indices=indices)
              for k in ('current', 'refresh', 'after')}
    strata = {k: {name: binary(p[k], y[k], base[k], mask, 1 if k == 'current' else r, indices)
                  for name, mask in (('observed', seen), ('unobserved', ~seen))} for k in ('current', 'refresh')}
    strata['after'] = {f'{sn}_{kn}': binary(p['after'], y['after'], base['after'], sm.unsqueeze(-2) & km.unsqueeze(-1), r, indices)
        for sn, sm in (('observed', seen), ('unobserved', ~seen)) for kn, km in (('known', known), ('unknown', ~known))}
    pooled['delivery'] = route_metric(p['delivery'], data['delivery_labels'], base['delivery'], indices=indices)
    strata['delivery'] = {name: route_metric(p['delivery'], data['delivery_labels'], base['delivery'], mask, indices)
                          for name, mask in (('known', known), ('unknown', ~known))}
    null = (data['history'].sum((-1, -2)) == 0).unsqueeze(-1).expand_as(known)
    strata['delivery']['zero_history_null'] = route_metric(p['delivery'], data['delivery_labels'], base['delivery'], null, indices)
    actual = F.one_hot(data['delivery_labels'], 4).double().mean(2)
    cells = {name: {f'command{c}_location{loc}': binary(p['delivery'][..., c, loc], actual[..., c, loc],
                base['delivery'][c, loc], mask[..., c], r, indices) for c in range(4) for loc in range(4)}
                for name, mask in (('known', known), ('unknown', ~known))}
    gates = {}
    if kind == 'structured':
        for key in ('current', 'refresh', 'after'):
            gates.update({f'{key}_{name}_calibration': qualified(metric) for name, metric in strata[key].items()})
            gates[f'{key}_pooled_brier'] = pooled[key]['brier'] <= pooled[key]['constant_brier']
        for name in ('known', 'unknown'):
            metric = strata['delivery'][name]
            gates[f'delivery_{name}_top_confidence_ece'] = metric['prediction_count'] > 0 and metric['top_confidence_ece'] <= .10
            gates.update({f'delivery_{name}_{cell}_calibration': qualified(metric) for cell, metric in cells[name].items()})
        gates['delivery_known_top1'] = strata['delivery']['known']['prediction_count'] > 0 and strata['delivery']['known']['top1'] >= .95
        gates['delivery_pooled_brier'] = pooled['delivery']['brier'] <= pooled['delivery']['constant_brier']
        gates['delivery_pooled_nll'] = pooled['delivery']['nll'] <= pooled['delivery']['constant_nll']
        metric = strata['delivery']['zero_history_null']
        gates['zero_history_gross_chance'] = metric['prediction_count'] > 0 and abs(metric['top1']-.25) <= .075
    return dict(kind=kind, snapshots=list(TIMES), world_count=n, repeat_count=r, pooled=pooled, strata=strata,
        delivery_probability_cells=cells, calibration_constants={k: v.tolist() for k, v in base.items()},
        gates=gates, engineering_gates_pass=all(gates.values()) if gates else None, bootstrap_count=256,
        bootstrap_seed=seed, top_confidence_ece_scope='reliability of predicted top location only; per-command/location binary reliability reported separately',
        goal_achieved=False, appearance_judgments_used=False)


def paired(a, b, selected, indices):
    result = dict(world_count=int(selected.sum()), original_success=None, comparison_success=None,
                  difference=None, difference_95=None)
    if selected.any():
        a, b = a.mean(-1), b.mean(-1)
        result.update(original_success=float(a[selected].mean()), comparison_success=float(b[selected].mean()),
            difference=float((a-b)[selected].mean()), difference_95=interval((a-b)[indices].mean(-1)))
    return result


def controller(data, p, seed, kind):
    n, r, step = len(data['sensory']), data['after_labels'].shape[2], 2
    g = rng(seed)
    unseen = ~data['sensory'][:, step, :, 65].bool()
    query = []
    for w in range(n):
        choices = torch.where(unseen[w])[0]
        if not len(choices): choices = torch.arange(4)
        query.append(choices[torch.randint(len(choices), (), generator=g)])
    query = torch.stack(query)
    rows = torch.arange(n)
    subset = unseen[rows, query]
    prob = p['after'][rows, step, :, query]
    action = prob.argmax(-1)
    # Explicit world loop avoids the production advanced-index expression.
    outcomes = torch.stack([data['after_labels'][w, step, :, action[w], query[w]] for w in range(n)]).double()
    random = torch.stack([data['after_labels'][w, step, :, :, query[w]].double().mean(-1) for w in range(n)])
    all_ix = torch.randint(n, (256, n), generator=g)
    selected = rows[subset]
    sub_ix = selected[torch.randint(len(selected), (256, len(selected)), generator=g)] if len(selected) else torch.empty(256, 0, dtype=torch.long)
    def summaries(comparison):
        return dict(all_worlds=paired(outcomes, comparison, torch.ones(n, dtype=torch.bool), all_ix),
                    unseen_query_subset=paired(outcomes, comparison, subset, sub_ix))
    gain = summaries(random)
    storage = dict(queries=query, actions=action, unseen_query_subset=subset, query_action_probabilities=prob,
        original_scores=outcomes, random_scores=random, all_world_bootstrap_indices=all_ix, unseen_query_bootstrap_indices=sub_ix)
    result = dict(snapshot=4, world_count=n, gain_vs_uniform_random_command=gain, bootstrap_count=256,
        seed=seed, gates={}, query_rule='uniform among unseen locations; if none, uniform among all four',
        ties_retained=True, goal_achieved=False, kind=kind)
    if kind == 'structured':
        composed = p['delivery']*p['refresh'].unsqueeze(-2)+(1-p['delivery'])*p['current'].unsqueeze(-2)
        corrupted_delivery = p['delivery'].roll(1, -2)
        corrupt_after = corrupted_delivery*p['refresh'].unsqueeze(-2)+(1-corrupted_delivery)*p['current'].unsqueeze(-2)
        restore_delivery = corrupted_delivery.roll(-1, -2)
        restore_after = restore_delivery*p['refresh'].unsqueeze(-2)+(1-restore_delivery)*p['current'].unsqueeze(-2)
        corrupt_action = corrupt_after[rows, step, :, query].argmax(-1)
        corrupted = torch.stack([data['after_labels'][w, step, :, corrupt_action[w], query[w]] for w in range(n)]).double()
        sorted_p = prob.sort(-1, descending=True).values
        unique = sorted_p[:, 0]-sorted_p[:, 1] > 1e-6
        rotation = bool((corrupt_action[unique] == (action[unique]+1)%4).all())
        restore = torch.equal(p['delivery'], restore_delivery) and torch.equal(p['after'], restore_after)
        corruption = summaries(corrupted)
        result['route_intervention'] = dict(operation='rotate represented command rows +1, keep sensory/current/refresh fixed',
            original_composition_exact=torch.equal(p['after'], composed), restoration_exact=restore,
            unique_winner_worlds=int(unique.sum()), unique_winner_actions_rotate=rotation,
            all_world_actions_changed=int((action != corrupt_action).sum()), original_minus_corrupted_success=corruption)
        a, b = gain['unseen_query_subset']['difference_95'], corruption['unseen_query_subset']['difference_95']
        result['gates'] = dict(unseen_query_controller_gain=a is not None and a[0] > 0,
            unseen_query_route_corruption_loss=b is not None and b[0] > 0, composition_exact=torch.equal(p['after'], composed),
            restoration_exact=restore, unique_winner_actions_rotate=rotation)
        storage.update(corrupted_actions=corrupt_action, corrupted_scores=corrupted, corrupted_delivery=corrupted_delivery,
            corrupted_after=corrupt_after, restored_delivery=restore_delivery, restored_after=restore_after, unique_winner=unique)
    result['engineering_gates_pass'] = all(result['gates'].values()) if result['gates'] else None
    return result, storage


def remap(seed, data, worlds):
    observed = archive(seed, 'remap-states')
    changed = {key: value[:64].clone() for key, value in worlds.items()}
    changed['command_mapping'] = (changed['command_mapping']+1)%4
    noise = .015*torch.randn(64, 4, 3, 16, 16, generator=rng(1000000000+seed))
    check(noise, archive(seed, 'remap-noise'), f'{seed}/remap/noise', 0.)
    image = pixels(changed, changed['command_mapping'], noise)
    check(image, archive(seed, 'remap-pixels'), f'{seed}/remap/pixels', 0.)
    h, z = decode(seed, image)
    check(dict(hidden=h, logits=z, locations=changed['command_mapping']), archive(seed, 'remap-decoded'), f'{seed}/remap/decoded', 0.)
    memory, history = data['sensory'][:64, -1].clone(), data['history'][:64, -1].clone()
    original = dict(sensory=memory[:, None].clone(), history=history[:, None].clone())
    states, routes, rows = [memory.clone()], [history.clone()], torch.arange(64)
    for cmd in range(4):
        loc = changed['command_mapping'][:, cmd]
        memory[rows, loc, :64] = h[:, cmd]
        memory[rows, loc, 64] = z[:, cmd]
        memory[rows, loc, 65] = 1
        history[:, cmd] = 0
        history[rows, cmd, loc] = 1
        states.append(memory.clone()); routes.append(history.clone())
    expected = dict(sensory=torch.stack(states, 1), history=torch.stack(routes, 1),
        delivery_labels=changed['command_mapping'][:, None].expand(64, 5, 4).clone(), restored=original)
    check(expected, observed, f'{seed}/remap/states', 0.)
    return observed


@torch.no_grad()
def main():
    assert (ROOT/'completion.json').exists(), 'registered run must finish before audit'
    torch.set_num_threads(2)
    before = torch.random.get_rng_state().clone()
    manifest = json.loads((ROOT/'manifest.json').read_text())
    completion = json.loads((ROOT/'completion.json').read_text())
    result = json.loads((ROOT/'results.json').read_text())
    assert completion['manifest_sha256'] == sha(ROOT/'manifest.json')
    assert completion['results_sha256'] == sha(ROOT/'results.json')
    for path, digest in {**manifest['source_sha256'], **manifest['frozen_input_sha256']}.items():
        assert sha(path) == digest, path
    for path, detail in completion['archive_files'].items():
        assert sha(path) == detail['sha256'] and Path(path).stat().st_size == detail['bytes'], path
    audited = {}
    for seed in SEEDS:
        print(f'independent reconstruction {seed}', flush=True)
        splits = [split_data(seed, split) for split in range(3)]
        datasets = [entry[0] for entry in splits]
        revision = remap(seed, datasets[2], splits[2][1])
        outcomes = {}
        for index, kind in enumerate(KINDS):
            instrument = archive(seed, 'instrument-'+kind)
            fitted, calibrated = instrument['fitted'], instrument['calibrated']
            assert fitted['seed'] == 1010000000+seed+index*10000 and fitted['steps'] == 1500
            assert fitted['calibration'] is None and calibrated['kind'] == kind
            check(fitted['state_dict'], calibrated['state_dict'], f'{seed}/{kind}/weights-preserved', 0.)
            check(fitted['normalization'], calibrated['normalization'], f'{seed}/{kind}/normalization-preserved', 0.)
            s, h = masked(datasets[0], kind)
            norm = dict(joint=normalization(torch.cat((s.flatten(-2), h.flatten(-2)), -1))) if kind == 'joint' else dict(sensory=normalization(s), history=normalization(h.flatten(-2)))
            check(norm, fitted['normalization'], f'{seed}/{kind}/training-normalization', 0.)
            assert sum(t.numel() for t in fitted['state_dict'].values()) == fitted['parameter_count'] == (20584 if kind == 'joint' else 4338)
            check(calibrators(fitted, datasets[1]), calibrated['calibration'], f'{seed}/{kind}/calibration')
            forecasts = [forecast(calibrated, data) for data in datasets]
            check(forecasts, instrument['forecasts'], f'{seed}/{kind}/forecasts', 0.)
            assessment = assess(datasets[1], datasets[2], forecasts[2], 1020000000+seed, kind)
            control_result, control_archive = controller(datasets[2], forecasts[2], 1030000000+seed, kind)
            check(control_archive, instrument['controller'], f'{seed}/{kind}/controller-archive', 0.)
            original = forecast(calibrated, revision['restored'])
            changed = forecast(calibrated, revision)
            restored = forecast(calibrated, revision['restored'])
            check(dict(original=original, remapped=changed, restored=restored), instrument['remap'], f'{seed}/{kind}/remap-forecasts', 0.)
            exact = all(torch.equal(original[key], restored[key]) for key in original)
            correct = changed['delivery'].argmax(-1) == revision['delivery_labels']
            revision_result = dict(top1_by_observations=correct.double().mean((0, 2)).tolist(),
                final_delivery_top1_pass=float(correct[:, -1].double().mean()) >= .95,
                restoration_exact=exact, intermediate_histories_out_of_distribution=True)
            gates = dict(**assessment['gates'], **control_result['gates'])
            if kind == 'structured': gates.update(remap_final_delivery=revision_result['final_delivery_top1_pass'], remap_restoration=exact)
            summary = dict(assessment=assessment, controller=control_result, remap=revision_result, gates=gates,
                engineering_gates_pass=all(gates.values()) if kind == 'structured' else None,
                parameter_count=fitted['parameter_count'], loss_trace=fitted['loss_trace'])
            check(summary, result['models'][str(seed)][kind], f'{seed}/{kind}/results')
            outcomes[kind] = dict(gate_count=len(gates), failed_gates=[name for name, value in gates.items() if not value],
                engineering_gates_pass=summary['engineering_gates_pass'], calibration_reconstructed=True,
                frozen_forecasts_exact=True, controller_reconstructed=True)
        audited[str(seed)] = outcomes
    assert torch.equal(before, torch.random.get_rng_state()), 'audit mutated ambient torch RNG'
    assert result['primary_all_engineering_gates_pass'] == all(x['structured']['engineering_gates_pass'] for x in audited.values())
    for flag in ('appearance_tested', 'consciousness_related_correspondence_established', 'reporter_tested', 'controller_is_reporter', 'goal_achieved'):
        assert result[flag] is False
    assert result['independent_visual_models'] == 3
    report = dict(status='independent_archived_reconstruction_pass', reviewer='unblinded AI subagent',
        archive_count=len(completion['archive_files']), checked_values=CHECKS, max_numeric_discrepancy=MAX_ERROR,
        models=audited, training_networks_refit=False, raw_target_and_forecast_reconstruction=True,
        subjective_experience_inferred=False, goal_achieved=False,
        results_sha256=sha(ROOT/'results.json'), manifest_sha256=sha(ROOT/'manifest.json'), audit_code_sha256=sha(__file__))
    (OUT/'numerical_reconstruction.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], checked_values=CHECKS, max_numeric_discrepancy=MAX_ERROR)), flush=True)


if __name__ == '__main__': main()
