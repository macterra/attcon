"""Learn forecasts from executed synthetic outcomes, without appearance targets.

The input contract contains stored sensory evidence and the most recently
observed command-delivery mapping. Correctness labels are accepted only by fit
and calibrate; predict accepts observations alone. These probabilities are task
outcome forecasts, not judgments of experience or confidence of a reporter.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F

STEPS = 1500
RATE, DECAY = .003, .001
CAL_STEPS, CAL_L2, CAL_TOL = 100, .001, 1e-8
TEMPERATURES = (.5, 1., 2., 4.)
KINDS = ('structured', 'joint', 'sensory_only', 'history_only')


def _inputs(sensory, history, kind):
    if kind not in KINDS:
        raise ValueError(f'unknown model kind: {kind}')
    sensory = sensory.detach().cpu().float().clone()
    history = history.detach().cpu().float().clone()
    if sensory.ndim != 4 or sensory.shape[-2:] != (4, 66):
        raise ValueError('sensory must have shape [N,S,4,66]')
    if history.shape != (*sensory.shape[:2], 4, 4):
        raise ValueError('history must have shape [N,S,4,4]')
    if not sensory.shape[0] or not sensory.shape[1]:
        raise ValueError('empty observation set')
    if not torch.isfinite(sensory).all() or not torch.isfinite(history).all():
        raise ValueError('nonfinite observations')
    if not ((sensory[..., 65] == 0) | (sensory[..., 65] == 1)).all():
        raise ValueError('observed flag must be binary')
    if not ((history == 0) | (history == 1)).all() or not (history.sum(-1) <= 1).all():
        raise ValueError('history rows must be zero or delivery one-hot')
    if kind == 'sensory_only':
        history.zero_()
    elif kind == 'history_only':
        sensory[..., :65] = 0
    return sensory, history


def _labels(shape, current, refresh, delivery, after):
    current, refresh, delivery, after = [x.detach().cpu() for x in (current, refresh, delivery, after)]
    if current.shape != (*shape, 4):
        raise ValueError('current labels must have shape [N,S,4]')
    if refresh.ndim != 4 or refresh.shape[:2] != shape or refresh.shape[-1] != 4 or not refresh.shape[2]:
        raise ValueError('refresh labels must have shape [N,S,R,4] with R>0')
    repeats = refresh.shape[2]
    if delivery.shape != (*shape, repeats, 4) or delivery.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('delivery labels must be integer [N,S,R,4]')
    if not ((delivery >= 0) & (delivery < 4)).all():
        raise ValueError('delivery locations must be in 0..3')
    if after.shape != (*shape, repeats, 4, 4):
        raise ValueError('after labels must have shape [N,S,R,4,4]')
    for value in (current, refresh, after):
        if not ((value == 0) | (value == 1)).all():
            raise ValueError('outcome labels must be actual binary correctness')
    return current.float(), refresh.float(), delivery.long(), after.float()


def _normalizer(x):
    flat = x.reshape(-1, x.shape[-1])
    mean = flat.mean(0)
    scale = (flat-mean).square().mean(0).sqrt()
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    return {'mean': mean.detach().clone(), 'scale': scale.detach().clone()}


def _standardize(x, stats):
    return (x-stats['mean'])/stats['scale']


class _OutcomeNetwork(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == 'joint':
            self.joint = nn.Sequential(nn.Linear(280, 64), nn.ReLU(), nn.Linear(64, 40))
        else:
            self.risk = nn.Sequential(nn.Linear(66, 32), nn.ReLU(), nn.Linear(32, 2))
            self.route = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 16))

    def forward(self, sensory, history, normalization):
        shape = sensory.shape[:2]
        if self.kind == 'joint':
            x = torch.cat((sensory.flatten(-2), history.flatten(-2)), -1)
            hidden = self.joint[:2](_standardize(x, normalization['joint']))
            out = self.joint[2](hidden)
            return {'delivery_logits': out[..., :16].reshape(*shape, 4, 4),
                    'current_logits': out[..., 16:20], 'refresh_logits': out[..., 20:24],
                    'after_logits': out[..., 24:].reshape(*shape, 4, 4),
                    'joint_hidden': hidden}
        risk_hidden = self.risk[:2](_standardize(sensory, normalization['sensory']))
        risk = self.risk[2](risk_hidden)
        route_hidden = self.route[:2](_standardize(history.flatten(-2), normalization['history']))
        route = self.route[2](route_hidden).reshape(*shape, 4, 4)
        return {'delivery_logits': route, 'current_logits': risk[..., 0],
                'refresh_logits': risk[..., 1], 'risk_hidden': risk_hidden,
                'route_hidden': route_hidden}


def _delivery_loss(logits, target):
    expanded = logits.unsqueeze(2).expand(*target.shape, 4)
    return F.cross_entropy(expanded.reshape(-1, 4), target.reshape(-1))


def _loss(raw, current, refresh, delivery, after, kind):
    route = _delivery_loss(raw['delivery_logits'], delivery)
    cur = F.binary_cross_entropy_with_logits(raw['current_logits'], current)
    ref = F.binary_cross_entropy_with_logits(raw['refresh_logits'], refresh.mean(2))
    if kind == 'joint':
        post = F.binary_cross_entropy_with_logits(raw['after_logits'], after.mean(2))
        return route + (cur + ref + post)/3
    return route + .5*(cur+ref)


def fit(sensory, history, current_labels, refresh_labels, delivery_labels, after_labels, seed, kind):
    """Fixed full-batch development fit; no calibration or evaluation inputs."""
    sensory, history = _inputs(sensory, history, kind)
    labels = _labels(sensory.shape[:2], current_labels, refresh_labels, delivery_labels, after_labels)
    normalization = ({'joint': _normalizer(torch.cat((sensory.flatten(-2), history.flatten(-2)), -1))}
                     if kind == 'joint' else {'sensory': _normalizer(sensory),
                                             'history': _normalizer(history.flatten(-2))})
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = _OutcomeNetwork(kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=RATE, betas=(.9, .999), eps=1e-8,
                                 weight_decay=DECAY, amsgrad=False, foreach=False)
    trace = []
    with torch.enable_grad():
        for step in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(sensory, history, normalization), *labels, kind)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training objective')
            if step % 500 == 0:
                trace.append([step, float(loss.detach())])
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        final = _loss(model(sensory, history, normalization), *labels, kind)
    if not torch.isfinite(final):
        raise ValueError('nonfinite final objective')
    trace.append([STEPS, float(final)])
    return {'kind': kind, 'seed': seed, 'steps': STEPS, 'normalization': normalization,
            'state_dict': {k: v.detach().clone() for k, v in model.state_dict().items()},
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'loss_trace': trace, 'final_loss': final.detach().clone(), 'calibration': None}


@torch.no_grad()
def _raw(readout, sensory, history):
    sensory, history = _inputs(sensory, history, readout['kind'])
    with torch.random.fork_rng():
        model = _OutcomeNetwork(readout['kind'])
    model.load_state_dict(readout['state_dict'])
    model.eval()
    return {k: v.detach().clone() for k, v in model(sensory, history, readout['normalization']).items()}


def _fit_signed_calibrator(scores, targets):
    scores, targets = scores.flatten().double(), targets.flatten().double()
    if scores.shape != targets.shape or not torch.isfinite(scores).all():
        raise ValueError('calibration scores/targets mismatch or nonfinite')
    mean = scores.mean()
    scale = (scores-mean).square().mean().sqrt()
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    x = torch.stack((torch.ones_like(scores), (scores-mean)/scale), 1)
    beta = torch.zeros(2, dtype=torch.float64)
    penalty = CAL_L2*torch.eye(2, dtype=torch.float64)
    for _ in range(CAL_STEPS):
        p = (x@beta).sigmoid()
        gradient = x.T@(p-targets)/len(x)+penalty@beta
        hessian = x.T@((p*(1-p))[:, None]*x)/len(x)+penalty
        beta -= torch.linalg.solve(hessian, gradient)
    gradient = x.T@((x@beta).sigmoid()-targets)/len(x)+penalty@beta
    gradient_max = gradient.abs().max()
    if not torch.isfinite(beta).all() or float(gradient_max) > CAL_TOL:
        raise ValueError('signed outcome calibration failed its fixed gradient criterion')
    objective = (F.softplus(x@beta)-targets*(x@beta)).mean()+CAL_L2/2*beta.square().sum()
    return {'mean': mean.detach(), 'scale': scale.detach(), 'beta': beta.detach(),
            'baseline': targets.mean().detach(), 'gradient_max': gradient_max.detach(),
            'objective': objective.detach(), 'count': len(scores), 'steps': CAL_STEPS}


def _calibrated(scores, mapping):
    return (mapping['beta'][0]+mapping['beta'][1]*(scores.double()-mapping['mean'])/mapping['scale']).sigmoid()


@torch.no_grad()
def calibrate(readout, sensory, history, current_labels, refresh_labels, delivery_labels, after_labels):
    """Return a new readout; preserve all fitted network weights and statistics."""
    raw = _raw(readout, sensory, history)
    current, refresh, delivery, after = _labels(sensory.shape[:2], current_labels, refresh_labels, delivery_labels, after_labels)
    losses = [_delivery_loss(raw['delivery_logits'].double()/temp, delivery) for temp in TEMPERATURES]
    chosen = min(range(len(TEMPERATURES)), key=lambda i: float(losses[i]))
    calibration = {'delivery_temperature': TEMPERATURES[chosen],
                   'delivery_temperature_candidates': list(TEMPERATURES),
                   'delivery_temperature_losses': torch.stack(losses),
                   'current': _fit_signed_calibrator(raw['current_logits'], current),
                   'refresh': _fit_signed_calibrator(raw['refresh_logits'], refresh.double().mean(2))}
    if readout['kind'] == 'joint':
        calibration['after'] = _fit_signed_calibrator(raw['after_logits'], after.double().mean(2))
    result = copy.deepcopy(readout)
    result['calibration'] = calibration
    return result


@torch.no_grad()
def predict(readout, sensory, history):
    """Forecast outcomes from input evidence only, independently per state."""
    raw = _raw(readout, sensory, history)
    calibration = readout.get('calibration')
    if calibration is None:
        delivery = raw['delivery_logits'].double().softmax(-1)
        current = raw['current_logits'].double().sigmoid()
        refresh = raw['refresh_logits'].double().sigmoid()
    else:
        delivery = (raw['delivery_logits'].double()/calibration['delivery_temperature']).softmax(-1)
        current = _calibrated(raw['current_logits'], calibration['current'])
        refresh = _calibrated(raw['refresh_logits'], calibration['refresh'])
    if readout['kind'] == 'joint':
        after = (raw['after_logits'].double().sigmoid() if calibration is None
                 else _calibrated(raw['after_logits'], calibration['after']))
    else:
        after = delivery*refresh.unsqueeze(-2)+(1-delivery)*current.unsqueeze(-2)
    return {'delivery': delivery, 'current': current, 'refresh': refresh, 'after': after, **raw}
