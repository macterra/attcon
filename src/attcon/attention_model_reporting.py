"""Read the existing learned inspection model; no change to controller computation.

The authoritative state m is sigmoid(hidden_self_model_head(h)), consumed by the
policy_self_model_head. Model-preferred means argmax of that contribution alone.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .data import TaskConfig, generate_batch
from .models import ModelConfig, RecurrentAttentionController

SEEDS = (107, 207, 307)
FAMILIES = ('state', 'answer', 'scene', 'physical', 'hidden')
SPLITS = {'fit': (512, 4100), 'validation': (128, 5100), 'test': (256, 6100)}
STEPS = 6
INTERVENTION_STEP = 3


def fingerprint(*tensors):
    digest = hashlib.sha256()
    for t in tensors:
        t = t.detach().cpu().contiguous()
        digest.update(str((str(t.dtype), tuple(t.shape))).encode())
        digest.update(t.numpy().tobytes())
    return digest.hexdigest()


@dataclass
class Contexts:
    scene: torch.Tensor
    cues: torch.Tensor
    ids: list[str]

    def subset(self, indexes):
        indexes = torch.as_tensor(indexes).long()
        return Contexts(self.scene[indexes], self.cues[indexes], [self.ids[i] for i in indexes.tolist()])


def make_contexts(task: TaskConfig, seed: int, split: str, count=None):
    size, offset = SPLITS[split]
    size = size if count is None else count
    batch = generate_batch(size, STEPS, task, generator=torch.Generator().manual_seed(offset + seed))
    # Half switch; training and test directed cue combinations are disjoint.
    index = torch.arange(size)
    switching = index % 2 == 0
    source = torch.where(switching, (index // 2 % 2) * 2 + int(split == 'test'), index // 2 % 4)
    cues = source[:, None].expand(-1, STEPS).clone()
    cues[switching, 3:] = (source[switching, None] + 1) % 4
    ids = [fingerprint(scene, cue) for scene, cue in zip(batch.scene, cues)]
    if len(set(ids)) != size:
        raise ValueError('duplicate scene/schedule')
    return Contexts(batch.scene, cues, ids)


def pad(x):
    if x.shape[-1] > 128:
        raise ValueError('report feature exceeds capacity')
    return F.pad(x, (0, 128 - x.shape[-1]))


@torch.no_grad()
def capture(agent, contexts, override=None):
    """Capture state by a read-only hook; all decisions remain in the original model."""
    hidden = []
    handle = agent.hidden_self_model_head.register_forward_pre_hook(
        lambda module, args: hidden.append(args[0].detach().clone()))
    intervention = None if override is None else {'step': INTERVENTION_STEP, 'hidden_self_model_override': override}
    try:
        out = agent(contexts.scene, contexts.cues[:, 0], cue_seq=contexts.cues,
                    num_steps=STEPS, target=None, intervention=intervention)
    finally:
        handle.remove()
    # Native output converts an override through logit/sigmoid. This output is
    # authoritative at replay precision; causal comparison allows FP tolerance.
    m = out['hidden_self_model_seq']
    contribution = agent.policy_self_model_head(m)
    belief = m >= .5
    preference = contribution.argmax(-1)
    n, t, cells = m.shape
    previous_fixation = torch.zeros_like(m)
    previous_fixation[:, 1:] = F.one_hot(out['attention_seq'][:, :-1].argmax(-1), cells).float()
    previous_answer = torch.zeros_like(out['logits_seq'])
    previous_answer[:, 1:] = out['logits_seq'][:, :-1]
    physical = out['inspection_seq'] >= .5
    visible = contexts.scene[:, :, :agent.num_types].flatten(1)[:, None].expand(-1, t, -1)
    features = {
        'state': pad(m),
        'answer': pad(previous_answer),
        'scene': pad(torch.cat((visible, F.one_hot(contexts.cues, agent.num_types).float()), -1)),
        'physical': pad(torch.cat((physical.float(), previous_fixation), -1)),
        'hidden': pad(torch.stack(hidden, 1)),
    }
    return {'m': m, 'belief': belief, 'preference': preference, 'features': features,
            'physical': physical, 'attention': out['attention_seq'], 'contribution': contribution,
            'answer': out['logits_seq'], 'hidden': torch.stack(hidden, 1)}


def make_agent(config, state):
    agent = RecurrentAttentionController(TaskConfig.from_dict(config['task']), ModelConfig.from_dict(config['model']))
    agent.load_state_dict(state)
    return agent.eval().requires_grad_(False)


class Reporter(nn.Module):
    def __init__(self, mean=None, scale=None):
        super().__init__()
        self.register_buffer('mean', torch.zeros(128) if mean is None else mean)
        self.register_buffer('scale', torch.ones(128) if scale is None else scale)
        self.belief = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 25))
        self.preference = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 25))

    def forward(self, x):
        x = (x - self.mean) / self.scale
        return self.belief(x), self.preference(x)

    @torch.no_grad()
    def report(self, x):
        belief, preference = self(x)
        return {'belief': belief >= 0, 'preference': preference.argmax(-1)}


def per_case(pred, truth):
    correct = pred['belief'] == truth['belief']
    positive = truth['belief']
    preference = pred['preference'] == truth['preference']
    return {'belief_correct': correct, 'positive': positive, 'preference': preference,
            'exact_map': correct.all(-1), 'exact_report': correct.all(-1) & preference}


def scores(pred, truth):
    c = per_case(pred, truth)
    pos, neg = c['positive'], ~c['positive']
    recall = c['belief_correct'][pos].float().mean().item() if pos.any() else None
    specificity = c['belief_correct'][neg].float().mean().item() if neg.any() else None
    return {'preference_accuracy': c['preference'].float().mean().item(),
            'positive_recall': recall, 'negative_recall': specificity,
            'balanced_belief_accuracy': (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            'exact_map_accuracy': c['exact_map'].float().mean().item(),
            'exact_report_accuracy': c['exact_report'].float().mean().item(),
            'count': c['preference'].numel(), 'positive_cells': int(pos.sum()), 'negative_cells': int(neg.sum())}


def fit_reporter(train, validation, family, seed, steps=1000):
    torch.manual_seed(7100 + seed)
    x = train['features'][family].flatten(0, 1)
    y = train['belief'].flatten(0, 1).float()
    target = train['preference'].flatten()
    model = Reporter(x.mean(0), x.std(0).clamp_min(.01))
    opt = torch.optim.Adam(model.parameters(), lr=.003)
    best, best_score, candidates = None, None, []
    for step in range(1, steps + 1):
        belief, preference = model(x)
        loss = F.binary_cross_entropy_with_logits(belief, y) + F.cross_entropy(preference, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in (250, 500, 750, 1000) or step == steps:
            metric = scores(model.report(validation['features'][family]), validation)
            key = (min(metric['balanced_belief_accuracy'], metric['preference_accuracy']), metric['exact_report_accuracy'])
            candidates.append({'step': step, 'loss': loss.item(), 'validation': metric})
            if best_score is None or key > best_score:
                best_score = key
                best = {k: v.detach().clone() for k, v in model.state_dict().items()}
                selected = step
    model.load_state_dict(best)
    return model.eval().requires_grad_(False), {'selected_step': selected, 'candidates': candidates}


def renderer(belief, preference):
    cells = torch.where(torch.as_tensor(belief).bool())[0].tolist()
    return f'The attention model favors cell {int(preference)}. It represents cells {cells} as inspected.'


def parse_rendered(text):
    import ast
    import re
    match = re.fullmatch(r'The attention model favors cell (\d+)\. It represents cells (\[.*\]) as inspected\.', text)
    if match is None:
        raise ValueError('invalid report syntax')
    cell = int(match[1])
    cells = ast.literal_eval(match[2])
    if not 0 <= cell < 25 or not isinstance(cells, list) or any(type(c) is not int or not 0 <= c < 25 for c in cells):
        raise ValueError('invalid report content')
    if sorted(set(cells)) != cells:
        raise ValueError('duplicate or unordered cells')
    belief = torch.zeros(25, dtype=torch.bool)
    belief[cells] = True
    return {'belief': belief, 'preference': cell}


def scene_bootstrap(values, seed, draws=1000):
    values = torch.as_tensor(values).float()
    if values.ndim > 1:
        values = values.flatten(1).mean(1)
    generator = torch.Generator().manual_seed(8100 + seed)
    indexes = torch.randint(len(values), (draws, len(values)), generator=generator)
    boot = values[indexes].mean(1)
    return [float(torch.quantile(boot, .025)), float(torch.quantile(boot, .975))]


def fidelity_gates(metric):
    return {'preference': metric['preference_accuracy'] >= .95,
            'belief': metric['balanced_belief_accuracy'] is not None and metric['balanced_belief_accuracy'] >= .95,
            'exact_report': metric['exact_report_accuracy'] >= .90}
