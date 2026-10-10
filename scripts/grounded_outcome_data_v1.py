"""Fresh executed outcomes; truth is separate from pre-command model inputs."""
import gzip
from pathlib import Path
import torch
from attcon.bound_content import VisualEncoder
from grounded_attention_world_v1 import OrientationWorld
from orientation_transfer_diagnostic_v1 import checkpoint, LUMINANCE

COUNTS = (512, 256, 512)
SNAPSHOTS = (0, 2, 4, 8)
REPEATS = (2, 2, 8)
CHUNK = 256


def generator(seed):
    return torch.Generator().manual_seed(seed)


def world_batch(seed, split, count):
    first = 960000000 + seed*1000 + split*100000
    worlds = [OrientationWorld.sample(first+i) for i in range(count)]
    return {name: torch.stack([getattr(w, name) for w in worlds])
            for name in ('labels', 'contrasts', 'mask', 'phase', 'command_mapping')}


def render_locations(worlds, locations, noise):
    """Render actual delivered locations only, with caller-supplied independent noise."""
    n = len(worlds['labels'])
    assert locations.shape[0] == n and noise.shape == (*locations.shape, 3, 16, 16)
    rows = torch.arange(n).reshape(n, *([1]*(locations.ndim-1)))
    labels = worlds['labels'][rows, locations]
    phase = worlds['phase'][rows, locations]
    mask = worlds['mask'][rows, locations]
    contrasts = worlds['contrasts'][rows, locations]
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing='ij')
    axis = torch.where(labels[..., None, None] == 0, yy, xx)
    texture = torch.sin(axis*torch.pi/2+phase) >= 0
    levels = (torch.tensor(LUMINANCE)/50)[contrasts]
    target = torch.where(texture, levels[..., 0, None, None], levels[..., 1, None, None])
    image = torch.where(mask, target, torch.tensor(35.5/50))
    return (image[..., None, :, :].expand(*locations.shape, 3, 16, 16)+noise).clamp(0, 1)


class FrozenSensor:
    """Only checkpoint weights and frozen raw orientation readout; no calibrators."""
    def __init__(self, seed):
        with torch.random.fork_rng():
            self.encoder = VisualEncoder().eval()
            self.readout = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).eval()
        self.encoder.load_state_dict(torch.load(checkpoint(seed), weights_only=True)['state_dict'])
        with gzip.open(Path(f'audits/sensory_task_adapter_v1/states-{seed}.pt.gz'), 'rb') as stream:
            frozen = torch.load(stream, weights_only=True)['instruments']['visual_hidden']['readout']
        self.readout.load_state_dict(frozen['state_dict'])
        self.mean, self.scale = frozen['mean'].clone(), frozen['scale'].clone()
        for model in (self.encoder, self.readout):
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, pixels):
        shape = pixels.shape[:-3]
        flat = pixels.reshape(-1, 3, 16, 16)
        hidden, logits = [], []
        for chunk in flat.split(CHUNK):
            h = self.encoder.net[:-1](chunk)
            logit = self.readout((h-self.mean)/self.scale)[:, 0]
            hidden.append(h); logits.append(logit)
        return torch.cat(hidden).reshape(*shape, 64), torch.cat(logits).reshape(shape)


def record_state(memory, history, labels):
    # Export evidence first; evaluator truth never enters these two tensors.
    sensory = memory.clone()
    route_history = history.clone()
    correctness = ((sensory[..., 64] > 0).long() == labels).clone()
    return sensory, route_history, correctness


@torch.no_grad()
def collect(seed, split, sensor, archive, count=None, repeats=None):
    count = COUNTS[split] if count is None else count
    repeats = REPEATS[split] if repeats is None else repeats
    worlds = world_batch(seed, split, count)
    base = seed*1000 + split*100000
    commands = torch.randint(4, (count, 8), generator=generator(970000000+base))
    delivered = worlds['command_mapping'].gather(1, commands)
    noise = .015*torch.randn(count, 8, 3, 16, 16, generator=generator(980000000+base))
    pixels = render_locations(worlds, delivered, noise)
    hidden, logits = sensor(pixels)
    archive('history-noise', noise); archive('history-pixels', pixels)
    archive('history-decoded', {'commands': commands, 'delivered': delivered, 'hidden': hidden, 'logits': logits})
    archive('world-truth', worlds)
    del noise, pixels
    memory = torch.zeros(count, 4, 66)
    history = torch.zeros(count, 4, 4)
    rows = torch.arange(count)
    out = {key: [] for key in ('sensory', 'history', 'current_labels', 'refresh_labels', 'delivery_labels', 'after_labels')}
    for time in range(9):
        if time in SNAPSHOTS:
            snapshot = SNAPSHOTS.index(time)
            sensory, route, correct = record_state(memory, history, worlds['labels'])
            out['sensory'].append(sensory); out['history'].append(route); out['current_labels'].append(correct)
            branch_delivered = worlds['command_mapping'][:, None, :].expand(count, repeats, 4).clone()
            branch_noise = .015*torch.randn(count, repeats, 4, 3, 16, 16,
                generator=generator(990000000+base+snapshot*1000))
            branch_pixels = render_locations(worlds, branch_delivered, branch_noise)
            branch_hidden, branch_logits = sensor(branch_pixels)
            branch_correct = ((branch_logits > 0).long() == worlds['labels'][rows[:, None, None], branch_delivered])
            refresh = torch.empty(count, repeats, 4, dtype=torch.bool)
            refresh.scatter_(2, branch_delivered, branch_correct)
            after = correct[:, None, None, :].expand(count, repeats, 4, 4).clone()
            after.scatter_(3, branch_delivered[..., None], branch_correct[..., None])
            out['refresh_labels'].append(refresh); out['delivery_labels'].append(branch_delivered); out['after_labels'].append(after)
            archive(f'branch-{time}-noise', branch_noise); archive(f'branch-{time}-pixels', branch_pixels)
            archive(f'branch-{time}-decoded', {'hidden': branch_hidden, 'logits': branch_logits,
                'delivered': branch_delivered, 'refresh_correct': refresh, 'after_correct': after})
            del branch_noise, branch_pixels, branch_hidden, branch_logits
        if time < 8:
            loc = delivered[:, time]
            memory[rows, loc, :64] = hidden[:, time]
            memory[rows, loc, 64] = logits[:, time]
            memory[rows, loc, 65] = 1
            cmd = commands[:, time]
            history[rows, cmd] = 0
            history[rows, cmd, loc] = 1
    data = {key: torch.stack(values, 1) for key, values in out.items()}
    data['snapshots'] = list(SNAPSHOTS)
    archive('dataset', data)
    return data, worlds


@torch.no_grad()
def remap_data(seed, evaluation, worlds, sensor, archive):
    """Physical remap; all four chronological deliveries retained, no weight fitting."""
    count = 64
    worlds = {key: value[:count].clone() for key, value in worlds.items()}
    worlds['command_mapping'] = (worlds['command_mapping']+1) % 4
    memory = evaluation['sensory'][:count, -1].clone()
    history = evaluation['history'][:count, -1].clone()
    original = {'sensory': memory[:, None].clone(), 'history': history[:, None].clone()}
    states, routes = [memory.clone()], [history.clone()]
    noises = .015*torch.randn(count, 4, 3, 16, 16, generator=generator(1000000000+seed))
    locations = worlds['command_mapping'].clone()
    pixels = render_locations(worlds, locations, noises)
    hidden, logits = sensor(pixels)
    rows = torch.arange(count)
    for cmd in range(4):
        loc = locations[:, cmd]
        memory[rows, loc, :64] = hidden[:, cmd]
        memory[rows, loc, 64] = logits[:, cmd]
        memory[rows, loc, 65] = 1
        history[:, cmd] = 0
        history[rows, cmd, loc] = 1
        states.append(memory.clone()); routes.append(history.clone())
    result = {'sensory': torch.stack(states, 1), 'history': torch.stack(routes, 1),
        'delivery_labels': locations[:, None].expand(count, 5, 4).clone(), 'restored': original}
    archive('remap-noise', noises); archive('remap-pixels', pixels)
    archive('remap-decoded', {'hidden': hidden, 'logits': logits, 'locations': locations})
    archive('remap-states', result)
    return result
