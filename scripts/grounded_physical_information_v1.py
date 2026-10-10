"""Fresh physical-information diagnostic of frozen grounded outcome states.

All decoders are external instruments. Pixel statistics and stored-answer fidelity
are never clarity, awareness, appearance or task-correctness targets.
"""
import argparse
import copy
import datetime
import gzip
import hashlib
import json
from pathlib import Path

import torch

from grounded_attention_world_v1 import OrientationWorld
from grounded_outcome_data_v1 import FrozenSensor, render_locations
from grounded_outcome_models_v1 import predict
from orientation_transfer_diagnostic_v1 import checkpoint
from verify_isolated_orientation_task_v1 import compare

ROOT = Path('audits/grounded_physical_information_v1')
PROTOCOL = 'docs/GROUNDED_PHYSICAL_INFORMATION_V1_PROTOCOL.md'
SEEDS = (1301, 1311, 1321)
COUNTS = (512, 256, 512)
ALPHAS = (.001, .01, .1, 1., 10.)
BOOTSTRAPS, STEPS, FLOOR = 256, 8, 1e-6
PERM_ATOL, PERM_RTOL, DECODE_ROUNDOFF = 1e-6, 1e-5, 1e-12
TARGET_NAMES = ('image_value_std', 'nearest_neighbor_rms')
LOCAL_VIEWS = ('risk_hidden', 'sensory_hidden', 'outcomes_only',
               'rawlogit_outcomes', 'observed_only')
GLOBAL_VIEWS = ('joint_hidden', 'route_hidden')
VIEW_DIMS = {'risk_hidden': 32, 'joint_hidden': 64, 'sensory_hidden': 64,
             'outcomes_only': 2, 'rawlogit_outcomes': 3, 'route_hidden': 64,
             'observed_only': 1}
SOURCES = ('scripts/grounded_physical_information_v1.py',
           'tests/test_grounded_physical_information.py', PROTOCOL,
           'scripts/grounded_attention_world_v1.py', 'scripts/grounded_outcome_data_v1.py',
           'scripts/grounded_outcome_models_v1.py', 'scripts/isolated_orientation_task_v1.py',
           'scripts/orientation_transfer_diagnostic_v1.py',
           'scripts/verify_isolated_orientation_task_v1.py', 'src/attcon/bound_content.py',
           'audits/grounded_physical_information_v1/execution_wrapper.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def rng(seed):
    return torch.Generator().manual_seed(seed)


def seed_for(namespace, visual_seed, split=0):
    # Checkpoints are 10 apart, leaving 100000 between checkpoint blocks.
    return namespace + visual_seed*10000 + split*1000


def instrument_path(seed, kind):
    return Path(f'audits/grounded_attention_outcomes_v1/model-{seed}/instrument-{kind}.pt.gz')


def input_paths():
    return [p for seed in SEEDS for p in
            (checkpoint(seed), Path(f'audits/sensory_task_adapter_v1/states-{seed}.pt.gz'),
             instrument_path(seed, 'structured'), instrument_path(seed, 'joint'))]


def register():
    ROOT.mkdir(parents=True, exist_ok=True)
    required_review = (ROOT/'design_review/review.md', ROOT/'design_review/metadata.json')
    assert all(p.is_file() for p in required_review), 'design review required before registration'
    review = tuple(p for p in sorted((ROOT/'design_review').glob('*')) if p.is_file())
    assert Path(PROTOCOL).is_file(), 'protocol required before registration'
    assert not (ROOT/'attempt.json').exists(), 'attempt already reserved'
    assert not (ROOT/'source_snapshots.json').exists(), 'source snapshots already exist'
    snapshot = {str(p): Path(p).read_text() for p in SOURCES}
    manifest = {'kind': 'frozen_grounded_physical_information', 'visual_seeds': SEEDS,
                'counts_train_calibration_evaluation': COUNTS, 'commands': STEPS,
                'snapshot': STEPS, 'alphas': ALPHAS, 'bootstrap_worlds': BOOTSTRAPS,
                'normalization_rms_floor': FLOOR, 'targets': TARGET_NAMES,
                'permutation_source_tolerance': {'atol': PERM_ATOL, 'rtol': PERM_RTOL},
                'linear_decoder_roundoff_floor': DECODE_ROUNDOFF,
                'view_dimensions': VIEW_DIMS, 'shared_local_views': LOCAL_VIEWS,
                'per_location_global_views': GLOBAL_VIEWS,
                'seed_namespaces': {'world': 1300000000, 'commands': 1400000000,
                                    'noise': 1500000000, 'bootstrap': 1600000000},
                'source_sha256': {str(p): digest(p) for p in SOURCES},
                'frozen_input_sha256': {str(p): digest(p) for p in input_paths()},
                'design_review_sha256': {str(p): digest(p) for p in review},
                'torch_version': str(torch.__version__), 'torch_cpu_threads': 2,
                'intervention_permutation_atol': PERM_ATOL, 'intervention_permutation_rtol': PERM_RTOL,
                'appearance_targets': 0, 'new_reports': 0, 'source_retraining': False,
                'goal_achieved': False, 'no_retries_or_selection': True}
    # Exclusive manifest is the registration reservation; no overwritten freeze.
    dump(ROOT/'manifest.json', manifest, exclusive=True)
    dump(ROOT/'source_snapshots.json', snapshot, exclusive=True)


def check_manifest():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for path, expected in {**manifest['source_sha256'], **manifest['frozen_input_sha256'],
                           **manifest['design_review_sha256']}.items():
        assert digest(path) == expected, path
    snapshots = json.loads((ROOT/'source_snapshots.json').read_text())
    assert snapshots.keys() == manifest['source_sha256'].keys()
    for path, text in snapshots.items():
        assert hashlib.sha256(text.encode()).hexdigest() == manifest['source_sha256'][path], path
    assert str(torch.__version__) == manifest['torch_version'], 'torch version changed'
    assert torch.get_num_threads() == manifest['torch_cpu_threads'], 'thread count changed'
    return manifest


class Archive:
    def __init__(self, verify=False):
        self.verify, self.files = verify, {}

    def write(self, relative, value):
        path = ROOT/(relative+'.pt.gz')
        if self.verify:
            with gzip.open(path, 'rb') as stream:
                compare(torch.load(stream, weights_only=True), value, str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as raw:
                with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
                    torch.save(value, stream)
        assert path.stat().st_size < 100000000, f'archive exceeds 100MB: {path}'
        self.files[str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}


@torch.no_grad()
def physical_targets(pixels):
    """Pooled RGB/spatial population SD; valid spatial neighbor edges only.

    SD uses all 3x16x16 image values, not purely spatial luminance. Neighbor
    RMS averages horizontal and vertical mean squared differences equally,
    without padding. Channels are not differences/edges.
    """
    value = pixels.double()
    mean = value.mean((-3, -2, -1), keepdim=True)
    std = (value-mean).square().mean((-3, -2, -1)).sqrt()
    horizontal = (value[..., :, 1:]-value[..., :, :-1]).square().mean((-3, -2, -1))
    vertical = (value[..., 1:, :]-value[..., :-1, :]).square().mean((-3, -2, -1))
    return torch.stack((std, ((horizontal+vertical)/2).sqrt()), -1)


@torch.no_grad()
def collect(seed, split, sensor, archive, count=None):
    """Exactly eight actual random deliveries; no counterfactual or truth inputs."""
    count = COUNTS[split] if count is None else count
    world_seeds = [seed_for(1300000000, seed, split)+i for i in range(count)]
    sampled = [OrientationWorld.sample(s) for s in world_seeds]
    worlds = {k: torch.stack([getattr(w, k) for w in sampled])
              for k in ('labels', 'contrasts', 'mask', 'phase', 'command_mapping')}
    commands = torch.randint(4, (count, STEPS), generator=rng(seed_for(1400000000, seed, split)))
    delivered = worlds['command_mapping'].gather(1, commands)
    noise = .015*torch.randn(count, STEPS, 3, 16, 16,
                            generator=rng(seed_for(1500000000, seed, split)))
    pixels = render_locations(worlds, delivered, noise)
    hidden, logits = sensor(pixels)
    assert hidden.shape == (count, STEPS, 64) and logits.shape == (count, STEPS)
    assert torch.isfinite(hidden).all() and torch.isfinite(logits).all()
    sensory, history = torch.zeros(count, 4, 66), torch.zeros(count, 4, 4)
    last_pixels, last_step = torch.zeros(count, 4, 3, 16, 16), torch.full((count, 4), -1)
    rows = torch.arange(count)
    for step in range(STEPS):
        loc, cmd = delivered[:, step], commands[:, step]
        sensory[rows, loc, :64] = hidden[:, step]
        sensory[rows, loc, 64] = logits[:, step]
        sensory[rows, loc, 65] = 1
        history[rows, cmd] = 0
        history[rows, cmd, loc] = 1
        last_pixels[rows, loc] = pixels[:, step]
        last_step[rows, loc] = step
    seen = sensory[..., 65].bool()
    targets = physical_targets(last_pixels)
    # Unseen zeros are archive placeholders only; probe masks are mandatory.
    assert torch.equal(last_step >= 0, seen)
    state = {'sensory': sensory[:, None], 'history': history[:, None], 'seen': seen,
             'last_pixels': last_pixels, 'last_step': last_step, 'physical_targets': targets,
             'stored_answers': (sensory[..., 64] > 0).long(), 'snapshot': STEPS,
             'world_seeds': torch.tensor(world_seeds, dtype=torch.long)}
    archive('world_truth', worlds)
    archive('noise', noise)
    archive('pixels', pixels)
    archive('decoded', {'commands': commands, 'delivered': delivered,
                        'hidden': hidden, 'raw_logits': logits})
    archive('snapshot', state)
    return state, worlds


def load_frozen(seed, kind):
    # Only this known key is consumed. Archived forecasts/results are never inputs.
    with gzip.open(instrument_path(seed, kind), 'rb') as stream:
        instrument = torch.load(stream, weights_only=True)
    model = instrument['calibrated']
    assert model['kind'] == kind and model['calibration'] is not None
    return model


@torch.no_grad()
def features(state, primary, joint):
    memory = state['sensory'][:, 0]
    n = len(memory)
    values = {'risk_hidden': primary['risk_hidden'][:, 0],
              'joint_hidden': joint['joint_hidden'][:, 0, None].expand(n, 4, -1).clone(),
              'sensory_hidden': memory[..., :64].clone(),
              'outcomes_only': torch.stack((primary['current'][:, 0], primary['refresh'][:, 0]), -1),
              'rawlogit_outcomes': torch.stack((memory[..., 64], primary['current'][:, 0],
                                               primary['refresh'][:, 0]), -1),
              'route_hidden': primary['route_hidden'][:, 0, None].expand(n, 4, -1).clone(),
              'observed_only': memory[..., 65:66].clone()}
    assert set(values) == set(VIEW_DIMS)
    for name, value in values.items():
        assert value.shape == (n, 4, VIEW_DIMS[name]) and torch.isfinite(value).all(), name
    return values


def moments(x):
    mean = x.double().mean(0)
    rms = (x.double()-mean).square().mean(0).sqrt()
    return mean, torch.where(rms > FLOOR, rms, torch.ones_like(rms))


def ridge_candidates(train_x, train_y, train_answer, cal_x, cal_y):
    """Centered float64 ridge; alpha selection sees physical calibration only."""
    if not len(train_x) or not len(cal_x):
        raise ValueError('empty seen training/calibration decoder rows')
    xm, xs = moments(train_x)
    ym, ys = moments(train_y)
    x = (train_x.double()-xm)/xs
    y = (train_y.double()-ym)/ys
    answer = train_answer.double()
    am = answer.mean()
    outputs = torch.cat((y, (answer-am)[:, None]), -1)
    cx, cy = (cal_x.double()-xm)/xs, (cal_y.double()-ym)/ys
    gram, cross = x.T@x/len(x), x.T@outputs/len(x)
    weights, losses = [], []
    for alpha in ALPHAS:
        weight = torch.linalg.solve(gram+alpha*torch.eye(x.shape[-1], dtype=torch.float64), cross)
        weights.append(weight)
        losses.append(((cx@weight[:, :2]-cy).square().mean()).item())
    chosen = min(range(len(ALPHAS)), key=lambda i: losses[i])  # first tie
    return {'feature_mean': xm, 'feature_scale': xs, 'target_mean': ym,
            'target_scale': ys, 'answer_mean': am, 'weights': weights[chosen],
            'alpha': ALPHAS[chosen], 'alpha_candidates': list(ALPHAS),
            'calibration_standardized_physical_mse': losses,
            'train_seen_rows': len(train_x), 'calibration_seen_rows': len(cal_x),
            'input_dimensions': x.shape[-1], 'output_dimensions': 3,
            'physical_output_dimensions': 2, 'auxiliary_output': 'stored_raw_logit_sign', 'auxiliary_score_threshold': .5,
            'normalization_fit': 'train_seen_only'}


def fit_view(name, train, calibration, train_state, cal_state):
    seen, cs = train_state['seen'], cal_state['seen']
    y, cy = train_state['physical_targets'], cal_state['physical_targets']
    answer = train_state['stored_answers']
    if name in LOCAL_VIEWS:
        fits = [ridge_candidates(train[seen], y[seen], answer[seen], calibration[cs], cy[cs])]
        layout = 'shared_pooled_seen_rows'
    elif name in GLOBAL_VIEWS:
        fits = [ridge_candidates(train[:, loc][seen[:, loc]], y[:, loc][seen[:, loc]],
                                 answer[:, loc][seen[:, loc]], calibration[:, loc][cs[:, loc]],
                                 cy[:, loc][cs[:, loc]]) for loc in range(4)]
        layout = 'independent_decoder_per_location'
    else:
        raise ValueError('unknown view')
    return {'view': name, 'layout': layout, 'fits': fits,
            'decoder_count': len(fits), 'feature_dimensions': VIEW_DIMS[name],
            'parameter_count': sum(f['weights'].numel()+3 for f in fits)}


@torch.no_grad()
def decode(fit, x):
    models = fit['fits'] if len(fit['fits']) == 4 else fit['fits']*4
    output, answer, baseline = [], [], []
    for loc, model in enumerate(models):
        raw = ((x[:, loc].double()-model['feature_mean'])/model['feature_scale'])@model['weights']
        output.append(raw[:, :2]*model['target_scale']+model['target_mean'])
        answer.append(raw[:, 2]+model['answer_mean'])
        baseline.append(model['target_mean'].expand(len(x), -1))
    return {'physical': torch.stack(output, 1), 'answer_score': torch.stack(answer, 1),
            'answer': (torch.stack(answer, 1) > .5).long(),
            'train_mean_baseline': torch.stack(baseline, 1)}


def finite_list(x):
    return [float(v) if bool(torch.isfinite(v)) else None for v in x]


def interval(values):
    values = values.double()
    result, counts = [], []
    for col in values.T:
        valid = col[torch.isfinite(col)]
        counts.append(len(valid))
        result.append(torch.quantile(valid, torch.tensor([.025, .975], dtype=torch.float64)).tolist()
                      if len(valid) else None)
    return result, counts


def grouped_scores(state, decoded, mask, resamples):
    """Paired whole-world bootstrap, retaining all rows of each drawn world."""
    y, prediction = state['physical_targets'], decoded['physical']
    m = mask.double()
    counts = m.sum(0)
    square = (prediction-y).square()*m[..., None]
    absolute = (prediction-y).abs()*m[..., None]
    energy = (decoded['train_mean_baseline']-y).square()*m[..., None]
    correct = ((decoded['answer'] == state['stored_answers']).double()*m)

    def metrics(sq, ab, en, count, cr):
        total = count.sum(-1)
        pooled = {'r2_vs_train_mean': 1-sq.sum(-2)/en.sum(-2),
                  'mae': ab.sum(-2)/total[..., None],
                  'rmse': (sq.sum(-2)/total[..., None]).sqrt(),
                  'stored_answer_fidelity': cr.sum(-1)/total}
        local = {'r2_vs_train_mean': 1-sq/en, 'mae': ab/count[..., None],
                 'rmse': (sq/count[..., None]).sqrt(), 'stored_answer_fidelity': cr/count}
        # Undefined empty/constant location scores stay missing, including macros.
        def valid_mean(value, axis):
            good = torch.isfinite(value)
            return torch.where(good, value, 0).sum(axis)/good.sum(axis)
        macro = {k: valid_mean(v, -2 if k != 'stored_answer_fidelity' else -1)
                 for k, v in local.items()}
        return pooled, local, macro

    point = metrics(square.sum(0), absolute.sum(0), energy.sum(0), counts, correct.sum(0))
    boot = metrics(square[resamples].sum(1), absolute[resamples].sum(1),
                   energy[resamples].sum(1), m[resamples].sum(1), correct[resamples].sum(1))
    target_mean = (y*m[..., None]).sum((0, 1))/m.sum()
    target_var = ((y-target_mean).square()*m[..., None]).sum((0, 1))/m.sum()
    out = {'target_mean': finite_list(target_mean), 'target_population_variance': finite_list(target_var),
           'seen_rows': int(mask.sum()), 'worlds_with_seen_rows': int(mask.any(-1).sum()),
           'seen_rows_by_location': counts.long().tolist(), 'baseline': 'training_seen_target_mean',
           'bootstrap_world_count': resamples.shape[1], 'bootstrap_repetitions': len(resamples)}
    for i, label in enumerate(('pooled', 'per_location', 'macro_location')):
        scores = {}
        for metric, value in point[i].items():
            if i == 1:
                point_values = [finite_list(v.reshape(-1)) for v in value]
                bounds, valid = zip(*(interval(v.reshape(len(resamples), -1)) for v in boot[i][metric].unbind(1)))
                scores[metric] = {'value': point_values, 'conditional_world_bootstrap_95': list(bounds),
                                  'valid_bootstrap_counts': list(valid)}
            else:
                bounds, valid = interval(boot[i][metric].reshape(len(resamples), -1))
                scores[metric] = {'value': finite_list(value.reshape(-1)),
                                  'conditional_world_bootstrap_95': bounds, 'valid_bootstrap_counts': valid}
        out[label] = scores
    return out


def assess(state, world, decoded, resamples, current_success):
    seen = state['seen']
    risk_bin = torch.bucketize(current_success.double().contiguous(),
                              torch.tensor([.2, .4, .6, .8], dtype=torch.float64), right=True)
    return {'all_seen': grouped_scores(state, decoded, seen, resamples),
            'per_contrast': {str(c): grouped_scores(state, decoded, seen & (world['contrasts'] == c), resamples)
                             for c in range(6)},
            'per_orientation': {str(c): grouped_scores(state, decoded, seen & (world['labels'] == c), resamples)
                                for c in range(2)},
            'current_success_bin_edges': [0., .2, .4, .6, .8, 1.],
            'current_success_bins': {str(c): grouped_scores(state, decoded, seen & (risk_bin == c), resamples)
                                     for c in range(5)}}


def risk_associations(state, world, primary, predictions, resamples):
    """Descriptive associations/error contrasts; not a qualified dissociation."""
    seen = state['seen']
    risk = torch.stack((primary['current'][:, 0], primary['refresh'][:, 0],
                        state['sensory'][:, 0, :, 64].abs()), -1).double()
    y = state['physical_targets']
    def correlations(mask):
        x, target = risk[mask], y[mask]
        if len(x) < 2:
            return {'seen_rows': len(x), 'pearson': None}
        x, target = x-x.mean(0), target-target.mean(0)
        corr = x.T@target/(x.square().sum(0).sqrt()[:, None]*target.square().sum(0).sqrt()[None])
        return {'seen_rows': len(x), 'pearson': [finite_list(row) for row in corr]}
    differences = {}
    reference = (predictions['risk_hidden']['physical']-y).square()
    for name in ('outcomes_only', 'rawlogit_outcomes', 'sensory_hidden'):
        difference = ((predictions[name]['physical']-y).square()-reference)*seen[..., None]
        total = seen.sum()
        boot = difference[resamples].sum((1, 2))/seen[resamples].sum((1, 2))[:, None]
        bounds, counts = interval(boot)
        differences[name] = {'other_minus_risk_hidden_mse': finite_list(difference.sum((0, 1))/total),
                             'conditional_world_bootstrap_95': bounds, 'valid_bootstrap_counts': counts}
    return {'risk_columns': ['current_success', 'refresh_success', 'absolute_raw_logit'],
            'physical_columns': list(TARGET_NAMES), 'all_seen': correlations(seen),
            'per_contrast': {str(c): correlations(seen & (world['contrasts'] == c)) for c in range(6)},
            'per_orientation': {str(c): correlations(seen & (world['labels'] == c)) for c in range(2)},
            'paired_error_comparisons': differences,
            'qualifies_appearance_or_confidence_dissociation': False,
            'limitations': 'Associations and decoder error contrasts do not isolate appearance from decision uncertainty.'}


@torch.no_grad()
def intervention(primary_model, state, original, fits, original_features):
    """Primary only: native row readdressing and architectural forecast patch."""
    rotated_state = copy.deepcopy(state)
    rotated_state['sensory'] = state['sensory'].roll(1, -2)
    changed = predict(primary_model, rotated_state['sensory'], state['history'])
    restored_memory = rotated_state['sensory'].roll(-1, -2)
    restored = predict(primary_model, restored_memory, state['history'])
    checks = {'memory_restore_exact': torch.equal(restored_memory, state['sensory']),
              'delivery_equal_after_memory_readdressing': torch.equal(changed['delivery'], original['delivery']),
              'raw_content_permutation_exact': torch.equal(rotated_state['sensory'], state['sensory'].roll(1, -2)),
              'all_forecasts_restored_exact': all(torch.equal(restored[k], original[k]) for k in original)}
    permutation_errors = {}
    for name in ('risk_hidden', 'current', 'refresh', 'current_logits', 'refresh_logits'):
        axis = -2 if name == 'risk_hidden' else -1
        expected = original[name].roll(1, axis)
        checks[name+'_permutation_within_tolerance'] = torch.allclose(
            changed[name], expected, atol=PERM_ATOL, rtol=PERM_RTOL)
        permutation_errors[name] = float((changed[name]-expected).abs().max())
    # Build local features without constructing/predicting any joint intervention.
    local = {'risk_hidden': changed['risk_hidden'][:, 0],
             'sensory_hidden': rotated_state['sensory'][:, 0, :, :64],
             'outcomes_only': torch.stack((changed['current'][:, 0], changed['refresh'][:, 0]), -1),
             'rawlogit_outcomes': torch.stack((rotated_state['sensory'][:, 0, :, 64],
                                               changed['current'][:, 0], changed['refresh'][:, 0]), -1),
             'observed_only': rotated_state['sensory'][:, 0, :, 65:66]}
    decoded = {name: decode(fits[name], value) for name, value in local.items()}
    original_decoded = {name: decode(fits[name], original_features[name]) for name in LOCAL_VIEWS}
    decode_error_bounds, decode_error_summaries = {}, {}
    for name in LOCAL_VIEWS:
        expected_features = original_features[name].roll(1, 1)
        checks[name+'_features_permutation_within_tolerance'] = torch.allclose(
            local[name].double(), expected_features.double(), atol=PERM_ATOL, rtol=PERM_RTOL)
        delta_features = (local[name].double()-expected_features.double()).abs()
        model = fits[name]['fits'][0]  # shared local decoder only
        bound = (delta_features/model['feature_scale'])@model['weights'][:, :2].abs()*model['target_scale']
        delta_output = (decoded[name]['physical']-original_decoded[name]['physical'].roll(1, 1)).abs()
        checks[name+'_decoded_physical_permutation_within_propagation_bound'] = bool(
            (delta_output <= bound+DECODE_ROUNDOFF).all())
        decode_error_bounds[name] = {'absolute_feature_difference': delta_features,
                                    'absolute_output_difference': delta_output,
                                    'propagated_bound': bound}
        decode_error_summaries[name] = {'max_absolute_feature_difference': float(delta_features.max()),
                                       'max_absolute_physical_difference': float(delta_output.max()),
                                       'max_linear_propagation_bound': float(bound.max())}

    restored_local = {'risk_hidden': restored['risk_hidden'][:, 0],
                      'sensory_hidden': restored_memory[:, 0, :, :64],
                      'outcomes_only': torch.stack((restored['current'][:, 0], restored['refresh'][:, 0]), -1),
                      'rawlogit_outcomes': torch.stack((restored_memory[:, 0, :, 64],
                                                       restored['current'][:, 0], restored['refresh'][:, 0]), -1),
                      'observed_only': restored_memory[:, 0, :, 65:66]}
    restored_decoded = {name: decode(fits[name], value) for name, value in restored_local.items()}
    for name in LOCAL_VIEWS:
        checks[name+'_readout_restoration_exact'] = all(torch.equal(restored_decoded[name][key], value)
                                                       for key, value in original_decoded[name].items())
    patched = {k: v.clone() for k, v in original.items()}
    patched['delivery'] = original['delivery'].roll(1, -2)
    patched['after'] = patched['delivery']*patched['refresh'].unsqueeze(-2)+(1-patched['delivery'])*patched['current'].unsqueeze(-2)
    # No network re-inference; patching output rows cannot change upstream risk.
    for name in ('risk_hidden', 'current', 'refresh', 'current_logits', 'refresh_logits'):
        checks['delivery_patch_'+name+'_invariant_exact'] = torch.equal(patched[name], original[name])
    restored_patch = patched['delivery'].roll(-1, -2)
    restored_after = restored_patch*patched['refresh'].unsqueeze(-2)+(1-restored_patch)*patched['current'].unsqueeze(-2)
    checks['delivery_patch_restoration_exact'] = torch.equal(restored_patch, original['delivery']) and torch.equal(restored_after, original['after'])
    patch_local = dict(original_features)
    patch_local['risk_hidden'] = patched['risk_hidden'][:, 0]
    patch_local['outcomes_only'] = torch.stack((patched['current'][:, 0], patched['refresh'][:, 0]), -1)
    patch_local['rawlogit_outcomes'] = torch.stack((state['sensory'][:, 0, :, 64], patched['current'][:, 0], patched['refresh'][:, 0]), -1)
    patch_decoded = {name: decode(fits[name], patch_local[name]) for name in LOCAL_VIEWS}
    for name in LOCAL_VIEWS:
        checks[name+'_delivery_patch_decoded_physical_invariant_exact'] = torch.equal(
            patch_decoded[name]['physical'], original_decoded[name]['physical'])
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}
    summary = {'checks': checks, 'native_operation': 'all four sensory-location rows rolled +1 at fixed history',
               'patch_operation': 'delivery command rows rolled +1; after recomposed; no hidden-state re-inference',
               'joint_intervened': False, 'appearance_effect_tested': False,
               'synthetic_patched_source_not_natural_state': True,
               'current_memory_precedence': True, 'history_retains_actual_past': True,
               'source_precedence': 'Readdressed sensory memory and patched represented forecasts define current diagnostic contents; retained history describes actual past deliveries, not proof of current content placement or patched future outcomes.',
               'permutation_tolerance': {'atol': PERM_ATOL, 'rtol': PERM_RTOL},
               'raw_permutation_max_absolute_differences': permutation_errors,
               'decoded_permutation_errors': decode_error_summaries,
               'decoder_propagation_roundoff_floor': DECODE_ROUNDOFF,
               'source_change_max': float((state['sensory']-rotated_state['sensory']).abs().max()),
               'patched_after_change_max': float((original['after']-patched['after']).abs().max())}
    trace = {'original_sensory': state['sensory'], 'history': state['history'],
             'readdressed_sensory': rotated_state['sensory'], 'restored_sensory': restored_memory,
             'original': original, 'readdressed': changed, 'restored': restored,
             'readdressed_local_features': local, 'original_local_decoded': original_decoded,
             'readdressed_local_decoded': decoded, 'decode_permutation_error_bounds': decode_error_bounds, 'restored_local_decoded': restored_decoded,
             'patched_forecasts': patched,
             'patch_local_decoded': patch_decoded, 'restored_patch_delivery': restored_patch,
             'restored_patch_after': restored_after}
    return summary, trace


@torch.no_grad()
def build(archive):
    summaries = {}
    for seed in SEEDS:
        sensor = FrozenSensor(seed)
        models = {kind: load_frozen(seed, kind) for kind in ('structured', 'joint')}
        states, worlds, views, primary = [], [], [], []
        for split in range(3):
            prefix = f'model-{seed}/split-{split}/'
            state, world = collect(seed, split, sensor, lambda n, v: archive.write(prefix+n, v))
            forecasts = {kind: predict(model, state['sensory'], state['history']) for kind, model in models.items()}
            view = features(state, forecasts['structured'], forecasts['joint'])
            archive.write(prefix+'features', view)
            archive.write(prefix+'forecasts', forecasts)
            states.append(state); worlds.append(world); views.append(view); primary.append(forecasts['structured'])
        assert len(set(int(v) for s in states for v in s['world_seeds'])) == sum(COUNTS)
        resamples = torch.randint(COUNTS[2], (BOOTSTRAPS, COUNTS[2]), generator=rng(seed_for(1600000000, seed)))
        fits, estimates, scores = {}, {}, {}
        for name in VIEW_DIMS:
            fit = fit_view(name, views[0][name], views[1][name], states[0], states[1])
            prediction = decode(fit, views[2][name])
            scores[name] = {'layout': fit['layout'], 'decoder_count': fit['decoder_count'],
                            'input_dimensions': VIEW_DIMS[name], 'physical_outputs': 2, 'auxiliary_outputs': 1,
                            'parameter_count': fit['parameter_count'],
                            'train_seen_rows_per_decoder': [f['train_seen_rows'] for f in fit['fits']],
                            'calibration_seen_rows_per_decoder': [f['calibration_seen_rows'] for f in fit['fits']],
                            'chosen_alpha_per_decoder': [f['alpha'] for f in fit['fits']],
                            'assessment': assess(states[2], worlds[2], prediction, resamples, primary[2]['current'][:, 0])}
            fits[name], estimates[name] = fit, prediction
        causal, causal_trace = intervention(models['structured'], states[2], primary[2], fits, views[2])
        associations = risk_associations(states[2], worlds[2], primary[2], estimates, resamples)
        archive.write(f'model-{seed}/decoders', fits)
        archive.write(f'model-{seed}/evaluation_predictions', estimates)
        archive.write(f'model-{seed}/bootstrap_world_indices', resamples)
        archive.write(f'model-{seed}/interventions', causal_trace)
        summaries[str(seed)] = {'worlds_train_calibration_evaluation': list(COUNTS),
                                'seen_rows_by_split': [int(s['seen'].sum()) for s in states],
                                'seen_rows_by_location_by_split': [s['seen'].sum(0).tolist() for s in states],
                                'readouts': scores, 'interventions': causal, 'risk_associations': associations}
        print(f'completed physical diagnostic {seed}', flush=True)
    return {'kind': 'frozen_grounded_physical_information_descriptive', 'models': summaries,
            'targets': list(TARGET_NAMES), 'appearance_tested': False, 'new_reports': 0,
            'participant_data_used': False, 'source_retraining': False, 'qualia_established': False,
            'consciousness_related_correspondence_established': False, 'goal_achieved': False,
            'qualification_gate': None,
            'interval_scope': '256 world-cluster bootstrap draws conditional on frozen checkpoints and fitted decoders; no model-population claim',
            'view_budget_limit': 'Local views use shared pooled-seen decoders; global views use four location decoders. Budgets and dimensions differ.'}


def execute(verify=False):
    check_manifest()
    if verify:
        completion = json.loads((ROOT/'completion.json').read_text())
        assert digest(ROOT/'results.json') == completion['results_sha256']
        for p, entry in completion['archive_files'].items():
            assert digest(p) == entry['sha256'], p
    else:
        assert not (ROOT/'results.json').exists() and not (ROOT/'completion.json').exists()
        dump(ROOT/'attempt.json', {'status': 'reserved', 'no_retry': True,
                                   'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}, exclusive=True)
    archive = Archive(verify)
    try:
        results = build(archive)
        if verify:
            assert results == json.loads((ROOT/'results.json').read_text()), 'summary replay mismatch'
            assert archive.files == completion['archive_files'], 'archive replay mismatch'
            dump(ROOT/'verification.json', {'status': 'exact_collection_features_decoder_selection_interventions_results_replay',
                                           'archive_count': len(archive.files), 'all_files_verified': True,
                                           'manifest_sha256': digest(ROOT/'manifest.json'),
                                           'results_sha256': digest(ROOT/'results.json')})
        else:
            dump(ROOT/'results.json', results, exclusive=True)
            dump(ROOT/'completion.json', {'status': 'single_registered_attempt_completed',
                                         'archive_files': archive.files, 'manifest_sha256': digest(ROOT/'manifest.json'),
                                         'results_sha256': digest(ROOT/'results.json')}, exclusive=True)
    except BaseException as error:
        if not verify:
            dump(ROOT/'failure.json', {'status': 'terminal_failure_no_retry', 'error_type': type(error).__name__,
                                       'error': str(error), 'archive_files': archive.files}, exclusive=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('register', 'run', 'verify'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == 'register':
        register()
    else:
        execute(verify=args.stage == 'verify')


if __name__ == '__main__':
    main()
