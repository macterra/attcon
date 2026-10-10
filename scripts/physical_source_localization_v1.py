"""Frozen physical-only diagnostic; never labels pixels as subjective appearance."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from attcon.bound_content import VisualEncoder, bind, scenes
from attcon.functional_controls import simulate_controls, last_forecast
from attcon.functional_model import buffer_contents
from attcon.predictive_attention import PredictiveAttention

ROOT = Path('audits/physical_source_localization_v1')
PAIRS = ((1301, 2011), (1311, 2021), (1321, 2031))
COUNT, RIDGE, BOOTSTRAPS = 128, .01, 256
VARIANTS = tuple((scale, blur) for scale in (.70, .85, 1.) for blur in (False, True))
SOURCES = ('scripts/physical_source_localization_v1.py',
    'docs/PHYSICAL_SOURCE_LOCALIZATION_V1_PROTOCOL.md', 'tests/test_physical_source_localization.py',
    'src/attcon/bound_content.py', 'src/attcon/functional_model.py',
    'src/attcon/functional_controls.py', 'src/attcon/predictive_attention.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_paths(visual, attention):
    return (Path(f'audits/bound_content/confirmation_v2/seed{visual}.pt'),
            Path(f'audits/functional_controls_v1/seed{attention}.pt'))


def prepare():
    if (ROOT/'manifest.json').exists():
        raise SystemExit('refusing to overwrite registration')
    ROOT.mkdir(parents=True, exist_ok=True)
    checkpoints = [p for pair in PAIRS for p in checkpoint_paths(*pair)]
    manifest = {'kind': 'physical_only_source_localization_development',
        'pairs': PAIRS, 'scenes_per_split_per_pair': COUNT, 'variants': VARIANTS,
        'targets': ['pixel_mean', 'neighbor_squared_difference'],
        'ridge': RIDGE, 'bootstrap_count': BOOTSTRAPS, 'theory_gate': None,
        'participant_data_used': False, 'new_reports': 0, 'new_training': False,
        'source_sha256': {p: digest(p) for p in SOURCES},
        'checkpoint_sha256': {str(p): digest(p) for p in checkpoints},
        'torch_version': torch.__version__, 'no_retries_or_selection': True}
    (ROOT/'source_snapshots.json').write_text(json.dumps(
        {p: Path(p).read_text() for p in SOURCES}, indent=2)+'\n')
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Physical diagnostic registered; no model output collected')


def check_manifest():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for p, h in {**manifest['source_sha256'], **manifest['checkpoint_sha256']}.items():
        assert digest(p) == h, p
    snapshots = json.loads((ROOT/'source_snapshots.json').read_text())
    for p, h in manifest['source_sha256'].items():
        assert hashlib.sha256(snapshots[p].encode()).hexdigest() == h, p
    return manifest


def transform(patches, scale, blur):
    x = patches.reshape(-1, 3, 16, 16) * scale
    if blur:
        points = torch.arange(-2, 3, dtype=x.dtype)
        kernel = torch.exp(-points.square()/(2*.7**2))
        kernel = kernel/kernel.sum()
        weight = (kernel[:, None]*kernel[None, :])[None, None].expand(3, 1, 5, 5)
        x = F.conv2d(F.pad(x, (2, 2, 2, 2), mode='replicate'), weight, groups=3)
    return x.reshape_as(patches)


def physical_targets(patches):
    mean = patches.mean((-3, -2, -1))
    edge = ((patches[..., 1:, :]-patches[..., :-1, :]).square().mean((-3,-2,-1)) +
            (patches[..., :, 1:]-patches[..., :, :-1]).square().mean((-3,-2,-1)))/2
    return torch.stack((mean, edge), -1)


def centered(x):
    # Shape: [base scene, variant, view, object, feature].
    value = x.double()
    return value-value.mean(1, keepdim=True)


def fit_readout(features, targets):
    x, y = centered(features).flatten(0, -2), centered(targets).flatten(0, -2)
    scale = x.square().mean(0).sqrt()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    x = x/scale
    weights = torch.linalg.solve(x.T@x/len(x) + RIDGE*torch.eye(x.shape[-1], dtype=x.dtype),
                                 x.T@y/len(x))
    return {'scale': scale, 'weights': weights}


def evaluate_readout(fit, features, targets, resamples):
    y = centered(targets)
    prediction = centered(features)/fit['scale'] @ fit['weights']
    error = (prediction-y).square().sum((1,2,3))
    energy = y.square().sum((1,2,3))
    assert bool((energy.sum(0) > 0).all())
    r2 = 1-error.sum(0)/energy.sum(0)
    bootstrap = 1-error[resamples].sum(1)/energy[resamples].sum(1)
    return {'r2': r2.tolist(), 'conditional_scene_bootstrap_95':
            torch.quantile(bootstrap, torch.tensor([.025,.975],dtype=bootstrap.dtype), dim=0).T.tolist()}, prediction


def forecast_vector(current):
    return torch.cat([x.flatten(1) for x in
                      (current.allocation, current.access, current.effects)], -1)


@torch.no_grad()
def collect_split(encoder, model, visual_seed, attention_seed, split):
    scene_seed = 860000000+visual_seed+10000*split
    process_seed = 870000000+attention_seed+10000*split
    patches, colors, shapes = scenes(scene_seed, COUNT)
    process = simulate_controls(process_seed, COUNT, 12, condition=0)
    values, targets, categories, hidden_states, heads = {}, [], [], [], []
    current = None
    for scale, blur in VARIANTS:
        images = transform(patches, scale, blur)
        visual_hidden = encoder.net[:-1](images.reshape(-1,3,16,16)).reshape(COUNT,2,4,64)
        visual = encoder(images)
        sequence, hidden = model(process.observations)
        current = last_forecast(sequence)
        bound = bind(encoder, images, current, process.allocation)
        q = current.access[:,0]
        buffer = buffer_contents(visual, q)
        maxprob = torch.stack((visual[...,:4].max(-1).values,
                               visual[...,4:].max(-1).values), -1)
        entropy = torch.stack((-(visual[...,:4]*visual[...,:4].clamp_min(1e-30).log()).sum(-1),
                              -(visual[...,4:]*visual[...,4:].clamp_min(1e-30).log()).sum(-1)), -1)
        features = {'visual_hidden': visual_hidden, 'category_probabilities': visual,
            'category_confidence': torch.cat((maxprob,entropy),-1),
            'bound_contents': bound.contents, 'buffered_contents': buffer,
            'category_q_interaction': torch.cat((visual,q[...,None],visual*q[...,None]),-1),
            'attention_hidden': hidden[0,:,None,None,:].expand(-1,2,4,-1),
            'attention_forecast': forecast_vector(current)[:,None,None,:].expand(-1,2,4,-1)}
        for key, value in features.items(): values.setdefault(key,[]).append(value.clone())
        targets.append(physical_targets(images))
        categories.append(torch.stack((visual[...,:4].argmax(-1),visual[...,4:].argmax(-1)),-1))
        hidden_states.append(hidden.clone()); heads.append(forecast_vector(current))
        baseline = features['category_q_interaction']
        reconstructed = baseline[...,9:] + .25*(1-baseline[...,8:9])
        assert torch.equal(reconstructed, buffer), 'deterministic bridge changed'
    features = {k:torch.stack(v,1) for k,v in values.items()}
    target = torch.stack(targets,1)
    predicted_categories = torch.stack(categories,1)
    truth = torch.stack((colors,shapes),-1)[:,None]
    invariance = all(torch.equal(hidden_states[0],v) for v in hidden_states)
    invariance = invariance and all(torch.equal(heads[0],v) for v in heads)
    assert invariance, 'pixel intervention changed identical-history attention state'
    different = process.observations.clone(); different[:,:,8:16] = 0
    different_sequence, different_hidden = model(different)
    attention_control = float((different_hidden-hidden_states[0]).abs().max())
    assert attention_control > 0, 'attention extractor is unresponsive'
    # Last variant; physical content stays fixed during model-state interventions.
    original = buffer_contents(visual, current.access[:,0])
    altered = current.intervene('access', current.access*.5)
    changed = buffer_contents(visual, altered.access[:,0])
    restored = altered.intervene('access',current.access)
    assert torch.equal(buffer_contents(visual,restored.access[:,0]),original)
    assert torch.equal(current.allocation,altered.allocation)
    assert torch.equal(current.effects,altered.effects)
    rolled = bound.replace_binding(bound.binding.roll(1,-2))
    assert torch.equal(rolled.replace_binding(bound.binding).contents,bound.contents)
    assert torch.equal(rolled.visual,bound.visual)
    assert bool((changed != original).any()), 'access intervention has no measured effect'
    assert bool((rolled.contents != bound.contents).any()), 'binding intervention has no measured effect'
    assert bool((centered(target).square().sum((0,1,2,3)) > 0).all()), 'physical targets do not vary'
    confidence = features['category_confidence'][...,:2]
    checks = {'attention_state_exactly_invariant_across_pixels': invariance,
        'attention_hidden_change_when_acquisition_zeroed_max': attention_control,
        'forecast_change_when_acquisition_zeroed_max': float((forecast_vector(last_forecast(different_sequence))-heads[0]).abs().max()),
        'bridge_reconstruction_exact': True, 'access_restoration_exact': True,
        'binding_restoration_exact': True,
        'access_attenuation_buffer_change_max': float((changed-original).abs().max()),
        'binding_roll_content_change_max': float((rolled.contents-bound.contents).abs().max()),
        'bound_known_fraction': float(bound.known.double().mean()),
        'category_joint_accuracy_by_variant': (predicted_categories==truth).all(-1).double().mean((0,2,3)).tolist(),
        'category_argmax_retained_vs_unblurred_scale1_by_variant': (predicted_categories==predicted_categories[:,4:5]).all(-1).double().mean((0,2,3)).tolist(),
        'mean_category_max_probability_by_variant': confidence.double().mean((0,2,3)).tolist(),
        'max_category_probability_change_within_object': float((confidence-confidence[:,4:5]).abs().max()),
        'physical_within_object_target_energy': centered(target).square().sum((0,1,2,3)).tolist(),
        'scene_seed': scene_seed, 'process_seed': process_seed}
    return {'features': features, 'targets': target, 'checks': checks,
        'category_predictions': predicted_categories, 'category_truth': truth,
        'source_inputs': {'base_patches':patches, 'process_observations':process.observations,
            'observed_allocation':process.allocation},
        'last_variant_interventions': {'visual':visual, 'attention_hidden':hidden_states[-1],
            'allocation':current.allocation, 'effects':current.effects,
            'access':current.access, 'attenuated_access':altered.access,
            'restored_access':restored.access, 'buffer_before':original,
            'buffer_attenuated':changed, 'buffer_restored':buffer_contents(visual,restored.access[:,0]),
            'bound_visual':bound.visual, 'remembered':bound.remembered,
            'binding':bound.binding, 'rolled_binding':rolled.binding,
            'restored_binding':rolled.replace_binding(bound.binding).binding}}


@torch.no_grad()
def build():
    trace, summary = {}, {}
    for visual_seed, attention_seed in PAIRS:
        encoder, model = VisualEncoder().eval(), PredictiveAttention().eval()
        visual_path, attention_path = checkpoint_paths(visual_seed,attention_seed)
        encoder.load_state_dict(torch.load(visual_path,weights_only=True)['state_dict'])
        model.load_state_dict(torch.load(attention_path,weights_only=True)['state_dict'])
        dev, evaluation = [collect_split(encoder,model,visual_seed,attention_seed,i) for i in (0,1)]
        assert dev['checks']['scene_seed'] != evaluation['checks']['scene_seed']
        resamples = torch.randint(COUNT,(BOOTSTRAPS,COUNT),
            generator=torch.Generator().manual_seed(880000000+attention_seed))
        fits, estimates, scores = {}, {}, {}
        for key, features in dev['features'].items():
            fit = fit_readout(features,dev['targets'])
            score, predictions = evaluate_readout(fit,evaluation['features'][key],evaluation['targets'],resamples)
            fits[key], estimates[key], scores[key] = fit, predictions, score
        pair = f'{visual_seed}_{attention_seed}'
        trace[pair] = {'development':dev,'evaluation':evaluation,'fits':fits,
                       'predictions':estimates,'bootstrap_scene_indices':resamples}
        summary[pair] = {'development_checks':dev['checks'],
                         'evaluation_checks':evaluation['checks'],'readout_scores':scores}
    return trace, {'kind':'physical_only_descriptive_results','pairs':summary,
        'targets':['pixel_mean','neighbor_squared_difference'],
        'goal_achieved':False,'appearance_judgments_tested':False,
        'new_reports':0,'participant_data_used':False}


def compare(saved, current, path='root'):
    if isinstance(saved,torch.Tensor):
        assert torch.equal(saved,current),path
    elif isinstance(saved,dict):
        assert saved.keys()==current.keys(),path
        for key in saved: compare(saved[key],current[key],path+'/'+str(key))
    else: assert saved==current,path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=('prepare','run','verify'))
    args=parser.parse_args(); torch.set_num_threads(2)
    if args.stage=='prepare': prepare(); return
    check_manifest()
    if args.stage=='run':
        if any((ROOT/name).exists() for name in ('attempt.json','states.pt.gz','results.json')):
            raise SystemExit('refusing to retry or overwrite frozen attempt')
        (ROOT/'attempt.json').write_text(json.dumps({'status':'reserved','no_retry':True})+'\n')
        trace, summary=build()
        with (ROOT/'states.pt.gz').open('wb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as stream:
                torch.save(trace,stream)
        (ROOT/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
        (ROOT/'completion.json').write_text(json.dumps({'status':'completed',
            'states_sha256':digest(ROOT/'states.pt.gz'),'results_sha256':digest(ROOT/'results.json'),
            'goal_achieved':False},indent=2)+'\n')
        print(json.dumps(summary,indent=2))
    else:
        completion=json.loads((ROOT/'completion.json').read_text())
        assert digest(ROOT/'states.pt.gz')==completion['states_sha256']
        assert digest(ROOT/'results.json')==completion['results_sha256']
        trace,summary=build()
        with gzip.open(ROOT/'states.pt.gz','rb') as stream:
            compare(torch.load(stream,weights_only=True),trace)
        assert json.loads((ROOT/'results.json').read_text())==summary
        print('Exact feature/target/coefficient/summary replay and source/checkpoint hashes verified')


if __name__=='__main__': main()
