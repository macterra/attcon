"""Independent pre-output calibrated toy fixture; no checkpoints or worlds.

Original reviewed implementation e509d487... uses bitwise permutation gates.
This fixture measures raw gaps and decoder propagation before invoking those gates.
Toy physical outputs use random sensory coordinates, not human/appearance data.
"""
import hashlib
import json
from pathlib import Path
import torch
import grounded_physical_information_v1 as study
from test_grounded_physical_information import fixture_model
from grounded_outcome_models_v1 import predict

torch.set_num_threads(2)
generator = torch.Generator().manual_seed(44019)
memory = torch.randn(16, 1, 4, 66, generator=generator)
memory[..., 65] = 1
history = torch.eye(4)[None, None].expand(16, 1, -1, -1).clone()
state = {'sensory': memory, 'history': history,
         'seen': torch.ones(16, 4, dtype=torch.bool),
         'physical_targets': torch.stack((memory[:, 0, :, 0], memory[:, 0, :, 1]), -1).double(),
         'stored_answers': (memory[:, 0, :, 64] > 0).long()}


def calibration():
    return {'mean': torch.tensor(.2, dtype=torch.float64),
            'scale': torch.tensor(1.7, dtype=torch.float64),
            'beta': torch.tensor([-.3, 1.2], dtype=torch.float64)}


primary_model, joint_model = fixture_model('structured'), fixture_model('joint')
primary_model['calibration'] = {'delivery_temperature': 2.,
                              'current': calibration(), 'refresh': calibration()}
joint_model['calibration'] = {'delivery_temperature': 2., 'current': calibration(),
                            'refresh': calibration(), 'after': calibration()}
primary_model['normalization']['sensory']['mean'] = torch.linspace(-.2, .3, 66)
primary_model['normalization']['sensory']['scale'] = torch.linspace(.5, 1.5, 66)
original = predict(primary_model, memory, history)
joint = predict(joint_model, memory, history)
changed = predict(primary_model, memory.roll(1, -2), history)
restored = predict(primary_model, memory.roll(1, -2).roll(-1, -2), history)
features = study.features(state, original, joint)
fits = {name: study.fit_view(name, value, value, state, state)
        for name, value in features.items()}
changed_local = {
    'risk_hidden': changed['risk_hidden'][:, 0],
    'sensory_hidden': memory.roll(1, -2)[:, 0, :, :64],
    'outcomes_only': torch.stack((changed['current'][:, 0], changed['refresh'][:, 0]), -1),
    'rawlogit_outcomes': torch.stack((memory.roll(1, -2)[:, 0, :, 64],
                                     changed['current'][:, 0], changed['refresh'][:, 0]), -1),
    'observed_only': memory.roll(1, -2)[:, 0, :, 65:66]}
receipt = {'fixture': 'calibrated_shared_nonidentity_normalizer',
           'source_sha256': hashlib.sha256(Path('scripts/grounded_physical_information_v1.py').read_bytes()).hexdigest(),
           'production_checkpoint_loads': 0, 'worlds_generated': 0,
           'raw_permutation': {}, 'decoded_permutation': {}}
for name in ('risk_hidden', 'current_logits', 'refresh_logits', 'current', 'refresh'):
    axis = -2 if name == 'risk_hidden' else -1
    expected = original[name].roll(1, axis)
    delta = (changed[name]-expected).abs()
    receipt['raw_permutation'][name] = {
        'max_absolute_gap': float(delta.max()),
        'max_relative_gap_nonzero': float((delta.double()/expected.abs().double().clamp_min(1e-12)).max()),
        'bitwise_equal': torch.equal(changed[name], expected),
        'allclose_atol_1e6_rtol_1e5': torch.allclose(changed[name], expected, atol=1e-6, rtol=1e-5),
        'inverse_restoration_bitwise': torch.equal(restored[name], original[name])}
for name in study.LOCAL_VIEWS:
    expected = study.decode(fits[name], features[name])['physical'].roll(1, 1)
    decoded = study.decode(fits[name], changed_local[name])['physical']
    delta = (decoded-expected).abs()
    model = fits[name]['fits'][0]
    source_delta = (changed_local[name].double()-features[name].roll(1, 1).double()).abs()
    bound = (source_delta/model['feature_scale'])@model['weights'][:, :2].abs()*model['target_scale']
    receipt['decoded_permutation'][name] = {
        'max_absolute_gap': float(delta.max()),
        'max_relative_gap_nonzero': float((delta/expected.abs().clamp_min(1e-12)).max()),
        'max_source_roundoff_propagation_bound': float(bound.max()),
        'gap_within_source_bound_plus_1e12': bool((delta <= bound+1e-12).all()),
        'allclose_atol_1e6_rtol_1e5': torch.allclose(decoded, expected, atol=1e-6, rtol=1e-5)}
print(json.dumps(receipt, indent=2, allow_nan=False), flush=True)
summary, _ = study.intervention(primary_model, state, original, fits, features)
assert all(summary['checks'].values())
print('All declared intervention checks passed.', flush=True)
