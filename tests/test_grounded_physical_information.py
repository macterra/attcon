"""Generated diagnostic fixtures; no production checkpoint/output access."""
import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import grounded_physical_information_v1 as study
from grounded_outcome_models_v1 import _OutcomeNetwork, predict


def fake_sensor(pixels):
    # Image-only callback; deterministic source information, no world closure.
    value = pixels.mean((-3, -2, -1))-.7
    return value[..., None].expand(*value.shape, 64).clone(), value


def decoder_fixture(seed=34, n=40, dim=32):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(n, 4, dim, generator=generator, dtype=torch.float64)
    target = torch.stack((2*features[..., 0]+3, -features[..., 1]+.7), -1)
    seen = torch.rand(n, 4, generator=generator) > .2
    state = {'seen': seen, 'physical_targets': target,
             'stored_answers': (features[..., 0] > 0).long()}
    return features, state


def fixture_model(kind):
    with torch.random.fork_rng():
        torch.manual_seed(32)
        model = _OutcomeNetwork(kind)
    normalization = ({'joint': {'mean': torch.zeros(280), 'scale': torch.ones(280)}}
                     if kind == 'joint' else {'sensory': {'mean': torch.zeros(66), 'scale': torch.ones(66)},
                                             'history': {'mean': torch.zeros(16), 'scale': torch.ones(16)}})
    return {'kind': kind, 'state_dict': model.state_dict(), 'normalization': normalization,
            'calibration': None}


class GroundedPhysicalInformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_physical_statistics_have_declared_spatial_channel_boundary(self):
        # Constant planes of differing colors have pooled image-value variation.
        pixels = torch.tensor([.1, .5, .9])[:, None, None].expand(3, 16, 16)
        target = study.physical_targets(pixels)
        self.assertAlmostEqual(float(target[0]), (.4**2*2/3)**.5, places=7)
        self.assertEqual(float(target[1]), 0.)
        self.assertTrue(torch.equal(study.physical_targets(torch.full((3, 16, 16), .5)),
                                    torch.zeros(2, dtype=torch.float64)))
        checker = ((torch.arange(16)[:, None]+torch.arange(16)[None, :]) % 2).double()
        targets = study.physical_targets(checker.expand(3, 16, 16))
        self.assertTrue(torch.equal(targets, torch.tensor([.5, 1.], dtype=torch.float64)))
        scaled = study.physical_targets(.3+.4*checker.expand(3, 16, 16))
        self.assertTrue(torch.allclose(scaled, targets*.4, atol=1e-15, rtol=0))

    def test_fresh_deliveries_latest_pixel_targets_and_rng_replay(self):
        archive = {}
        before = torch.random.get_rng_state().clone()
        state, world = study.collect(1301, 0, fake_sensor, lambda k, v: archive.__setitem__(k, v), count=16)
        replay, _ = study.collect(1301, 0, fake_sensor, lambda k, v: None, count=16)
        for key in state:
            if isinstance(state[key], torch.Tensor):
                self.assertTrue(torch.equal(state[key], replay[key]), key)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(state['sensory'].shape, (16, 1, 4, 66))
        self.assertFalse(any('correct' in key for key in state))
        for row in range(16):
            for loc in range(4):
                steps = torch.where(archive['decoded']['delivered'][row] == loc)[0]
                if len(steps):
                    latest = steps[-1]
                    self.assertEqual(state['last_step'][row, loc], latest)
                    self.assertTrue(torch.equal(state['last_pixels'][row, loc], archive['pixels'][row, latest]))
                    self.assertTrue(torch.equal(state['sensory'][row, 0, loc, :64], archive['decoded']['hidden'][row, latest]))
                    self.assertEqual(state['sensory'][row, 0, loc, 64], archive['decoded']['raw_logits'][row, latest])
                else:
                    self.assertEqual(state['last_step'][row, loc], -1)
                    self.assertFalse(state['seen'][row, loc])
                    self.assertFalse(state['sensory'][row, 0, loc].any())
        self.assertTrue(torch.equal(state['physical_targets'], study.physical_targets(state['last_pixels'])))
        rerender = study.render_locations(world, archive['decoded']['delivered'], archive['noise'])
        self.assertTrue(torch.equal(rerender, archive['pixels']))
        all_seeds = [study.seed_for(1300000000, seed, split)+i
                     for seed in study.SEEDS for split, count in enumerate(study.COUNTS) for i in range(count)]
        self.assertEqual(len(all_seeds), len(set(all_seeds)))
        self.assertTrue(all(v > 1100000000 for v in all_seeds))

    def test_training_and_calibration_exclude_unseen_even_if_poisoned(self):
        x, train = decoder_fixture()
        cx, cal = decoder_fixture(35)
        fitted = study.fit_view('risk_hidden', x, cx, train, cal)
        poisoned_train, poisoned_cal = copy.deepcopy(train), copy.deepcopy(cal)
        px, pcx = x.clone(), cx.clone()
        for features, state in ((px, poisoned_train), (pcx, poisoned_cal)):
            features[~state['seen']] = 1e12
            state['physical_targets'][~state['seen']] = -1e12
            state['stored_answers'][~state['seen']] = 1-state['stored_answers'][~state['seen']]
        poisoned = study.fit_view('risk_hidden', px, pcx, poisoned_train, poisoned_cal)
        study.compare(fitted, poisoned)
        self.assertEqual(fitted['decoder_count'], 1)
        self.assertEqual(fitted['fits'][0]['train_seen_rows'], int(train['seen'].sum()))
        tx, evaluation = decoder_fixture(36)
        decoded = study.decode(fitted, tx)
        mse = (decoded['physical'][evaluation['seen']]-evaluation['physical_targets'][evaluation['seen']]).square().mean()
        self.assertLess(mse, .03)

    def test_calibration_chooses_physical_loss_not_answer_labels_or_eval(self):
        x, train = decoder_fixture(n=25)
        cx, cal = decoder_fixture(35, n=25)
        fit = study.fit_view('risk_hidden', x, cx, train, cal)
        model = fit['fits'][0]
        expected = min(range(len(study.ALPHAS)), key=lambda i: model['calibration_standardized_physical_mse'][i])
        self.assertEqual(model['alpha'], study.ALPHAS[expected])
        changed = copy.deepcopy(train)
        changed['stored_answers'] = 1-train['stored_answers']
        other = study.fit_view('risk_hidden', x, cx, changed, cal)
        self.assertEqual(model['alpha'], other['fits'][0]['alpha'])
        self.assertTrue(torch.equal(model['weights'][:, :2], other['fits'][0]['weights'][:, :2]))
        original = copy.deepcopy(fit)
        study.decode(fit, x*1000)
        study.compare(original, fit)
        # Constant physical features tie all alpha candidates and preserve scale=1.
        constant = torch.ones(12, 1)
        target = torch.zeros(12, 2)
        tied = study.ridge_candidates(constant, target, torch.zeros(12), constant, target)
        self.assertEqual(tied['alpha'], study.ALPHAS[0])
        self.assertTrue(torch.equal(tied['feature_scale'], torch.ones(1, dtype=torch.float64)))
        self.assertTrue(torch.equal(tied['target_scale'], torch.ones(2, dtype=torch.float64)))

    def test_global_location_decoders_preserve_location_specific_targets(self):
        generator = torch.Generator().manual_seed(39)
        def global_fixture():
            x = torch.randn(40, 64, generator=generator, dtype=torch.float64)
            expanded = x[:, None].expand(-1, 4, -1).clone()
            y = torch.stack([torch.stack((x[:, loc]+loc, 2*x[:, loc+4]-loc), -1) for loc in range(4)], 1)
            return expanded, {'seen': torch.ones(40, 4, dtype=torch.bool), 'physical_targets': y,
                              'stored_answers': (y[..., 0] > 0).long()}
        x, train = global_fixture()
        cx, cal = global_fixture()
        tx, evaluation = global_fixture()
        fit = study.fit_view('joint_hidden', x, cx, train, cal)
        self.assertEqual(fit['decoder_count'], 4)
        self.assertTrue(all(f['train_seen_rows'] == 40 for f in fit['fits']))
        self.assertFalse(torch.equal(fit['fits'][0]['target_mean'], fit['fits'][1]['target_mean']))
        decoded = study.decode(fit, tx)
        self.assertEqual(decoded['physical'].shape, (40, 4, 2))
        self.assertTrue(torch.isfinite(decoded['physical']).all())

    def test_world_cluster_scores_baseline_unseen_poison_and_empty_strata(self):
        _, state = decoder_fixture(n=6)
        baseline = torch.zeros_like(state['physical_targets'])
        decoded = {'physical': state['physical_targets'].clone(), 'answer': state['stored_answers'].clone(),
                   'train_mean_baseline': baseline}
        # Each sampled index represents a whole world, all four locations retained.
        resamples = torch.tensor([[0, 0, 1, 1, 2, 2], [3, 3, 4, 4, 5, 5]])
        score = study.grouped_scores(state, decoded, state['seen'], resamples)
        self.assertEqual(score['pooled']['r2_vs_train_mean']['value'], [1., 1.])
        self.assertEqual(score['pooled']['rmse']['value'], [0., 0.])
        self.assertEqual(score['pooled']['stored_answer_fidelity']['value'], [1.])
        changed = copy.deepcopy(state)
        changed['physical_targets'][~state['seen']] += 1e9
        other = study.grouped_scores(changed, decoded, state['seen'], resamples)
        self.assertEqual(score, other)
        empty = study.grouped_scores(state, decoded, torch.zeros_like(state['seen']), resamples)
        self.assertEqual(empty['seen_rows'], 0)
        self.assertEqual(empty['pooled']['r2_vs_train_mean']['value'], [None, None])
        self.assertEqual(empty['pooled']['r2_vs_train_mean']['valid_bootstrap_counts'], [0, 0])
        baseline_decoded = dict(decoded, physical=baseline)
        baseline_score = study.grouped_scores(state, baseline_decoded, state['seen'], resamples)
        self.assertEqual(baseline_score['pooled']['r2_vs_train_mean']['value'], [0., 0.])

    def test_native_readdress_and_output_patch_are_distinct_and_exact(self):
        state, _ = study.collect(1301, 0, fake_sensor, lambda k, v: None, count=24)
        primary_model, joint_model = fixture_model('structured'), fixture_model('joint')
        primary = predict(primary_model, state['sensory'], state['history'])
        joint = predict(joint_model, state['sensory'], state['history'])
        views = study.features(state, primary, joint)
        fits = {name: study.fit_view(name, value, value, state, state) for name, value in views.items()}
        before = copy.deepcopy(state)
        summary, trace = study.intervention(primary_model, state, primary, fits, views)
        self.assertTrue(all(summary['checks'].values()))
        self.assertFalse(summary['joint_intervened'])
        self.assertTrue(summary['synthetic_patched_source_not_natural_state'])
        self.assertTrue(summary['current_memory_precedence'])
        self.assertTrue(summary['history_retains_actual_past'])
        self.assertEqual(set(trace['decode_permutation_error_bounds']), set(study.LOCAL_VIEWS))
        for record in trace['decode_permutation_error_bounds'].values():
            self.assertTrue(bool((record['absolute_output_difference'] <=
                                  record['propagated_bound']+study.DECODE_ROUNDOFF).all()))
        self.assertGreater(summary['source_change_max'], 0)
        self.assertGreater(summary['patched_after_change_max'], 0)
        self.assertTrue(torch.equal(trace['readdressed_sensory'], state['sensory'].roll(1, -2)))
        self.assertTrue(torch.equal(trace['patched_forecasts']['delivery'], primary['delivery'].roll(1, -2)))
        study.compare(before, state)

    def test_exclusive_archives_and_readonly_replay(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(study, 'ROOT', Path(directory)):
            original = {'tensor': torch.arange(4), 'nested': [torch.ones(2), (1, 2)]}
            archive = study.Archive()
            archive.write('fixture/state', original)
            with self.assertRaises(FileExistsError):
                archive.write('fixture/state', original)
            replay = study.Archive(verify=True)
            replay.write('fixture/state', original)
            self.assertEqual(archive.files, replay.files)
            bad = dict(original, tensor=torch.arange(4)+1)
            with self.assertRaises(AssertionError):
                replay.write('fixture/state', bad)
            study.dump(Path(directory)/'reservation.json', {'reserved': True}, exclusive=True)
            with self.assertRaises(FileExistsError):
                study.dump(Path(directory)/'reservation.json', {'reserved': True}, exclusive=True)


    def test_registration_requires_review_and_detects_source_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, source, inputs = root/'protocol.md', root/'fixture.py', root/'input.weights'
            protocol.write_text('Synthetic protocol only')
            source.write_text('# Synthetic fixture only')
            inputs.write_bytes(b'No checkpoint weights')
            with patch.object(study, 'ROOT', root), patch.object(study, 'PROTOCOL', str(protocol)), \
                 patch.object(study, 'SOURCES', (str(protocol), str(source))), \
                 patch.object(study, 'input_paths', return_value=[inputs]):
                with self.assertRaises(AssertionError):
                    study.register()
                (root/'design_review').mkdir()
                (root/'design_review/review.md').write_text('Synthetic review')
                (root/'design_review/metadata.json').write_text('{}')
                study.register()
                manifest = study.check_manifest()
                self.assertEqual(manifest['intervention_permutation_atol'],
                                 manifest['permutation_source_tolerance']['atol'])
                self.assertEqual(manifest['intervention_permutation_rtol'],
                                 manifest['permutation_source_tolerance']['rtol'])
                with self.assertRaises(AssertionError):
                    study.register()
                source.write_text('# Changed synthetic fixture')
                with self.assertRaises(AssertionError):
                    study.check_manifest()

    def test_failure_reservation_prevents_retry(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(study, 'ROOT', Path(directory)), \
             patch.object(study, 'check_manifest', return_value={}), \
             patch.object(study, 'build', side_effect=ValueError('synthetic failure')) as build:
            with self.assertRaises(ValueError):
                study.execute()
            self.assertEqual(json.loads((Path(directory)/'failure.json').read_text())['status'],
                             'terminal_failure_no_retry')
            with self.assertRaises(FileExistsError):
                study.execute()
            self.assertEqual(build.call_count, 1)

    def test_generated_end_to_end_collection_fit_assessment_and_archive_replay(self):
        # Entire pipeline uses toy sensor/random fixture networks and tiny worlds.
        with tempfile.TemporaryDirectory() as directory, patch.object(study, 'ROOT', Path(directory)), \
             patch.object(study, 'COUNTS', (24, 12, 24)), patch.object(study, 'SEEDS', (1301,)), \
             patch.object(study, 'BOOTSTRAPS', 8), patch.object(study, 'FrozenSensor', return_value=fake_sensor), \
             patch.object(study, 'load_frozen', side_effect=lambda seed, kind: fixture_model(kind)), \
             contextlib.redirect_stdout(io.StringIO()):
            archive = study.Archive()
            result = study.build(archive)
            replay_archive = study.Archive(verify=True)
            replay = study.build(replay_archive)
            self.assertEqual(result, replay)
            self.assertEqual(archive.files, replay_archive.files)
            self.assertFalse(result['goal_achieved'])
            self.assertFalse(result['appearance_tested'])
            scores = result['models']['1301']['readouts']['risk_hidden']['assessment']
            self.assertEqual(len(scores['current_success_bins']), 5)
            self.assertIn('target_population_variance', scores['all_seen'])
            # JSON serialization excludes any nonfinite metric, empty groups explicit.
            json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
