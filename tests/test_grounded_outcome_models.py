"""Small generated fixtures only: no study checkpoint or evaluation data."""
import copy
import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import grounded_outcome_models_v1 as models


def fixture():
    generator = torch.Generator().manual_seed(208)
    sensory = torch.randn(8, 2, 4, 66, generator=generator)
    sensory[..., 65] = torch.randint(2, (8, 2, 4), generator=generator)
    locations = torch.randint(4, (8, 2, 4), generator=generator)
    history = torch.nn.functional.one_hot(locations, 4).float()
    history[:, 0] = 0
    current = torch.randint(2, (8, 2, 4), generator=generator).float()
    refresh = torch.randint(2, (8, 2, 3, 4), generator=generator).float()
    delivery = torch.randint(4, (8, 2, 3, 4), generator=generator)
    after = torch.randint(2, (8, 2, 3, 4, 4), generator=generator).float()
    return sensory, history, current, refresh, delivery, after


class GroundedOutcomeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fit_fixture(self, data, kind='structured'):
        # Tests reduce iterations only on generated fixtures. Production uses
        # the fixed module constant, not an exposed fit hyperparameter.
        with patch.object(models, 'STEPS', 12):
            return models.fit(*data, seed=917, kind=kind)

    def test_shapes_composition_and_batch_rng_independence(self):
        data = fixture()
        before = torch.random.get_rng_state().clone()
        first = self.fit_fixture(data)
        second = self.fit_fixture(data)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for key in first['state_dict']:
            self.assertTrue(torch.equal(first['state_dict'][key], second['state_dict'][key]))
        result = models.predict(first, *data[:2])
        self.assertEqual(result['delivery'].shape, (8, 2, 4, 4))
        self.assertEqual(result['current'].shape, (8, 2, 4))
        self.assertEqual(result['refresh'].shape, (8, 2, 4))
        self.assertEqual(result['after'].shape, (8, 2, 4, 4))
        self.assertTrue(torch.allclose(result['delivery'].sum(-1), torch.ones(8, 2, 4, dtype=torch.float64)))
        composed = result['delivery']*result['refresh'].unsqueeze(-2)+(1-result['delivery'])*result['current'].unsqueeze(-2)
        self.assertTrue(torch.equal(result['after'], composed))
        single = models.predict(first, data[0][3:4, 1:2], data[1][3:4, 1:2])
        for key in result:
            self.assertTrue(torch.allclose(result[key][3:4, 1:2], single[key], atol=1e-6, rtol=1e-6), key)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(first['parameter_count'], 4338)
        self.assertEqual(first['loss_trace'][0][0], 0)
        self.assertEqual(first['loss_trace'][-1][0], 12)
        self.assertEqual(list(inspect.signature(models.predict).parameters), ['readout', 'sensory', 'history'])

    def test_source_masks_applied_before_normalization(self):
        data = fixture()
        original_sensory, original_history = data[0].clone(), data[1].clone()
        sensory_only = self.fit_fixture(data, 'sensory_only')
        self.assertTrue(torch.equal(sensory_only['normalization']['history']['mean'], torch.zeros(16)))
        self.assertTrue(torch.equal(sensory_only['normalization']['history']['scale'], torch.ones(16)))
        changed_history = data[1].roll(1, -1)
        a = models.predict(sensory_only, *data[:2])
        b = models.predict(sensory_only, data[0], changed_history)
        self.assertTrue(all(torch.equal(a[key], b[key]) for key in a))
        history_only = self.fit_fixture(data, 'history_only')
        self.assertTrue(torch.equal(history_only['normalization']['sensory']['mean'][:65], torch.zeros(65)))
        changed_sensory = data[0].clone()
        changed_sensory[..., :65] += 1000
        a = models.predict(history_only, *data[:2])
        b = models.predict(history_only, changed_sensory, data[1])
        self.assertTrue(all(torch.equal(a[key], b[key]) for key in a))
        self.assertTrue(torch.equal(data[0], original_sensory))
        self.assertTrue(torch.equal(data[1], original_history))
        self.assertTrue(torch.equal(history_only['normalization']['sensory']['mean'][65], data[0][..., 65].mean()))

    def test_current_and_refresh_risks_are_not_after_targets(self):
        data = fixture()
        changed = (*data[:5], 1-data[5])
        a, b = self.fit_fixture(data), self.fit_fixture(changed)
        self.assertTrue(all(torch.equal(a['state_dict'][key], b['state_dict'][key]) for key in a['state_dict']))
        a, b = self.fit_fixture(data, 'joint'), self.fit_fixture(changed, 'joint')
        self.assertTrue(any(not torch.equal(a['state_dict'][key], b['state_dict'][key]) for key in a['state_dict']))
        self.assertEqual(a['parameter_count'], 20584)
        self.assertEqual(models.predict(a, *data[:2])['after_logits'].shape, (8, 2, 4, 4))

    def test_calibration_fractional_labels_and_preserved_weights(self):
        data = fixture()
        for kind in ('structured', 'joint'):
            readout = self.fit_fixture(data, kind)
            snapshot = copy.deepcopy(readout)
            calibrated = models.calibrate(readout, *data)
            self.assertIsNone(readout['calibration'])
            for key in snapshot['state_dict']:
                self.assertTrue(torch.equal(snapshot['state_dict'][key], calibrated['state_dict'][key]))
                self.assertTrue(torch.equal(snapshot['state_dict'][key], readout['state_dict'][key]))
            for name in snapshot['normalization']:
                for key in ('mean', 'scale'):
                    self.assertTrue(torch.equal(snapshot['normalization'][name][key], calibrated['normalization'][name][key]))
            cal = calibrated['calibration']
            self.assertIn(cal['delivery_temperature'], models.TEMPERATURES)
            self.assertLessEqual(float(cal['refresh']['gradient_max']), 1e-8)
            self.assertAlmostEqual(float(cal['refresh']['baseline']), float(data[3].double().mean()), places=14)
            p = models.predict(calibrated, *data[:2])
            self.assertTrue(all(bool(((p[key] >= 0) & (p[key] <= 1)).all()) for key in ('delivery', 'current', 'refresh', 'after')))
            if kind == 'structured':
                self.assertNotIn('after', cal)
                self.assertTrue(torch.equal(p['after'], p['delivery']*p['refresh'].unsqueeze(-2)+(1-p['delivery'])*p['current'].unsqueeze(-2)))
            else:
                self.assertIn('after', cal)
        # Calibration uses signed task-correctness logits, not absolute margins.
        scores = torch.tensor([-2., -1., 0., 1., 2.])
        targets = torch.tensor([0., 1/3, .5, 2/3, 1.], dtype=torch.float64)
        mapping = models._fit_signed_calibrator(scores, targets)
        probabilities = models._calibrated(scores, mapping)
        self.assertGreater(float(probabilities[-1]), float(probabilities[0]))
        self.assertLessEqual(float(mapping['gradient_max']), 1e-8)

    def test_invalid_targets_and_history_are_rejected(self):
        data = fixture()
        bad_delivery = data[4].clone()
        bad_delivery[0, 0, 0, 0] = 4
        with self.assertRaises(ValueError):
            self.fit_fixture((*data[:4], bad_delivery, data[5]))
        with self.assertRaises(ValueError):
            self.fit_fixture((*data[:3], data[3][:, :, 0], *data[4:]))
        history = data[1].clone()
        history[0, 0, 0] = 1
        with self.assertRaises(ValueError):
            self.fit_fixture((data[0], history, *data[2:]))
        sensory = data[0].clone()
        sensory[0, 0, 0, 65] = .5
        with self.assertRaises(ValueError):
            self.fit_fixture((sensory, *data[1:]))


if __name__ == '__main__':
    unittest.main()
