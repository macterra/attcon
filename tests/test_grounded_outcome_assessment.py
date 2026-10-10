"""Generated task fixtures; no checkpoints or collected study outcomes."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import grounded_outcome_assessment_v1 as assessment


def deterministic_fixture(n=32, all_seen=False):
    sensory = torch.zeros(n, 4, 4, 66)
    sensory[:, 1, :1, 65] = 1
    sensory[:, 2, :2, 65] = 1
    sensory[:, 3, :, 65] = 1
    if all_seen:
        sensory[..., 65] = 1
    history = torch.eye(4).expand(n, 4, 4, 4).clone()
    history[:, 0] = 0
    history[:, 1, 0] = 0
    current = sensory[..., 65].clone()
    refresh = torch.ones(n, 4, 2, 4)
    mapping = torch.arange(4).expand(n, 4, 2, 4).clone()
    # The zero-history null has balanced actual command-delivery destinations.
    mapping[:, 0] = (torch.arange(n)[:, None, None]+torch.arange(4)[None, None])%4
    d = torch.nn.functional.one_hot(mapping[:, :, 0], 4).double()
    d[:, 0] = .25
    cur, ref = current.double(), torch.ones_like(current).double()
    after = current[:, :, None, None, :].expand(n, 4, 2, 4, 4).clone()
    after.scatter_(-1, mapping[..., None], 1)
    forecast_after = d*ref.unsqueeze(-2)+(1-d)*cur.unsqueeze(-2)
    data = {'sensory': sensory, 'history': history, 'current_labels': current,
            'refresh_labels': refresh, 'delivery_labels': mapping, 'after_labels': after}
    forecasts = {'current': cur, 'refresh': ref, 'delivery': d, 'after': forecast_after, 'kind': 'structured'}
    return data, forecasts


class GroundedAssessmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_binary_repeat_scoring_and_bin_edges(self):
        p = torch.tensor([[.1, .9], [0., 1.]], dtype=torch.float64)
        y = torch.tensor([[0., 1.], [0., 1.]], dtype=torch.float64)
        result = assessment._binary(p, y, torch.tensor(.5))
        self.assertAlmostEqual(result['brier'], .005)
        self.assertAlmostEqual(result['ece'], .05)
        self.assertEqual(result['reliability'][0]['prediction_count'], 1)
        self.assertEqual(result['reliability'][1]['prediction_count'], 1)
        self.assertEqual(result['reliability'][9]['prediction_count'], 2)
        mixed = assessment._binary(torch.full((2, 2), .5, dtype=torch.float64),
                                   torch.full((2, 2), .5, dtype=torch.float64), torch.tensor(.5), repeats=2)
        self.assertEqual(mixed['brier'], .25)  # Not zero from scoring the mean.
        self.assertEqual(mixed['outcome_count'], 8)
        self.assertEqual(mixed['world_count'], 2)

    def test_multiclass_scoring_and_calibration_only_constants(self):
        data, forecasts = deterministic_fixture()
        before = torch.random.get_rng_state().clone()
        result = assessment.assess(data, data, forecasts, forecasts, 18, 'structured')
        self.assertTrue(result['engineering_gates_pass'])
        self.assertEqual(result['strata']['delivery']['zero_history_null']['top1'], .25)
        self.assertEqual(result['strata']['delivery']['known']['top1'], 1.)
        self.assertEqual(result['strata']['delivery']['known']['nll'], 0.)
        self.assertEqual(result['strata']['delivery']['zero_history_null']['brier'], .75)
        expected_baseline = torch.nn.functional.one_hot(data['delivery_labels'], 4).double().mean((0, 1, 2)).tolist()
        self.assertEqual(result['calibration_constants']['delivery'], expected_baseline)
        self.assertEqual(result['calibration_constants']['refresh'], 1.)
        self.assertEqual(result['pooled']['refresh']['brier_difference_95'], [0., 0.])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        changed = dict(data)
        changed['current_labels'] = 1-data['current_labels']
        second = assessment.assess(data, changed, forecasts, forecasts, 18, 'structured')
        self.assertEqual(result['calibration_constants'], second['calibration_constants'])
        self.assertFalse(second['engineering_gates_pass'])
        self.assertEqual(len(result['delivery_probability_cells']['known']), 16)

    def test_controller_gain_corruption_restoration_and_world_units(self):
        data, forecasts = deterministic_fixture()
        before = torch.random.get_rng_state().clone()
        result, archive = assessment.control(data, forecasts, 892)
        repeat, second_archive = assessment.control(data, forecasts, 892)
        self.assertTrue(result['engineering_gates_pass'])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(result, repeat)
        self.assertTrue(torch.equal(archive['queries'], second_archive['queries']))
        self.assertEqual(archive['original_scores'].shape, (32, 2))
        self.assertEqual(archive['all_world_bootstrap_indices'].shape, (256, 32))
        self.assertTrue(archive['unseen_query_subset'].all())
        self.assertTrue((archive['queries'] >= 2).all())
        self.assertTrue(torch.equal(archive['actions'], archive['queries']))
        self.assertTrue(torch.equal(archive['corrupted_actions'], (archive['actions']+1)%4))
        self.assertTrue(torch.equal(archive['restored_after'], forecasts['after']))
        gain = result['gain_vs_uniform_random_command']['unseen_query_subset']
        self.assertEqual(gain['difference'], .75)
        self.assertEqual(gain['difference_95'], [.75, .75])
        corruption = result['route_intervention']['original_minus_corrupted_success']['unseen_query_subset']
        self.assertEqual(corruption['difference'], 1.)
        self.assertEqual(corruption['difference_95'], [1., 1.])

    def test_empty_subset_ties_and_nonprimary_controls(self):
        data, forecasts = deterministic_fixture(all_seen=True)
        result, archive = assessment.control(data, forecasts, 39)
        self.assertFalse(result['gates']['unseen_query_controller_gain'])
        self.assertFalse(result['gates']['unseen_query_route_corruption_loss'])
        self.assertEqual(archive['unseen_query_bootstrap_indices'].shape, (256, 0))
        self.assertEqual(result['route_intervention']['unique_winner_worlds'], 0)
        self.assertEqual(len(archive['actions']), len(data['sensory']))
        self.assertTrue((archive['actions'] == 0).all())
        for kind in ('joint', 'sensory_only', 'history_only'):
            other = {**forecasts, 'kind': kind}
            result, archive = assessment.control(data, other, 39, kind)
            self.assertEqual(result['gates'], {})
            self.assertNotIn('corrupted_scores', archive)
            self.assertIsNone(result['engineering_gates_pass'])

    def test_fixed_subset_and_label_independent_query_action_choices(self):
        data, forecasts = deterministic_fixture()
        sensory = data['sensory'].clone()
        sensory[:8, 2, :, 65] = 1
        mixed = {**data, 'sensory': sensory}
        result, archive = assessment.control(mixed, forecasts, 321)
        self.assertEqual(result['gain_vs_uniform_random_command']['unseen_query_subset']['world_count'], 24)
        self.assertTrue((archive['unseen_query_bootstrap_indices'] >= 8).all())
        changed = {**mixed, 'after_labels': 1-data['after_labels']}
        _, changed_archive = assessment.control(changed, forecasts, 321)
        self.assertTrue(torch.equal(archive['queries'], changed_archive['queries']))
        self.assertTrue(torch.equal(archive['actions'], changed_archive['actions']))
        self.assertTrue(torch.equal(archive['all_world_bootstrap_indices'], changed_archive['all_world_bootstrap_indices']))
        self.assertFalse(torch.equal(archive['original_scores'], changed_archive['original_scores']))

    def test_paired_world_brier_interval_matches_explicit_resampling(self):
        p = torch.tensor([[.1, .7], [.8, .6], [.4, .9]], dtype=torch.float64)
        y = torch.tensor([[0., 1.], [1., 0.], [1., 1.]], dtype=torch.float64)
        constant = torch.tensor(.5, dtype=torch.float64)
        indices = torch.tensor([[0, 0, 2], [2, 1, 1], [1, 0, 2]])
        metric = assessment._binary(p, y, constant, indices=indices)
        differences = []
        for row in indices:
            differences.append(((p[row]-y[row]).square()-(constant-y[row]).square()).mean())
        expected = torch.quantile(torch.stack(differences), torch.tensor([.025, .975], dtype=torch.float64)).tolist()
        self.assertTrue(torch.allclose(torch.tensor(metric['brier_difference_95']), torch.tensor(expected), atol=1e-14, rtol=0))


if __name__ == '__main__':
    unittest.main()
