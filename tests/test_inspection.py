from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from attcon.inspection import inspection_returns, context_return_interval, select_policy, policy_features, immediate_rewards, policy_switches
from attcon.regulation import agent_for, weight_fingerprint, WIDTH
from attcon.regulation_interventions import choice_null_directions, paired_indices, projection_transplant
from attcon.history_reporting import make_history_splits


class InspectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(15)

    def test_environment_return_and_strict_cost_threshold(self):
        returns, inspect = inspection_returns(torch.tensor([0.1, 0.6, 0.9]), torch.tensor([1., 0., 1.]), 0.4)
        self.assertEqual(inspect.tolist(), [True, False, False])
        self.assertTrue(torch.allclose(returns, torch.tensor([0.6, 0., 1.])))

    def test_bootstrap_resamples_contexts_and_preserves_constant_gain(self):
        result = context_return_interval(torch.tensor([0., 1., 0., 1.]), torch.tensor([4, 4, 9, 9]), 4)
        for name in ('mean', 'low', 'high'):
            self.assertEqual(result[name], 0.5)
        self.assertEqual(result['contexts'], 2)

    def test_reward_fitting_freezes_agent_and_uses_only_fit_normalization(self):
        agent, _ = agent_for('gru')
        before = weight_fingerprint(agent)
        states = torch.randn(24, 64)
        features = policy_features(agent, states)['state']
        reward = immediate_rewards(agent, states, torch.arange(24) % 6)
        model, selection = select_policy(features, reward, features + 100, reward, seed=3, steps=2)
        self.assertTrue(torch.equal(model.center, features.mean(0)))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4161)
        self.assertEqual(before, weight_fingerprint(agent))
        self.assertEqual(len(selection['candidates']), 2)
        self.assertTrue(all(p.grad is None for p in agent.parameters()))

    def test_choice_null_change_keeps_action_confidence_policy_invariant(self):
        agent, _ = agent_for('gru')
        data = make_history_splits({'fit': 4}, seed=4)['fit']
        states = torch.randn(len(data), 64)
        directions, _ = choice_null_directions(agent.choice.weight, states, data, 8)
        seen, unseen = paired_indices(data)
        recipients = states[seen]
        changed = recipients + projection_transplant(recipients, states[unseen], directions['access'])
        features = policy_features(agent, states)
        rewards = immediate_rewards(agent, states, data.value)
        models = {name: select_policy(value, rewards, value, rewards, seed=2, steps=2)[0] for name, value in features.items()}
        result = policy_switches(models, agent, recipients, changed)
        for name in ('action', 'confidence'):
            self.assertLess(result[name]['max_feature_residual'], 1e-5)
            self.assertLess(result[name]['max_probability_residual'], 1e-5)
            self.assertTrue(all(cost['switch_rate'] == 0 for cost in result[name]['costs'].values()))

if __name__ == '__main__':
    unittest.main()
