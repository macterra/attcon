import unittest
import torch
from attcon.functional_task_interface import task_record, remap_task, without_task_relation
from attcon.predictive_attention import Forecast


class FunctionalTaskInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.visual = torch.zeros(1, 2, 4, 8)
        self.visual[..., 2] = 1; self.visual[..., 5] = 1
        a = torch.nn.functional.one_hot(torch.tensor([[1, 2]]), 4).float()
        effects = a[:, None].expand(-1, 4, -1, -1).clone()
        self.current = Forecast(a, torch.full((1, 3, 2, 4), .8), effects)
        self.future = torch.full((1, 4, 2, 4), .8)
        self.observed = torch.cat((a.flatten(1), (.8*a).flatten(1),
            torch.nn.functional.one_hot(torch.tensor([3]), 4).float()), -1)[:, None]
        self.query, self.command = torch.tensor([1]), torch.tensor([3])

    def view(self, **kwargs):
        return task_record(self.visual, self.current, self.future, self.observed,
                           0, self.query, self.command, **kwargs)

    def test_task_actions_have_explicit_pending_and_abstention_meanings(self):
        pending = self.view()['task_decision']
        self.assertFalse(pending['command_executed'])
        self.assertEqual(pending['response']['status'], 'pending')
        abstained = self.view(response=torch.tensor([[-1, -1]]), executed=True)['task_decision']
        self.assertTrue(abstained['command_executed'])
        self.assertEqual(abstained['response']['status'], 'abstained')
        answered = self.view(response=torch.tensor([[2, 1]]), executed=True)['task_decision']
        self.assertEqual(answered['response'], {'status': 'answered', 'color': 'blue', 'shape': 'square'})
        with self.assertRaises(ValueError): self.view(response=torch.tensor([[2, -1]]), executed=True)
        with self.assertRaises(ValueError): self.view(response=torch.tensor([[2, 1]]))

    def test_task_remapping_follows_readout_and_all_command_references(self):
        original = self.view(response=torch.tensor([[2, 1]]), executed=True, order=(2, 1, 0))
        nodes = {'n0':'q7', 'n1':'q2', 'n2':'q9'}
        commands = {'k0':'m7', 'k1':'m2', 'k2':'m9', 'k3':'m4'}
        changed = remap_task(original, nodes, commands)
        self.assertEqual(changed['task_decision']['query']['node'], changed['output_node'])
        self.assertEqual(changed['task_decision']['selected_command'], changed['observed_history'][-1]['command'])
        self.assertEqual(remap_task(changed, {v:k for k,v in nodes.items()},
            {v:k for k,v in commands.items()}), original)
        self.assertEqual(original['task_decision']['query']['node'], 'n0')
        removed = without_task_relation(changed)
        self.assertEqual(removed['task_decision'], changed['task_decision'])
        self.assertIsNone(removed['observed_history'])
        self.assertIsNone(removed['predicted_by_command'])

    def test_executed_task_command_cannot_be_fabricated_from_a_forecast(self):
        self.command = torch.tensor([0])
        with self.assertRaisesRegex(ValueError, 'actual observation'):
            self.view(response=torch.tensor([[2, 1]]), executed=True)


if __name__ == '__main__':
    unittest.main()
