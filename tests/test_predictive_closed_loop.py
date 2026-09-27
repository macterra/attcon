import unittest
import torch
from attcon.predictive_attention import PredictiveAttention
from attcon.predictive_closed_loop import rollout


class ClosedLoopTests(unittest.TestCase):
    def test_restore_reproduces_entire_trajectory(self):
        torch.manual_seed(4)
        model = PredictiveAttention()
        ordinary, a = rollout(model, 7, count=8, steps=8)
        restored, b = rollout(model, 7, count=8, steps=8, intervention='restore')
        self.assertEqual(ordinary, restored)
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))

    def test_rotation_changes_commands_after_shared_prefix(self):
        torch.manual_seed(9)
        model = PredictiveAttention()
        _, a = rollout(model, 7, count=16, steps=8)
        _, b = rollout(model, 7, count=16, steps=8, intervention='rotate_effects')
        self.assertTrue(torch.equal(a['command'][:, :4], b['command'][:, :4]))
        self.assertTrue(torch.equal((a['command'][:, 4] + 1) % 4, b['command'][:, 4]))
        self.assertFalse(torch.equal(a['physical_access'][:, 4], b['physical_access'][:, 4]))
