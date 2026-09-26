from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from attcon.history_reporting import make_history_splits
from attcon.regulation_interventions import paired_indices, choice_null_directions, projection_transplant


class ChoiceNullTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        self.data = make_history_splits({'fit': 4}, seed=3)['fit']
        self.states = torch.randn(len(self.data), 64)
        self.weight = torch.randn(6, 64)

    def test_pairing_survives_shuffled_rows(self):
        data = self.data.subset(torch.randperm(len(self.data)))
        seen, unseen = paired_indices(data)
        self.assertTrue(data.seen[seen].all())
        self.assertTrue((~data.seen[unseen]).all())
        self.assertTrue(torch.equal(data.group[seen], data.group[unseen]))
        self.assertTrue(torch.equal(data.value[seen], data.value[unseen]))

    def test_choice_preservation_norm_matching_and_restoration(self):
        directions, _ = choice_null_directions(self.weight, self.states, self.data, 90)
        seen, unseen = paired_indices(self.data)
        recipient, donor = self.states[seen], self.states[unseen]
        delta = projection_transplant(recipient, donor, directions['access'])
        random = projection_transplant(recipient, donor, directions['random'], delta)
        self.assertTrue(torch.allclose(delta.norm(dim=1), random.norm(dim=1), atol=1e-6))
        for change in (delta, random):
            self.assertLess((change @ self.weight.T).abs().max().item(), 1e-5)
            self.assertTrue(torch.allclose(recipient + change - change, recipient, atol=1e-6))
        self.assertTrue(torch.allclose((recipient + delta) @ directions['access'], donor @ directions['access'], atol=1e-6))

    def test_degenerate_access_direction_rejected(self):
        with self.assertRaisesRegex(ValueError, 'degenerate'):
            choice_null_directions(self.weight, torch.zeros_like(self.states), self.data, 90)

    def test_missing_counterpart_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unpaired'):
            paired_indices(self.data.subset(torch.arange(len(self.data) - 1)))

    def test_rank_deficiency_keeps_null_projection_valid(self):
        self.weight[1] = self.weight[0]
        directions, metadata = choice_null_directions(self.weight, self.states, self.data, 90)
        self.assertEqual(metadata['choice_rank'], 5)
        for direction in directions.values():
            self.assertLess((self.weight @ direction).abs().max().item(), 1e-5)

if __name__ == '__main__':
    unittest.main()
