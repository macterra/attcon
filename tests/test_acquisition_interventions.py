from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from attcon.active_inspection import make_splits, AcquisitionAgent, initial_events
from attcon.acquisition_interventions import condition_pairs, fitted_directions
from attcon.regulation import weight_fingerprint


class AcquisitionInterventionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)
        self.data = make_splits(12)['report_fit'].subset(torch.arange(108))
        self.agent = AcquisitionAgent().eval().requires_grad_(False)

    def test_pairs_survive_shuffling_and_match_cost_value_context(self):
        data = self.data.subset(torch.randperm(len(self.data)))
        fresh, stale = condition_pairs(data)
        for field in ('group', 'value', 'cost'):
            self.assertTrue(torch.equal(getattr(data, field)[fresh], getattr(data, field)[stale]))
        self.assertTrue((data.condition[fresh] == 0).all())
        self.assertTrue((data.condition[stale] == 1).all())

    def test_directions_preserve_answer_logits(self):
        directions, _ = fitted_directions(self.agent, self.data, 5)
        for direction in directions.values():
            self.assertLess((self.agent.answer.weight @ direction).abs().max().item(), 1e-5)

if __name__ == '__main__':
    unittest.main()
