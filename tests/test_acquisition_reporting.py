from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from attcon.active_inspection import make_splits, AcquisitionAgent, initial_events
from attcon.acquisition_reporting import measurement_data, report_scores, fit_report
from attcon.regulation import weight_fingerprint


class AcquisitionReportingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)
        self.data = make_splits(12)['report_fit'].subset(torch.arange(108))
        self.agent = AcquisitionAgent().eval().requires_grad_(False)

    def test_report_training_cannot_mutate_controller(self):
        before = weight_fingerprint(self.agent)
        features, labels, visited = measurement_data(self.agent, self.data)
        model, selection = fit_report(features['state'], labels, features['state'], labels, 4, steps=2)
        self.assertEqual(weight_fingerprint(self.agent), before)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4359)
        self.assertEqual(len(visited), 3 * len(self.data))
        self.assertEqual(len(selection['candidates']), 2)

    def test_unverified_is_not_scored_as_a_value(self):
        metrics = report_scores(torch.tensor([1, 6, 0, 3]), torch.tensor([1, 6, 6, 3]))
        self.assertEqual(metrics['verified_value_accuracy'], 1)
        self.assertEqual(metrics['unverified_accuracy'], .5)
        self.assertEqual(metrics['balanced_accuracy'], .75)
        empty = report_scores(torch.tensor([6]), torch.tensor([6]))
        self.assertIsNone(empty['balanced_accuracy'])

if __name__ == '__main__':
    unittest.main()
