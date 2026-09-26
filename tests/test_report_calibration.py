from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attcon.history_reporting import HistoryAgent, HistoryConfig, make_history_splits, pad_features
from attcon.report_calibration import (
    ActionScoreReporter, EntropyReporter, NormalizedReadout, fit_readout,
    select_entropy_reporter, select_readout, selection_score,
)


class ReportCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_normalization_uses_fitting_data_and_is_immutable_on_new_inputs(self) -> None:
        fit = torch.tensor([[1.0, 4.0], [3.0, 4.0]])
        model = NormalizedReadout(fit, 3, True)
        self.assertTrue(torch.equal(model.center, torch.tensor([2.0, 4.0])))
        self.assertTrue(torch.allclose(model.scale, torch.tensor([1.0, 0.01])))
        center, scale = model.center.clone(), model.scale.clone()
        model(torch.full((10, 2), 1000.0))
        self.assertTrue(torch.equal(center, model.center))
        self.assertTrue(torch.equal(scale, model.scale))

    def test_selection_is_validation_only_with_documented_tie_breaking(self) -> None:
        data = make_history_splits({"fit": 2, "validation": 2}, seed=9)
        features = torch.randn(len(data["fit"]), 64)
        validation = torch.randn(len(data["validation"]), 64)
        low = dict(seen_value_accuracy=1.0, unseen_unknown_accuracy=0.1, paired_accuracy=0.1, balanced_accuracy=0.55)
        good = dict(seen_value_accuracy=0.7, unseen_unknown_accuracy=0.7, paired_accuracy=0.5, balanced_accuracy=0.7)
        better_pair = dict(good, paired_accuracy=0.6)
        # No test inputs exist in select_readout's API. Make validation ranking
        # disagree with content-only accuracy and assert deterministic selection.
        with patch("attcon.report_calibration.report_metrics", side_effect=[low, good, better_pair, better_pair, low, low, low, low]):
            result = select_readout(features, data["fit"].report_labels(HistoryConfig()),
                                    validation, data["validation"], HistoryConfig(), seed=3, steps=1)
        self.assertEqual(result.selected["l2"], 0.001)
        self.assertFalse(result.selected["standardize"])
        self.assertEqual(len(result.candidates), 8)
        self.assertGreater(selection_score(good), selection_score(low))

    def test_report_fitting_detaches_features_and_cannot_change_agent(self) -> None:
        agent = HistoryAgent()
        batch = make_history_splits({"fit": 2}, seed=13)["fit"]
        before = {k: v.clone() for k, v in agent.state_dict().items()}
        # Even accidentally supplied graph-connected features must be detached.
        features = agent.state(batch.events)
        fit_readout(features, batch.report_labels(HistoryConfig()), 7,
                    standardize=True, l2=0.001, seed=3, steps=3)
        self.assertTrue(all(p.grad is None for p in agent.parameters()))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in agent.state_dict().items()))

    def test_action_adapter_matches_score_readout_and_copies_frozen_choice(self) -> None:
        choice = torch.nn.Linear(64, 6)
        states = torch.randn(10, 64)
        features = pad_features(choice(states), 64)
        readout = NormalizedReadout(features, 7, True)
        adapter = ActionScoreReporter(choice, readout, 64)
        expected = readout(features)
        self.assertTrue(torch.allclose(adapter(states), expected))
        with torch.no_grad():
            choice.weight.add_(100)
        self.assertTrue(torch.allclose(adapter(states), expected))
        self.assertEqual(sum(p.numel() for p in readout.parameters()), 455)

    def test_entropy_reports_uniform_unknown_and_concentrated_content(self) -> None:
        choice = torch.nn.Linear(6, 6)
        with torch.no_grad():
            choice.weight.copy_(torch.eye(6))
            choice.bias.zero_()
        reporter = EntropyReporter(choice, 0.5)
        states = torch.zeros(2, 6)
        states[1, 3] = 20
        self.assertEqual(reporter(states).argmax(-1).tolist(), [6, 3])
        self.assertIn("threshold", reporter.state_dict())

    def test_entropy_threshold_selection_fits_validation_without_mutation(self) -> None:
        batch = make_history_splits({"validation": 2}, seed=13)["validation"]
        states = torch.zeros(len(batch), 6)
        states[batch.seen, batch.value[batch.seen]] = 20
        choice = torch.nn.Linear(6, 6)
        with torch.no_grad():
            choice.weight.copy_(torch.eye(6))
            choice.bias.zero_()
        result = select_entropy_reporter(choice, states, batch, HistoryConfig())
        self.assertEqual(len(result.candidates), 101)
        self.assertEqual(result.selected["validation"]["paired_accuracy"], 1.0)
        self.assertTrue(all(p.grad is None for p in choice.parameters()))


if __name__ == "__main__":
    unittest.main()
