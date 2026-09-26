from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attcon.history_reporting import HistoryConfig, make_history_splits
from attcon.report_uncertainty import context_bootstrap, context_scores


class ReportUncertaintyTests(unittest.TestCase):
    def test_context_statistics_preserve_pairs_under_row_reordering(self):
        data = make_history_splits({"test": 8}, seed=4)["test"]
        predictions = data.report_labels(HistoryConfig()).clone()
        predictions[:12] = 6
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(4))
        original = context_scores(predictions, data, HistoryConfig())
        reordered = context_scores(predictions[order], data.subset(order), HistoryConfig())
        self.assertTrue(torch.equal(original, reordered))
        self.assertEqual(original.shape, (8, 4))

    def test_constant_clusters_have_zero_interval_width(self):
        result = context_bootstrap(torch.ones(8, 4), seed=4)
        self.assertTrue(all(v == {"mean": 1.0, "low": 1.0, "high": 1.0} for v in result.values()))

    def test_duplicate_pairs_are_rejected_even_with_correct_row_count(self):
        data = make_history_splits({"test": 2}, seed=4)["test"]
        order = torch.arange(len(data))
        order[1] = 0
        bad = data.subset(order)
        with self.assertRaises(ValueError):
            context_scores(bad.report_labels(HistoryConfig()), bad, HistoryConfig())

    def test_resampling_is_deterministic_and_rejects_single_context(self):
        scores = torch.arange(32).reshape(8, 4).float() / 32
        self.assertEqual(context_bootstrap(scores, seed=4), context_bootstrap(scores, seed=4))
        with self.assertRaises(ValueError):
            context_bootstrap(scores[:1], seed=4)


if __name__ == "__main__":
    unittest.main()
