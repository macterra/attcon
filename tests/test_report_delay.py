from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attcon.history_reporting import HistoryConfig, make_history_splits
from attcon.report_delay import insert_delay
from attcon.report_sufficiency import history_oracle


class ReportDelayTests(unittest.TestCase):
    def test_delays_preserve_labels_query_and_all_historical_records(self):
        batch = make_history_splits({"test": 8}, seed=7)["test"]
        original = batch.events.clone()
        for delay in (0, 1, 3, 6):
            changed = insert_delay(batch, delay)
            self.assertTrue(torch.equal(changed.events[:, -1], batch.events[:, -1]))
            self.assertTrue(torch.equal(changed.events[:, :5], batch.events[:, :5]))
            self.assertTrue(torch.equal(history_oracle(changed.events, HistoryConfig()), batch.report_labels(HistoryConfig())))
            self.assertEqual(changed.events.shape[1], 6 + delay)
        self.assertTrue(torch.equal(batch.events, original))

    def test_zero_delay_is_identity_and_negative_delay_is_rejected(self):
        batch = make_history_splits({"test": 2}, seed=7)["test"]
        self.assertIs(insert_delay(batch, 0), batch)
        with self.assertRaises(ValueError):
            insert_delay(batch, -1)


if __name__ == "__main__":
    unittest.main()
