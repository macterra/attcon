from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attcon.history_reporting import HistoryConfig, make_history_splits, pad_features
from attcon.report_sufficiency import WIDTH, NonlinearReadout, SequenceAgent, StateInputReporter, batch_fingerprint, history_oracle, make_sufficiency_splits, select_nonlinear


class SufficiencyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_more_reporter_data_preserves_training_validation_and_test(self):
        small = make_sufficiency_splits(1901, 128)
        large = make_sufficiency_splits(1901, 512)
        for name in ("agent_train", "validation", "test"):
            self.assertEqual(batch_fingerprint(small[name]), batch_fingerprint(large[name]))
        self.assertTrue(torch.equal(small["report_fit"].events, large["report_fit"].events[:len(small["report_fit"])]))

    def test_oracle_uses_input_history_and_resolves_last_matching_record(self):
        config = HistoryConfig()
        batch = make_history_splits({"test": 5}, seed=44)["test"]
        self.assertTrue(torch.equal(history_oracle(batch.events, config), batch.report_labels(config)))
        altered = batch.events.clone()
        query = batch.query[0].item()
        altered[0, -2] = 0
        altered[0, -2, query] = 1
        altered[0, -2, config.keys + 4] = 1
        self.assertEqual(history_oracle(altered, config)[0].item(), 4)

    def test_all_feature_families_have_equal_capacity_without_truncation(self):
        for width in (6, 15, 64, 90):
            features = torch.randn(12, width)
            padded = pad_features(features, WIDTH)
            self.assertTrue(torch.equal(padded[:, :width], features))
            model = NonlinearReadout(padded)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 6663)

    def test_recurrent_architectures_accept_variable_delay_and_freeze(self):
        for architecture in ("gru", "rnn"):
            model = SequenceAgent(HistoryConfig(), architecture).requires_grad_(False)
            for length in (6, 9):
                state = model.state(torch.zeros(3, length, 15))
                self.assertEqual(state.shape, (3, 64))
                self.assertFalse(state.requires_grad)

    def test_fitting_reporters_does_not_update_agent_and_adapter_matches(self):
        batch = make_history_splits({"fit": 2}, seed=44)["fit"]
        agent = SequenceAgent(HistoryConfig())
        state = agent.state(batch.events)
        features = pad_features(state, WIDTH)
        result = select_nonlinear(features, batch.report_labels(HistoryConfig()), features, batch,
                                  HistoryConfig(), seed=3, steps=2)
        self.assertTrue(all(p.grad is None for p in agent.parameters()))
        self.assertTrue(torch.equal(StateInputReporter(result.model)(state), result.model(features)))
        self.assertEqual(len(result.candidates), 2)


if __name__ == "__main__":
    unittest.main()
