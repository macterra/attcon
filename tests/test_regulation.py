from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attcon.history_reporting import make_history_splits, pad_features
from attcon.regulation import WIDTH, Readout, agent_for, assigned_delays, confidence_features, fit_readout, mixed_states, train_delay_agent, weight_fingerprint


class RegulationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.data = make_history_splits({"fit": 8}, seed=8)["fit"]

    def test_delays_are_balanced_and_shared_by_every_context_variant(self):
        delays = assigned_delays(self.data, 4)
        for group in self.data.group.unique():
            self.assertEqual(len(delays[self.data.group == group].unique()), 1)
        self.assertTrue(torch.equal(delays, assigned_delays(self.data, 4)))
        self.assertEqual(delays.unique(return_counts=True)[1].tolist(), [24] * 4)

    def test_architecture_parameter_counts_and_reporter_capacity(self):
        counts = {}
        for architecture in ("gru", "rnn_matched"):
            agent, config = agent_for(architecture)
            counts[architecture] = sum(p.numel() for p in agent.parameters())
            features = pad_features(torch.randn(12, config.hidden), WIDTH)
            self.assertEqual(sum(p.numel() for p in Readout(features).parameters()), 8711)
        self.assertEqual(counts, {"gru": 15942, "rnn_matched": 15876})

    def test_paired_training_preserves_initialization_and_update_budget(self):
        results = []
        for recipe in ("fixed", "variable"):
            torch.manual_seed(4)
            agent, _ = agent_for("gru")
            results.append(train_delay_agent(agent, self.data, seed=9, recipe=recipe, epochs=4))
            self.assertTrue(all(not p.requires_grad for p in agent.parameters()))
        self.assertEqual(results[0]["initial_sha256"], results[1]["initial_sha256"])
        self.assertEqual(results[0]["updates"], results[1]["updates"])
        self.assertGreater(results[1]["recurrent_example_steps"], results[0]["recurrent_example_steps"])

    def test_mixed_states_match_individual_delayed_forwards(self):
        from attcon.report_delay import insert_delay
        agent, _ = agent_for("gru")
        states, delays = mixed_states(agent, self.data, 7)
        for delay in (0, 1, 3, 6):
            indices = torch.where(delays == delay)[0]
            with torch.no_grad():
                expected = agent.state(insert_delay(self.data.subset(indices), delay).events)
            self.assertTrue(torch.equal(states[indices], expected))

    def test_readout_fitting_cannot_update_agent(self):
        agent, _ = agent_for("gru")
        before = weight_fingerprint(agent)
        features = pad_features(agent.state(self.data.events), WIDTH)
        fit_readout(features, self.data.value, seed=7, l2=0.001, steps=2)
        self.assertEqual(before, weight_fingerprint(agent))
        self.assertTrue(all(p.grad is None for p in agent.parameters()))

    def test_confidence_features_distinguish_uniform_and_sharp_distributions(self):
        logits = torch.zeros(2, 6)
        logits[1, 2] = 20
        features = confidence_features(logits)
        self.assertAlmostEqual(features[0, 1].item(), 1, places=5)
        self.assertLess(features[1, 1].item(), 1e-5)
        self.assertEqual(features[:, 2:].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
