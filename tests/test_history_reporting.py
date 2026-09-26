from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attcon.history_reporting import (
    HistoryAgent, HistoryConfig, content_basis, fit_probe,
    intervention_metrics, make_history_splits, pad_features, split_checks, train_agent, transplant,
)


class HistoryReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.config = HistoryConfig()
        self.splits = make_history_splits({"train": 12, "test": 8}, seed=27)

    def test_paired_histories_hide_unseen_answer_and_match_final_observation(self) -> None:
        self.assertTrue(all(split_checks(self.splits, self.config).values()))
        for data in self.splits.values():
            for group in data.group.unique():
                batch = data.subset(torch.where(data.group == group)[0])
                unseen = batch.subset(torch.where(~batch.seen)[0])
                self.assertEqual(set(unseen.value.tolist()), set(range(self.config.values)))
                self.assertTrue(torch.equal(unseen.events, unseen.events[:1].expand_as(unseen.events)))
                query = batch.query[0].item()
                self.assertEqual(unseen.events[:, :-1, query].sum().item(), 0)
                self.assertTrue((batch.events[batch.seen, :-1, query].sum(dim=1) == 1).all())

    def test_full_protocol_has_no_cross_split_input_duplicates(self) -> None:
        for seed in (1729, 1730, 1731):
            splits = make_history_splits(
                {"train": 256, "fit": 128, "validation": 64, "test": 128}, seed=seed,
            )
            self.assertTrue(all(split_checks(splits, self.config).values()), seed)

    def test_fitting_reporter_cannot_update_frozen_agent(self) -> None:
        model = HistoryAgent(self.config)
        data = self.splits["train"]
        train_agent(model, data, seed=3, epochs=1)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        features = model.state(data.events)
        self.assertFalse(features.requires_grad)
        fit_probe(features, data.report_labels(self.config), self.config.values + 1, seed=5, steps=2)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value))

    def test_transplant_changes_only_selected_subspace(self) -> None:
        torch.manual_seed(8)
        states, donors = torch.randn(10, 64), torch.randn(10, 64)
        basis = torch.linalg.qr(torch.randn(64, 5)).Q
        result = transplant(states, donors, basis)
        self.assertTrue(torch.allclose(result @ basis, donors @ basis, atol=2e-6))
        complement = torch.eye(64) - basis @ basis.T
        self.assertTrue(torch.allclose(result @ complement, states @ complement, atol=2e-6))
        self.assertTrue(torch.equal(transplant(states, states, basis), states))

    def test_content_basis_uses_seen_labels_and_expected_rank(self) -> None:
        data = self.splits["train"]
        states = torch.zeros(len(data), self.config.hidden)
        states[data.seen, data.value[data.seen]] = 1
        states[~data.seen] = torch.randn_like(states[~data.seen]) * 100
        basis = content_basis(states, data, self.config)
        self.assertEqual(basis.shape, (64, 5))
        self.assertTrue(torch.allclose(basis.T @ basis, torch.eye(5), atol=1e-6))
        # Arbitrary unseen states must not contaminate the content directions.
        self.assertLess(basis[6:].abs().max().item(), 1e-6)

    def test_causal_scoring_recovers_known_content_and_matches_null_norms(self) -> None:
        data = self.splits["test"]
        states = torch.zeros(len(data), 64)
        states[data.seen, data.value[data.seen]] = 1
        states[torch.arange(len(data)), 6 + data.query] = 1
        agent = HistoryAgent()
        reporter = torch.nn.Linear(64, 7)
        query_probe = torch.nn.Linear(64, 8)
        with torch.no_grad():
            for module in (agent.choice, reporter, query_probe):
                module.weight.zero_()
                module.bias.zero_()
            agent.choice.weight[:, :6] = torch.eye(6)
            reporter.weight[:6, :6] = 3 * torch.eye(6)
            reporter.bias[6] = 1
            query_probe.weight[:, 6:14] = torch.eye(8)
        basis = content_basis(states, data, self.config)
        scores = intervention_metrics(agent, reporter, query_probe, states, data, basis, self.config)
        for key in ("joint_donor_follow", "eligible_fraction", "query_identity_stability", "report_access_stability"):
            self.assertEqual(scores[key], 1.0, key)
        random_basis = torch.linalg.qr(torch.randn(64, 5)).Q
        null = intervention_metrics(agent, reporter, query_probe, states, data, random_basis, self.config,
                                    norm_reference_basis=basis)
        self.assertAlmostEqual(null["mean_state_change_norm"], scores["mean_state_change_norm"], places=5)

    def test_capacity_matching_preserves_input_and_rejects_truncation(self) -> None:
        x = torch.randn(12, 15)
        padded = pad_features(x, 64)
        self.assertTrue(torch.equal(x, padded[:, :15]))
        self.assertEqual(padded[:, 15:].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            pad_features(x, 10)


if __name__ == "__main__":
    unittest.main()
