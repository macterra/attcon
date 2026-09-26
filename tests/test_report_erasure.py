from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attcon.report_erasure import erasure_delta, loss_report_metrics


class ReportErasureTests(unittest.TestCase):
    def test_erasure_removes_only_selected_displacement_and_restores_state(self):
        torch.manual_seed(2)
        states, center = torch.randn(12, 64), torch.randn(64)
        basis = torch.linalg.qr(torch.randn(64, 5)).Q
        delta = erasure_delta(states, center, basis, 1.0)
        changed = states + delta
        self.assertTrue(torch.allclose((changed - center) @ basis, torch.zeros(12, 5), atol=2e-6))
        complement = torch.eye(64) - basis @ basis.T
        self.assertTrue(torch.allclose(changed @ complement, states @ complement, atol=2e-6))
        self.assertTrue(torch.allclose(changed - delta, states, atol=1e-6))

    def test_nulls_match_individual_norms_and_zero_strength_is_identity(self):
        torch.manual_seed(2)
        states, center = torch.randn(12, 64), torch.randn(64)
        basis, null = torch.linalg.qr(torch.randn(64, 5)).Q, torch.linalg.qr(torch.randn(64, 5)).Q
        for strength in (0.0, 0.25, 0.5, 1.0):
            target = erasure_delta(states, center, basis, strength)
            control = erasure_delta(states, center, null, strength, basis)
            self.assertTrue(torch.allclose(target.norm(dim=1), control.norm(dim=1), atol=1e-6))
        self.assertEqual(erasure_delta(states, center, basis, 0).abs().sum().item(), 0)

    def test_correct_reports_on_choice_errors_are_not_called_unavailable(self):
        choice = torch.tensor([1, 1, 1])
        report = torch.tensor([0, 6, 2])
        target = torch.zeros(3, dtype=torch.long)
        result = loss_report_metrics(choice, report, target, 6, torch.ones(3, dtype=torch.bool))
        self.assertAlmostEqual(result["true_value_report_on_choice_error"], 1 / 3)
        self.assertAlmostEqual(result["unavailable_on_choice_error"], 1 / 3)
        self.assertAlmostEqual(result["incorrect_value_report_on_choice_error"], 1 / 3)
        empty = loss_report_metrics(choice, report, target, 6, torch.zeros(3, dtype=torch.bool))
        self.assertEqual(empty, {"count": 0})


if __name__ == "__main__":
    unittest.main()
