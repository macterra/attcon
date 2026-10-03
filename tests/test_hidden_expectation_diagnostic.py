import unittest
import torch
from scripts.diagnose_hidden_expectation import head_projector, paired_streams


class HiddenExpectationTests(unittest.TestCase):
    def test_projector_separates_readable_and_null_changes(self):
        g = torch.Generator().manual_seed(14)
        weight = torch.randn(5, 12, generator=g)
        weight[-1] = weight[0]
        projector, rank = head_projector(weight)
        self.assertEqual(rank, 4)
        delta = torch.randn(17, 12, generator=g)
        row = delta @ projector
        null = delta - row
        torch.testing.assert_close(row @ weight.T, delta @ weight.T)
        torch.testing.assert_close(null @ weight.T, torch.zeros(17, 5), atol=2e-6, rtol=0)

    def test_paired_histories_share_commands_and_physical_replay(self):
        streams, tables = paired_streams(42, count=16)
        self.assertTrue(torch.equal(streams[0][..., -4:], streams[1][..., -4:]))
        torch.testing.assert_close(tables[0][:, :, :, 1], tables[1][:, :, :, 0])
        for owner in (0, 1):
            for c in range(4):
                self.assertTrue(bool((tables[owner][:, :, c, owner].argmax(-1) == c).all()))


if __name__ == '__main__':
    unittest.main()
