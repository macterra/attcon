import unittest
import torch
from attcon.functional_model import paired_process, execute_alternatives, buffer_contents, neutral_record
from attcon.predictive_attention import Forecast


class FunctionalModelTests(unittest.TestCase):
    def test_physical_alternatives_isolate_controlled_channel(self):
        for owner, process in enumerate(paired_process(73, count=16)):
            executed = execute_alternatives(process)
            independent = 1-owner
            self.assertTrue(torch.equal(executed[:, 0, independent], executed[:, 3, independent]))
            self.assertFalse(torch.equal(executed[:, 0, owner], executed[:, 3, owner]))
            for c in range(4):
                torch.testing.assert_close(executed[:, c, owner, c], torch.full((16,), .8))
            before = process.recovery.clone()
            execute_alternatives(process, effects=process.next_effects.flip(-2))
            self.assertTrue(torch.equal(before, process.recovery))

    def test_pair_shares_commands_quality_and_initial_uncertainty(self):
        a, b = paired_process(73, count=16)
        self.assertTrue(torch.equal(a.observations[..., -4:], b.observations[..., -4:]))
        qa = a.observations[..., 8:16].reshape(16, 8, 2, 4).sum(-1)
        qb = b.observations[..., 8:16].reshape(16, 8, 2, 4).sum(-1)
        self.assertTrue(torch.equal(qa, qb))
        visual = torch.softmax(torch.randn(16, 2, 4, 4), -1).repeat(1, 1, 1, 2)
        initial = buffer_contents(visual, torch.zeros(16, 2, 4))
        self.assertTrue(bool((initial == .25).all()))

    def test_neutral_render_preserves_distributions_and_readout(self):
        process = paired_process(73, count=1)[0]
        current = Forecast(process.allocation[:, -1],
                           process.recovery[:, -1, None].expand(-1, 3, -1, -1), process.next_effects)
        visual = torch.softmax(torch.randn(1, 2, 4, 4), -1).repeat(1, 1, 1, 2)
        recovery = execute_alternatives(process)
        payload = neutral_record(visual, current, recovery, process, 0, (2, 1, 0))
        self.assertEqual(payload['output_node'], 'n0')
        for trial in payload['predicted_by_command']:
            node = {x['node']: x['color_and_shape_distributions'] for x in trial['nodes']}
            self.assertEqual(node['n0'], node['n2'])
            for values in node.values():
                for row in values:
                    self.assertAlmostEqual(sum(row[:4]), 1, places=4)
                    self.assertAlmostEqual(sum(row[4:]), 1, places=4)
        self.assertNotIn('physical_owner', payload)
        self.assertNotIn('condition', payload)


if __name__ == '__main__':
    unittest.main()
