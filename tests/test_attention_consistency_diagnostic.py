import unittest
import torch
from scripts.diagnose_attention_consistency import process, SWITCH, STEPS, COUNT


class ConsistencyDiagnosticTests(unittest.TestCase):
    def test_transition_preserves_prefix_and_changes_command_mapping(self):
        x, tables, owners, commands = process(42, 'unchanged')
        shifted, changed, _, commands2 = process(42, 'command_offset')
        self.assertTrue(torch.equal(commands, commands2))
        self.assertTrue(torch.equal(x[:, :SWITCH], shifted[:, :SWITCH]))
        rows = torch.arange(COUNT)
        for t in range(SWITCH, STEPS):
            for c in range(4):
                self.assertTrue(torch.equal(changed[t][rows, c, owners[t]].argmax(-1),
                                            torch.full((COUNT,), (c + 2) % 4)))
                self.assertTrue(torch.equal(changed[t][rows, c, 1-owners[t]],
                                            tables[t][rows, c, 1-owners[t]]))

    def test_swap_changes_owner_and_observation_matches_executed_command(self):
        x, tables, owners, commands = process(42, 'owner_swap')
        _, _, old_owners, _ = process(42, 'unchanged')
        rows = torch.arange(COUNT)
        for t in range(STEPS):
            expected_owner = old_owners[t] if t < SWITCH else 1-old_owners[t]
            self.assertTrue(torch.equal(owners[t], expected_owner))
            self.assertTrue(torch.equal(x[:, t, :8], tables[t][rows, commands[:, t]].flatten(1)))


if __name__ == '__main__':
    unittest.main()
