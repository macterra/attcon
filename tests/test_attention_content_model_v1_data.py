"""Handcrafted evaluator fixtures only; no registered worlds or source artifacts."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import attention_content_model_v1_data as data


class EpisodeDataTests(unittest.TestCase):
    def setUp(self):
        self.v = torch.arange(2 * 4 * 24 * 4 * 4, dtype=torch.float32).reshape(2, 4, 24, 4, 4) / 100
        self.mapping = torch.tensor(data.MAPPINGS[:2], dtype=torch.long)[:, None].repeat(1, 4, 1)
        self.addresses = torch.arange(15).remainder(4)[None].repeat(2, 1)
        self.symbols = torch.arange(15).remainder(3)[None].repeat(2, 1)

    def test_same_v_fork_recovers_exact_input_after_lossy_command(self):
        addresses = torch.tensor([0, 1])
        degraded = data.execute_command(self.v, self.v, addresses, torch.tensor([2, 1]), self.mapping)
        self.assertFalse(torch.equal(degraded[0, 0], self.v[0, 0]))
        recovered = data.execute_command(self.v, degraded, addresses, torch.tensor([0, 0]), self.mapping)
        self.assertTrue(torch.equal(recovered, self.v))
        self.assertTrue(torch.equal(self.v, torch.arange(self.v.numel(), dtype=torch.float32).reshape_as(self.v) / 100))

    def test_common_spine_is_exact_matching_alternative_with_nonselected_copy(self):
        episode = data.actual_episode(self.v, self.addresses, self.symbols, self.mapping)
        self.assertEqual(episode['actual_completed_w'].shape, (2, 16, 4, 24, 4, 4))
        self.assertEqual(episode['actual_forks'].shape, (2, 15, 3, 4, 24, 4, 4))
        for time in range(15):
            for world in range(2):
                self.assertTrue(torch.equal(episode['actual_completed_w'][world, time + 1],
                    episode['actual_forks'][world, time, int(self.symbols[world, time])]))
                for address in range(4):
                    if address != int(self.addresses[world, time]):
                        self.assertTrue(torch.equal(episode['actual_forks'][world, time, :, address],
                            episode['actual_completed_w'][world, time, address].expand(3, 24, 4, 4)))

    def test_donor_has_t_transitions_but_only_t_minus_one_permitted_reveals(self):
        schedule = {'evaluator_mapping_indices': torch.tensor([[0] * 4, [1] * 4]),
                    'command_addresses': self.addresses, 'command_symbols': self.symbols}
        for time in (12, 13, 14):
            donor = data.donor_prefix(self.v, schedule, time, torch.tensor([0, 2]))
            self.assertEqual(donor['actual_completed_w'].shape[1], time + 1)
            self.assertEqual(donor['executor_command_indices'], tuple(range(time)))
            self.assertEqual(donor['permitted_feedback_command_indices'], tuple(range(time - 1)))
            self.assertEqual(donor['evaluator_mapping_indices'][0].tolist(), [1, 0, 0, 0])
            self.assertEqual(donor['evaluator_mapping_indices'][1].tolist(), [1, 1, 2, 1])

    def test_sealed_split_is_rejected_before_renderer_or_extractor(self):
        with patch.object(data, 'render_world', side_effect=AssertionError('sealed renderer called')) as renderer:
            with self.assertRaises(ValueError):
                data.collect_worlds(1301, 'heldout', object(), released_split='fit')
            renderer.assert_not_called()

    def test_actual_operator_family_matches_executor_selected_maps(self):
        family = data.operator_family(self.v)
        self.assertEqual(family.shape, (2, 4, 3, 24, 4, 4))
        for symbol in range(3):
            actual = data.execute_command(self.v, self.v, torch.tensor([0, 1]),
                                          torch.full((2,), symbol), self.mapping)
            for world, address in enumerate((0, 1)):
                band = int(self.mapping[world, address, symbol])
                self.assertTrue(torch.equal(actual[world, address],
                    family[world, address, (1, 2, 4).index(band)]))

    def test_invalid_common_commands_are_rejected_before_any_fork(self):
        for invalid in (-1, 3):
            symbols = self.symbols.clone()
            symbols[0, 0] = invalid
            with patch.object(data, 'execute_command', side_effect=AssertionError('fork called')) as executor:
                with self.assertRaises(ValueError):
                    data.actual_episode(self.v, self.addresses, symbols, self.mapping)
                executor.assert_not_called()
        for addresses, symbols in ((self.addresses.float(), self.symbols),
                                   (self.addresses, self.symbols.float())):
            with self.assertRaises(ValueError):
                data.actual_episode(self.v, addresses, symbols, self.mapping)

    def test_invalid_mapping_dtype_permutation_and_donor_indices_are_rejected(self):
        fractional = self.mapping.float()
        fractional[0, 0, 0] = 1.9
        duplicate = self.mapping.clone()
        duplicate[0, 0] = torch.tensor([1, 1, 4])
        for mapping in (fractional, duplicate, self.mapping.float()):
            with self.assertRaises(ValueError):
                data.execute_command(self.v, self.v, torch.tensor([0, 1]),
                                     torch.tensor([0, 0]), mapping)
        for invalid in (-1, 6):
            indices = torch.tensor([[0] * 4, [1] * 4])
            indices[0, 0] = invalid
            schedule = {'evaluator_mapping_indices': indices,
                        'command_addresses': self.addresses, 'command_symbols': self.symbols}
            with self.assertRaises(ValueError):
                data.donor_prefix(self.v, schedule, 12, torch.tensor([0, 2]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
