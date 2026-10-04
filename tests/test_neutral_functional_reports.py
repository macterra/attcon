import copy
import unittest
import torch
from scripts.neutral_functional_reports import remap, model_swap, variants


class NeutralReportingTests(unittest.TestCase):
    def sample(self):
        payload = {'output_node': 'n2', 'observed_history': [{'command': 'k0'}],
            'predicted_current': [{'node': f'n{i}', 'color_and_shape_distributions': [[.25]*8]*4} for i in range(3)],
            'predicted_by_command': [{'command': f'k{c}', 'nodes': [{'node': f'n{i}',
                'color_and_shape_distributions': [[.25]*8]*4} for i in range(3)]} for c in range(4)]}
        return {'row': 0, 'physical_owner': 0, 'visual_seed': 1301, 'payload': payload}

    def test_identifier_roundtrip_changes_no_values(self):
        payload = self.sample()['payload']
        shifted = remap(payload, {'n0': 'a', 'n1': 'b', 'n2': 'c'})
        self.assertEqual(payload, remap(shifted, {'a': 'n0', 'b': 'n1', 'c': 'n2'}))
        self.assertEqual(shifted['observed_history'], payload['observed_history'])

    def test_model_swap_leaves_history_and_current_unchanged(self):
        payload = self.sample()['payload']
        changed = model_swap(payload, torch.ones(4, 2, 4, 8)*.125, (0, 1, 2))
        self.assertEqual(changed['predicted_current'], payload['predicted_current'])
        self.assertEqual(changed['observed_history'], payload['observed_history'])
        self.assertNotEqual(changed['predicted_by_command'], payload['predicted_by_command'])
        self.assertEqual(changed['predicted_by_command'][0]['nodes'][0]['color_and_shape_distributions'],
                         changed['predicted_by_command'][0]['nodes'][2]['color_and_shape_distributions'])

    def test_missing_and_conflict_remove_or_change_only_intended_fields(self):
        sample = self.sample(); before = copy.deepcopy(sample)
        stored = {1301: {'visual': torch.ones(1, 2, 4, 8)*.25,
                         0: {'model_only_recovery': torch.ones(1, 4, 2, 4)}}}
        out = variants(sample, stored)
        self.assertEqual(sample, before)
        self.assertEqual(out['restored'], out['neutral'])
        self.assertIsNone(out['missing_relation']['observed_history'])
        self.assertEqual(out['missing_relation']['predicted_current'], sample['payload']['predicted_current'])
        note = out['conflicting_description'].pop('informal_operator_note')
        self.assertTrue(note)
        self.assertEqual(out['conflicting_description'], out['neutral'])


if __name__ == '__main__':
    unittest.main()
