"""Independent first-pass toy checks only: no encoder, checkpoint, world or model."""
import dataclasses
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import attention_content_form_primitives_v1 as p


def maps(dtype=torch.float32):
    return torch.arange(-64, 64, dtype=dtype).reshape(2, 8, 8)


def state(dtype=torch.float32):
    value = maps(dtype)
    return p.AddressedState((
        p.AddressedItem.from_tensors('east', value, value.flip(-1), True, 'E', -1.125, 2, 3),
        p.AddressedItem.from_tensors('west', -value, value, True, 'W', 2.25, 1, 4),
        p.AddressedItem.from_tensors('unseen', value, value, False, None, 0., None, None),
    ), 5)


class IndependentToyChecks(unittest.TestCase):
    def test_all_dtypes_rectangular_noncontiguous_reference(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            value = maps(dtype).transpose(-1, -2)[:, :, :4]
            self.assertFalse(value.is_contiguous())
            for bandwidth in (1, 2, 4):
                actual = p.block_average(value, bandwidth)
                expected = torch.empty_like(value)
                for c in range(value.shape[0]):
                    for y in range(0, value.shape[1], bandwidth):
                        for x in range(0, value.shape[2], bandwidth):
                            samples = [float(value[c, yy, xx])
                                       for yy in range(y, y + bandwidth)
                                       for xx in range(x, x + bandwidth)]
                            mean = sum(samples) / len(samples)
                            expected[c, y:y+bandwidth, x:x+bandwidth] = mean
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(actual.dtype, dtype)
                self.assertEqual(actual.shape, value.shape)
                self.assertNotEqual(actual.data_ptr(), value.data_ptr())

    def test_copied_source_accessors_and_both_exports_own_their_values(self):
        supplied = maps()
        item = p.AddressedItem.from_tensors('only', supplied, supplied, True, 'id', 1., 0, 0)
        before = p.AddressedState((item,), 0)
        expected = maps()
        supplied.fill_(9.)
        item.source_v.tensor().fill_(10.)
        item.actual_w.tensor().fill_(11.)
        current = p.current_export(before)
        current['items'][0]['actual_current_w'][0][0][0] = 12345.
        prospective = p.prospective_export(before, p.PendingOperator('only', 4, 0, 1))
        prospective['operator_result_from_saved_v'][0][0][0] = 12345.
        self.assertTrue(torch.equal(item.source_v.tensor(), expected))
        self.assertTrue(torch.equal(item.actual_w.tensor(), expected))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            item.raw_logit = 99.
        with self.assertRaises(TypeError):
            item.source_v.values[0] = 99.

    def test_sequence_fixed_source_logit_and_address_isolation(self):
        before = state()
        first = p.execute_endogenous(before, p.PendingOperator('east', 4, 5, 8))
        second = p.execute_endogenous(first, p.PendingOperator('east', 1, 8, 9))
        for old, new in zip(before.items, second.items):
            self.assertEqual(old.source_v, new.source_v)
            self.assertEqual(old.identity, new.identity)
            self.assertEqual(old.raw_logit, new.raw_logit)
            self.assertEqual(old.delivered_at, new.delivered_at)
        self.assertEqual(before.item('west'), second.item('west'))
        self.assertEqual(before.item('unseen'), second.item('unseen'))
        self.assertTrue(torch.equal(second.item('east').actual_w.tensor(), before.item('east').source_v.tensor()))
        self.assertEqual(before.time, 5)
        self.assertEqual(before.history, ())
        self.assertEqual(first.history, second.history[:1])
        self.assertEqual(second.item('east').w_completed_at, 9)
        self.assertEqual(len(second.history), 2)

    def test_actual_exogenous_matching_has_audit_only_driver(self):
        for address in ('east', 'west'):
            for bandwidth in (1, 2, 4):
                before = state()
                pending = p.PendingOperator(address, bandwidth, 5, 10)
                endo = p.execute_endogenous(before, pending)
                exo = p.execute_exogenous(before, pending)
                self.assertEqual(endo.items, exo.items)
                self.assertEqual(endo.time, exo.time)
                self.assertEqual(p.current_export(endo), p.current_export(exo))
                self.assertEqual(p.audit_history(endo)[0]['driver'], 'endogenous')
                self.assertEqual(p.audit_history(exo)[0]['driver'], 'exogenous')
                self.assertNotIn('driver', str(p.current_export(endo)))
                self.assertNotIn('driver', str(p.prospective_export(before, pending)))

    def test_current_actual_and_analytic_prospective_are_distinct(self):
        before = state()
        pending = p.PendingOperator('east', 2, 5, 12)
        current = p.current_export(before)
        prospective = p.prospective_export(before, pending)
        self.assertNotEqual(current['schema_version'], prospective['schema_version'])
        self.assertEqual(current['snapshot_time'], 5)
        self.assertEqual(current['items'][0]['actual_w_completed_at'], 3)
        self.assertEqual(prospective['source_snapshot_time'], 5)
        self.assertEqual(prospective['issued_at'], 5)
        self.assertEqual(prospective['prospective_at'], 12)
        self.assertFalse(prospective['learned_prediction'])
        self.assertEqual(prospective['quantity'], 'analytic_operator_result_not_observed_or_learned')
        self.assertEqual(before, state())
        self.assertEqual(prospective['operator_result_from_saved_v'],
                         p.execute_endogenous(before, pending).item('east').actual_w.tensor().tolist())
        self.assertNotEqual(current['items'][0]['actual_current_w'], prospective['operator_result_from_saved_v'])

    def test_unseen_unknown_stale_and_nonforward_refused(self):
        before = state()
        row = p.current_export(before)['items'][2]
        for field in ('identity_record', 'delivered_at', 'actual_w_completed_at', 'actual_current_w'):
            self.assertIsNone(row[field])
        for operation in (p.execute_endogenous, p.execute_exogenous, p.prospective_export):
            for address in ('unseen', 'absent'):
                with self.assertRaises(ValueError):
                    operation(before, p.PendingOperator(address, 2, 5, 6))
            with self.assertRaises(ValueError):
                operation(before, p.PendingOperator('east', 2, 4, 6))
        for end in (4, 5):
            with self.assertRaises(ValueError):
                p.PendingOperator('east', 2, 5, end)

    def test_known_mapping_baseline_discloses_oracle(self):
        before = state()
        baseline = p.known_operator_baseline(before, 'probe', {'probe': ('west', 2)}, 7)
        self.assertEqual(baseline['mapping_knowledge'], 'oracle_not_empirically_learned')
        self.assertEqual(baseline['baseline_kind'], 'analytic_known_operator')
        self.assertFalse(baseline['learned_prediction'])
        self.assertEqual(before, state())
        self.assertEqual(baseline['operator_result_from_saved_v'],
                         p.execute_exogenous(before, p.PendingOperator('west', 2, 5, 7)).item('west').actual_w.tensor().tolist())

    def test_saved_restore_exact_for_tensor_factories_with_history(self):
        saved = p.execute_exogenous(state(), p.PendingOperator('west', 2, 5, 8))
        altered = p.execute_endogenous(saved, p.PendingOperator('east', 4, 8, 9))
        restored = p.restore_saved(saved)
        self.assertEqual(restored, saved)
        self.assertIsNot(restored, saved)
        self.assertEqual(p.current_export(restored), p.current_export(saved))
        self.assertEqual(p.audit_history(restored), p.audit_history(saved))
        self.assertNotEqual(altered, saved)

    def test_saved_restore_exact_for_valid_direct_spatial_buffer(self):
        # The public constructor accepts finite noncanonical float32 values.
        # Exact saved-state restoration must also cover that accepted input.
        direct = p.SpatialBuffer((1, 4, 4), (0.1,) * 16, torch.float32)
        item = p.AddressedItem('direct', direct, direct, True, 'literal', 1., 0, 0)
        saved = p.AddressedState((item,), 0)
        restored = p.restore_saved(saved)
        self.assertEqual(p.current_export(restored), p.current_export(saved))
        self.assertTrue(torch.equal(restored.item('direct').source_v.tensor(), saved.item('direct').source_v.tensor()))
        self.assertEqual(restored, saved)


if __name__ == '__main__':
    unittest.main(verbosity=2)
