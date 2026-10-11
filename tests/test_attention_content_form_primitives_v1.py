import dataclasses
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from attention_content_form_primitives_v1 import (
    AddressedItem, AddressedState, PendingOperator, SpatialBuffer, audit_history,
    block_average, current_export, execute_endogenous, execute_exogenous,
    known_operator_baseline, prospective_export, restore_saved,
)


def toy_map():
    # The left/right addressed relation survives block-2 pooling, while within-
    # block alternating arrangement is removed. Neither relation is qualia.
    return torch.tensor([[[1., 3., 9., 11.], [3., 1., 11., 9.],
                          [1., 3., 9., 11.], [3., 1., 11., 9.]]], dtype=torch.float64)


def state():
    first, second = toy_map(), toy_map().flip(-1)
    return AddressedState((
        AddressedItem.from_tensors('a', first, first, True, 'left-low/right-high', 1.25, 2, 2),
        AddressedItem.from_tensors('b', second, second, True, 'left-high/right-low', -3.5, 3, 3),
        AddressedItem.from_tensors('unseen', torch.zeros_like(first), torch.zeros_like(first),
                                  False, None, 0.0, None, None),
    ), 3)


class ContentFormPrimitiveTests(unittest.TestCase):
    def test_exact_block_means_preserve_shape_dtype_and_coarse_relation(self):
        source = toy_map()
        fine = block_average(source, 1)
        coarse = block_average(source, 2)
        whole = block_average(source, 4)
        self.assertTrue(torch.equal(fine, source))
        self.assertNotEqual(fine.data_ptr(), source.data_ptr())
        self.assertTrue(torch.equal(coarse, torch.tensor([[[2., 2., 10., 10.]] * 4], dtype=source.dtype)))
        self.assertTrue(torch.equal(whole, torch.full_like(source, 6.)))
        for result in (fine, coarse, whole):
            self.assertEqual(result.shape, source.shape)
            self.assertEqual(result.dtype, source.dtype)
            self.assertEqual(result.mean().item(), source.mean().item())
        self.assertLess(coarse[..., :2].mean(), coarse[..., 2:].mean())
        self.assertNotEqual(source[0, 0, 0].item(), source[0, 0, 1].item())
        self.assertEqual(coarse[0, 0, 0].item(), coarse[0, 0, 1].item())

    def test_content_specific_relation_and_address_isolation(self):
        before = state()
        after = execute_endogenous(before, PendingOperator('a', 2, 3, 4))
        self.assertTrue(torch.equal(after.item('b').actual_w.tensor(), before.item('b').actual_w.tensor()))
        self.assertEqual(after.item('b'), before.item('b'))
        self.assertEqual(after.item('unseen'), before.item('unseen'))
        self.assertLess(after.item('a').actual_w.tensor()[..., :2].mean(),
                        after.item('a').actual_w.tensor()[..., 2:].mean())
        other = execute_endogenous(before, PendingOperator('b', 2, 3, 4))
        self.assertGreater(other.item('b').actual_w.tensor()[..., :2].mean(),
                           other.item('b').actual_w.tensor()[..., 2:].mean())

    def test_raw_source_identity_logit_and_prior_times_are_invariant(self):
        before = state()
        after = execute_endogenous(before, PendingOperator('a', 2, 3, 7))
        for old, new in zip(before.items, after.items):
            self.assertEqual(old.source_v, new.source_v)
            self.assertEqual(old.identity, new.identity)
            self.assertEqual(old.raw_logit, new.raw_logit)
            self.assertEqual(old.delivered_at, new.delivered_at)
        self.assertEqual(before.time, 3)
        self.assertEqual(before.item('a').w_completed_at, 2)
        self.assertEqual(before.history, ())
        self.assertEqual(after.time, 7)
        self.assertEqual(after.item('a').w_completed_at, 7)
        self.assertEqual(after.history[0].issued_at, 3)
        self.assertEqual(after.history[0].completed_at, 7)

    def test_operations_use_saved_v_not_cumulative_lossy_w(self):
        before = state()
        flattened = execute_endogenous(before, PendingOperator('a', 4, 3, 4))
        finer = execute_endogenous(flattened, PendingOperator('a', 1, 4, 5))
        self.assertTrue(torch.equal(finer.item('a').actual_w.tensor(), before.item('a').source_v.tensor()))
        self.assertEqual(finer.history[:1], flattened.history)
        self.assertEqual(flattened.time, 4)
        self.assertEqual(len(flattened.history), 1)

    def test_exogenous_control_is_numerically_matched_with_audit_only_provenance(self):
        before = state()
        pending = PendingOperator('a', 2, 3, 4)
        endogenous = execute_endogenous(before, pending)
        exogenous = execute_exogenous(before, pending)
        self.assertEqual(endogenous.items, exogenous.items)
        self.assertEqual(current_export(endogenous), current_export(exogenous))
        self.assertNotEqual(audit_history(endogenous), audit_history(exogenous))
        self.assertNotIn('driver', str(current_export(endogenous)))
        self.assertNotIn('driver', str(prospective_export(before, pending)))

    def test_saved_restoration_is_exact_without_inverting_average(self):
        saved = execute_endogenous(state(), PendingOperator('b', 2, 3, 4))
        changed = execute_endogenous(saved, PendingOperator('a', 4, 4, 5))
        restored = restore_saved(saved)
        self.assertEqual(restored, saved)
        self.assertIsNot(restored, saved)
        self.assertEqual(current_export(restored), current_export(saved))
        self.assertNotEqual(current_export(changed), current_export(saved))
        self.assertEqual(restored.history, saved.history)
        self.assertEqual(restored.time, saved.time)

    def test_current_and_prospective_exports_have_separate_time_semantics(self):
        before = state()
        current = current_export(before)
        pending = PendingOperator('a', 2, 3, 9)
        alternative = prospective_export(before, pending)
        self.assertNotEqual(current['schema_version'], alternative['schema_version'])
        self.assertEqual(current['snapshot_time'], 3)
        self.assertEqual(current['items'][0]['actual_w_completed_at'], 2)
        self.assertIsNone(current['items'][2]['actual_current_w'])
        self.assertEqual(alternative['source_snapshot_time'], 3)
        self.assertEqual(alternative['prospective_at'], 9)
        self.assertFalse(alternative['learned_prediction'])
        self.assertNotIn('actual_current_w', alternative)
        self.assertEqual(before, state())
        self.assertEqual(alternative['operator_result_from_saved_v'],
                         execute_endogenous(before, pending).item('a').actual_w.tensor().tolist())

    def test_analytic_baseline_discloses_mapping_oracle(self):
        before = state()
        baseline = known_operator_baseline(before, 'k', {'k': ('a', 2)}, 4)
        self.assertEqual(baseline['mapping_knowledge'], 'oracle_not_empirically_learned')
        self.assertEqual(baseline['baseline_kind'], 'analytic_known_operator')
        self.assertFalse(baseline['learned_prediction'])
        with self.assertRaises(ValueError):
            known_operator_baseline(before, 'unknown', {'k': ('a', 2)}, 4)

    def test_inputs_properties_outputs_and_exports_cannot_alias_mutate_state(self):
        source = toy_map()
        item = AddressedItem.from_tensors('a', source, source, True, 'identity', 1., 0, 0)
        before = AddressedState((item,), 0)
        source.zero_()
        access = item.source_v.tensor()
        access.zero_()
        self.assertTrue(torch.equal(item.source_v.tensor(), toy_map()))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            before.time = 12
        with self.assertRaises(TypeError):
            item.source_v.values[0] = 123.
        after = execute_endogenous(before, PendingOperator('a', 2, 0, 1))
        changed_view = after.item('a').actual_w.tensor()
        changed_view.zero_()
        exported = current_export(after)
        exported['items'][0]['actual_current_w'][0][0][0] = 999.
        self.assertEqual(after.item('a').actual_w.tensor()[0, 0, 0].item(), 2.)
        self.assertEqual(before.item('a').actual_w.tensor()[0, 0, 0].item(), 1.)

    def test_refuse_unknown_unseen_or_stale_pending_operator(self):
        for address in ('missing', 'unseen'):
            pending = PendingOperator(address, 2, 3, 4)
            for operation in (execute_endogenous, execute_exogenous, prospective_export):
                with self.assertRaises(ValueError):
                    operation(state(), pending)
        with self.assertRaises(ValueError):
            execute_endogenous(state(), PendingOperator('a', 2, 2, 4))
        with self.assertRaises(ValueError):
            PendingOperator('a', 2, 3, 3)

    def test_refuse_invalid_shapes_bandwidths_and_nonfinite_values(self):
        for bandwidth in (0, 3, 8, 2.0, True):
            with self.assertRaises(ValueError):
                block_average(toy_map(), bandwidth)
            with self.assertRaises(ValueError):
                PendingOperator('a', bandwidth, 3, 4)
        for source in (torch.zeros(4, 4), torch.zeros(1, 0, 4), torch.zeros(1, 5, 4),
                       torch.zeros(1, 4, 4, dtype=torch.int64)):
            with self.assertRaises(ValueError):
                block_average(source, 2)
        for value in (float('nan'), float('inf'), -float('inf')):
            source = toy_map()
            source[0, 0, 0] = value
            with self.assertRaises(ValueError):
                SpatialBuffer.from_tensor(source)
            with self.assertRaises(ValueError):
                block_average(source, 2)
        with self.assertRaises(ValueError):
            AddressedItem.from_tensors('a', torch.zeros(1, 2, 4), torch.zeros(1, 2, 4),
                                      True, 'identity', 1., 0, 0)

    def test_refuse_state_alias_containers_and_invalid_event_times(self):
        before = state()
        with self.assertRaises(ValueError):
            AddressedState(list(before.items), before.time)
        with self.assertRaises(ValueError):
            AddressedState(before.items + (before.items[0],), before.time)
        with self.assertRaises(ValueError):
            AddressedState(before.items, 1)
        with self.assertRaises(ValueError):
            AddressedItem.from_tensors('a', toy_map(), toy_map(), True, 'id', float('nan'), 0, 0)


if __name__ == '__main__':
    unittest.main()
