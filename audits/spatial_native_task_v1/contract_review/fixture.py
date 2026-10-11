"""Independent tiny software fixtures; no renderer/checkpoint/dataset/fit calls."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from attcon.bound_content import VisualEncoder
from spatial_native_task_v1 import (
    AddressedItem, AddressedState, ChannelStandardizer, FrozenSpatialExtractor,
    LocalOccupancyHead, NativeController, NativeItemEvidence, NativeSnapshot,
    PendingOperator, SpatialBuffer, execute_endogenous, execute_exogenous,
    occupancy_loss, restore_saved, role_fields,
)


def fixed_head():
    with torch.random.fork_rng():
        head = LocalOccupancyHead()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.net[0].weight[0, 0] = 1
        head.net[0].weight[1, 1] = 1
        head.net[2].weight[1, 0] = 2
        head.net[2].weight[2, 1] = 3
    return head


def one_item(value, address='l0'):
    item = AddressedItem.from_tensors(address, value, value, True,
                                     'opaque-role-pair', 13., 0, 0)
    return AddressedState((item,), 0)


def snapshot_with_fields(fields):
    return NativeSnapshot(0, (NativeItemEvidence('l0',
        SpatialBuffer.from_tensor(torch.zeros(3, 4, 4)),
        SpatialBuffer.from_tensor(fields)),))


class IndependentNativeFixtures(unittest.TestCase):
    def test_conditional_softmax_all_background_and_independent_odds_reference(self):
        logits = torch.empty(3, 4, 4)
        logits[0].fill_(100)
        logits[1].fill_(-100)
        logits[2].fill_(-200)
        fields = role_fields(logits)
        self.assertTrue(torch.equal(fields, torch.full((2, 4, 4), .0625)))
        logits[0, 2, 1] = 90
        contrast = (logits[1:].double() - logits[:1].double()).flatten(1)
        odds = (contrast - contrast.max(1, keepdim=True).values).exp()
        expected = (odds / odds.sum(1, keepdim=True)).reshape(2, 4, 4).float()
        self.assertTrue(torch.equal(role_fields(logits), expected))
        self.assertEqual(tuple(fields.sum((1, 2)).tolist()), (1., 1.))
        self.assertEqual(int(role_fields(logits)[0].argmax()), 9)

    def test_cross_entropy_matches_manual_reference_and_does_not_mutate_source(self):
        logits = torch.linspace(-2, 3, 96).reshape(2, 3, 4, 4).clone().requires_grad_()
        original = logits.detach().clone()
        labels = torch.zeros(2, 4, 4, dtype=torch.long)
        labels[0, 1, 0] = 1
        labels[0, 2, 3] = 2
        labels[1, 0, 3] = 1
        labels[1, 3, 0] = 2
        reference = (logits.logsumexp(1) - logits.gather(1, labels[:, None]).squeeze(1)).mean()
        measured = occupancy_loss(logits, labels)
        self.assertTrue(torch.allclose(measured, reference, atol=1e-7, rtol=1e-6))
        fields_before = role_fields(logits)
        occupancy_loss(logits, labels.roll(1, -1)).backward()
        self.assertTrue(torch.equal(logits.detach(), original))
        self.assertTrue(torch.equal(role_fields(logits), fields_before))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_centroid_order_disagrees_with_identical_half_masses(self):
        fields = torch.zeros(2, 4, 4, dtype=torch.float64)
        fields[0, 0, 0], fields[0, 0, 2] = .6, .4
        fields[1, 0, 1], fields[1, 0, 3] = .6, .4
        self.assertTrue(torch.equal(fields[0, :, :2].sum(), fields[1, :, :2].sum()))
        result = NativeController(fixed_head()).coarse_order(snapshot_with_fields(fields), 'l0')
        self.assertEqual(result.order, 'red_left_of_blue')
        self.assertAlmostEqual(result.expected_columns[0], .8)
        self.assertAlmostEqual(result.expected_columns[1], 1.8)

    def test_coarse_unknown_band_and_reverse_from_fields(self):
        controller = NativeController(fixed_head())
        for shift, answer in ((1e-7, 'unknown'), (1e-5, 'blue_left_of_red')):
            fields = torch.zeros(2, 4, 4, dtype=torch.float64)
            fields[0, 0, 1], fields[0, 0, 2] = 1 - shift, shift
            fields[1, 0, 1] = 1
            self.assertEqual(controller.coarse_order(snapshot_with_fields(fields), 'l0').order, answer)

    def test_t2_four_ties_and_balanced_fine_accuracy_ceiling_without_worlds(self):
        controller = NativeController(fixed_head())
        successes = 0
        for row in range(4):
            for column in range(4):
                value = torch.zeros(24, 4, 4)
                value[0, row, column] = 8
                value[1, (row + 2) % 4, (column + 2) % 4] = 8
                source = one_item(value)
                transformed = execute_endogenous(source, PendingOperator('l0', 2, 0, 1))
                snapshot = controller.consume(transformed)
                action = controller.target(snapshot, 'l0', 'red')
                self.assertEqual(len(action.tied_cells), 4)
                self.assertEqual((action.row % 2, action.column % 2), (0, 0))
                self.assertEqual(set(action.tied_cells),
                    {(r, c) for r in range(2*(row//2), 2*(row//2)+2)
                            for c in range(2*(column//2), 2*(column//2)+2)})
                successes += (action.row, action.column) == (row, column)
                logits = snapshot.item('l0').local_logits.tensor()
                for r in (0, 2):
                    for c in (0, 2):
                        block = logits[:, r:r+2, c:c+2]
                        self.assertTrue(torch.equal(block, block[:, :1, :1].expand_as(block)))
        self.assertEqual(successes, 4)

    def test_t4_all_ties_coarse_unknown_and_same_w_driver_actions(self):
        value = torch.zeros(24, 4, 4)
        value[0, 3, 0], value[1, 0, 3] = 12, 12
        source = one_item(value)
        controller = NativeController(fixed_head())
        for bandwidth in (1, 2, 4):
            operation = PendingOperator('l0', bandwidth, 0, 1)
            endogenous = execute_endogenous(source, operation)
            exogenous = execute_exogenous(source, operation)
            a, b = controller.consume(endogenous), controller.consume(exogenous)
            self.assertEqual(a, b)
            self.assertNotEqual(endogenous.history, exogenous.history)
            for role in ('red', 'blue'):
                self.assertEqual(controller.target(a, 'l0', role), controller.target(b, 'l0', role))
            self.assertEqual(controller.coarse_order(a, 'l0'), controller.coarse_order(b, 'l0'))
            if bandwidth == 4:
                self.assertEqual(len(controller.target(a, 'l0', 'red').tied_cells), 16)
                self.assertEqual(controller.coarse_order(a, 'l0').order, 'unknown')
        self.assertEqual(restore_saved(source), source)

    def test_consumer_ignores_metadata_truth_and_v_but_follows_actual_w(self):
        value = torch.zeros(24, 4, 4)
        value[0, 1, 2], value[1, 2, 0] = 9, 9
        source = one_item(value)
        controller = NativeController(fixed_head())
        before = controller.consume(source)
        decoy_item = replace(source.items[0], source_v=SpatialBuffer.from_tensor(torch.ones_like(value)),
                             identity='blue-before-red-decoy', raw_logit=-1234.)
        self.assertEqual(controller.consume(replace(source, items=(decoy_item,))), before)
        with self.assertRaises(TypeError):
            controller.consume(source, evaluator_truth='decoy')
        changed = replace(decoy_item, actual_w=SpatialBuffer.from_tensor(value.flip(-1)))
        after = controller.consume(replace(source, items=(changed,)))
        self.assertEqual(controller.target(before, 'l0', 'red').column, 2)
        self.assertEqual(controller.target(after, 'l0', 'red').column, 1)
        with self.assertRaises(ValueError):
            controller.target(before, 'absent', 'red')

    def test_spatial_permutation_preserves_fields_but_tie_action_not_equivariant(self):
        controller = NativeController(fixed_head())
        value = torch.zeros(24, 4, 4)
        value[0, 0, 0], value[0, 0, 1] = 7, 7
        original = controller.consume(one_item(value))
        moved_value = value.flatten(1).roll(1, -1).reshape_as(value)
        moved = controller.consume(one_item(moved_value))
        original_fields = original.item('l0').role_fields.tensor()
        self.assertTrue(torch.allclose(moved.item('l0').role_fields.tensor(),
            original_fields.flatten(1).roll(1, -1).reshape_as(original_fields), atol=1e-6, rtol=1e-5))
        # Put tied maxima at indices 0 and 15: rolling by +1 changes the first
        # selected maximum from 0 to 0 (old index15), rather than moving 0 to1.
        value[0].zero_()
        value[0, 0, 0], value[0, 3, 3] = 7, 7
        original = controller.consume(one_item(value))
        moved = controller.consume(one_item(value.flatten(1).roll(1, -1).reshape_as(value)))
        self.assertEqual(controller.target(original, 'l0', 'red').column, 0)
        action = controller.target(moved, 'l0', 'red')
        self.assertEqual((action.row, action.column), (0, 0))
        self.assertNotEqual((action.row, action.column), (0, 1))

    def test_standardizer_all_leading_spatial_axes_population_and_fixed_application(self):
        base = torch.arange(96, dtype=torch.float32).reshape(2, 3, 4, 4)
        offsets = torch.arange(24, dtype=torch.float32)
        maps = base.unsqueeze(2) + offsets.reshape(1, 1, 24, 1, 1)
        maps[:, :, 23].fill_(5)
        standardizer = ChannelStandardizer.fit(maps)
        expected_mean = tuple(float(base.mean()) + i for i in range(23)) + (5.,)
        expected_scale = (float(base.std(correction=0)),) * 23 + (1.,)
        self.assertEqual(standardizer.mean, expected_mean)
        self.assertEqual(standardizer.scale, expected_scale)
        held = torch.zeros_like(maps)
        applied = standardizer.apply(held)
        self.assertAlmostEqual(float(applied[1, 2, 0, 0, 0]), -expected_mean[0]/expected_scale[0], places=6)
        self.assertEqual(standardizer.mean, expected_mean)
        self.assertTrue(torch.equal(applied[:, :, 23], torch.full((2, 3, 4, 4), -5.)))

    def test_random_toy_extractor_native_slices_frozen_copies_and_orientation_scaling(self):
        with torch.random.fork_rng():
            torch.manual_seed(911)
            encoder = VisualEncoder()
            orientation = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        pixels = torch.linspace(0, 1, 1536).reshape(2, 3, 16, 16)
        mean, scale = torch.linspace(-.5, .5, 64), torch.linspace(.25, 2, 64)
        extractor = FrozenSpatialExtractor(encoder, orientation, mean, scale)
        parameters = {name: p.detach().clone() for name, p in extractor.encoder.named_parameters()}
        spatial = extractor.extract(pixels)
        measured = extractor.controls(spatial)
        with torch.no_grad():
            direct_spatial = encoder.net[:6](pixels)
            hidden = encoder.net[6:9](direct_spatial)
            self.assertTrue(torch.equal(spatial, direct_spatial))
            self.assertTrue(torch.equal(measured['category_logits'], encoder.net(pixels)))
            self.assertTrue(torch.equal(measured['orientation_logits'], orientation((hidden-mean)/scale)[:, 0]))
            next(encoder.parameters()).add_(100)
            mean.add_(100)
        self.assertTrue(torch.equal(extractor.extract(pixels), spatial))
        for name, parameter in extractor.encoder.named_parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
            self.assertTrue(torch.equal(parameter, parameters[name]))
        self.assertFalse(spatial.requires_grad)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
