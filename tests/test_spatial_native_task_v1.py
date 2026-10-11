"""Software fixtures only: handcrafted tensors and a fresh random toy encoder."""
from collections import Counter
from dataclasses import replace
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from attcon.bound_content import VisualEncoder
from spatial_native_task_v1 import (
    ADDRESSES, AddressedItem, AddressedState, ChannelStandardizer,
    EvaluatorGridTruth, FrozenSpatialExtractor, LocalOccupancyHead,
    NativeController, PendingOperator, balanced_layouts, block_average,
    execute_endogenous, execute_exogenous, layout_for_world, occupancy_loss,
    render_world, restore_saved, role_fields,
)


def toy_head():
    head = LocalOccupancyHead()
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.net[0].weight[0, 0] = 1.
        head.net[0].weight[1, 1] = 1.
        head.net[2].weight[1, 0] = 1.
        head.net[2].weight[2, 1] = 1.
    return head


def toy_state():
    items = []
    for index, address in enumerate(ADDRESSES):
        value = torch.zeros(24, 4, 4)
        value[0, index, 0] = 8.
        value[1, 3 - index, 3] = 8.
        items.append(AddressedItem.from_tensors(address, value, value, True,
                                                'ordered-red-blue', 11., 0, 0))
    return AddressedState(tuple(items), 0)


class SpatialNativeTaskTests(unittest.TestCase):
    def test_balancing_per_address_role_and_vertical_order_without_rendering(self):
        # Metadata only. These are NOT generated fit/diagnostic images.
        layouts = balanced_layouts(32)
        for layout in layouts:
            self.assertEqual(sum(r[1] < b[1] for r, b in layout.cells), 2)
        for address in range(4):
            for role in range(2):
                self.assertEqual(Counter(x.cells[address][role] for x in layouts),
                                 Counter({(r, c): 2 for r in range(4) for c in range(4)}))
            vertical = Counter((r[0] > b[0]) - (r[0] < b[0])
                               for r, b in (x.cells[address] for x in layouts))
            self.assertEqual(vertical, {-1: 12, 0: 8, 1: 12})
        for address in range(4):
            self.assertEqual(len({layout_for_world(i).cells[address] for i in range(128)}), 128)

    def test_four_patch_renderer_fixture_deterministic_and_rng_isolated(self):
        before_rng = torch.get_rng_state().clone()
        first, second = render_world(17, 0), render_world(17, 0)
        self.assertTrue(torch.equal(first.pixels, second.pixels))
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(first.truth, second.truth)
        self.assertEqual(first.pixels.shape, (4, 3, 16, 16))
        self.assertEqual(first.pixels.dtype, torch.float32)
        self.assertTrue(((first.pixels >= 0) & (first.pixels <= 1)).all())
        self.assertTrue(all(.65 <= x <= 1 for row in first.intensities for x in row))
        self.assertTrue(.005 <= first.noise_sigma <= .02)
        for item, pair in enumerate(first.truth.cells):
            for role, (r, c) in enumerate(pair):
                cell = first.pixels[item, :, 4*r:4*r+4, 4*c:4*c+4]
                self.assertGreater(float(cell[0 if role == 0 else 2].max()), .6)

    def test_truth_occupancy_exact_one_each_and_validity(self):
        truth = layout_for_world(0)
        labels = truth.occupancy()
        self.assertEqual(labels.shape, (4, 4, 4))
        self.assertEqual(labels.dtype, torch.long)
        for item in range(4):
            self.assertEqual(torch.bincount(labels[item].flatten(), minlength=3).tolist(), [14, 1, 1])
        labels.zero_()
        self.assertEqual((truth.occupancy() == 1).sum().item(), 4)
        for cells in ([((0, 0), (0, 3))] * 4,
                      (((0, 0), (1, 1)),) * 4,
                      (((0, 0), (4, 3)),) * 4):
            with self.assertRaises(ValueError):
                EvaluatorGridTruth(cells)
        for count in (0, 1, 31, 33, True):
            with self.assertRaises(ValueError):
                balanced_layouts(count)

    def test_shared_head_parameter_count_and_location_equivariance(self):
        head = toy_head()
        self.assertEqual(sum(p.numel() for p in head.parameters()), 899)
        self.assertEqual(sum(p.numel() for p in LocalOccupancyHead(3).parameters()), 227)
        source = torch.arange(384, dtype=torch.float32).reshape(24, 4, 4)
        permutation = torch.tensor([15, 0, 6, 1, 8, 11, 3, 12, 2, 4, 13, 5, 7, 10, 14, 9])
        moved = source.flatten(-2)[:, permutation].reshape_as(source)
        expected = head(source).flatten(-2)[:, permutation].reshape(3, 4, 4)
        self.assertTrue(torch.equal(head(moved), expected))
        expected_fields = role_fields(head(source)).flatten(-2)[:, permutation].reshape(2, 4, 4)
        self.assertTrue(torch.allclose(role_fields(head(moved)), expected_fields, atol=1e-6, rtol=1e-5))

    def test_ce_targets_only_enter_supervision_function(self):
        logits = torch.zeros(4, 3, 4, 4, requires_grad=True)
        loss = occupancy_loss(logits, layout_for_world(0).occupancy())
        self.assertAlmostEqual(float(loss.detach()), float(torch.tensor(3.).log()), places=6)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        with self.assertRaises(ValueError):
            occupancy_loss(logits, torch.zeros(4, 4, 4))
        with self.assertRaises(ValueError):
            occupancy_loss(logits, torch.full((4, 4, 4), 3, dtype=torch.long))

    def test_role_minus_background_and_uniform_negative_evidence(self):
        logits = torch.zeros(3, 4, 4)
        logits[0] = 10.
        logits[1] = -10.
        logits[2] = -20.
        fields = role_fields(logits)
        self.assertTrue(torch.equal(fields, torch.full((2, 4, 4), 1 / 16)))
        logits[0, 0, 1] = -10.
        self.assertEqual(int(role_fields(logits)[0].argmax()), 1)
        shifted = logits + torch.arange(16).reshape(1, 4, 4)
        self.assertTrue(torch.equal(role_fields(logits), role_fields(shifted)))
        extreme = torch.full((3, 4, 4), torch.finfo(torch.float32).max)
        extreme[0] *= -1
        self.assertTrue(torch.isfinite(role_fields(extreme)).all())

    def test_uniform_controller_ties_are_row_major_and_coarse_unknown(self):
        source = toy_state()
        uniform = replace(source, items=tuple(replace(x, actual_w=x.actual_w.from_tensor(
            torch.zeros(24, 4, 4))) for x in source.items))
        controller = NativeController(toy_head())
        snapshot = controller.consume(uniform)
        action = controller.target(snapshot, 'l0', 'red')
        self.assertEqual((action.row, action.column), (0, 0))
        self.assertEqual(action.tied_cells, tuple((r, c) for r in range(4) for c in range(4)))
        coarse = controller.coarse_order(snapshot, 'l0')
        self.assertEqual(coarse.order, 'unknown')
        self.assertEqual(coarse.expected_columns, (1.5, 1.5))

    def test_native_target_and_coarse_paths_use_actual_w_only(self):
        controller = NativeController(toy_head())
        source = toy_state()
        snapshot = controller.consume(source)
        self.assertEqual(controller.target(snapshot, 'l1', 'red').row, 1)
        self.assertEqual(controller.target(snapshot, 'l1', 'blue').row, 2)
        self.assertEqual(controller.coarse_order(snapshot, 'l1').order, 'red_left_of_blue')
        decoy = replace(source, items=tuple(replace(x, identity='wrong-renderer-label', raw_logit=-999.,
                                                   source_v=x.source_v.from_tensor(torch.ones(24, 4, 4)))
                                           for x in source.items))
        self.assertEqual(controller.consume(decoy), snapshot)
        swapped = replace(source, items=tuple(replace(x, actual_w=x.actual_w.from_tensor(
            x.actual_w.tensor().flip(-1))) for x in source.items))
        changed = controller.consume(swapped)
        self.assertEqual(controller.target(changed, 'l1', 'red').column, 3)
        self.assertEqual(controller.coarse_order(changed, 'l1').order, 'blue_left_of_red')
        with self.assertRaises(TypeError):
            controller.consume(source, layout_for_world(0))
        with self.assertRaises(ValueError):
            controller.consume(PendingOperator('l0', 2, 0, 1))

    def test_address_permutation_moves_evidence_and_action(self):
        controller = NativeController(toy_head())
        source = toy_state()
        before = controller.consume(source)
        indices = (2, 0, 3, 1)
        moved = AddressedState(tuple(replace(source.items[i], address=ADDRESSES[a])
                                     for a, i in enumerate(indices)), 0)
        after = controller.consume(moved)
        for a, i in enumerate(indices):
            self.assertEqual(after.item(ADDRESSES[a]).role_fields, before.item(ADDRESSES[i]).role_fields)
            old = controller.target(before, ADDRESSES[i], 'red')
            new = controller.target(after, ADDRESSES[a], 'red')
            self.assertEqual((old.row, old.column), (new.row, new.column))

    def test_t2_repetition_t4_uniformity_saved_v_and_driver_parity(self):
        source = toy_state()
        standardizer = ChannelStandardizer.fit(torch.stack([x.actual_w.tensor() for x in source.items]))
        controller = NativeController(toy_head(), standardizer)
        t2 = execute_endogenous(source, PendingOperator('l0', 2, 0, 1))
        t4 = execute_endogenous(t2, PendingOperator('l0', 4, 1, 2))
        for state, bandwidth in ((t2, 2), (t4, 4)):
            evidence = controller.consume(state).item('l0')
            for value in (evidence.local_logits.tensor(), evidence.role_fields.tensor()):
                for r in range(0, 4, bandwidth):
                    for c in range(0, 4, bandwidth):
                        block = value[:, r:r+bandwidth, c:c+bandwidth]
                        self.assertTrue(torch.equal(block, block[:, :1, :1].expand_as(block)))
            self.assertEqual(state.item('l1'), source.item('l1'))
        self.assertTrue(torch.equal(controller.consume(t4).item('l0').role_fields.tensor(),
                                    torch.full((2, 4, 4), 1 / 16)))
        restored_by_t1 = execute_endogenous(t4, PendingOperator('l0', 1, 2, 3))
        self.assertEqual(restored_by_t1.item('l0').actual_w, source.item('l0').source_v)
        exogenous = execute_exogenous(source, PendingOperator('l0', 2, 0, 1))
        self.assertEqual(controller.consume(t2), controller.consume(exogenous))
        self.assertEqual(restore_saved(source), source)

    def test_unseen_items_do_not_supply_default_native_answers(self):
        source = toy_state()
        unseen = replace(source.items[0], observed=False, identity=None,
                         delivered_at=None, w_completed_at=None)
        controller = NativeController(toy_head())
        snapshot = controller.consume(replace(source, items=(unseen,) + source.items[1:]))
        for address in ('l0', 'missing'):
            with self.assertRaises(ValueError):
                controller.target(snapshot, address, 'red')
        with self.assertRaises(ValueError):
            controller.target(snapshot, 'l1', 'green')

    def test_standardizer_population_scale_and_constant_channels(self):
        value = torch.ones(2, 24, 4, 4)
        value[0, 0] = -1.
        value[1, 0] = 1.
        standardizer = ChannelStandardizer.fit(value)
        self.assertEqual(standardizer.mean[0], 0.)
        self.assertEqual(standardizer.scale[0], 1.)
        self.assertEqual(standardizer.scale[1:], (1.,) * 23)
        self.assertTrue(torch.equal(standardizer.apply(value)[:, 1:], torch.zeros(2, 23, 4, 4)))

    def test_fresh_random_toy_encoder_extraction_controls_and_no_training(self):
        # New random weights, no production checkpoints or task fit/inference.
        with torch.random.fork_rng():
            torch.manual_seed(23)
            encoder = VisualEncoder().eval()
            readout = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)).eval()
        weights = {name: p.detach().clone() for name, p in encoder.named_parameters()}
        pixels = torch.arange(4*3*16*16, dtype=torch.float32).reshape(4, 3, 16, 16) / 3072
        mean, scale = torch.linspace(-1, 1, 64), torch.linspace(.5, 1.5, 64)
        frozen = FrozenSpatialExtractor(encoder, readout, mean, scale)
        spatial = frozen.extract(pixels)
        self.assertEqual(spatial.shape, (4, 24, 4, 4))
        self.assertFalse(spatial.requires_grad)
        controls = frozen.controls(spatial)
        with torch.no_grad():
            hidden = encoder.net[:-1](pixels)
            self.assertTrue(torch.equal(controls['category_logits'], encoder.net(pixels)))
            self.assertTrue(torch.equal(controls['category_probabilities'], encoder(pixels)))
            self.assertTrue(torch.equal(controls['orientation_hidden'], hidden))
            self.assertTrue(torch.equal(controls['orientation_logits'], readout((hidden - mean) / scale)[:, 0]))
        head = toy_head()
        occupancy_loss(head(spatial), layout_for_world(0).occupancy()).backward()
        for module in (frozen.encoder, frozen.orientation_head):
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.parameters()))
        for name, p in encoder.named_parameters():
            self.assertTrue(torch.equal(p, weights[name]))
            self.assertIsNone(p.grad)
        mean.zero_(); scale.zero_()
        self.assertFalse(torch.equal(mean, frozen.mean64))
        self.assertTrue((frozen.scale64 > 0).all())

    def test_bad_shapes_nonfinite_maps_and_invalid_controls_rejected(self):
        for value in (torch.zeros(24, 5, 4), torch.zeros(24, 4, 4, dtype=torch.long),
                      torch.full((24, 4, 4), float('nan'))):
            with self.assertRaises(ValueError):
                toy_head()(value)
        with self.assertRaises(ValueError):
            role_fields(torch.full((3, 4, 4), float('inf')))
        with self.assertRaises(ValueError):
            ChannelStandardizer((0.,) * 24, (0.,) * 24)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
