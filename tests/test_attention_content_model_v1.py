"""Tiny handcrafted software fixtures; no renderer, archive, checkpoint or fit."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from attention_content_model_v1 import (
    AnalyticMappingSource, CommandBatch, ExactAlias, FitNormalizer,
    LearnedContentModel, NativePredictionReadout, ObservationBatch,
    PARAMETER_COUNT, PredictionBatch, RevealedFeedback, build_learned_sources,
    choose_from_maxima, patch_addressed_state, restore_state, snapshot_state,
)
from attention_content_form_primitives_v1 import block_average
from spatial_native_task_v1 import ChannelStandardizer, LocalOccupancyHead, role_fields


def maps(batch=2):
    # Structured numerical fixture, not a task world or source encoder output.
    return torch.arange(batch * 4 * 384, dtype=torch.float32).reshape(batch, 4, 24, 4, 4) / 3072


def command(batch=2, address=1, symbol=0):
    return CommandBatch(torch.full((batch,), address, dtype=torch.long),
                        torch.full((batch,), symbol, dtype=torch.long))


def observation(v, time=0, last=None, previous=None):
    return ObservationBatch(v, v.clone() if last is None else last, time, max(0, time-1),
                            previous, torch.ones(v.shape[0], 4, dtype=torch.bool))


def feedback(v, cmd, completion=1, bandwidth=2):
    completed = v.clone()
    for row in range(v.shape[0]):
        address = int(cmd.address[row])
        completed[row, address] = block_average(v[row, address], bandwidth)
    return RevealedFeedback(completed, cmd, completion, completion-1, completion)


def toy_head():
    head = LocalOccupancyHead()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.net[0].weight[0, 0] = 1
        head.net[0].weight[1, 1] = 1
        head.net[2].weight[1, 0] = 1
        head.net[2].weight[2, 1] = 1
    return head


class AttentionContentCoreTests(unittest.TestCase):
    def model(self, v):
        return LearnedContentModel(FitNormalizer.fit(v, split='fit'), 31)

    def test_exact_architecture_initialization_and_ordinary_alias(self):
        v = maps()
        rng = torch.get_rng_state().clone()
        attention, ordinary, alias = build_learned_sources(FitNormalizer.fit(v, split='fit'), 1301)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in attention.parameters()), PARAMETER_COUNT)
        self.assertEqual(sum(p.numel() for p in ordinary.parameters()), PARAMETER_COUNT)
        self.assertEqual((attention.gru.input_size, attention.gru.hidden_size), (771, 48))
        self.assertEqual((attention.decoder[0].in_features, attention.decoder[0].out_features), (435, 96))
        self.assertEqual(attention.decoder[2].out_features, 384)
        self.assertTrue(all(p.dtype == torch.float32 and p.device.type == 'cpu' for p in attention.parameters()))
        self.assertFalse(torch.equal(attention.decoder[0].weight, ordinary.decoder[0].weight))
        state, obs, address = attention.initial_state(2), observation(v), torch.tensor([0, 3])
        direct, duplicate = attention.predict(obs, state, address), alias.predict(obs, state, address)
        self.assertTrue(torch.equal(direct.current, duplicate.current))
        self.assertTrue(torch.equal(direct.next, duplicate.next))
        self.assertIs(alias.model, attention)

    def test_normalizer_population_statistics_and_fit_only_guard(self):
        v = torch.ones(2, 4, 24, 4, 4)
        v[0, :, 0] = -1
        normalizer = FitNormalizer.fit(v, split='fit')
        self.assertEqual(normalizer.mean[0], 0)
        self.assertEqual(normalizer.scale[0], 1)
        self.assertEqual(normalizer.scale[1:], (1.,) * 23)
        self.assertTrue(torch.equal(normalizer.denormalize(normalizer.normalize(v)), v))
        v.zero_()
        self.assertEqual(normalizer.mean[1], 1)
        for split in ('dev', 'heldout', None):
            with self.assertRaises(ValueError):
                FitNormalizer.fit(maps(), split=split)

    def test_current_and_next_assembly_preserve_delayed_uncertainty(self):
        v = maps(); model = self.model(v); state = model.initial_state(2)
        cmd = command(); candidate = torch.tensor([2, 3])
        initial = model.predict(observation(v), state, candidate)
        self.assertTrue(torch.equal(initial.current, v))
        pred = model.predict(observation(v, 1, v.clone(), cmd), state, candidate)
        expected = model.decode(v[:, 1], state.hidden[:, 1], cmd.symbol)
        self.assertTrue(torch.equal(pred.current[:, 1], expected))
        self.assertTrue(torch.equal(pred.current[:, [0, 2, 3]], v[:, [0, 2, 3]]))
        for row, address in enumerate(candidate):
            for symbol in range(3):
                self.assertTrue(torch.equal(pred.next[row, symbol, 1], pred.current[row, 1]))
                for other in set(range(4)) - {int(address)}:
                    self.assertTrue(torch.equal(pred.next[row, symbol, other], pred.current[row, other]))
        self.assertEqual(pred.next.shape, (2, 3, 4, 24, 4, 4))
        # A prediction cannot silently accept a current completed target argument.
        with self.assertRaises(TypeError):
            model.predict(observation(v, 1, v, cmd), state, candidate, actual_w=feedback(v, cmd).completed_w)

    def test_update_is_address_local_nonmutating_and_differentiable(self):
        v = maps(); model = self.model(v); original = model.initial_state(2)
        cmd = CommandBatch(torch.tensor([0, 3]), torch.tensor([1, 2]))
        updated = model.update(original, v, feedback(v, cmd))
        self.assertEqual(updated.update_command_indices, (0,))
        self.assertEqual(updated.revealed_through, 1)
        self.assertTrue(torch.equal(original.hidden, torch.zeros_like(original.hidden)))
        for row, address in enumerate(cmd.address):
            self.assertFalse(torch.equal(updated.hidden[row, address], original.hidden[row, address]))
            for other in set(range(4)) - {int(address)}:
                self.assertTrue(torch.equal(updated.hidden[row, other], original.hidden[row, other]))
        model.decode(v[:, 0], updated.hidden[:, 0], torch.tensor([0, 0])).square().mean().backward()
        self.assertIsNotNone(model.gru.weight_ih.grad)
        self.assertGreater(float(model.gru.weight_ih.grad.abs().sum()), 0)
        # Backward only, with no optimizer, parameter update or fit loop.

    def test_reveal_and_pre_reveal_timestamp_guards(self):
        v = maps(); model = self.model(v); state = model.initial_state(2); cmd = command()
        fb = feedback(v, cmd); updated = model.update(state, v, fb)
        for bad in (replace(fb, completed_at=0), replace(fb, command_index=1),
                    replace(fb, scored_decision_at=0)):
            with self.assertRaises(ValueError):
                model.update(state, v, bad)
        with self.assertRaises(ValueError):
            model.update(updated, v, fb)
        for obs in (replace(observation(v, 1, v, cmd), last_revealed_at=1),
                    observation(v, 2, v, cmd), replace(observation(v), previous_command=cmd)):
            with self.assertRaises(ValueError):
                model.predict(obs, state, torch.tensor([0, 1]))
        with self.assertRaises(ValueError):
            model.predict(observation(v, 1, v, cmd), updated, torch.tensor([0, 1]))

    def test_nonfinite_unseen_bad_shapes_and_unknown_commands_rejected(self):
        v = maps(); model = self.model(v); state = model.initial_state(2)
        for value in (v.double(), v[:, :3], torch.full_like(v, float('nan'))):
            with self.assertRaises(ValueError):
                model.predict(observation(value), state, torch.tensor([0, 0]))
        missing = observation(v); missing.observed[0, 2] = False
        with self.assertRaises(ValueError):
            model.predict(missing, state, torch.tensor([0, 0]))
        for cmd in (command(address=4), command(symbol=3),
                    CommandBatch(torch.tensor([0., 1.]), torch.tensor([0, 1]))):
            with self.assertRaises(ValueError):
                model.predict(observation(v, 1, v, cmd), state, torch.tensor([0, 0]))

    def test_outer_address_equivariance(self):
        v = maps(); model = self.model(v); cmd = command(); state = model.initial_state(2)
        updated = model.update(state, v, feedback(v, cmd))
        previous = command(address=2, symbol=1)
        obs = observation(v, 2, feedback(v, cmd).completed_w, previous)
        query = torch.tensor([0, 3]); before = model.predict(obs, updated, query)
        shifted_cmd = CommandBatch((previous.address + 1) % 4, previous.symbol)
        moved = replace(obs, v=v.roll(1, 1), last_revealed_w=obs.last_revealed_w.roll(1, 1),
                        previous_command=shifted_cmd)
        shifted = replace(updated, hidden=updated.hidden.roll(1, 1))
        after = model.predict(moved, shifted, (query + 1) % 4)
        self.assertTrue(torch.equal(after.current, before.current.roll(1, 1)))
        self.assertTrue(torch.equal(after.next, before.next.roll(1, 2)))

    def test_donor_patch_isolation_inheritance_and_exact_restoration(self):
        v = maps(); model = self.model(v); cmd = command(address=1)
        recipient = model.update(model.initial_state(2), v, feedback(v, cmd, bandwidth=1))
        donor = model.update(model.initial_state(2), v, feedback(v, cmd, bandwidth=4))
        obs = observation(v, 2, feedback(v, cmd, bandwidth=1).completed_w, cmd)
        query, address = torch.tensor([2, 2]), torch.tensor([1, 1])
        saved = snapshot_state(recipient); baseline = model.predict(obs, recipient, query)
        patched = patch_addressed_state(recipient, donor, address)
        changed = model.predict(obs, patched, query)
        self.assertTrue(torch.equal(patched.hidden[:, [0, 2, 3]], recipient.hidden[:, [0, 2, 3]]))
        self.assertTrue(torch.equal(patched.hidden[:, 1], donor.hidden[:, 1]))
        self.assertTrue(torch.equal(changed.next[:, :, 2], baseline.next[:, :, 2]))
        self.assertTrue(torch.equal(changed.next[:, :, 1], changed.current[:, None, 1].expand(-1, 3, -1, -1, -1)))
        self.assertFalse(torch.equal(changed.current[:, 1], baseline.current[:, 1]))
        restored = restore_state(saved); repeated = model.predict(obs, restored, query)
        self.assertTrue(torch.equal(restored.hidden, recipient.hidden))
        self.assertTrue(torch.equal(repeated.current, baseline.current))
        self.assertTrue(torch.equal(repeated.next, baseline.next))
        patched.hidden.zero_()
        self.assertTrue(torch.equal(restored.hidden, recipient.hidden))
        for bad in (replace(donor, source_key='ordinary'), replace(donor, revealed_through=0, update_command_indices=())):
            with self.assertRaises(ValueError):
                patch_addressed_state(recipient, bad, address)

    def test_analytic_preserves_uninformative_ambiguity(self):
        v = torch.ones(2, 4, 24, 4, 4); source = AnalyticMappingSource(); state = source.initial_state(2)
        updated = source.update(state, v, feedback(v, command(), bandwidth=4))
        self.assertTrue(updated.survivors.all())
        self.assertEqual(updated.summary()['survivor_count'].tolist(), [[6]*4]*2)
        self.assertEqual(updated.summary()['revealed_symbol_count'][:, 1].tolist(), [1, 1])
        pred = source.predict(observation(v, 2, v, command()), updated, torch.tensor([0, 1]))
        self.assertTrue(torch.equal(pred.current, v))
        self.assertTrue(torch.equal(pred.next, v[:, None].expand(-1, 3, -1, -1, -1, -1)))

    def test_analytic_learns_mapping_from_reveals_without_oracle(self):
        v = maps(); source = AnalyticMappingSource(); cmd = command(symbol=0)
        state = source.update(source.initial_state(2), v, feedback(v, cmd, bandwidth=2))
        self.assertEqual(state.survivors[:, 1].sum(-1).tolist(), [2, 2])
        second = command(symbol=1)
        state = source.update(state, v, feedback(v, second, completion=2, bandwidth=4))
        self.assertEqual(state.survivors[:, 1].sum(-1).tolist(), [1, 1])
        predicted = source.decode(v[:, 1], state.survivors[:, 1], torch.tensor([2, 2]))
        self.assertTrue(torch.equal(predicted, v[:, 1]))
        impossible = replace(feedback(v, cmd), completed_w=torch.full_like(v, -999))
        with self.assertRaises(ValueError):
            source.update(source.initial_state(2), v, impossible)

    def test_analytic_addressed_patch_restore_and_timing(self):
        v = maps(); source = AnalyticMappingSource(); cmd = command()
        initial = source.initial_state(2)
        recipient = source.update(initial, v, feedback(v, cmd, bandwidth=1))
        donor = source.update(initial, v, feedback(v, cmd, bandwidth=4))
        patched = patch_addressed_state(recipient, donor, torch.tensor([1, 1]))
        self.assertTrue(torch.equal(patched.survivors[:, 1], donor.survivors[:, 1]))
        self.assertTrue(torch.equal(patched.survivors[:, [0, 2, 3]], recipient.survivors[:, [0, 2, 3]]))
        restored = restore_state(snapshot_state(recipient))
        self.assertTrue(torch.equal(restored.survivors, recipient.survivors))
        with self.assertRaises(ValueError):
            source.predict(observation(v, 1, v, cmd), recipient, torch.tensor([0, 0]))

    def test_native_predicted_readout_itemwise_matches_frozen_reference(self):
        v = maps(); head = toy_head(); standardizer = ChannelStandardizer((0.,)*24, (1.,)*24)
        readout = NativePredictionReadout(head, standardizer)
        evidence = readout.consume(v)
        for row in range(2):
            for address in range(4):
                expected = head(standardizer.apply(v[row, address]))
                self.assertTrue(torch.equal(evidence.local_logits[row, address], expected))
                self.assertTrue(torch.equal(evidence.role_fields[row, address], role_fields(expected)))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in readout.head.parameters()))
        with torch.no_grad():
            head.net[2].weight.zero_()
        self.assertTrue(torch.equal(readout.consume(v).local_logits, evidence.local_logits))

    def test_native_symbol_policy_and_confidence_duplicate_ties(self):
        raw = torch.zeros(2, 3, 4, 24, 4, 4)
        raw[:, 1, 2, 0, 0, 0] = 2
        raw[:, 2, 2, 0, 0, 0] = 8
        prediction = PredictionBatch(torch.zeros(2, 4, 24, 4, 4), raw, 3, 'A')
        readout = NativePredictionReadout(toy_head(), ChannelStandardizer((0.,)*24, (1.,)*24))
        choice = readout.choose(prediction, torch.tensor([2, 2]), torch.tensor([0, 1]))
        self.assertEqual(choice.symbol.tolist(), [2, 0])
        self.assertTrue(torch.equal(choice.symbol, choose_from_maxima(choice.maxima)))
        self.assertEqual(choice.candidate_fields.shape, (2, 3, 2, 4, 4))
        self.assertEqual(choose_from_maxima(torch.ones(2, 3)).tolist(), [0, 0])
        with self.assertRaises(ValueError):
            readout.choose(prediction, torch.tensor([2, 2]), torch.tensor([0, 2]))


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
