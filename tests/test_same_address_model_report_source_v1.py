"""Bounded synthetic checks only: never read native archives or target outputs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from copy import deepcopy

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import same_address_model_report_source_v1 as pilot
from attention_content_model_v1 import (ObservationBatch, CommandBatch, SourceState,
                                       PredictionBatch, _assemble)


class ToySource:
    source_key = 'A'

    def decode(self, v, hidden, symbols):
        return v + hidden[:, :1, None, None] + symbols[:, None, None, None].float()

    def predict(self, observation, state, candidate):
        rows = torch.arange(observation.v.shape[0])
        return _assemble(observation, state, candidate,
                         lambda address, symbol: self.decode(observation.v[rows, address],
                                                           state.hidden[rows, address], symbol), self.source_key)


def toy(batch=4):
    v = torch.arange(batch * 4 * 24 * 4 * 4).float().reshape(batch, 4, 24, 4, 4) / 1000
    w11 = v + 10
    address = torch.full((batch,), 3, dtype=torch.long)
    obs = ObservationBatch(v, w11, 12, 11, CommandBatch(address.clone(), torch.full((batch,), 2, dtype=torch.long)),
                           torch.ones(batch, 4, dtype=torch.bool))
    hidden = torch.zeros(batch, 4, 48)
    recipient = SourceState(hidden, 11, tuple(range(11)), 'A')
    donor_h = hidden.clone()
    donor_h[:, 3] = 7
    donor = SourceState(donor_h, 11, tuple(range(11)), 'A')
    return obs, recipient, donor, address


def test_import_defines_only_stdlib_no_torch_or_archive_load():
    source = "import sys;sys.path.insert(0," + repr(str(SCRIPTS)) + ");import same_address_model_report_source_v1;assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, '-c', source], check=True)


def test_inventory_is_exact_frozen_design():
    rows, pairs = pilot.expected_inventory()
    inventory = json.loads((SCRIPTS.parent / pilot.INVENTORY).read_text())
    assert rows == inventory['opportunities']
    assert pairs == inventory['pairs']
    assert len(rows) == 72 and len(pairs) == 84


def registration_fixture(tmp_path):
    def put(name, value=b'bound bytes'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return {'path': name, 'sha256': hashlib.sha256(value).hexdigest()}
    plan = put(pilot.PLAN)
    rows, pairs = pilot.expected_inventory()
    inventory = put(pilot.INVENTORY, json.dumps(dict(opportunities=rows, pairs=pairs, opportunity_count=72,
                                                   pair_count=84, plan=plan)).encode())
    sources = {name: put(name)['sha256'] for name in pilot.REQUIRED_SOURCES}
    return dict(schema=pilot.SCHEMA, approved=True, scope=pilot.SCOPE, source_sha256=sources,
                plan=plan, inventory=inventory, original_inputs=put(pilot.ORIGINAL_INPUTS),
                artifacts={name: put(path) for name, path in pilot.ARTIFACT_PATHS.items()},
                receipts={name: put(path) for name, path in pilot.RECEIPT_PATHS.items()},
                output_dir=pilot.OUTPUT, execution_batch_context='original_archived_96_rows',
                cases=dict(native_split='heldout', encoder_seed=1311, world_indices=[0, 1, 2, 3], decision_time=12, address=3))


def test_registration_separate_release_exact_bindings_before_loading(tmp_path):
    registration = registration_fixture(tmp_path)
    assert len(pilot.validate_registration(registration, tmp_path)['opportunities']) == 72
    for key, value in [('approved', False), ('scope', 'old_export_approval'),
                       ('execution_batch_context', 'new_batch4')]:
        bad = deepcopy(registration)
        bad[key] = value
        with pytest.raises(ValueError):
            pilot.validate_registration(bad, tmp_path)
    bad = deepcopy(registration)
    bad['cases']['world_indices'] = [3]
    with pytest.raises(ValueError, match='cases'):
        pilot.validate_registration(bad, tmp_path)
    (tmp_path / pilot.ARTIFACT_PATHS['fit']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='bytes differ'):
        pilot.validate_registration(registration, tmp_path)


def test_path_binding_rejects_escape(tmp_path):
    for value in ('../outside', '/tmp/absolute', './notcanonical'):
        with pytest.raises(ValueError):
            pilot.relative_path(value, tmp_path)


def test_pre_reveal_source_current_and_inheritance_are_owned():
    obs, recipient, donor, address = toy()
    before = pilot.observation_hashes(obs, address)
    result, receipt = pilot.predict_checked(ToySource(), obs, recipient, address)
    assert receipt['current_source_decoder_exact']
    assert torch.equal(result.current[:, :3], obs.last_revealed_w[:, :3])
    assert torch.equal(result.current[:, 3], obs.v[:, 3] + 2)
    assert not torch.equal(result.current[:, 3], obs.last_revealed_w[:, 3])
    assert torch.equal(result.next[:, 0, 3], obs.v[:, 3])
    assert before == pilot.observation_hashes(obs, address)
    for invalid in (SourceState(recipient.hidden, 12, tuple(range(12)), 'A'),
                    SourceState(recipient.hidden, 11, tuple(range(11)), 'ordinary')):
        with pytest.raises(ValueError):
            pilot.predict_checked(ToySource(), obs, invalid, address)
    with pytest.raises(ValueError, match='candidates'):
        pilot.predict_checked(ToySource(), obs, recipient, torch.zeros(4, dtype=torch.long))


def test_full_context_other_candidates_keep_original_inheritance():
    obs, recipient, donor, address = toy(96)
    address[4:] = torch.arange(92) % 4
    result = pilot.source_triplet(ToySource(), obs, recipient, donor, address)
    assert result['baseline'].current.shape[0] == 96
    assert torch.equal(result['baseline'].current, result['source_restored'].current)
    assert torch.equal(result['baseline'].next[:, :, :3], result['source_donor'].next[:, :, :3])
    assert pilot.selected_private({'maps': result['baseline'].current})['maps'].shape[0] == 4


def test_same_address_donor_transfers_only_hidden_and_exact_saved_restore():
    obs, recipient, donor, address = toy()
    before = pilot.observation_hashes(obs, address)
    original_state = pilot.state_hashes(recipient)
    progress = {}
    result = pilot.source_triplet(ToySource(), obs, recipient, donor, address, progress)
    assert torch.equal(result['source_donor'].current[:, 3], obs.v[:, 3] + 7 + 2)
    assert torch.equal(result['source_donor'].next[:, :, 3], torch.stack([obs.v[:, 3] + 7 + n for n in range(3)], 1))
    assert torch.equal(result['baseline'].next, result['source_restored'].next)
    assert pilot.state_hashes(recipient) == original_state
    assert pilot.observation_hashes(obs, address) == before
    assert progress['source_restored']['receipt']['state'] == progress['baseline']['receipt']['state']
    incompatible = SourceState(donor.hidden, 11, tuple(range(11)), 'ordinary')
    with pytest.raises(ValueError, match='donor source'):
        pilot.source_triplet(ToySource(), obs, recipient, incompatible, address)


def test_copied_W_cannot_pass_current_source_ownership():
    class CopySource(ToySource):
        def predict(self, observation, state, candidate):
            copied = observation.last_revealed_w.clone()
            return PredictionBatch(copied, copied[:, None].expand(-1, 3, -1, -1, -1, -1).clone(), 12, 'A')
    obs, state, _, address = toy()
    with pytest.raises(ValueError, match='not the addressed source decoder'):
        pilot.predict_checked(CopySource(), obs, state, address)


def test_input_mutation_and_wrong_source_are_rejected():
    class Mutator(ToySource):
        def predict(self, observation, state, candidate):
            result = super().predict(observation, state, candidate)
            observation.v.add_(1)
            return result
    obs, state, _, address = toy()
    with pytest.raises(ValueError, match='mutated inputs'):
        pilot.predict_checked(Mutator(), obs, state, address)
    class WrongOwner(ToySource):
        def predict(self, observation, state, candidate):
            result = super().predict(observation, state, candidate)
            return PredictionBatch(result.current, result.next, 12, 'ordinary')
    obs, state, _, address = toy()
    with pytest.raises(ValueError, match='ownership'):
        pilot.predict_checked(WrongOwner(), obs, state, address)


def test_native_k0_counterfactual_and_exact_restore_keep_future_fixed():
    current = torch.zeros(4, 4, 24, 4, 4)
    alternatives = torch.stack([torch.ones(4, 24, 4, 4) * n for n in (8, 3, 5)], 1)
    before = current.clone()
    result = pilot.native_controls(current, alternatives)
    assert torch.equal(result['W_changed']['current'][:, 3], alternatives[:, 0])
    assert torch.equal(result['W_changed']['current'][:, :3], current[:, :3])
    assert torch.equal(result['W_changed']['next'], result['baseline']['next'])
    assert torch.equal(result['W_restored']['current'], result['baseline']['current'])
    assert torch.equal(current, before)
    assert pilot.origin('W', 'current', 'W_changed') == dict(kind='counterfactual_native_snapshot', represented_time=12,
                  archived_origin_time=13, executed_branch_claim=False, fixed_command='k0')
    assert pilot.origin('A', 'candidate', 'source_donor')['represented_time'] == 13


def test_one_role_public_fields_and_exact_native_score_ties():
    fields = torch.full((3, 4, 4), 1 / 16, dtype=torch.float32)
    panel = pilot.public_panel('p0', 'ordinary', 'source_donor', 'blue', fields[0], fields,
                               torch.tensor(0), torch.tensor([1 / 16] * 3))
    assert panel['panel_role'] == 'predictor_estimate'
    assert panel['current']['role'] == 'blue'
    assert panel['processing_scores']['tied_winners'] == ['k0', 'k1', 'k2']
    assert panel['processing_scores']['first_index_choice'] == 'k0'
    assert panel['processing_scores']['top_two_margin'] == 0
    assert 'ordinary' not in json.dumps(panel) and 'source_donor' not in json.dumps(panel)
    with pytest.raises(ValueError, match='score reconstruction'):
        pilot.scores(fields, torch.tensor(0), torch.tensor([0.1, 0.1, 0.1]))
    with pytest.raises(ValueError, match='first-index'):
        pilot.scores(fields, torch.tensor(1), torch.tensor([1 / 16] * 3))


def test_complete_synthetic_source_export_retains72_rows84_pairs(monkeypatch):
    """Exercise the complete assembly with invented maps/checkpoints, no archives."""
    from attention_content_model_v1 import (FitNormalizer, LearnedContentModel, AnalyticMappingSource,
        AnalyticState, NativePredictionReadout, snapshot_state, patch_addressed_state)
    from spatial_native_task_v1 import LocalOccupancyHead, ChannelStandardizer
    import attention_relational_report_v1 as relational
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(93421)
    v = torch.randn(96, 4, 24, 4, 4, generator=generator)
    family = torch.randn(96, 4, 3, 24, 4, 4, generator=generator)
    w11 = v.clone()
    w12 = w11.clone()
    w12[:, 3] = family[:, 3, 2]
    addresses = torch.zeros(96, 15, dtype=torch.long)
    addresses[:, 11] = 3
    addresses[:, 12] = torch.arange(96) % 4
    symbols = torch.zeros_like(addresses)
    symbols[:, 11] = 2
    roles = torch.zeros_like(addresses)
    roles[:, 12] = torch.arange(96) % 2
    mappings = torch.tensor([1, 2, 4]).expand(96, 4, 3).clone()
    episode = dict(split='heldout', encoder_seed=1311, V=v, command_addresses=addresses,
                   command_symbols=symbols, query_roles=roles, evaluator_mappings=mappings)
    head = LocalOccupancyHead().float().eval()
    coordinates = dict(mean=[0.0] * 24, scale=[1.0] * 24)
    native = dict(native_head_fit=dict(weights=head.state_dict(), fit_only_standardizer=coordinates))
    readout = NativePredictionReadout(head, ChannelStandardizer(tuple(coordinates['mean']), tuple(coordinates['scale'])))
    obs = ObservationBatch(v, w11, 12, 11, CommandBatch(addresses[:, 11], symbols[:, 11]),
                           torch.ones(96, 4, dtype=torch.bool))
    normalizer = FitNormalizer(tuple([0.0] * 24), tuple([1.0] * 24))
    fit = dict(normalizer=dict(mean=normalizer.mean, scale=normalizer.scale), checkpoints={})
    retained = {}
    with torch.no_grad():
        for key in pilot.SOURCES:
            if key == 'analytic':
                model = AnalyticMappingSource()
                state = AnalyticState(torch.ones(96, 4, 6, dtype=torch.bool),
                                      torch.zeros(96, 4, 3, dtype=torch.bool), 11, tuple(range(11)))
            else:
                model = LearnedContentModel(normalizer, 482, key).eval()
                fit['checkpoints'][key] = deepcopy(model.state_dict())
                state = SourceState(torch.randn(96, 4, 48, generator=generator), 11, tuple(range(11)), key)
            donor = snapshot_state(state)
            if key == 'analytic':
                donor.survivors[:, 3, 0] = False
            else:
                donor.hidden[:, 3] += 1
            baseline = model.predict(obs, state, addresses[:, 12])
            patched = model.predict(obs, donor, addresses[:, 12])
            source = dict(selected_current_maps=torch.zeros(96, 14, 24, 4, 4),
                          selected_current_fields=torch.zeros(96, 14, 2, 4, 4),
                          selected_next_maps=torch.zeros(96, 15, 3, 24, 4, 4),
                          selected_next_fields=torch.zeros(96, 15, 3, 2, 4, 4))
            source['selected_current_maps'][:, 11] = baseline.current[:, 3]
            source['selected_current_fields'][:, 11] = readout.consume(baseline.current[:, 3]).role_fields
            selected = baseline.next[torch.arange(96)[:, None], torch.arange(3)[None, :], addresses[:, 12, None]]
            donor_selected = patched.next[torch.arange(96)[:, None], torch.arange(3)[None, :], addresses[:, 12, None]]
            source['selected_next_maps'][:, 12] = selected
            source['selected_next_fields'][:, 12] = readout.consume(selected).role_fields
            ledger = dict(selected_current_maps=patched.current[:, None, 3],
                          selected_current_fields=readout.consume(patched.current[:, 3]).role_fields[:, None],
                          selected_next_maps=donor_selected[:, None],
                          policy_candidate_fields=readout.consume(donor_selected).role_fields[:, None])
            for attribute in ('hidden',) if key != 'analytic' else ('survivors', 'revealed_symbols'):
                saved = getattr(state, attribute)
                expanded = saved[:, None].expand(-1, 15, *saved.shape[1:]).clone()
                source['pre_reveal_' + attribute] = expanded
                ledger['donor_' + ('hidden' if attribute == 'hidden' else attribute) + '_at_patched_address'] = getattr(donor, attribute)[:, None, 3]
            source['donors'] = {'current': ledger}
            retained[key] = source
        current_fields = torch.zeros(96, 13, 4, 2, 4, 4)
        current_fields[:, 12] = readout.consume(w12).role_fields
        shared = dict(actual_operator_family=family, actual_completed_fields=current_fields,
                      actual_family_fields=readout.consume(family).role_fields)
        bundle = dict(episode=episode, evaluation=dict(tensors=dict(shared=shared, sources=retained)))
        monkeypatch.setattr(relational, '_actual_maps', lambda episode, shared, torch, world, time: (w11 if time == 11 else w12)[world].clone())
        monkeypatch.setattr(relational, '_donor_evidence', lambda *args: dict(patched_address=3, synthetic_receipt=True))
        rows, pairs = pilot.expected_inventory()
        progress = {}
        result = pilot.build_source_export(bundle, fit, native, dict(opportunities=rows, pairs=pairs), progress)
    assert len(result['packets']) == 72 and len(result['pairs']) == 84
    assert result['fresh_independent_sample'] is False
    assert result['execution_batch_context'] == 'original_archived_96_rows'
    assert all('source' not in item['content'] and 'world_index' not in item['content'] for item in result['packets'])
    assert all(item['public_input_identical'] for item in result['pairs'] if item['kind'].endswith('baseline_to_restored'))
    assert pilot.selected_private(progress)['source_states']['A']['saved_state']['hidden'].shape == (4, 4, 48)
