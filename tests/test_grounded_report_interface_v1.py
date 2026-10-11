"""Source-only toy fixtures; no real archive/world export or Reporter calls."""
import copy
import gzip
import tempfile
from pathlib import Path
import json
import unittest
from unittest.mock import patch

import torch

import grounded_report_interface_v1 as interface


def fixture(n=20):
    commands = torch.tensor([0, 1, 0, 1, 2, 0, 1, 2])[None].expand(n, -1).clone()
    delivered = (commands+torch.arange(n)[:, None]) % 4
    sensory = torch.zeros(n, 1, 4, 66)
    history = torch.zeros(n, 1, 4, 4)
    last = torch.full((n, 4), -1, dtype=torch.long)
    for row in range(n):
        for step in range(8):
            loc, cmd = int(delivered[row, step]), int(commands[row, step])
            sensory[row, 0, loc, :64] = torch.arange(64)/100+step/10+row/100
            sensory[row, 0, loc, 64] = (cmd-1)*2
            sensory[row, 0, loc, 65] = 1
            history[row, 0, cmd] = 0
            history[row, 0, cmd, loc] = 1
            last[row, loc] = step
    delivery = history.double().clone()
    delivery[:, :, 3] = .25

    def model(memory):
        current = memory[..., 64].double().sigmoid()
        refresh = .6+.2*memory[..., 0].double().sigmoid()
        return {'delivery': delivery.clone(), 'current': current, 'refresh': refresh,
                'after': interface.compose_after(delivery, current, refresh),
                'risk_hidden': memory[..., :32].clone()}

    p, rotated_memory = model(sensory), sensory.roll(1, -2)
    moved = model(rotated_memory)
    patched = copy.deepcopy(p)
    patched['delivery'] = p['delivery'].roll(1, -2)
    patched['after'] = interface.compose_after(patched['delivery'], p['current'], p['refresh'])
    physical = torch.stack((sensory[:, 0, :, 0].double()-1.3, sensory[:, 0, :, 1].double()), -1)
    error = {metric: {'value': [.01, .02], 'conditional_world_bootstrap_95': [[.005, .015], [.01, .03]],
                      'valid_bootstrap_counts': [256, 256]} for metric in ('mae', 'rmse')}
    weights = torch.zeros(32, 3, dtype=torch.float64)
    weights[0, 0], weights[1, 1] = 1., 1.
    parameters = {'feature_scale': torch.ones(32, dtype=torch.float64), 'weights': weights,
                  'target_scale': torch.ones(2, dtype=torch.float64)}
    bounds = {'absolute_feature_difference': torch.zeros(n, 4, 32, dtype=torch.float64),
              'absolute_output_difference': torch.zeros(n, 4, 2, dtype=torch.float64),
              'propagated_bound': torch.zeros(n, 4, 2, dtype=torch.float64)}
    return interface.SourceBundle(1301, sensory, history, commands, delivered, last,
                                  {'natural': sensory.clone(), 'sham': sensory.clone(),
                                   'memory_readdressed': rotated_memory, 'delivery_patched': sensory.clone(),
                                   'memory_restored': rotated_memory.roll(-1, -2), 'delivery_restored': sensory.clone()},
                                  {'natural': p, 'sham': copy.deepcopy(p), 'memory_readdressed': moved,
                                   'delivery_patched': patched, 'memory_restored': copy.deepcopy(p),
                                   'delivery_restored': copy.deepcopy(p)},
                                  {'natural': physical, 'sham': physical.clone(),
                                   'memory_readdressed': physical.roll(1, 1), 'delivery_patched': physical.clone(),
                                   'memory_restored': physical.clone(), 'delivery_restored': physical.clone()},
                                  {'fixture_only': True, 'grounded_model_sha256': 'toy-hash'}, error, parameters, bounds)


class GroundedReportInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_preselection_retains_all_locations_and_unseen_has_no_content(self):
        source = fixture()
        self.assertEqual(interface.selection_indices(source), tuple(range(16)))
        record = interface.record(source, 0)
        value = record['reporter_input']
        self.assertEqual(len(value['locations']), 4)
        unseen = value['locations'][3]
        self.assertFalse(unseen['observed'])
        for key in ('raw_orientation_logit', 'logit_sign', 'stored_decision', 'evidence_item'):
            self.assertIsNone(unseen['stored_evidence'][key])
        self.assertEqual(unseen['stored_evidence']['decision_if_no_sample'],
                         {'value': 'horizontal', 'basis': 'fixed default guess'})
        self.assertIsNone(unseen['external_image_statistic_readouts']['image_value_std_estimate'])
        self.assertIsNone(unseen['external_image_statistic_readouts']['nearest_neighbor_rms_estimate'])
        # Non-content task forecasts remain supplied for unseen locations.
        self.assertEqual(unseen['estimated_current_decision_success'], .5)
        with self.assertRaises(ValueError):
            interface.record(source, 16)
        with self.assertRaises(ValueError):
            interface.selection_indices(fixture(n=15))

    def test_raw_sign_decision_and_exact_composition_are_distinct_from_truth(self):
        source = fixture()
        value = interface.record(source, 0)['reporter_input']
        evidence = [node['stored_evidence'] for node in value['locations']]
        self.assertEqual([evidence[i]['logit_sign'] for i in range(3)], ['negative', 'zero', 'positive'])
        self.assertEqual([evidence[i]['stored_decision'] for i in range(3)], ['horizontal', 'horizontal', 'vertical'])
        for command in value['command_forecasts']:
            for node in value['locations']:
                loc = node['location']
                route = command['predicted_delivery_distribution'][loc]
                current = node['estimated_current_decision_success']
                refresh = node['estimated_fresh_glimpse_decision_success']
                self.assertEqual(command['estimated_decision_success_after_command'][loc],
                                 route*refresh+(1-route)*current)
        malformed = copy.deepcopy(source)
        malformed.forecasts_by_variant['natural']['after'][0, 0, 0, 0] += .01
        with self.assertRaises(ValueError):
            interface.record(malformed, 0)

    def test_native_history_is_latest_only_and_chronology_is_audit_only(self):
        record = interface.record(fixture(), 0)
        value, audit = record['reporter_input'], record['audit_provenance']
        self.assertNotIn('observed_delivery_history', value)
        self.assertNotIn('external_observed_history_instrumentation', value)
        self.assertEqual(len(value['native_route_history']), 4)
        native = value['native_route_history']
        self.assertEqual(native[0]['last_observed_delivery_one_hot'], {'l0': 1, 'l1': 0, 'l2': 0, 'l3': 0})
        self.assertEqual(native[3]['status'], 'no_delivery_observed')
        self.assertFalse(any(native[3]['last_observed_delivery_one_hot'].values()))
        history = audit['external_observed_history_instrumentation']
        self.assertEqual(len(history['chronological_events']), 8)
        self.assertFalse(history['native_model_input'])
        self.assertFalse(history['reporter_context'])

    def test_readdressing_preserves_actual_past_and_restoration_is_exact(self):
        source = fixture()
        original = interface.record(source, 0)
        moved = interface.record(source, 0, 'memory_readdressed')
        natural, readdressed = original['reporter_input'], moved['reporter_input']
        self.assertEqual(natural['native_route_history'], readdressed['native_route_history'])
        self.assertEqual(original['audit_provenance']['external_observed_history_instrumentation'],
                         moved['audit_provenance']['external_observed_history_instrumentation'])
        for loc in range(4):
            donor = (loc-1) % 4
            self.assertEqual(readdressed['locations'][loc]['stored_evidence'], natural['locations'][donor]['stored_evidence'])
            self.assertEqual(readdressed['locations'][loc]['external_image_statistic_readouts'],
                             natural['locations'][donor]['external_image_statistic_readouts'])
        self.assertTrue(moved['audit_provenance']['synthetic_patched_source_not_natural_state'])
        self.assertTrue(moved['audit_provenance']['history_retains_actual_past'])
        for variant in ('sham', 'memory_restored', 'delivery_restored'):
            self.assertEqual(natural, interface.record(source, 0, variant)['reporter_input'])

    def test_route_patch_is_export_only_and_retains_memory_risk_and_readouts(self):
        source = fixture()
        original = interface.record(source, 0)['reporter_input']
        patched = interface.record(source, 0, 'delivery_patched')
        self.assertEqual(original['locations'], patched['reporter_input']['locations'])
        self.assertEqual(original['native_route_history'], patched['reporter_input']['native_route_history'])
        for cmd in range(4):
            self.assertEqual(patched['reporter_input']['command_forecasts'][cmd]['predicted_delivery_distribution'],
                             original['command_forecasts'][(cmd-1) % 4]['predicted_delivery_distribution'])
        self.assertEqual(patched['audit_provenance']['forecast_origin'], 'export_only_delivery_patch')
        self.assertNotIn('source_variant', patched['reporter_input'])

    def test_same_information_table_roundtrip_and_no_mutation(self):
        source = fixture()
        before = copy.deepcopy(source.original_sensory)
        record = interface.record(source, 0)['reporter_input']
        original = copy.deepcopy(record)
        tables = interface.address_aware_tables(record)
        self.assertEqual(len(tables['tables']), 5)
        self.assertEqual(interface.join_address_aware_tables(tables), record)
        self.assertEqual(record, original)
        self.assertTrue(torch.equal(before, source.original_sensory))
        tables['tables']['risk_forecasts'][0]['location'] = 'wrong'
        with self.assertRaises(ValueError):
            interface.join_address_aware_tables(tables)

    def test_external_readouts_are_unclipped_and_extra_summary_fields_are_excluded(self):
        source = fixture()
        self.assertLess(float(source.physical_by_variant['natural'][0, 0, 0]), 0)
        source.physical_error_summary['truth_orientation'] = 'FORBIDDEN'
        source.physical_error_summary['mae']['contrast'] = 'FORBIDDEN'
        source.physical_error_summary['scope'] = 'FORBIDDEN'
        value = interface.record(source, 0)['reporter_input']
        self.assertLess(value['locations'][0]['external_image_statistic_readouts']['image_value_std_estimate'], 0)
        self.assertNotIn('FORBIDDEN', json.dumps(value))
        self.assertEqual(value['locations'][0]['risk_feature_source']['dimensions'], 32)
        self.assertIn('not per-item error bounds', value['external_readout_evaluation_summary']['scope'])
        for forbidden in ('physical_targets', 'world_truth', 'contrast', 'labels', 'correctness',
                          'pixels', 'future_images', 'human_ratings', 'calibrated_sensory_probability'):
            self.assertNotIn('"'+forbidden+'"', json.dumps(value))
        json.dumps(value, allow_nan=False)

    def test_invalid_sources_fail_before_record_and_no_implicit_loader(self):
        malformed = fixture()
        malformed.native_history[0, 0, 0] = 0
        with self.assertRaises(ValueError):
            interface.record(malformed, 0)
        malformed = fixture()
        malformed.physical_by_variant['natural'][0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            interface.record(malformed, 0)
        with patch.object(interface, 'load_bundle', side_effect=AssertionError('no production loading')):
            self.assertEqual(interface.record(fixture(), 0)['reporter_input']['snapshot_after_commands'], 8)


    def test_fake_archive_loader_extracts_only_whitelist_and_requires_verified_hashes(self):
        source = fixture()
        trace = {'original_sensory': source.original_sensory, 'original': source.forecasts_by_variant['natural'],
                 'readdressed_sensory': source.sensory_by_variant['memory_readdressed'],
                 'restored_sensory': source.sensory_by_variant['memory_restored'],
                 'readdressed': source.forecasts_by_variant['memory_readdressed'],
                 'restored': source.forecasts_by_variant['memory_restored'],
                 'patched_forecasts': source.forecasts_by_variant['delivery_patched'],
                 'restored_patch_delivery': source.forecasts_by_variant['natural']['delivery'],
                 'restored_patch_after': source.forecasts_by_variant['natural']['after'],
                 'decode_permutation_error_bounds': {'risk_hidden': source.decode_permutation_error_bounds}}
        for key, variant in (('original_local_decoded', 'natural'), ('readdressed_local_decoded', 'memory_readdressed'),
                             ('restored_local_decoded', 'memory_restored'), ('patch_local_decoded', 'delivery_patched')):
            trace[key] = {'risk_hidden': {'physical': source.physical_by_variant[variant]}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_files = {}
            def archive(relative, value):
                path = root/'model-1301'/(relative+'.pt.gz')
                path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(path, 'wb') as stream:
                    torch.save(value, stream)
                archive_files[str(path)] = {'sha256': interface.digest(path)}
            archive('split-2/snapshot', {'sensory': source.original_sensory, 'history': source.native_history,
                                        'last_step': source.last_step, 'physical_targets': torch.full((20, 4, 2), 999.)})
            archive('split-2/decoded', {'commands': source.commands, 'delivered': source.delivered,
                                       'hidden': torch.full((20, 8, 64), 999.)})
            archive('split-2/forecasts', {'structured': source.forecasts_by_variant['natural']})
            archive('evaluation_predictions', {'risk_hidden': {'physical': source.physical_by_variant['natural']}})
            archive('interventions', trace)
            archive('decoders', {'risk_hidden': {'layout': 'shared_pooled_seen_rows', 'decoder_count': 1,
                                                'feature_dimensions': 32, 'fits': [source.risk_decoder_parameters]}})
            model_path = 'audits/grounded_attention_outcomes_v1/model-1301/instrument-structured.pt.gz'
            (root/'manifest.json').write_text(json.dumps({'frozen_input_sha256': {model_path: 'toy-frozen-model'},
                                                         'permutation_source_tolerance': {'atol': 1e-6, 'rtol': 1e-5},
                                                         'linear_decoder_roundoff_floor': 1e-12}))
            (root/'results.json').write_text(json.dumps({'models': {'1301': {'readouts': {'risk_hidden': {
                'assessment': {'all_seen': {'pooled': source.physical_error_summary}}}}}}}))
            completion = {'status': 'single_registered_attempt_completed', 'archive_files': archive_files,
                          'manifest_sha256': interface.digest(root/'manifest.json'),
                          'results_sha256': interface.digest(root/'results.json')}
            (root/'completion.json').write_text(json.dumps(completion))
            verification = {'status': 'exact_collection_features_decoder_selection_interventions_results_replay',
                            'all_files_verified': True, 'manifest_sha256': completion['manifest_sha256'],
                            'results_sha256': completion['results_sha256']}
            (root/'verification.json').write_text(json.dumps(verification))
            real_digest = interface.digest
            with patch.object(interface, 'digest', side_effect=lambda path: 'toy-frozen-model'
                              if str(path) == model_path else real_digest(path)):
                loaded = interface.load_bundle(1301, root)
                self.assertTrue(torch.equal(loaded.original_sensory, source.original_sensory))
                self.assertFalse(hasattr(loaded, 'physical_targets'))
                self.assertEqual(interface.record(loaded, 0)['reporter_input'],
                                 interface.record(source, 0)['reporter_input'])
                verification['all_files_verified'] = False
                (root/'verification.json').write_text(json.dumps(verification))
                with self.assertRaises(ValueError):
                    interface.load_bundle(1301, root)


    def test_memory_readout_corruption_and_inflated_bounds_are_rejected(self):
        for recipient in (0, 1):  # Both unseen and observed recipients validated.
            source = fixture()
            source.physical_by_variant['memory_readdressed'][0, recipient, 0] += 10
            with self.assertRaises(ValueError):
                interface.record(source, 0, 'memory_readdressed')
        source = fixture()
        source.decode_permutation_error_bounds['propagated_bound'] += 100
        with self.assertRaises(ValueError):
            interface.record(source, 0)
        source = fixture()
        source.risk_decoder_parameters['feature_scale'][0] = 0
        with self.assertRaises(ValueError):
            interface.record(source, 0)

    def test_registered_roundoff_envelope_accepts_measured_small_source_delta(self):
        source = fixture()
        original = source.forecasts_by_variant['natural']['risk_hidden'][:, 0].roll(1, 1).double()
        moved = source.forecasts_by_variant['memory_readdressed']['risk_hidden']
        moved[0, 0, 1, 0] += 1e-7
        delta = (moved[:, 0].double()-original).abs()
        params = source.risk_decoder_parameters
        bound = delta/params['feature_scale']@params['weights'][:, :2].abs()*params['target_scale']
        source.physical_by_variant['memory_readdressed'][0, 1, 0] += float(delta[0, 1, 0])
        actual = (source.physical_by_variant['memory_readdressed']-source.physical_by_variant['natural'].roll(1, 1)).abs()
        source.decode_permutation_error_bounds = {'absolute_feature_difference': delta,
                                                 'absolute_output_difference': actual, 'propagated_bound': bound}
        value = interface.record(source, 0, 'memory_readdressed')
        self.assertFalse(value['audit_provenance']['decoder_parameters_and_bounds_reporter_context'])
        self.assertNotIn('risk_decoder_parameters', value['reporter_input'])
        self.assertNotIn('decode_permutation_error_bounds', value['reporter_input'])
        source.physical_by_variant['memory_readdressed'][0, 1, 0] += .001
        source.decode_permutation_error_bounds['absolute_output_difference'] = (
            source.physical_by_variant['memory_readdressed']-source.physical_by_variant['natural'].roll(1, 1)).abs()
        with self.assertRaises(ValueError):
            interface.record(source, 0)

    def test_all_table_boundaries_reject_duplicates_missing_addresses_and_extra_fields(self):
        value = interface.record(fixture(), 0)['reporter_input']
        tables = interface.address_aware_tables(value)
        for name in interface.TABLE_SCHEMAS:
            for operation in ('duplicate', 'missing', 'extra', 'wrong'):
                changed = copy.deepcopy(tables)
                rows = changed['tables'][name]
                if operation == 'duplicate':
                    rows.append(copy.deepcopy(rows[0]))
                elif operation == 'missing':
                    rows.pop()
                elif operation == 'extra':
                    rows[0]['physical_targets'] = 'FORBIDDEN'
                else:
                    rows[0][interface.TABLE_SCHEMAS[name][0]] = 'wrong'
                with self.subTest(table=name, operation=operation), self.assertRaises(ValueError):
                    interface.join_address_aware_tables(changed)
        for target in ('top', 'metadata', 'nested'):
            changed = copy.deepcopy(tables)
            if target == 'top':
                changed['physical_targets'] = 'FORBIDDEN'
            elif target == 'metadata':
                changed['frozen_sources']['physical_targets'] = 'FORBIDDEN'
            else:
                changed['tables']['external_readouts'][0]['external_image_statistic_readouts']['physical_targets'] = 'FORBIDDEN'
            with self.assertRaises(ValueError):
                interface.join_address_aware_tables(changed)
        value['locations'][0]['physical_targets'] = 'FORBIDDEN'
        with self.assertRaises(ValueError):
            interface.address_aware_tables(value)

    def test_record_complete_opportunity_and_task_only_legend(self):
        with self.assertRaises(ValueError):
            interface.record(fixture(n=15), 0)
        value = interface.record(fixture(), 0)['reporter_input']
        pieces = list(value['source_semantics'].values())
        pieces += [node['external_image_statistic_readouts']['interpretation'] for node in value['locations']]
        for phrase in ('felt confidence', 'native presentation', 'appearance', 'consciousness', 'qualia'):
            self.assertFalse(any(phrase in text.lower() for text in pieces))


if __name__ == '__main__':
    unittest.main()
