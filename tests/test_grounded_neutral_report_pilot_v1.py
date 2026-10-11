"""Generated source-only fixtures; never load selected real source archives."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

import grounded_neutral_report_pilot_v1 as pilot
import grounded_report_interface_v1 as interface
from test_grounded_report_interface_v1 import fixture


def registration(output_root='targets'):
    return pilot.make_registration(
        source_sha256={path: 'a'*64 for path in pilot.expected_source_paths()},
        code_sha256={path: 'b'*64 for path in ('scripts/grounded_report_interface_v1.py',
                                             'scripts/grounded_neutral_report_pilot_v1.py')},
        protocol_sha256={'docs/protocol.md': 'c'*64},
        review_receipts={name: {'id': name+'-generated-only', 'path': 'docs/'+name+'.md', 'sha256': 'd'*64}
                         for name in ('inventory', 'independent_correction_review', 'exporter_independent_review')},
        output_root=output_root)


def bundles():
    result = {}
    for seed in interface.SEEDS:
        source = fixture(n=16)
        source.visual_seed = seed
        result[seed] = source
    return result


class GroundedNeutralPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.bundles = bundles()
        cls.registration = registration()
        with patch.object(interface, 'load_bundle', side_effect=AssertionError('no real sources')):
            cls.prepared = pilot.build_prepared(cls.bundles, cls.registration)

    def test_exact_factorial_export_composition_and_no_mutation(self):
        moved = interface.record(self.bundles[1301], 0, 'memory_readdressed')['reporter_input']
        original = copy.deepcopy(moved)
        joint = pilot.derive_memory_route(moved)
        self.assertEqual(moved, original)
        self.assertEqual(joint['locations'], moved['locations'])
        self.assertEqual(joint['native_route_history'], moved['native_route_history'])
        for index, row in enumerate(joint['command_forecasts']):
            self.assertEqual(row['predicted_delivery_distribution'],
                             moved['command_forecasts'][(index-1) % 4]['predicted_delivery_distribution'])
            for node in joint['locations']:
                loc = node['location']
                d = row['predicted_delivery_distribution'][loc]
                self.assertEqual(row['estimated_decision_success_after_command'][loc],
                                 d*node['estimated_fresh_glimpse_decision_success']
                                 +(1-d)*node['estimated_current_decision_success'])

    def test_all_nine_source_ids_and_addresses_have_exact_inverse(self):
        value = interface.record(self.bundles[1301], 0)['reporter_input']
        mapping = pilot.identifier_map(34092)
        for seed in interface.SEEDS:
            for source in (f'visual-{seed}', f'structured-{seed}', f'risk-hidden-ridge-{seed}'):
                self.assertIn(source, mapping)
        remapped = pilot.remapped_record(value, mapping)
        self.assertEqual(pilot.undo_relabel(remapped, mapping), value)
        self.assertNotEqual(remapped['frozen_sources'], value['frozen_sources'])
        with self.assertRaises(ValueError):
            interface.validate_reporter_input(remapped)
        # The frozen canonical whitelist must run before any generic transformation.
        injected = copy.deepcopy(value)
        injected['world_truth'] = 'FORBIDDEN'
        with self.assertRaises(ValueError):
            pilot.remapped_record(injected, mapping)
        with self.assertRaises(ValueError):
            pilot.relabel(value, {'l0': 'x', 'l1': 'x'})

    def test_opaque_tables_remain_exactly_joinable(self):
        value = interface.record(self.bundles[1301], 0)['reporter_input']
        mapping = pilot.identifier_map(405)
        tables = pilot.remapped_tables(value, mapping)
        self.assertEqual(interface.join_address_aware_tables(pilot.undo_relabel(tables, mapping)), value)
        inverse = pilot.undo_relabel(tables, mapping)
        inverse['tables']['external_readouts'][0]['location'] = 'l1'
        with self.assertRaises(ValueError):
            interface.join_address_aware_tables(inverse)

    def test_budget_assignment_full_association_and_all_query_pairs(self):
        prepared = self.prepared
        self.assertEqual(len(prepared['packets']), 624)
        self.assertEqual(len(prepared['audit']), 624)
        self.assertEqual(len({packet['packet_id'] for packet in prepared['packets']}), 624)
        counts = {}
        pairs = {seed: set() for seed in interface.SEEDS}
        for packet, audit in zip(prepared['packets'], prepared['audit']):
            self.assertEqual(packet['packet_id'], audit['packet_id'])
            self.assertEqual(audit['packet_sha256'], pilot.sha256(pilot.encoded(packet)))
            counts[audit['arm']] = counts.get(audit['arm'], 0)+1
            pairs[audit['visual_seed']].add(tuple(audit['canonical_query'].values()))
            self.assertEqual(pilot.undo_relabel(packet['source_record'], audit['identifier_map']),
                             audit['source_view_before_aliases'])
            self.assertEqual(pilot.undo_relabel(packet['indexed_query'], audit['identifier_map']),
                             audit['canonical_query'])
            text = json.dumps(packet)
            for forbidden in ('chronological_events', 'world_index', 'source_variant', 'identifier_map',
                              'required_facts', 'truth_orientation', 'pixels', '1301', '1311', '1321'):
                self.assertNotIn('"'+forbidden+'"', text)
            for key in ('visual_model', 'outcome_model', 'external_readout'):
                if key in packet['source_record']['frozen_sources']:
                    self.assertTrue(packet['source_record']['frozen_sources'][key].startswith('x'))
        self.assertEqual(counts, dict.fromkeys(pilot.ARMS, 48))
        self.assertTrue(all(len(value) == 16 for value in pairs.values()))
        self.assertNotEqual([(row['visual_seed'], row['world_index'], row['arm']) for row in prepared['audit']],
                            sorted((row['visual_seed'], row['world_index'], row['arm']) for row in prepared['audit']))

    def test_restoration_apparatus_and_alias_controls_preserve_sources(self):
        rows = {row['arm']: (packet, row) for packet, row in zip(self.prepared['packets'], self.prepared['audit'])
                if row['visual_seed'] == 1301 and row['world_index'] == 0}
        natural = rows['natural'][0]
        for arm in ('sham', 'restore_memory', 'restore_route'):
            self.assertEqual(rows[arm][0]['source_record'], natural['source_record'])
            self.assertEqual(rows[arm][0]['indexed_query'], natural['indexed_query'])
        apparatus = rows['apparatus'][0]
        self.assertEqual(apparatus['source_record'], natural['source_record'])
        self.assertNotEqual(apparatus['instruction'], natural['instruction'])
        self.assertEqual(rows['id_remap'][1]['source_view_before_aliases'],
                         rows['natural'][1]['source_view_before_aliases'])
        self.assertNotEqual(rows['id_remap'][0]['indexed_query'], natural['indexed_query'])

    def test_ablations_drop_exact_source_classes_and_are_unmatched(self):
        canonical = interface.record(self.bundles[1301], 0)['reporter_input']
        sensory = pilot.ablate(canonical, 'sensory_only')
        self.assertNotIn('command_forecasts', sensory)
        self.assertNotIn('native_route_history', sensory)
        self.assertIn('external_readout_evaluation_summary', sensory)
        self.assertIn('external_image_statistic_readouts', sensory['locations'][0])
        self.assertNotIn('estimated_current_decision_success', sensory['locations'][0])
        route = pilot.ablate(canonical, 'route_only')
        self.assertNotIn('locations', route)
        self.assertNotIn('external_readout_evaluation_summary', route)
        self.assertEqual(set(route['command_forecasts'][0]), {'command', 'predicted_delivery_distribution'})
        absent = pilot.ablate(canonical, 'no_external')
        self.assertNotIn('external_readout', absent['frozen_sources'])
        self.assertNotIn('external_readout_evaluation_summary', absent)
        self.assertNotIn('external_image_statistic_readouts', absent['locations'][0])
        self.assertEqual(absent['command_forecasts'], canonical['command_forecasts'])
        self.assertIn('risk_feature_source', absent['locations'][0])
        self.assertTrue(all(row['unmatched_ablation'] for row in self.prepared['audit']
                            if row['arm'] in pilot.UNMATCHED_ARMS))

    def test_source_first_fact_paths_select_query_only(self):
        value = interface.record(self.bundles[1301], 0)['reporter_input']
        facts = pilot.required_facts(value, {'location': 'l2', 'command': 'k3'})
        for fact in facts['facts']:
            self.assertTrue(fact['source_path'].startswith(('/locations/2/', '/command_forecasts/3/', '/native_route_history/3', '/source_semantics/')))
        self.assertEqual(facts['required_unit_count'], 21)
        self.assertEqual(facts['source_absence_unit_count'], 0)
        self.assertFalse(any('risk_feature_source' in item['source_path'] for item in facts['facts']))
        self.assertFalse(any('evaluation_summary' in item['source_path'] for item in facts['facts']))
        text = json.dumps(facts)
        self.assertNotIn('/locations/0/', text)
        self.assertNotIn('/command_forecasts/0/', text)
        expected = {'sensory_only': (9, 3), 'route_only': (6, 3), 'no_external': (18, 1)}
        for arm, counts in expected.items():
            masked = pilot.required_facts(pilot.ablate(value, arm), {'location': 'l2', 'command': 'k3'})
            self.assertEqual((masked['required_unit_count'], masked['source_absence_unit_count']), counts)

    def test_hand_entered_perfect_qualification_counts_cannot_release_reporters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'qualification.json'
            path.write_bytes(pilot.encoded({'schema': pilot.SCHEMA, 'status': 'factual_qualification_passed',
                'synthetic_cases': 12, 'assertions': 48, 'auditors': 2, 'correct_by_auditor': [48, 48],
                'coverage_correct_by_auditor': [12, 12]}))
            with self.assertRaisesRegex(ValueError, 'independently reviewed reporting'):
                pilot.reporter_release('registration.json', 'targets', path, repo_root=directory,
                                       freeze_checkout=directory, freeze_commit='1'*40)

    def test_replay_rejects_full_source_association_tamper(self):
        tampered = copy.deepcopy(self.prepared)
        tampered['audit'][0]['source_audit']['history_retains_actual_past'] = False
        with self.assertRaisesRegex(ValueError, 'deterministic replay'):
            pilot.verify_prepared(tampered, self.bundles, self.registration)

    def test_single_attempt_failure_is_preserved_and_alternate_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'registration.json'
            path.write_bytes(pilot.encoded(self.registration))
            output = root/'targets'
            with patch.object(pilot, 'check_frozen_registration', side_effect=ValueError('generated failure')):
                with self.assertRaisesRegex(ValueError, 'generated failure'):
                    pilot.export_once(path, output, repo_root=root, freeze_checkout=root, freeze_commit='1'*40)
                self.assertTrue((output/'attempt.json').is_file())
                self.assertFalse(json.loads((output/'failure.json').read_bytes())['retry_allowed'])
                with self.assertRaises(FileExistsError):
                    pilot.export_once(path, output, repo_root=root, freeze_checkout=root, freeze_commit='1'*40)
                with self.assertRaisesRegex(ValueError, 'sole registered attempt'):
                    pilot.export_once(path, root/'second', repo_root=root, freeze_checkout=root, freeze_commit='1'*40)
                self.assertFalse((root/'second').exists())

    def test_source_hash_check_precedes_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/next(iter(self.registration['source_sha256']))
            path.parent.mkdir(parents=True)
            path.write_text('Generated wrong-hash source receipt only.\n')
            with patch.object(Path, 'cwd', return_value=root), \
                    patch.object(interface, 'load_bundle', side_effect=AssertionError('must not load')) as loader:
                with self.assertRaisesRegex(ValueError, 'registered source hash changed'):
                    pilot._load_registered_bundles(self.registration, root)
                loader.assert_not_called()

    def test_registration_checks_budget_receipts_and_input_hashes(self):
        for field, value in (('max_target_packets', 625), ('arms', ['natural']),
                             ('export_attempts', 2), ('query_rule', {'location': 'outcome_selected'})):
            malformed = copy.deepcopy(self.registration)
            malformed[field] = value
            with self.assertRaises(ValueError):
                pilot.validate_registration(malformed)
        malformed = copy.deepcopy(self.registration)
        malformed['review_receipts'].pop('exporter_independent_review')
        with self.assertRaises(ValueError):
            pilot.validate_registration(malformed)
        malformed = copy.deepcopy(self.registration)
        malformed['source_sha256'].pop(next(iter(malformed['source_sha256'])))
        with self.assertRaises(ValueError):
            pilot.validate_registration(malformed)

    def test_real_git_freeze_and_disk_replay_with_generated_data_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_hashes = {}
            for name in pilot.expected_source_paths():
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'generated placeholder, not an archive')
                source_hashes[name] = pilot.digest(path)
            code_hashes = {}
            for name, source_path in (('scripts/grounded_report_interface_v1.py', interface.__file__),
                                      ('scripts/grounded_neutral_report_pilot_v1.py', pilot.__file__)):
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(Path(source_path).read_bytes())
                code_hashes[name] = pilot.digest(path)
            (root/'docs').mkdir()
            protocol = root/'docs/protocol.md'
            protocol.write_text('Generated protocol fixture only.\n')
            reviews = {}
            for name in ('inventory', 'independent_correction_review', 'exporter_independent_review'):
                path = root/'docs'/(name+'.md')
                path.write_text('Generated accepted review fixture only.\n')
                reviews[name] = {'id': name+'-fixture', 'path': str(path.relative_to(root)), 'sha256': pilot.digest(path)}
            manifest = pilot.make_registration(source_sha256=source_hashes, code_sha256=code_hashes,
                protocol_sha256={'docs/protocol.md': pilot.digest(protocol)}, review_receipts=reviews,
                output_root='targets')
            registration_path = root/'registration.json'
            registration_path.write_bytes(pilot.encoded(manifest))
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True).stdout.decode().strip()
            git('init', '-q')
            git('add', 'scripts', 'docs', 'registration.json')
            git('-c', 'user.name=Generated Fixture', '-c', 'user.email=fixture@example.invalid',
                'commit', '-qm', 'Generated pilot fixture freeze')
            commit = git('rev-parse', 'HEAD')
            options = {'repo_root': root, 'freeze_checkout': root/'.git', 'freeze_commit': commit}
            loaded, _ = pilot.check_frozen_registration(registration_path, **options)
            self.assertEqual(loaded, manifest)
            with patch.object(pilot, '_load_registered_bundles', return_value=self.bundles):
                pilot.export_once(registration_path, root/'targets', **options)
                receipt = pilot.verify_export(registration_path, root/'targets', **options)
                self.assertEqual(receipt['packet_count'], 624)
                self.assertFalse(receipt['reporter_run'])
                packet_path = next((root/'targets/reporter_packets').iterdir())
                packet_path.write_bytes(packet_path.read_bytes()+b' ')
                with self.assertRaisesRegex(ValueError, 'packet hash'):
                    pilot.verify_export(registration_path, root/'targets', **options)
            protocol.write_text('Changed after freeze.\n')
            with self.assertRaisesRegex(ValueError, 'code/protocol/review changed'):
                pilot.check_frozen_registration(registration_path, **options)


if __name__ == '__main__':
    unittest.main()
