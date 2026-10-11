"""Independent first-pass fake-only review; no selected archive or Reporter.

Written after the author's READY notice and root's immutable draft commit receipt.
All disk source objects below are generated placeholders in temporary directories.
"""
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
from test_grounded_report_interface_v1 import fixture as toy_source


def generated_registration():
    return pilot.make_registration(
        source_sha256={name: 'a'*64 for name in pilot.expected_source_paths()},
        code_sha256={name: 'b'*64 for name in
                     ('scripts/grounded_report_interface_v1.py', 'scripts/grounded_neutral_report_pilot_v1.py')},
        protocol_sha256={'docs/generated_protocol.md': 'c'*64},
        review_receipts={name: {'id': name+'-FAKE-ONLY', 'path': 'docs/'+name+'.md', 'sha256': 'd'*64}
                         for name in ('inventory', 'independent_correction_review', 'exporter_independent_review')},
        output_root='targets')


def resolve(record, pointer):
    value = record
    for part in pointer.split('/')[1:]:
        part = part.replace('~1', '/').replace('~0', '~')
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


class IndependentReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.registration = generated_registration()
        cls.bundles = {}
        for seed in (1301, 1311, 1321):
            value = toy_source(16)
            value.visual_seed = seed
            cls.bundles[seed] = value
        with patch.object(interface, 'load_bundle', side_effect=AssertionError('REAL SOURCE READ FORBIDDEN')):
            cls.prepared = pilot.build_prepared(cls.bundles, cls.registration)
        cls.rows = {(audit['visual_seed'], audit['world_index'], audit['arm']): (packet, audit)
                    for packet, audit in zip(cls.prepared['packets'], cls.prepared['audit'])}

    def test_01_fixed_624_slots_all_pairs_and_128_bit_ids(self):
        self.assertEqual(len(self.rows), 624)
        self.assertEqual(len(self.prepared['packets']), 624)
        self.assertEqual(set(pilot.ARMS), {'natural', 'memory', 'route', 'memory_route', 'sham',
                         'restore_memory', 'restore_route', 'id_remap', 'apparatus', 'tables',
                         'sensory_only', 'route_only', 'no_external'})
        for seed in (1301, 1311, 1321):
            for world in range(16):
                for arm in pilot.ARMS:
                    packet, audit = self.rows[seed, world, arm]
                    self.assertRegex(packet['packet_id'], r'^P[0-9a-f]{32}$')
                    self.assertEqual(audit['canonical_query'],
                                     {'location': f'l{world % 4}', 'command': f'k{world // 4}'})
                    self.assertEqual(pilot.undo_relabel(packet['indexed_query'], audit['identifier_map']),
                                     audit['canonical_query'])
                    self.assertEqual(set(packet), {'packet_id', 'instruction', 'indexed_query', 'source_record'})
                    self.assertEqual(audit['packet_sha256'], pilot.sha256(pilot.encoded(packet)))
                    if arm != 'route_only':
                        source = audit['source_view_before_aliases']
                        locations = source['tables']['stored_evidence'] if arm == 'tables' else source['locations']
                        self.assertEqual({node['location'] for node in locations}, {'l0', 'l1', 'l2', 'l3'})
        self.assertEqual(len({packet['packet_id'] for packet in self.prepared['packets']}), 624)

    def test_02_full_inventory_semantics_scalars_and_null_default(self):
        for packet, audit in zip(self.prepared['packets'], self.prepared['audit']):
            if audit['arm'] in pilot.UNMATCHED_ARMS:
                continue
            facts = audit['required_facts']
            self.assertEqual(facts['required_unit_count'], 21)
            self.assertEqual(facts['source_absence_unit_count'], 0)
            units = facts['required_units']
            self.assertEqual(len({unit['unit_id'] for unit in units}), 21)
            self.assertEqual(sum(unit['source_group'] == 'delivery_forecasts' for unit in units), 5)
            self.assertEqual(sum(unit['source_group'] == 'outcome_forecasts' for unit in units), 6)
            names = {unit['unit_id'] for unit in units}
            self.assertTrue({'stored_decision_meaning', 'forecast_meaning', 'external_estimate_meaning'} <= names)
            for unit in units:
                for fact in unit['source_facts']:
                    self.assertEqual(resolve(packet['source_record'], fact['source_path']), fact['value'])
        canonical = interface.record(self.bundles[1301], 0)['reporter_input']
        facts = pilot.required_facts(canonical, {'location': 'l3', 'command': 'k3'})
        self.assertEqual(facts['required_unit_count'], 21)
        by_id = {unit['unit_id']: unit for unit in facts['required_units']}
        self.assertIsNone(by_id['raw_orientation_logit']['source_facts'][0]['value'])
        self.assertEqual(by_id['decision_or_default']['source_facts'][1]['value'],
                         {'value': 'horizontal', 'basis': 'fixed default guess'})
        self.assertEqual(by_id['latest_observed_delivery']['source_facts'][0]['value']['status'], 'no_delivery_observed')

    def test_03_sham_inverses_and_apparatus_match_exact_input(self):
        for seed in (1301, 1311, 1321):
            for world in range(16):
                packet, audit = self.rows[seed, world, 'natural']
                substantive = {key: value for key, value in packet.items() if key != 'packet_id'}
                for arm in ('sham', 'restore_memory', 'restore_route'):
                    matched, other = self.rows[seed, world, arm]
                    self.assertEqual({key: value for key, value in matched.items() if key != 'packet_id'}, substantive)
                    self.assertEqual(other['identifier_map'], audit['identifier_map'])
                apparatus, _ = self.rows[seed, world, 'apparatus']
                self.assertEqual(apparatus['source_record'], packet['source_record'])
                self.assertEqual(apparatus['indexed_query'], packet['indexed_query'])
                self.assertEqual(apparatus['instruction'].replace('ordinary apparatus', 'system'), packet['instruction'])

    def test_04_crossed_route_inverse_then_archived_memory_restoration(self):
        for seed in (1301, 1311, 1321):
            for world in range(16):
                _, audit = self.rows[seed, world, 'memory_route']
                joint = audit['source_view_before_aliases']
                moved = self.rows[seed, world, 'memory'][1]['source_view_before_aliases']
                natural = self.rows[seed, world, 'natural'][1]['source_view_before_aliases']
                self.assertFalse(audit['derived_joint_network_inference'])
                self.assertEqual(joint['locations'], moved['locations'])
                self.assertEqual(joint['native_route_history'], natural['native_route_history'])
                inverse = copy.deepcopy(joint)
                for command, row in enumerate(inverse['command_forecasts']):
                    row['predicted_delivery_distribution'] = copy.deepcopy(
                        joint['command_forecasts'][(command+1) % 4]['predicted_delivery_distribution'])
                    for node in inverse['locations']:
                        loc = node['location']
                        d = row['predicted_delivery_distribution'][loc]
                        row['estimated_decision_success_after_command'][loc] = (
                            d*node['estimated_fresh_glimpse_decision_success']+(1-d)*node['estimated_current_decision_success'])
                self.assertEqual(inverse, moved)
                interface.validate_reporter_input(inverse)
                restored = interface.record(self.bundles[seed], world, 'memory_restored')['reporter_input']
                self.assertEqual(restored, natural)

    def test_05_aliases_cover_every_reference_and_tables_join_exactly(self):
        for seed in (1301, 1311, 1321):
            for world in range(16):
                natural = self.rows[seed, world, 'natural'][1]['source_view_before_aliases']
                packet, audit = self.rows[seed, world, 'id_remap']
                self.assertEqual(pilot.undo_relabel(packet['source_record'], audit['identifier_map']), natural)
                self.assertNotEqual(audit['identifier_map'], self.rows[seed, world, 'natural'][1]['identifier_map'])
                table_packet, table_audit = self.rows[seed, world, 'tables']
                tables = pilot.undo_relabel(table_packet['source_record'], table_audit['identifier_map'])
                for rows in tables['tables'].values():
                    rows.reverse()
                joined = interface.join_address_aware_tables(tables)
                joined['locations'].sort(key=lambda row: row['location'])
                joined['command_forecasts'].sort(key=lambda row: row['command'])
                self.assertEqual(joined, natural)
                tokens = json.dumps(packet['source_record'])
                for token in ('l0', 'l1', 'l2', 'l3', 'k0', 'k1', 'k2', 'k3', 'e0', 'e1', 'e2', 'e3',
                              f'visual-{seed}', f'structured-{seed}', f'risk-hidden-ridge-{seed}'):
                    self.assertNotIn('"'+token+'"', tokens)

    def test_06_typed_missing_groups_not_supplied_and_unmatched(self):
        expected = {'sensory_only': (9, {'native_history', 'delivery_forecasts', 'outcome_forecasts'}),
                    'route_only': (6, {'stored_evidence', 'outcome_forecasts', 'external_readouts'}),
                    'no_external': (18, {'external_readouts'})}
        for packet, audit in zip(self.prepared['packets'], self.prepared['audit']):
            if audit['arm'] not in expected:
                continue
            count, missing = expected[audit['arm']]
            facts = audit['required_facts']
            self.assertEqual(facts['required_unit_count'], count)
            self.assertEqual({unit['source_group'] for unit in facts['source_absence_units']}, missing)
            self.assertTrue(all(unit['status'] == 'not_supplied' for unit in facts['source_absence_units']))
            self.assertTrue(audit['unmatched_ablation'])
            self.assertFalse(audit['matched_information_control'])
            if audit['arm'] == 'route_only':
                self.assertFalse(facts['source_groups']['stored_evidence'])

    def test_07_truth_injection_rejected_unclipped_estimates_preserved(self):
        source = interface.record(self.bundles[1301], 0)['reporter_input']
        self.assertLess(source['locations'][0]['external_image_statistic_readouts']['image_value_std_estimate'], 0)
        for transform in (lambda value: pilot.derive_memory_route(value),
                          lambda value: pilot.remapped_record(value, pilot.identifier_map(42)),
                          lambda value: pilot.ablate(value, 'no_external')):
            contaminated = copy.deepcopy(source)
            contaminated['command_mapping'] = {'truth': 'FORBIDDEN'}
            with self.assertRaises(ValueError):
                transform(contaminated)
        for packet in self.prepared['packets']:
            text = json.dumps(packet)
            for token in ('command_mapping', 'world_truth', 'pixels', 'chronological_events', 'physical_targets',
                          'identifier_map', 'world_index', 'source_variant', 'required_facts'):
                self.assertNotIn('"'+token+'"', text)

    def test_08_upstream_manifest_dependency_checked_before_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root/interface.PHYSICAL_ROOT/'manifest.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({'frozen_input_sha256': {'generated-upstream.pt': 'f'*64}}))
            with patch.object(Path, 'cwd', return_value=root), patch.object(pilot, '_check_sources'), \
                    patch.object(interface, 'load_bundle', side_effect=AssertionError('MUST NOT LOAD')) as loader:
                with self.assertRaisesRegex(ValueError, 'upstream frozen input absent'):
                    pilot._load_registered_bundles(self.registration, root)
                loader.assert_not_called()

    def test_09_sole_attempt_failure_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registration = root/'registration.json'
            registration.write_bytes(pilot.encoded(self.registration))
            with patch.object(pilot, 'check_frozen_registration', side_effect=ValueError('GENERATED PRELOAD FAILURE')), \
                    patch.object(interface, 'load_bundle', side_effect=AssertionError('MUST NOT LOAD')) as loader:
                with self.assertRaisesRegex(ValueError, 'GENERATED PRELOAD FAILURE'):
                    pilot.export_once(registration, root/'targets', repo_root=root,
                                      freeze_checkout=root, freeze_commit='0'*40)
                failure = json.loads((root/'targets/failure.json').read_bytes())
                self.assertFalse(failure['retry_allowed'])
                self.assertTrue((root/'targets/attempt.json').exists())
                with self.assertRaises(FileExistsError):
                    pilot.export_once(registration, root/'targets', repo_root=root,
                                      freeze_checkout=root, freeze_commit='0'*40)
                with self.assertRaisesRegex(ValueError, 'sole registered attempt'):
                    pilot.export_once(registration, root/'other', repo_root=root,
                                      freeze_checkout=root, freeze_commit='0'*40)
                self.assertFalse((root/'other').exists())
                loader.assert_not_called()

    def test_10_reporting_release_always_fails_closed(self):
        with patch.object(interface, 'load_bundle', side_effect=AssertionError('MUST NOT LOAD')) as loader:
            with self.assertRaisesRegex(ValueError, 'source-linked factual qualification verifier'):
                pilot.reporter_release('FAKE registration', 'FAKE output', 'FAKE perfect qualification',
                                       repo_root='FAKE repo', freeze_checkout='FAKE freeze', freeze_commit='f'*40)
            loader.assert_not_called()

    def test_11_verify_rejects_alternate_output_root(self):
        # Build complete fake-only on-disk output without invoking production loaders.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/'unregistered_copy'
            output.mkdir()
            (output/'reporter_packets').mkdir()
            registration_hash = 'e'*64
            commit = 'f'*40
            verification = {'schema': pilot.SCHEMA, 'status': 'exact_packets_sources_inverse_controls_verified',
                            'packet_count': 624, 'prepared_sha256': pilot.sha256(pilot.encoded(self.prepared)),
                            'reporter_run': False}
            (output/'audit.json').write_bytes(pilot.encoded(self.prepared['audit']))
            (output/'verification.json').write_bytes(pilot.encoded({**verification,
                'registration_sha256': registration_hash, 'freeze_commit': commit}))
            hashes = {}
            for packet in self.prepared['packets']:
                path = output/'reporter_packets'/(packet['packet_id']+'.json')
                path.write_bytes(pilot.encoded(packet))
                hashes[packet['packet_id']] = pilot.digest(path)
            (output/'completion.json').write_bytes(pilot.encoded({
                'status': 'single_registered_export_completed', 'registration_sha256': registration_hash,
                'freeze_commit': commit, 'audit_sha256': pilot.digest(output/'audit.json'),
                'verification_sha256': pilot.digest(output/'verification.json'), 'packet_sha256': hashes}))
            with patch.object(pilot, 'check_frozen_registration', return_value=(self.registration, registration_hash)), \
                    patch.object(pilot, '_load_registered_bundles', return_value=self.bundles):
                with self.assertRaisesRegex(ValueError, 'sole registered attempt'):
                    pilot.verify_export(root/'registration.json', output, repo_root=root,
                                        freeze_checkout=root, freeze_commit=commit)

    def test_12_real_temporary_git_commit_binds_registration_code_protocol_and_reviews(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registration = copy.deepcopy(self.registration)
            for name, source in (('scripts/grounded_report_interface_v1.py', interface.__file__),
                                 ('scripts/grounded_neutral_report_pilot_v1.py', pilot.__file__)):
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(Path(source).read_bytes())
                registration['code_sha256'][name] = pilot.digest(path)
            for name in registration['protocol_sha256']:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('Independent generated protocol fixture, no target data.\n')
                registration['protocol_sha256'][name] = pilot.digest(path)
            for receipt in registration['review_receipts'].values():
                path = root/receipt['path']
                path.write_text('Independent generated review receipt, not scientific qualification.\n')
                receipt['sha256'] = pilot.digest(path)
            path = root/'registration.json'
            path.write_bytes(pilot.encoded(registration))
            def git(*arguments):
                return subprocess.run(['git', '-C', str(root), *arguments], check=True,
                                      capture_output=True).stdout.decode().strip()
            git('init', '-q')
            git('add', 'scripts', 'docs', 'registration.json')
            git('-c', 'user.name=Independent Generated Fixture', '-c',
                'user.email=fixture@example.invalid', 'commit', '-qm', 'Fake-only source registration')
            commit = git('rev-parse', 'HEAD')
            options = {'repo_root': root, 'freeze_checkout': root/'.git', 'freeze_commit': commit}
            self.assertEqual(pilot.check_frozen_registration(path, **options)[0], registration)
            path.write_bytes(path.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError, 'registration differs'):
                pilot.check_frozen_registration(path, **options)
            path.write_bytes(pilot.encoded(registration))
            for name in [*registration['code_sha256'], *registration['protocol_sha256'],
                         *(receipt['path'] for receipt in registration['review_receipts'].values())]:
                changed = root/name
                saved = changed.read_bytes()
                changed.write_bytes(saved+b'\nCHANGED\n')
                with self.assertRaisesRegex(ValueError, 'code/protocol/review changed'):
                    pilot.check_frozen_registration(path, **options)
                changed.write_bytes(saved)


if __name__ == '__main__':
    unittest.main(verbosity=2)
