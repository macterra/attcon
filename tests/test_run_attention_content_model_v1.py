"""Tiny phase-boundary fixtures; no worlds, weights, fitting or Reporter calls."""
import argparse
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1] / 'scripts/run_attention_content_model_v1.py'
spec = importlib.util.spec_from_file_location('phase_runner_fixture', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PhaseBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.root_patch = patch.object(runner, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def write(self, path, value):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))
        return target

    def phase_receipt(self, artifacts, registration='registered'):
        return self.write('run/fit/receipt.json', {'phase': 'fit', 'status': 'completed',
            'registration_sha256': registration, 'artifact_sha256': artifacts})

    def test_receipt_rejects_empty_artifact_inventory_before_binding(self):
        self.phase_receipt({})
        with patch.object(runner, 'bound_files') as bind:
            with self.assertRaisesRegex(ValueError, 'incomplete phase artifact inventory'):
                runner.receipt(self.root / 'run', 'fit', 'a' * 40, 'registered')
            bind.assert_not_called()

    def test_receipt_rejects_a_different_registration(self):
        self.phase_receipt({}, registration='foreign')
        with self.assertRaisesRegex(ValueError, 'different input registration'):
            runner.receipt(self.root / 'run', 'fit', 'a' * 40, 'registered')

    def test_receipt_binds_every_required_archive_and_log(self):
        names = [f'run/fit/states-{seed}.pt.gz' for seed in runner.SEEDS] + ['run/fit/first_run.log']
        hashes = {name: f'fixture-{index}' for index, name in enumerate(names)}
        receipt_path = self.phase_receipt(hashes)
        with patch.object(runner, 'bound_files') as bind:
            runner.receipt(self.root / 'run', 'fit', 'a' * 40, 'registered')
            bind.assert_called_once_with('a' * 40,
                {'run/fit/receipt.json': runner.digest(receipt_path), **hashes})

    def test_commit_binding_rejects_different_committed_bytes(self):
        path = self.root / 'source.py'
        path.write_bytes(b'current')
        completed = subprocess.CompletedProcess([], 0, stdout=b'old', stderr=b'')
        with patch.object(runner.subprocess, 'run', return_value=completed):
            with self.assertRaisesRegex(ValueError, 'committed binding failed'):
                runner.bound_files('a' * 40, {'source.py': hashlib.sha256(b'current').hexdigest()})

    def test_native_acceptance_pins_the_published_review_before_json(self):
        for name in ('metadata_manifest.json', 'metadata.json.gz'):
            path = self.root / runner.NATIVE_REVIEW_DIR / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'invalid-review-fixture')
        with patch.object(runner, 'bound_files', side_effect=ValueError('commit pin rejected')) as bind:
            with self.assertRaisesRegex(ValueError, 'commit pin rejected'):
                runner.native_acceptance()
            self.assertEqual(bind.call_args.args[0], runner.NATIVE_ACCEPTANCE_COMMIT)
            self.assertEqual(set(bind.call_args.args[1]), {
                f'{runner.NATIVE_REVIEW_DIR}/metadata_manifest.json',
                f'{runner.NATIVE_REVIEW_DIR}/metadata.json.gz'})

    def test_registration_rejects_an_omitted_source_before_review(self):
        registration = {'schema': runner.SCHEMA, 'review_path': 'approval.json', 'source_hashes': {}}
        with self.assertRaisesRegex(ValueError, 'reviewed source closure'):
            runner.validate_registration(registration)

    def test_original_import_failure_survives_a_retention_failure_and_blocks_retry(self):
        registration = {'schema': runner.SCHEMA, 'source_commit': 'a' * 40,
            'source_hashes': {}, 'input_hashes': {}, 'output_dir': 'run'}
        path = self.write('registration.json', registration)
        args = argparse.Namespace(phase='fit', registration=path, registration_commit='b' * 40,
                                  fit_commit=None, dev_commit=None, assessment=None)
        real_import = builtins.__import__
        def failing_import(name, *args, **kwargs):
            if name == 'torch':
                raise ImportError('original torch import failure fixture')
            return real_import(name, *args, **kwargs)
        with patch.object(runner, 'validate_registration'), patch.object(runner, 'ancestor'), \
             patch.object(runner, 'bound_files'), patch.object(runner, 'native_acceptance', return_value={}), \
             patch.object(runner, 'save_tensor_archive', side_effect=OSError('secondary retention fixture')), \
             patch('builtins.__import__', side_effect=failing_import):
            with self.assertRaisesRegex(ImportError, 'original torch import failure fixture'):
                runner.execute_phase(args)
            failure = json.loads((self.root / 'run/fit/failure.json').read_text())
            retention = json.loads((self.root / 'run/fit/retention_failure.json').read_text())
            self.assertIn('original torch import failure fixture', failure['traceback'])
            self.assertNotIn('secondary retention fixture', failure['traceback'])
            self.assertEqual(failure['registration_sha256'], runner.digest(path))
            self.assertIsNone(failure['seed'])
            self.assertFalse(failure['retry_allowed'])
            self.assertTrue(retention['original_failure_preserved'])
            self.assertIn('secondary retention fixture', retention['traceback'])
            self.assertFalse((self.root / 'run/fit/receipt.json').exists())
            self.assertFalse((self.root / 'run/fit/first_run.log').exists())
            with self.assertRaises(FileExistsError):
                runner.execute_phase(args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
