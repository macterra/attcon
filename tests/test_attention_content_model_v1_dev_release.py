"""Heldout-release fixtures with temporary JSON and mocked bindings only."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1] / 'scripts/run_attention_content_model_v1.py'
spec = importlib.util.spec_from_file_location('dev_release_runner_fixture', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class DevReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        root_patch = patch.object(runner, 'ROOT', self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.base = self.root / 'run'
        self.write('run/dev/receipt.json', {'synthetic_receipt': True})
        self.write('registration.json', {'synthetic_registration': True})
        self.args = argparse.Namespace(dev_commit='d' * 40, fit_commit='b' * 40,
            registration_commit='a' * 40, registration=self.root / 'registration.json')
        self.registration = {'source_hashes': {'scripts/verify_attention_content_model_v1.py': 'registered-verifier'}}
        self.fit_receipt = {'artifact_sha256': {'synthetic-fit-archive': 'synthetic-sha'}}
        self.report = {'schema': 'attention_content_model_v1_independent_phase_reconstruction',
            'phase': 'dev', 'status': 'passed_phase_reconstruction', 'hard_invariants_pass': True,
            'error_count': 0, 'errors': [], 'check_count': 1, 'checks': [{'passed': True}],
            'dev_receipt_sha256': runner.digest(self.base / 'dev/receipt.json'),
            'fitted_artifact_sha256': self.fit_receipt['artifact_sha256'],
            'verifier_sha256': 'registered-verifier',
            'command_bindings': {'registration_commit': self.args.registration_commit,
                'registration': str(self.args.registration), 'fit_commit': self.args.fit_commit,
                'phase_commit': 'c' * 40}}
        self.assessment = {'schema': 'attention_content_model_v1_dev_acceptance',
            'hard_invariants_pass': True, 'dev_receipt_sha256': self.report['dev_receipt_sha256'],
            'fitted_artifact_sha256': self.fit_receipt['artifact_sha256'],
            'reconstruction_report_path': 'dev-report.json', 'reporting_released': False,
            'source_semantic_dispositions': {source: {'literal_content_disposition': 'unqualified',
                'presentation_manner_qualified': False} for source in ('A', 'ordinary', 'analytic')}}

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def validate(self):
        report = self.write('dev-report.json', self.report)
        self.assessment['reconstruction_report_sha256'] = runner.digest(report)
        path = self.write('acceptance.json', self.assessment)
        with patch.object(runner, 'bound_files') as bind, patch.object(runner, 'ancestor') as ancestor:
            result = runner.validate_dev_assessment(path, self.args, self.registration, self.base, self.fit_receipt)
            return result, bind.call_args_list, ancestor.call_args_list

    def test_negative_semantics_permit_fixed_heldout_with_bound_reconstruction(self):
        result, bindings, ancestry = self.validate()
        self.assertEqual(result['source_semantic_dispositions'], self.assessment['source_semantic_dispositions'])
        self.assertFalse(result['reporting_released'])
        self.assertEqual(len(bindings), 2)
        self.assertEqual(bindings[1].args, (self.args.dev_commit,
            {'dev-report.json': self.assessment['reconstruction_report_sha256']}))
        self.assertEqual([call.args for call in ancestry],
            [(self.args.fit_commit, 'c' * 40), ('c' * 40, self.args.dev_commit)])

    def test_unbound_success_assertion_cannot_release_heldout(self):
        self.assessment.pop('reconstruction_report_path')
        with self.assertRaises(KeyError):
            self.validate()

    def test_failed_report_cannot_be_promoted_by_assessment_success(self):
        self.report.update(status='failed_first_phase_reconstruction', error_count=1,
            errors=['retained-negative-check'], checks=[{'passed': False}])
        with self.assertRaisesRegex(ValueError, 'registered independent reconstruction'):
            self.validate()

    def test_report_from_a_different_verifier_cannot_release_heldout(self):
        self.report['verifier_sha256'] = 'another-verifier'
        with self.assertRaisesRegex(ValueError, 'registered independent reconstruction'):
            self.validate()

    def test_report_from_different_fit_commit_cannot_release_heldout(self):
        self.report['command_bindings']['fit_commit'] = 'e' * 40
        with self.assertRaisesRegex(ValueError, 'different registration or fitted artifacts'):
            self.validate()


if __name__ == '__main__':
    unittest.main(verbosity=2)
