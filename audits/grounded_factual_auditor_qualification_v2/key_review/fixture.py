"""Independent v2 generated probes, including actual isolated Git preservation.

Reuse this reviewer's immutable v1 source/schema/mutation checks, adapted to v2.
Only disposable /tmp repositories receive test commits; no project/recovery Git
mutation, model/archive/Reporter access, live auditor response or live receipt.
Fresh-looking metadata below is explicitly a synthetic boundary emulation.
"""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import assess_grounded_factual_auditor_qualification_v2 as scorer

DRAFT = ROOT / 'audits/grounded_factual_auditor_qualification_v2/draft'
PREVIOUS = ROOT / 'audits/grounded_factual_auditor_qualification_v1/key_review/fixture.py'
spec = importlib.util.spec_from_file_location('immutable_prior_independent_fixture', PREVIOUS)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
# Redirect in-memory globals only, without modifying the immutable prior file.
prior.scorer = scorer
prior.DRAFT = DRAFT


def fixture_submission(cases, key, identity, emulate_fresh=False):
    texts = {c['claim_id']: c['text'] for c in cases['claims']}
    return {'schema': 'grounded_factual_auditor_qualification_firstpass_v2',
        'metadata': {'auditor_id': identity, 'separate_context': emulate_fresh,
            'source_access': 'qualification_sources_only' if emulate_fresh else 'generated_code_test_materials',
            'key_exposure_before_firstpass': not emulate_fresh,
            'peer_response_exposure_before_firstpass': False, 'actual_target_exposure': False,
            'model_identity': 'synthetic_boundary_emulation_no_model_called', 'created_at': 'synthetic',
            'response_kind': 'fresh_qualification_firstpass' if emulate_fresh else 'synthetic_code_test'},
        'answers': [{**copy.deepcopy(a), 'claim_text': texts[a['claim_id']]} for a in key['answers']],
        'source_inventories': copy.deepcopy(key['source_inventories'])}


class IndependentSourceTests(prior.IndependentReview):
    def setUp(self):
        self.sources, self.cases, self.key = [scorer.load(DRAFT / n) for n in ('sources.json', 'cases.json', 'author_proposed_key.json')]
        self.submission = fixture_submission(self.cases, self.key, 'synthetic_pure_grader_not_auditor')

    def test_hash_bound_assessment_mocks_preserved_bytes_without_live_rating(self):
        # Supersede the inherited mock boundary with actual Git tests below.
        self.assertEqual(self.grade()['response_kind'], 'synthetic_code_test')

    def test_new_explicit_cases_and_provenance_are_source_consistent(self):
        claims = {c['claim_id']: c['text'] for c in self.cases['claims']}
        self.assertIn('currently at l1 in the world', claims['Q07-C4'])
        self.assertEqual(claims['Q10-C3'], 'The supplied tables view omits command fields and location-keyed delivery maps.')
        key = {a['claim_id']: a for a in self.key['answers']}
        self.assertEqual(key['Q07-C4']['temporal_scope'], 'current_snapshot')
        self.assertEqual(key['Q10-C3']['content_class'], 'absence_scope')
        provenance = scorer.load(DRAFT / 'material_provenance.json')
        self.assertEqual(provenance['source_sha256'], scorer.digest(DRAFT / 'sources.json'))
        self.assertEqual((DRAFT / 'sources.json').read_bytes(), (prior.ROOT / 'audits/grounded_factual_auditor_qualification_v1/draft/sources.json').read_bytes())

    def test_all_48_text_mutations_fail_and_v1_schema_rejected(self):
        for i in range(48):
            probe = copy.deepcopy(self.submission)
            probe['answers'][i]['claim_text'] += ' changed'
            result = self.grade(probe)
            self.assertFalse(result['passed'])
            self.assertEqual(result['claim_text_exact'], 47)
        probe = copy.deepcopy(self.submission)
        probe['schema'] = 'grounded_factual_auditor_qualification_firstpass_v1'
        with self.assertRaises(ValueError):
            self.grade(probe)


class RealGitPreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='independent-v2-git-probe-')
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.paths = scorer.load(DRAFT / 'material_paths.json')
        for relative in self.paths.values():
            p = self.repo / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((ROOT / relative).read_bytes())
        self.write('materials.json', self.paths)
        spec = importlib.util.spec_from_file_location('temporary_saved_v2_scorer', self.repo / self.paths['scorer'])
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        cases, key = [scorer.load(DRAFT / n) for n in ('cases.json', 'author_proposed_key.json')]
        self.firstpasses = ['synthetic_boundary/A.json', 'synthetic_boundary/B.json']
        for i,p in enumerate(self.firstpasses):
            self.write(p, fixture_submission(cases, key, 'emulated_synthetic_auditor_' + str(i), True))
        self.key_path, self.evidence_path = 'synthetic_boundary/key_review.json', 'synthetic_boundary/evidence_review.json'
        self.write(self.key_path, {'schema': 'grounded_factual_auditor_qualification_key_review_v2',
            'status': 'independently_key_reviewed', 'reviewer_id': 'synthetic_key_review', 'separate_context': True,
            'approved_material_sha256': {role: {'path': p, 'sha256': scorer.digest(self.repo / p)} for role,p in self.paths.items()},
            'checked_claim_ids': scorer.CLAIM_IDS, 'checked_packet_ids': scorer.PACKET_IDS, 'substantive_issues': []})
        self.write(self.evidence_path, {'schema': 'grounded_factual_auditor_qualification_evidence_review_v2',
            'reviewer_id': 'synthetic_evidence_review', 'separate_context': True, 'checked_claim_ids': scorer.CLAIM_IDS,
            'reviewed_firstpasses': [{'path': p, 'sha256': scorer.digest(self.repo / p)} for p in self.firstpasses],
            'all_96_claim_bases_relevant_and_valid': True, 'issues': []})
        self.git('init', '-q')
        self.preserve()

    def git(self, *args):
        return subprocess.run(['git', '-c', 'user.name=Synthetic Fixture', '-c', 'user.email=synthetic@example.invalid',
            '-c', 'commit.gpgsign=false', *args], cwd=self.repo, check=True, capture_output=True).stdout.decode().strip()

    def write(self, relative, value):
        p = self.repo / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, sort_keys=True) + '\n')

    def preserve(self):
        self.git('add', '.')
        self.git('commit', '-q', '--allow-empty', '-m', 'Disposable synthetic scorer boundary fixture; not qualification')
        self.commit = self.git('rev-parse', 'HEAD')

    def assess(self, **overrides):
        args = dict(repo_root=self.repo, material_paths=self.paths, key_review_path=self.key_path,
            firstpass_paths=self.firstpasses, preservation_commit=self.commit, evidence_review_path=self.evidence_path)
        args.update(overrides)
        return self.module.assess(**args)

    def cli(self, output):
        return subprocess.run([sys.executable, str(self.repo / self.paths['scorer']), '--repo-root', str(self.repo),
            '--materials', str(self.repo / 'materials.json'), '--key-review', self.key_path,
            '--firstpass', self.firstpasses[0], '--firstpass', self.firstpasses[1],
            '--preservation-commit', self.commit, '--evidence-review', self.evidence_path, '--output', str(output)],
            cwd=self.repo, capture_output=True, text=True)

    def test_actual_git_show_all_material_and_review_bindings(self):
        result = self.assess()
        self.assertEqual(result['status'], 'factual_qualification_passed')
        self.assertEqual(len(result['material_sha256']), 7)
        self.assertFalse(result['actual_target_reporting_authorized_by_this_receipt_alone'])
        self.assertEqual(self.module.committed_hash(self.repo, self.commit, self.evidence_path), scorer.digest(self.repo / self.evidence_path))

    def test_all_eleven_bound_files_mutated_fail_before_any_grading(self):
        bound = [*self.paths.values(), self.key_path, self.evidence_path, *self.firstpasses]
        for relative in bound:
            with self.subTest(relative=relative):
                p = self.repo / relative
                old = p.read_bytes()
                p.write_bytes(old + b' ')
                with patch.object(self.module, 'grade_firstpass', side_effect=AssertionError('must not grade changed bytes')):
                    with self.assertRaises(ValueError):
                        self.assess()
                p.write_bytes(old)

    def test_missing_semantic_review_in_commit_fails_before_grading(self):
        self.git('rm', '--cached', self.evidence_path)
        self.git('commit', '-q', '-m', 'Synthetic missing-review boundary')
        self.commit = self.git('rev-parse', 'HEAD')
        with patch.object(self.module, 'grade_firstpass', side_effect=AssertionError('must not grade unpreserved review')):
            with self.assertRaises(subprocess.CalledProcessError):
                self.assess()

    def test_synthetic_and_exposed_submissions_cannot_qualify(self):
        value = scorer.load(self.repo / self.firstpasses[0])
        value['metadata']['response_kind'] = 'synthetic_code_test'
        self.write(self.firstpasses[0], value)
        self.preserve()
        with self.assertRaisesRegex(ValueError, 'synthetic code tests cannot qualify'):
            self.assess()
        value['metadata']['response_kind'] = 'fresh_qualification_firstpass'
        value['metadata']['key_exposure_before_firstpass'] = True
        self.write(self.firstpasses[0], value)
        review = scorer.load(self.repo / self.evidence_path)
        review['reviewed_firstpasses'][0]['sha256'] = scorer.digest(self.repo / self.firstpasses[0])
        self.write(self.evidence_path, review)
        self.preserve()
        with self.assertRaisesRegex(ValueError, 'fresh independent'):
            self.assess()

    def test_role_coverage_and_order_guards_before_grading(self):
        edits = [(self.key_path, 'reviewer_id', 'emulated_synthetic_auditor_0'),
            (self.evidence_path, 'reviewer_id', 'emulated_synthetic_auditor_0'),
            (self.evidence_path, 'checked_claim_ids', scorer.CLAIM_IDS[:-1]),
            (self.evidence_path, 'reviewed_firstpasses', []), (self.key_path, 'approved_material_sha256', {}),
            (self.key_path, 'substantive_issues', ['synthetic blocking issue'])]
        for relative,field,value in edits:
            with self.subTest(relative=relative, field=field):
                p = self.repo / relative
                old = p.read_bytes()
                obj = json.loads(old)
                obj[field] = value
                self.write(relative, obj)
                self.preserve()
                with patch.object(self.module, 'grade_firstpass', side_effect=AssertionError('must not grade invalid review')):
                    with self.assertRaises(ValueError):
                        self.assess()
                p.write_bytes(old)
                self.preserve()

    def test_semantically_invalid_resolving_pointer_fails_separate_review(self):
        obj = scorer.load(self.repo / self.firstpasses[0])
        obj['answers'][0]['evidence_references'] = ['/packet_id']
        obj['answers'][0]['basis'] = 'Packet ID proves an orientation logit.'
        self.write(self.firstpasses[0], obj)
        review = scorer.load(self.repo / self.evidence_path)
        review['reviewed_firstpasses'][0]['sha256'] = scorer.digest(self.repo / self.firstpasses[0])
        review['all_96_claim_bases_relevant_and_valid'] = False
        review['issues'] = ['synthetic irrelevant-evidence failure']
        self.write(self.evidence_path, review)
        self.preserve()
        result = self.assess()
        self.assertEqual(result['status'], 'factual_qualification_failed')
        self.assertFalse(result['evidence_review']['passed'])

    def test_path_escape_symlink_role_alias_and_mutable_commit_rejected(self):
        for value in ('HEAD', 'a' * 39):
            with self.assertRaises(ValueError):
                self.assess(preservation_commit=value)
        bad = {**self.paths, 'sources': '../escaped.json'}
        with self.assertRaises(ValueError):
            self.assess(material_paths=bad)
        bad = {**self.paths, 'builder': self.paths['scorer']}
        with self.assertRaises(ValueError):
            self.assess(material_paths=bad)
        (self.repo / 'escape').symlink_to(self.repo.parent)
        with self.assertRaisesRegex(ValueError, 'escapes checkout'):
            self.module.repository_path(self.repo, 'escape/anything')

    def test_cli_valid_once_no_overwrite_and_invalid_input_leaves_no_output(self):
        output = self.repo / 'temporary_synthetic_assessment.json'
        first = self.cli(output)
        self.assertEqual(first.returncode, 0, first.stderr)
        saved = output.read_bytes()
        second = self.cli(output)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn('no overwrite/regrading', second.stderr)
        self.assertEqual(output.read_bytes(), saved)
        p = self.repo / self.evidence_path
        p.write_bytes(p.read_bytes() + b' ')
        failed_output = self.repo / 'must_not_exist.json'
        failed = self.cli(failed_output)
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse(failed_output.exists())


if __name__ == '__main__':
    # Avoid collecting the imported base class itself as a second v1 suite.
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentSourceTests))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(RealGitPreservationTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
