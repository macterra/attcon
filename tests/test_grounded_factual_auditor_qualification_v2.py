"""Synthetic code probes only. No Git, network, live raters or assessment receipts.

The assessment-path fixtures emulate fresh metadata solely to test the real gate
under a mocked preserved-byte store. They are temporary generated key copies and
are never live qualification firstpasses. Only git-show's boundary is mocked.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import assess_grounded_factual_auditor_qualification_v2 as scorer

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT/'audits/grounded_factual_auditor_qualification_v2/draft'
V1 = ROOT/'audits/grounded_factual_auditor_qualification_v1/draft'
COMMIT = 'a'*40


def fixture(cases, key, auditor_id, *, emulate_fresh=False):
    texts = {c['claim_id']:c['text'] for c in cases['claims']}
    return {'schema':'grounded_factual_auditor_qualification_firstpass_v2',
        'metadata': {'auditor_id':auditor_id,'separate_context':emulate_fresh,
            'source_access':'qualification_sources_only' if emulate_fresh else 'generated_code_test_materials',
            'key_exposure_before_firstpass':not emulate_fresh,'peer_response_exposure_before_firstpass':False,
            'actual_target_exposure':False,'model_identity':'generated_code_test_no_model_called',
            'created_at':'synthetic_test_timestamp',
            'response_kind':'fresh_qualification_firstpass' if emulate_fresh else 'synthetic_code_test'},
        'answers':[{**copy.deepcopy(a),'claim_text':texts[a['claim_id']]} for a in key['answers']],
        'source_inventories':copy.deepcopy(key['source_inventories'])}


class GateTests(unittest.TestCase):
    def setUp(self):
        self.sources = scorer.load(DRAFT/'sources.json')
        self.cases = scorer.load(DRAFT/'cases.json')
        self.key = scorer.load(DRAFT/'author_proposed_key.json')
        self.submission = fixture(self.cases,self.key,'generated_code_probe')

    def grade(self):
        return scorer.grade_firstpass(self.sources,self.cases,self.key,self.submission)

    def test_only_two_claims_and_one_content_label_change_and_sources_identical(self):
        self.assertEqual((DRAFT/'sources.json').read_bytes(),(V1/'sources.json').read_bytes())
        old_cases, old_key = scorer.load(V1/'cases.json'), scorer.load(V1/'author_proposed_key.json')
        changed_claims = [b['claim_id'] for a,b in zip(old_cases['claims'],self.cases['claims']) if a != b]
        self.assertEqual(changed_claims,['Q07-C4','Q10-C3'])
        self.assertEqual(old_cases['source_inventory_queries'],self.cases['source_inventory_queries'])
        self.assertEqual(old_key['source_inventories'],self.key['source_inventories'])
        differences = [(b['claim_id'],f) for a,b in zip(old_key['answers'],self.key['answers'])
                       for f in a if a[f] != b[f]]
        self.assertEqual(differences,[('Q07-C4','basis'),('Q10-C3','basis'),('Q10-C3','content_class')])
        proposed = {r['claim_id']:r for r in self.key['answers']}
        self.assertEqual((proposed['Q07-C4']['supportedness'],proposed['Q07-C4']['content_class'],
                          proposed['Q07-C4']['temporal_scope']),('not_auditable','cross_level','current_snapshot'))
        self.assertEqual((proposed['Q10-C3']['supportedness'],proposed['Q10-C3']['content_class'],
                          proposed['Q10-C3']['temporal_scope']),('contradicted','absence_scope','record_scope'))

    def test_matching_synthetic_probe_counts_are_not_live_qualification(self):
        result = self.grade()
        self.assertTrue(result['passed'])
        self.assertEqual(result['response_kind'],'synthetic_code_test')
        for field in ('supportedness_exact','content_class_exact','temporal_scope_exact','claim_text_exact','evidence_pointer_valid_claims'):
            self.assertEqual(result[field],48)
        self.assertEqual(result['source_inventory_exact'],12)

    def test_every_label_and_text_cell_is_required(self):
        for field, changed in [('supportedness','contradicted'),('content_class','task_risk'),
                               ('temporal_scope','predicted_conditional'),('claim_text','rewritten assertion')]:
            with self.subTest(field=field):
                original = self.submission['answers'][0][field]
                self.submission['answers'][0][field] = changed
                result = self.grade()
                self.assertFalse(result['passed'])
                self.assertEqual(result[field+'_exact'],47)
                self.submission['answers'][0][field] = original

    def test_all_252_inventory_cells_and_null_versus_missing_are_required(self):
        self.assertEqual(sum(len(r['unit_status']) for r in self.submission['source_inventories']),252)
        units = self.submission['source_inventories'][2]['unit_status']
        self.assertEqual(units['memory.evidence_item'],'explicitly_undefined')
        units['memory.evidence_item'] = 'not_supplied'
        self.assertFalse(self.grade()['passed'])
        del units['memory.status']
        with self.assertRaisesRegex(ValueError,'inventory unit status fields differ'):
            self.grade()

    def test_invalid_empty_duplicate_evidence_and_blank_basis_are_rejected(self):
        row = self.submission['answers'][0]
        for references, message in [([], 'nonempty evidence'),(['/source_views/main/world_truth'],'does not resolve'),
                                    (['/packet_id','/packet_id'],'unique nonempty evidence'),(['/bad~2escape'],'invalid JSON-pointer escape')]:
            with self.subTest(references=references):
                row['evidence_references'] = references
                with self.assertRaisesRegex(ValueError,message):
                    self.grade()
        row['evidence_references'] = ['/packet_id']
        row['basis'] = ' '
        with self.assertRaisesRegex(ValueError,'evidence basis required'):
            self.grade()

    def test_exposures_and_nonindependent_live_metadata_are_rejected(self):
        self.submission = fixture(self.cases,self.key,'synthetic_live_metadata_probe',emulate_fresh=True)
        for field in ('key_exposure_before_firstpass','peer_response_exposure_before_firstpass','actual_target_exposure','separate_context'):
            with self.subTest(field=field):
                original = self.submission['metadata'][field]
                self.submission['metadata'][field] = not original
                with self.assertRaisesRegex(ValueError,'fresh independent'):
                    self.grade()
                self.submission['metadata'][field] = original

    def test_missing_duplicate_unknown_cells_and_changed_query_are_rejected(self):
        for change, message in [
            (lambda s:s['answers'][0].pop('content_class'),'answer fields differ'),
            (lambda s:s['answers'].__setitem__(1,copy.deepcopy(s['answers'][0])),'IDs differ'),
            (lambda s:s['answers'][0].__setitem__('content_class','unknown'),'unknown rating label'),
            (lambda s:s['source_inventories'][0].__setitem__('indexed_location','l1'),'source query changed'),
            (lambda s:s.__setitem__('supportedness_exact',48),'submission fields differ')]:
            with self.subTest(message=message):
                self.submission = fixture(self.cases,self.key,'generated_code_probe')
                change(self.submission)
                with self.assertRaisesRegex(ValueError,message):
                    self.grade()


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='grounded-v2-code-test-')
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.paths = scorer.load(DRAFT/'material_paths.json')
        for relative in self.paths.values():
            target = self.repo/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/relative).read_bytes())
        spec = importlib.util.spec_from_file_location('temporary_v2_scorer_code_test',self.repo/self.paths['scorer'])
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        cases, key = scorer.load(DRAFT/'cases.json'), scorer.load(DRAFT/'author_proposed_key.json')
        self.firstpasses = ['code_test_only/a.json','code_test_only/b.json']
        for n,path in enumerate(self.firstpasses):
            self.write(path,fixture(cases,key,'synthetic_boundary_probe_'+str(n),emulate_fresh=True))
        self.key_path, self.evidence_path = 'code_test_only/key_review.json', 'code_test_only/evidence_review.json'
        self.write(self.key_path,{'schema':'grounded_factual_auditor_qualification_key_review_v2',
            'status':'independently_key_reviewed','reviewer_id':'synthetic_key_review','separate_context':True,
            'approved_material_sha256':{r:{'path':p,'sha256':scorer.digest(self.repo/p)} for r,p in self.paths.items()},
            'checked_claim_ids':scorer.CLAIM_IDS,'checked_packet_ids':scorer.PACKET_IDS,'substantive_issues':[]})
        self.write(self.evidence_path,{'schema':'grounded_factual_auditor_qualification_evidence_review_v2',
            'reviewer_id':'synthetic_semantic_review','separate_context':True,'checked_claim_ids':scorer.CLAIM_IDS,
            'reviewed_firstpasses':[{'path':p,'sha256':scorer.digest(self.repo/p)} for p in self.firstpasses],
            'all_96_claim_bases_relevant_and_valid':True,'issues':[]})
        self.preserve()

    def write(self,path,value):
        target = self.repo/path
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(value,sort_keys=True)+'\n')

    def preserve(self):
        self.preserved = {p:scorer.digest(self.repo/p) for p in [*self.paths.values(),self.key_path,self.evidence_path,*self.firstpasses]}

    def assess(self, **overrides):
        args = dict(repo_root=self.repo,material_paths=self.paths,key_review_path=self.key_path,
                    firstpass_paths=self.firstpasses,preservation_commit=COMMIT,evidence_review_path=self.evidence_path)
        args.update(overrides)
        def preserved_hash(repo,commit,path):
            self.assertEqual(repo,self.repo)
            self.assertEqual(commit,COMMIT)
            return self.preserved.get(path,'not_committed')
        with patch.object(self.module,'committed_hash',side_effect=preserved_hash):
            return self.module.assess(**args)

    def test_complete_preserved_code_probe_and_changed_material_reviews_firstpasses_rejected_before_grading(self):
        result = self.assess()
        self.assertEqual(result['status'],'factual_qualification_passed')
        self.assertFalse(result['actual_target_reporting_authorized_by_this_receipt_alone'])
        for path in [*self.paths.values(),self.key_path,self.evidence_path,*self.firstpasses]:
            with self.subTest(path=path):
                target = self.repo/path
                original = target.read_bytes()
                target.write_bytes(original+b' ')
                with patch.object(self.module,'grade_firstpass',side_effect=AssertionError('must not score changed bytes')):
                    with self.assertRaises(ValueError):
                        self.assess()
                target.write_bytes(original)

    def test_unpreserved_semantic_review_is_rejected_before_any_grading(self):
        self.preserved.pop(self.evidence_path)
        with patch.object(self.module,'grade_firstpass',side_effect=AssertionError('must not score unpreserved review')):
            with self.assertRaisesRegex(ValueError,'evidence review must be preserved'):
                self.assess()

    def test_resolving_but_irrelevant_evidence_semantic_review_fails_gate(self):
        value = scorer.load(self.repo/self.firstpasses[0])
        value['answers'][0]['evidence_references'] = ['/packet_id']
        value['answers'][0]['basis'] = 'Packet identifier proves the stored logit and decision.'
        self.write(self.firstpasses[0],value)
        review = scorer.load(self.repo/self.evidence_path)
        review['reviewed_firstpasses'][0]['sha256'] = scorer.digest(self.repo/self.firstpasses[0])
        review['all_96_claim_bases_relevant_and_valid'] = False
        review['issues'] = [{'auditor_id':value['metadata']['auditor_id'],'claim_id':'Q01-C1','reason':'irrelevant evidence'}]
        self.write(self.evidence_path,review)
        self.preserve()
        result = self.assess()
        self.assertTrue(all(r['passed'] for r in result['auditor_results']))
        self.assertEqual(result['status'],'factual_qualification_failed')
        self.assertFalse(result['evidence_review']['passed'])

    def test_synthetic_submissions_cannot_qualify_and_fresh_exposure_is_rejected(self):
        value = scorer.load(self.repo/self.firstpasses[0])
        value['metadata']['response_kind'] = 'synthetic_code_test'
        self.write(self.firstpasses[0],value)
        self.preserve()
        with self.assertRaisesRegex(ValueError,'synthetic code tests cannot qualify'):
            self.assess()

    def test_review_hash_coverage_identity_and_full_commit_boundaries(self):
        for path, field, changed, message in [
            (self.key_path,'approved_material_sha256',{},'approve exact material'),
            (self.key_path,'checked_claim_ids',scorer.CLAIM_IDS[:-1],'complete key review'),
            (self.key_path,'reviewer_id','synthetic_boundary_probe_0','key reviewer may not qualify'),
            (self.evidence_path,'checked_claim_ids',scorer.CLAIM_IDS[:-1],'complete semantic evidence review'),
            (self.evidence_path,'reviewed_firstpasses',[],'firstpass hashes differ'),
            (self.evidence_path,'separate_context',False,'separate evidence reviewer'),
            (self.evidence_path,'reviewer_id','synthetic_boundary_probe_0','separate evidence reviewer')]:
            with self.subTest(field=field,path=path):
                original = (self.repo/path).read_bytes()
                value = scorer.load(self.repo/path)
                value[field] = changed
                self.write(path,value)
                self.preserve()
                with self.assertRaisesRegex(ValueError,message):
                    self.assess()
                (self.repo/path).write_bytes(original)
                self.preserve()
        with self.assertRaisesRegex(ValueError,'immutable preservation commit'):
            self.assess(preservation_commit='HEAD')
        wrong_paths = {**self.paths,'scorer':self.paths['builder'],'builder':self.paths['scorer']}
        with self.assertRaisesRegex(ValueError,'executing scorer'):
            self.assess(material_paths=wrong_paths)
        wrong_paths = {**self.paths,'sources':'../escaped.json'}
        with self.assertRaisesRegex(ValueError,'relative repository path'):
            self.assess(material_paths=wrong_paths)


if __name__ == '__main__':
    unittest.main()
