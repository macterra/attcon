"""Generated in-memory scorer probes; no live auditor output or assessment file."""
import copy
import json
from pathlib import Path
import unittest

import assess_grounded_factual_auditor_qualification_v1 as scorer

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT/'audits/grounded_factual_auditor_qualification_v1/draft'


class GeneratedScorerTests(unittest.TestCase):
    def setUp(self):
        self.sources = json.loads((DRAFT/'sources.json').read_text())
        self.cases = json.loads((DRAFT/'cases.json').read_text())
        self.key = json.loads((DRAFT/'author_proposed_key.json').read_text())
        texts = {c['claim_id']:c['text'] for c in self.cases['claims']}
        # Key-copying is permitted only for this code probe, never qualification.
        self.submission = {'schema':'grounded_factual_auditor_qualification_firstpass_v1',
            'metadata':{'auditor_id':'generated_code_probe_not_real_auditor','separate_context':True,
                        'source_access':'qualification_sources_only','key_exposure_before_firstpass':False,
                        'peer_response_exposure_before_firstpass':False,'actual_target_exposure':False,
                        'model_identity':'no_model_called_generated_in_memory','created_at':'synthetic_test_timestamp'},
            'answers':[{**copy.deepcopy(a),'claim_text':texts[a['claim_id']]} for a in self.key['answers']],
            'source_inventories':copy.deepcopy(self.key['source_inventories'])}

    def grade(self):
        return scorer.grade_firstpass(self.sources,self.cases,self.key,self.submission)

    def test_generated_matching_probe_and_all_cell_counts(self):
        result = self.grade()
        self.assertTrue(result['passed'])
        self.assertEqual(result['supportedness_exact'],48)
        self.assertEqual(result['content_class_exact'],48)
        self.assertEqual(result['temporal_scope_exact'],48)
        self.assertEqual(result['source_inventory_exact'],12)

    def test_unsupported_false_positive_fails_exact_gate(self):
        self.submission['answers'][3]['supportedness'] = 'supported'
        result = self.grade()
        self.assertFalse(result['passed'])
        self.assertEqual(result['supportedness_exact'],47)

    def test_substantive_source_class_and_time_fail_separately(self):
        self.submission['answers'][0]['content_class'] = 'task_risk'
        self.submission['answers'][0]['temporal_scope'] = 'predicted_conditional'
        result = self.grade()
        self.assertFalse(result['passed'])
        self.assertEqual(result['supportedness_exact'],48)
        self.assertEqual(result['content_class_exact'],47)
        self.assertEqual(result['temporal_scope_exact'],47)

    def test_null_versus_missing_inventory_cell_fails(self):
        unseen = self.submission['source_inventories'][2]['unit_status']
        self.assertEqual(unseen['memory.evidence_item'],'explicitly_undefined')
        unseen['memory.evidence_item'] = 'not_supplied'
        result = self.grade()
        self.assertFalse(result['passed'])
        self.assertEqual(result['source_inventory_exact'],11)

    def test_invalid_pointer_rejected(self):
        self.submission['answers'][0]['evidence_references'] = ['/source_views/main/world_truth']
        with self.assertRaisesRegex(ValueError,'pointer does not resolve'):
            self.grade()

    def test_missing_field_and_duplicate_claim_rejected(self):
        self.submission['answers'][0].pop('content_class')
        with self.assertRaisesRegex(ValueError,'answer fields differ'):
            self.grade()
        self.setUp()
        self.submission['answers'][1] = copy.deepcopy(self.submission['answers'][0])
        with self.assertRaisesRegex(ValueError,'IDs differ'):
            self.grade()

    def test_exposure_or_extra_handtyped_counts_rejected(self):
        self.submission['metadata']['key_exposure_before_firstpass'] = True
        with self.assertRaisesRegex(ValueError,'fresh independent'):
            self.grade()
        self.setUp()
        self.submission['supportedness_exact'] = 48
        with self.assertRaisesRegex(ValueError,'submission fields differ'):
            self.grade()

    def test_rewritten_claim_and_changed_query_rejected(self):
        self.submission['answers'][0]['claim_text'] += ' invented qualification wording'
        result = self.grade()
        self.assertFalse(result['passed'])
        self.assertEqual(result['claim_text_exact'],47)
        self.setUp()
        self.submission['source_inventories'][0]['indexed_location'] = 'l1'
        with self.assertRaisesRegex(ValueError,'source query changed'):
            self.grade()

    def test_mutable_commit_reference_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError,'immutable preservation commit'):
            scorer.assess(repo_root=ROOT,material_paths={},key_review_path='absent',firstpass_paths=[],
                          preservation_commit='HEAD',evidence_review_path='absent')


if __name__ == '__main__':
    unittest.main()
