"""Independent synthetic key/scorer review. No archives, ratings or live assessment.

Key-copy probes below are in-memory software tests, not qualifying firstpasses.
Hash contract tests use a mock preserved-byte reader, not a Git preservation claim.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import assess_grounded_factual_auditor_qualification_v1 as scorer
import grounded_report_interface_v1 as interface

DRAFT = ROOT / 'audits/grounded_factual_auditor_qualification_v1/draft'


def read(name):
    return json.loads((DRAFT / name).read_text())


class IndependentReview(unittest.TestCase):
    def setUp(self):
        self.sources, self.cases, self.key = [read(n) for n in ('sources.json', 'cases.json', 'author_proposed_key.json')]
        texts = {c['claim_id']: c['text'] for c in self.cases['claims']}
        self.submission = {'schema': 'grounded_factual_auditor_qualification_firstpass_v1',
            'metadata': {'auditor_id': 'synthetic_probe_A_not_auditor', 'separate_context': True,
                'source_access': 'qualification_sources_only', 'key_exposure_before_firstpass': False,
                'peer_response_exposure_before_firstpass': False, 'actual_target_exposure': False,
                'model_identity': 'no_model_called_fixture', 'created_at': 'synthetic'},
            'answers': [{**copy.deepcopy(a), 'claim_text': texts[a['claim_id']]} for a in self.key['answers']],
            'source_inventories': copy.deepcopy(self.key['source_inventories'])}

    def grade(self, submission=None):
        return scorer.grade_firstpass(self.sources, self.cases, self.key, submission or self.submission)

    def test_all_252_inventory_cells_independently_derived(self):
        by_id = {p['packet_id']: p for p in self.sources['packets']}
        key_inventory = {i['packet_id']: i for i in self.key['source_inventories']}
        for n, query in enumerate(self.cases['source_inventory_queries']):
            self.assertEqual(query, {'packet_id': f'Q{n+1:02d}', 'query_view': 'sensory_only' if n == 10 else 'main',
                'indexed_location': f'l{n%4}', 'indexed_command': f'k{n//4}'})
            view = by_id[query['packet_id']]['source_views'][query['query_view']]
            node = next(v for v in view['locations'] if v['location'] == query['indexed_location'])
            evidence = node['stored_evidence']
            expected = {}
            for unit in ('status', 'evidence_item', 'raw_orientation_logit', 'logit_sign'):
                expected['memory.' + unit] = 'explicitly_undefined' if evidence[unit] is None else 'present'
            expected.update({'memory.stored_decision_or_default': 'present',
                'history.latest_observed_delivery_or_unknown': 'present' if 'native_route_history' in view else 'not_supplied',
                'semantics.stored_decision_vs_default': 'present'})
            for address in ('l0', 'l1', 'l2', 'l3'):
                for field in ('delivery', 'after'):
                    expected[field + '.' + address] = 'present' if 'command_forecasts' in view else 'not_supplied'
            for unit, field in [('risk.current', 'estimated_current_decision_success'), ('risk.refresh', 'estimated_fresh_glimpse_decision_success')]:
                expected[unit] = 'present' if field in node else 'not_supplied'
            external = node.get('external_image_statistic_readouts')
            for unit, field in [('external.image_value_std', 'image_value_std_estimate'), ('external.nearest_neighbor_rms', 'nearest_neighbor_rms_estimate')]:
                expected[unit] = 'not_supplied' if external is None else 'explicitly_undefined' if external[field] is None else 'present'
            expected['semantics.forecasts_not_observed_future'] = 'present' if 'command_forecasts' in view or 'estimated_current_decision_success' in node else 'not_supplied'
            expected['semantics.external_objective_source_and_normalized_units'] = 'present' if external is not None else 'not_supplied'
            self.assertEqual(len(expected), 21)
            self.assertEqual(key_inventory[query['packet_id']]['unit_status'], expected)

    def test_generated_source_schema_tables_arithmetic_and_pointers(self):
        packets = {p['packet_id']: p for p in self.sources['packets']}
        claims = {c['claim_id']: c for c in self.cases['claims']}
        for packet in packets.values():
            self.assertIs(packet['synthetic_fixture'], True)
            self.assertIs(packet['audit_provenance']['generated_not_study_source'], True)
            for name, view in packet['source_views'].items():
                if name not in ('sensory_only', 'no_external', 'identifier_relations', 'tables'):
                    interface.validate_reporter_input(view)
            for name, label in packet['visibility'].items():
                if label == 'same_information_address_aware_tables':
                    self.assertEqual(interface.join_address_aware_tables(packet['source_views'][name]), packet['source_views']['main'])
        for row in self.key['answers']:
            for ref in row['evidence_references']:
                scorer.pointer(packets[claims[row['claim_id']]['packet_id']], ref)
        self.assertAlmostEqual(packets['Q04']['source_views']['main']['command_forecasts'][0]['estimated_decision_success_after_command']['l0'], 0.809824)
        self.assertAlmostEqual(packets['Q08']['source_views']['main']['command_forecasts'][0]['estimated_decision_success_after_command']['l2'], 0.684)

    def test_key_q04_c2_explicit_comparison_requires_mixed_time(self):
        self.assertEqual(self.key['answers'][13]['temporal_scope'], 'mixed_time')

    def test_key_q04_c3_equating_forecast_current_requires_mixed_time(self):
        self.assertEqual(self.key['answers'][14]['temporal_scope'], 'mixed_time')

    def test_key_q06_c3_native_vs_actual_chronology_requires_cross_level(self):
        self.assertEqual(self.key['answers'][22]['content_class'], 'cross_level')

    def test_key_q06_c3_native_vs_actual_chronology_requires_mixed_time(self):
        self.assertEqual(self.key['answers'][22]['temporal_scope'], 'mixed_time')

    def test_key_q12_c4_role_conflict_requires_contradicted(self):
        self.assertEqual(self.key['answers'][47]['supportedness'], 'contradicted')

    def test_scorer_all_valid_label_cell_mutations_fail(self):
        for i in range(48):
            for field, allowed in [('supportedness', scorer.SUPPORT), ('content_class', scorer.CONTENT), ('temporal_scope', scorer.TEMPORAL)]:
                probe = copy.deepcopy(self.submission)
                probe['answers'][i][field] = sorted(allowed - {probe['answers'][i][field]})[0]
                result = self.grade(probe)
                self.assertFalse(result['passed'])
                self.assertEqual(result[field + '_exact'], 47)

    def test_scorer_all_252_inventory_mutations_fail(self):
        for i in range(12):
            for unit, status in self.submission['source_inventories'][i]['unit_status'].items():
                probe = copy.deepcopy(self.submission)
                probe['source_inventories'][i]['unit_status'][unit] = 'not_supplied' if status != 'not_supplied' else 'present'
                result = self.grade(probe)
                self.assertFalse(result['passed'])
                self.assertEqual(result['source_inventory_exact'], 11)

    def test_scorer_all_answer_fields_and_unknown_labels_rejected(self):
        for field in self.submission['answers'][0]:
            probe = copy.deepcopy(self.submission)
            del probe['answers'][0][field]
            with self.assertRaises(ValueError):
                self.grade(probe)
        for field in ('supportedness', 'content_class', 'temporal_scope'):
            probe = copy.deepcopy(self.submission)
            probe['answers'][0][field] = 'unknown'
            with self.assertRaises(ValueError):
                self.grade(probe)
        for field, value in [('basis', ''), ('evidence_references', []), ('evidence_references', ['/absent']), ('evidence_references', ['/packet_id', '/packet_id'])]:
            probe = copy.deepcopy(self.submission)
            probe['answers'][0][field] = value
            with self.assertRaises(ValueError):
                self.grade(probe)

    def test_scorer_schema_metadata_extra_fields_queries_duplicates(self):
        mutations = [lambda p: p.update(schema='unknown'), lambda p: p.update(handwritten_perfect_score=48),
            lambda p: p['answers'][0].update(unsupported_extra='injection'),
            lambda p: p['source_inventories'][0].update(unsupported_extra='injection'),
            lambda p: p['source_inventories'][0]['unit_status'].update(extra='present'),
            lambda p: p['source_inventories'][0].update(indexed_command='k1'),
            lambda p: p['answers'].__setitem__(1, copy.deepcopy(p['answers'][0])),
            lambda p: p['source_inventories'].__setitem__(1, copy.deepcopy(p['source_inventories'][0]))]
        for mutate in mutations:
            probe = copy.deepcopy(self.submission)
            mutate(probe)
            with self.assertRaises(ValueError):
                self.grade(probe)
        for field in self.submission['metadata']:
            probe = copy.deepcopy(self.submission)
            del probe['metadata'][field]
            with self.assertRaises(ValueError):
                self.grade(probe)

    def test_pointer_escape_array_and_nonresolving_guards(self):
        fixture = {'x/y': {'~value': [1]}}
        self.assertEqual(scorer.pointer(fixture, '/x~1y/~0value/0'), 1)
        for ref in ('/', '', 'x', '/x~2y', '/x~1y/~0value/00', '/x~1y/~0value/1', '/x~1y/~0value/-1'):
            with self.assertRaises(ValueError):
                scorer.pointer(fixture, ref)

    def test_pointer_relevance_is_explicitly_deferred_not_claimed(self):
        probe = copy.deepcopy(self.submission)
        probe['answers'][0]['evidence_references'] = ['/packet_id']
        probe['answers'][0]['basis'] = 'Irrelevant synthetic test basis.'
        result = self.grade(probe)
        self.assertTrue(result['passed'])
        self.assertIn('requires_separate_review', result['evidence_semantic_validity'])

    def test_hash_bound_assessment_mocks_preserved_bytes_without_live_rating(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_key_review_') as temp:
            root = Path(temp)
            materials = read('material_paths.json')
            saved = {}
            for role, relative in materials.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT / relative).read_bytes())
                saved[relative] = scorer.digest(path)
            spec = importlib.util.spec_from_file_location('synthetic_scorer_copy', root / materials['scorer'])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            review = {'schema': 'grounded_factual_auditor_qualification_key_review_v1', 'status': 'independently_key_reviewed',
                'reviewer_id': 'synthetic_reviewer_not_actual_approval', 'separate_context': True,
                'approved_material_sha256': {r: {'path': p, 'sha256': saved[p]} for r,p in materials.items()},
                'checked_claim_ids': [c['claim_id'] for c in self.cases['claims']],
                'checked_packet_ids': [p['packet_id'] for p in self.sources['packets']], 'substantive_issues': []}
            firstpasses = ['probeA.json', 'probeB.json']
            for i,path in enumerate(firstpasses):
                probe = copy.deepcopy(self.submission)
                probe['metadata']['auditor_id'] += str(i)
                (root / path).write_text(json.dumps(probe))
                saved[path] = module.digest(root / path)
            evidence = {'schema': 'grounded_factual_auditor_qualification_evidence_review_v1',
                'reviewer_id': 'synthetic_evidence_reviewer', 'reviewed_firstpasses': [{'path': p, 'sha256': saved[p]} for p in firstpasses],
                'all_96_claim_bases_relevant_and_valid': True, 'issues': []}
            def write_reviews():
                (root / 'review.json').write_text(json.dumps(review))
                (root / 'evidence.json').write_text(json.dumps(evidence))
                saved['review.json'] = module.digest(root / 'review.json')
            write_reviews()
            args = dict(repo_root=root, material_paths=materials, key_review_path='review.json', firstpass_paths=firstpasses,
                preservation_commit='a' * 40, evidence_review_path='evidence.json')
            with patch.object(module, 'committed_hash', side_effect=lambda r,c,p: saved[p]):
                self.assertEqual(module.assess(**args)['status'], 'factual_qualification_passed')
                evidence['issues'] = ['synthetic semantic failure']
                write_reviews()
                self.assertEqual(module.assess(**args)['status'], 'factual_qualification_failed')
                evidence['issues'] = []
                write_reviews()
                (root / firstpasses[0]).write_text((root / firstpasses[0]).read_text() + '\n')
                with self.assertRaisesRegex(ValueError, 'firstpass differs'):
                    module.assess(**args)
                saved[firstpasses[0]] = module.digest(root / firstpasses[0])
                evidence['reviewed_firstpasses'][0]['sha256'] = saved[firstpasses[0]]
                write_reviews()
                review['substantive_issues'] = ['synthetic blocking key issue']
                write_reviews()
                with self.assertRaisesRegex(ValueError, 'issue-free'):
                    module.assess(**args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
