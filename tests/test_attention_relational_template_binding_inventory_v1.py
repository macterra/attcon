"""Fictional checks for lossless first-binding/source joins, never qualification."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/attention_relational_template_binding_inventory_v1.py'
SPEC = importlib.util.spec_from_file_location('binding_inventory', SOURCE)
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


def fixture():
    annotation = dict(features={}, composites=dict(COD=dict(status='explicit', bindings=[
        dict(referent='content', attribute='red', scope='model_organization', time='future_forecast')])) )
    components, evidence, witnesses = [], [], []
    for feature in ('C', 'O', 'D'):
        quote = dict(quote='fictional ' + feature, referent='content', attribute='red',
                     scope='model_organization', time='future_forecast')
        annotation['features'][feature] = dict(status='explicit', evidence=[quote])
        components.append(dict(component_path=['features', feature], original_status='explicit',
                               assessment='supported', notes='fictional first judgment'))
        path = ['features', feature, 'evidence', 0]
        witnesses.append(dict(feature=feature, evidence_paths=[path]))
        ref = (dict(unit_id='peaks', value_path=[], value=[[1, 2]], semantic_role='content_operand', reason='exact content')
               if feature == 'C' else dict(unit_id='origin', value_path=['kind'],
               value='copied_prior_record', semantic_role='status_operand', reason='status only'))
        evidence.append(dict(evidence_path=path, grounding='supported', source_meaning='source_specific',
                             properties=[ref], notes='fictional reasoning'))
    components.append(dict(component_path=['composites', 'COD'], original_status='explicit',
                           assessment='supported', notes='fictional composite reasoning'))
    path = ['composites', 'COD', 'bindings', 0]
    record = dict(id='fixture_template', unit_id='u', text_sha256='text', reference_established=True,
        case={'source':'A'}, canonical_first_annotation=annotation,
        bindings=[dict(path=path, required_features=['C','O','D'])],
        property_table=[dict(unit_id='peaks', kind='role_peaks', value=[[1,2]], context={'node':'n3'},
                            source_resolution={'panel_source_key':'A','origin_class':{'origin_source_key':'A'}}),
                        dict(unit_id='origin',kind='origin',value={'kind':'copied_prior_record','record_time':11},
                            context={'node':'n0'}, source_resolution={'panel_source_key':'A','origin_class':{'origin_source_key':'W'}})])
    judgment = dict(component_reviews=components, evidence_reviews=evidence,
        binding_reviews=[dict(binding_path=path, feature_witnesses=witnesses,
                             binding_evidence_paths=[], assessment='supported', notes='fictional compatible binding')])
    table = {'peak_identity':dict(construction_unit_id='peaks',identity={'value_path':[], 'actual_source':'A'}, value=[[1,2]]),
             'origin_identity':dict(construction_unit_id='origin',identity={'value_path':['kind'], 'actual_source':'W'}, value='copied_prior_record')}
    return record, judgment, table


class FirstBindingJoin(unittest.TestCase):
    def test_supported_graph_retains_scope_and_source_without_semantic_verdict(self):
        record, judgment, table = fixture()
        before = deepcopy((record, judgment, table))
        result = inventory.binding_inventory(record, judgment, table)[0]
        self.assertTrue(result['all_recorded_support_facts_true'])
        self.assertEqual(result['original_scope'], 'model_organization')
        self.assertEqual(result['original_time'], 'future_forecast')
        self.assertFalse(result['source_attributed_endpoint_established_by_code'])
        operands = [o for e in result['selected_evidence'] for o in e['operands']]
        self.assertEqual([o['atomic_identity']['actual_source'] for o in operands], ['A','W','W'])
        self.assertEqual((record, judgment, table), before)

    def test_grounding_failure_cannot_be_hidden_by_supported_binding(self):
        record, judgment, table = fixture()
        judgment['evidence_reviews'][1]['grounding'] = 'unresolved'
        result = inventory.binding_inventory(record, judgment, table)[0]
        self.assertEqual(result['first_binding_review']['assessment'], 'supported')
        self.assertFalse(result['recorded_support_facts']['selected_evidence_grounding_supported'])
        self.assertFalse(result['all_recorded_support_facts_true'])

    def test_supported_absence_and_ambiguity_not_upgraded_to_explicit(self):
        record, judgment, table = fixture()
        record['canonical_first_annotation']['composites']['COD']['status'] = 'ambiguous'
        judgment['component_reviews'][-1]['original_status'] = 'ambiguous'
        result = inventory.binding_inventory(record, judgment, table)[0]
        self.assertEqual(result['original_composite_status'], 'ambiguous')
        judgment['component_reviews'][0]['original_status'] = 'absent'
        self.assertFalse(inventory.binding_inventory(record, judgment, table)[0]['all_recorded_support_facts_true'])

    def test_container_is_retained_without_atomic_leaf_expansion(self):
        record, judgment, table = fixture()
        ref = judgment['evidence_reviews'][1]['properties'][0]
        ref['value_path'] = []
        ref['value'] = deepcopy(record['property_table'][1]['value'])
        result = inventory.binding_inventory(record, judgment, table)[0]
        operand = result['selected_evidence'][1]['operands'][0]
        self.assertIsNone(operand['atomic_identity'])
        self.assertEqual(operand['atomic_mapping_state'], 'no_exact_atomic_identity')
        self.assertFalse(operand['missing_atomic_mapping_is_feature_absence'])
        self.assertEqual(operand['first_review_property']['value'], ref['value'])

    def test_missing_required_feature_prevents_vacuous_support(self):
        record, judgment, table = fixture()
        judgment['binding_reviews'][0]['feature_witnesses'] = []
        with self.assertRaisesRegex(ValueError, 'required features'):
            inventory.binding_inventory(record, judgment, table)

    def test_changed_property_bytes_are_rejected(self):
        record, judgment, table = fixture()
        judgment['evidence_reviews'][0]['properties'][0]['value'] = [[2,1]]
        with self.assertRaisesRegex(ValueError, 'exact construction'):
            inventory.binding_inventory(record, judgment, table)
        record, judgment, table = fixture()
        table['peak_identity']['value'] = [[2,1]]
        with self.assertRaisesRegex(ValueError, 'bank property'):
            inventory.binding_inventory(record, judgment, table)

    def test_missing_first_row_is_unknown_and_never_salvaged(self):
        record, judgment, table = fixture()
        rows = [dict(reviewer_id=s, id=record['id'], unit_id='u', text_sha256='text', reference_established=True,
                     independent_context_id=s, chunk_number=0, chunk_structurally_valid=s=='reviewer_0',
                     first_judgment=judgment if s=='reviewer_0' else None, review_state='retained' if s=='reviewer_0' else 'unestablished')
                for s in inventory.STREAMS]
        result = inventory.join_positions([record], rows, {'fixture':table})[0]
        self.assertIsNone(result['first_streams']['reviewer_1']['bindings'])
        self.assertEqual(result['both_first_streams_recorded_support_candidates'], [])
        rows[1]['first_judgment'] = judgment
        with self.assertRaisesRegex(ValueError, 'salvage'):
            inventory.join_positions([record], rows, {'fixture':table})

    def test_different_compatible_evidence_paths_need_not_match(self):
        record, judgment, table = fixture()
        second = deepcopy(judgment)
        record['canonical_first_annotation']['features']['O']['evidence'].append(
            deepcopy(record['canonical_first_annotation']['features']['O']['evidence'][0]))
        alternative = deepcopy(second['evidence_reviews'][1])
        alternative['evidence_path'][-1] = 1
        second['evidence_reviews'].append(alternative)
        second['binding_reviews'][0]['feature_witnesses'][1]['evidence_paths'] = [alternative['evidence_path']]
        rows = [dict(reviewer_id=s, id=record['id'], unit_id='u', text_sha256='text', reference_established=True,
                     independent_context_id=s, chunk_number=0, chunk_structurally_valid=True,
                     first_judgment=j, review_state='retained')
                for s,j in zip(inventory.STREAMS, [judgment, second])]
        result = inventory.join_positions([record], rows, {'fixture':table})[0]
        self.assertEqual(len(result['both_first_streams_recorded_support_candidates']), 1)
        self.assertIsNone(result['semantic_or_qualia_verdict'])

    def test_numerical_null_and_restoration_cannot_supply_semantic_measurement(self):
        original = dict(first_id='before', second_id='restored', kind='current_donor_to_restored')
        numerical = dict(original_pair=deepcopy(original), counts={'matched':1,'unmatched':2,'value_changed':0},
                         by_actual_source={'A':{'peak_null_with_raw_distribution_change':1}}, literal_text_equal=True)
        positions = {name+'_template':dict(both_first_streams_recorded_support_candidates=[['composites','COD','bindings',0]])
                     for name in ('before','restored')}
        result = inventory.join_pairs(positions, [original], [numerical])[0]
        self.assertEqual(result['numerical_property_counts']['unmatched'], 2)
        self.assertIsNone(result['semantic_true_null'])
        self.assertIsNone(result['semantic_predicate_tracking'])
        self.assertIsNone(result['semantic_exact_restoration'])

    def test_pair_inventory_cannot_be_reordered_or_dropped(self):
        original = dict(first_id='before', second_id='after', kind='current_donor')
        numerical = dict(original_pair=dict(original, kind='W_changed'))
        with self.assertRaisesRegex(ValueError, 'pair inventory differs'):
            inventory.join_pairs({}, [original], [numerical])
        with self.assertRaises(ValueError):
            inventory.join_pairs({}, [original], [])


if __name__ == '__main__':
    unittest.main()
