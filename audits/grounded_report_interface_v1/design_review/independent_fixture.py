"""Independent post-draft adversarial checks on generated toy sources only.

No production archive loading/selection, checkpoint inference or record serialization.
First-pass failures are intentionally retained for correction review.
"""
import copy
import unittest
from unittest.mock import patch
import torch
import grounded_report_interface_v1 as interface
from test_grounded_report_interface_v1 import fixture

class IndependentEdges(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_memory_readout_donor_mismatch_must_be_rejected(self):
        source=fixture()
        source.physical_by_variant['memory_readdressed'][0,0,0]+=10
        with self.assertRaises(ValueError): interface.record(source,0,'memory_readdressed')

    def test_record_requires_entire_fixed_opportunity_set(self):
        with self.assertRaises(ValueError): interface.record(fixture(n=15),0)

    def test_duplicate_table_address_must_be_rejected(self):
        tables=interface.address_aware_tables(interface.record(fixture(),0)['reporter_input'])
        tables['tables']['risk_forecasts'].append(copy.deepcopy(tables['tables']['risk_forecasts'][0]))
        with self.assertRaises(ValueError): interface.join_address_aware_tables(tables)

    def test_extra_table_field_must_be_rejected(self):
        tables=interface.address_aware_tables(interface.record(fixture(),0)['reporter_input'])
        tables['tables']['risk_forecasts'][0]['physical_targets']='FORBIDDEN_TOY_MARKER'
        with self.assertRaises(ValueError): interface.join_address_aware_tables(tables)

    def test_neutral_task_source_legend_excludes_theory_vocabulary(self):
        value=interface.record(fixture(),0)['reporter_input']
        pieces=list(value['source_semantics'].values())
        pieces.extend(node['external_image_statistic_readouts']['interpretation'] for node in value['locations'])
        for phrase in ['felt confidence','native presentation','appearance','consciousness','qualia']:
            self.assertFalse(any(phrase in p.lower() for p in pieces),phrase)

    def test_inconsistent_latest_command_history_is_rejected(self):
        source=fixture(); source.native_history[0,0,0].zero_()
        with self.assertRaises(ValueError): interface.record(source,0)

    def test_unseen_hidden_and_logit_nonzero_are_rejected(self):
        for coord in [0,64]:
            source=fixture()
            for variant in interface.VARIANTS:
                memory=source.sensory_by_variant[variant]
                memory[...,coord][memory[...,65]==0]=.1
            source.original_sensory[...,coord][source.original_sensory[...,65]==0]=.1
            with self.assertRaises(ValueError): interface.record(source,0)

    def test_chronology_remains_audit_only_despite_memory_conflict(self):
        source=fixture()
        with patch.object(interface,'load_bundle',side_effect=AssertionError('no production loading')):
            natural=interface.record(source,0); moved=interface.record(source,0,'memory_readdressed')
        self.assertEqual(natural['reporter_input']['native_route_history'],moved['reporter_input']['native_route_history'])
        self.assertEqual(natural['audit_provenance']['external_observed_history_instrumentation'],moved['audit_provenance']['external_observed_history_instrumentation'])
        self.assertNotIn('external_observed_history_instrumentation',moved['reporter_input'])
        self.assertEqual(moved['audit_provenance']['evidence_original_location_by_current_slot'],[3,0,1,2])
        self.assertEqual(moved['audit_provenance']['evidence_acquisition_event_by_current_slot'],[None,6,7,8])
        self.assertEqual(moved['reporter_input']['locations'][1]['stored_evidence'],natural['reporter_input']['locations'][0]['stored_evidence'])

    def test_restores_and_patch_origin_match_factual_sources(self):
        source=fixture(); natural=interface.record(source,0)
        for variant in ['sham','memory_restored','delivery_restored']:
            self.assertEqual(natural['reporter_input'],interface.record(source,0,variant)['reporter_input'])
        self.assertEqual(interface.record(source,0,'delivery_patched')['audit_provenance']['forecast_origin'],'export_only_delivery_patch')
        self.assertEqual(interface.record(source,0,'memory_readdressed')['audit_provenance']['forecast_origin'],'frozen_network_inference')

if __name__=='__main__': unittest.main(verbosity=2)
