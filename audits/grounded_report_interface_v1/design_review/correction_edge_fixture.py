"""Additional independent correction checks; generated sources only.

No real archive loading, selected production rows, checkpoint inference or export.
Original first-pass fixture/log/source snapshots remain untouched.
"""
import copy
import unittest
import torch
import grounded_report_interface_v1 as interface
from test_grounded_report_interface_v1 import fixture

class CorrectionEdges(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    @staticmethod
    def envelope(source):
        f=source.forecasts_by_variant
        features=(f['memory_readdressed']['risk_hidden'][:,0].double()-f['natural']['risk_hidden'][:,0].roll(1,1).double()).abs()
        output=(source.physical_by_variant['memory_readdressed']-source.physical_by_variant['natural'].roll(1,1)).abs()
        p=source.risk_decoder_parameters
        bounds=features/p['feature_scale']@p['weights'][:,:2].abs()*p['target_scale']
        source.decode_permutation_error_bounds={'absolute_feature_difference':features,'absolute_output_difference':output,'propagated_bound':bounds}
        return features,output,bounds

    def test_original_observed_recipient_corruption_is_rejected(self):
        source=fixture()
        source.physical_by_variant['memory_readdressed'][0,1,0]+=10
        # Updating the saved discrepancy cannot legitimize mismatched output.
        self.envelope(source)
        with self.assertRaisesRegex(ValueError,'exceeds registered'):
            interface.record(source,0,'memory_readdressed')

    def test_floor_is_distinct_from_a_loose_physical_tolerance(self):
        for error,should_pass in [(8e-13,True),(2e-12,False)]:
            source=fixture()
            source.physical_by_variant['memory_readdressed'][0,1,0]+=error
            _,actual,bound=self.envelope(source)
            self.assertEqual(float(bound.max()),0.)
            if should_pass:
                interface.record(source,0,'memory_readdressed')
                self.assertLessEqual(float(actual.max()),1e-12)
            else:
                with self.assertRaisesRegex(ValueError,'exceeds registered'):
                    interface.record(source,0,'memory_readdressed')

    def test_signed_weights_and_heterogeneous_scales_use_absolute_bound(self):
        source=fixture(); p=source.risk_decoder_parameters
        p['feature_scale'][0]=2.; p['target_scale']=torch.tensor([.003,.007],dtype=torch.float64)
        p['weights'].zero_(); p['weights'][0,:2]=torch.tensor([-3.,5.],dtype=torch.float64)
        moved=source.forecasts_by_variant['memory_readdressed']['risk_hidden']
        before=float(moved[0,0,1,0]); moved[0,0,1,0]+=1e-7
        # Signed mathematical decoder shift, not an arbitrary output allowance.
        delta=float(moved[0,0,1,0])-before
        for output in range(2):
            shift=delta/float(p['feature_scale'][0])*float(p['weights'][0,output])*float(p['target_scale'][output])
            source.physical_by_variant['memory_readdressed'][0,1,output]+=shift
        df,dy,bounds=self.envelope(source)
        for output in range(2):
            scalar_bound=sum(float(df[0,1,j])/float(p['feature_scale'][j])*abs(float(p['weights'][j,output])) for j in range(32))*float(p['target_scale'][output])
            self.assertAlmostEqual(float(bounds[0,1,output]),scalar_bound,places=24)
        self.assertTrue(torch.all(dy<=bounds+1e-12))
        interface.record(source,0,'memory_readdressed')
        source.decode_permutation_error_bounds['propagated_bound']+=1e-3
        with self.assertRaisesRegex(ValueError,'differs from actual'):
            interface.record(source,0,'memory_readdressed')

    def test_nested_excluded_fields_and_legend_injection_fail(self):
        value=interface.record(fixture(),0)['reporter_input']
        changes=[('stored_evidence','human_ratings'),('external_image_statistic_readouts','pixels'),('risk_feature_source','future_outcomes')]
        for node,key in changes:
            malformed=copy.deepcopy(value); malformed['locations'][0][node][key]='FORBIDDEN_TOY'
            with self.assertRaises(ValueError): interface.address_aware_tables(malformed)
        malformed=copy.deepcopy(value); malformed['source_semantics']['success']='felt confidence'
        with self.assertRaisesRegex(ValueError,'legend differs'): interface.address_aware_tables(malformed)

if __name__=='__main__': unittest.main(verbosity=2)
