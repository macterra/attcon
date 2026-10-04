import unittest
from scripts.assess_functional_prose import assess


class FactualScoringTests(unittest.TestCase):
    def source(self):
        return {'output_node':'n2','predicted_current':[{'node':'n2','objects':[
            {'position':'p0','identified_color':'blue','identified_shape':None,
             'color_distribution':{'blue':.7,'red':.3},'shape_distribution':{'circle':.5,'square':.5}}]}],
            'predicted_by_command':None}

    def extracted(self, **changes):
        c={'node':'output','position':'p0','scope':'current','command':None,'dimension':'color',
           'status':'identified','value':'blue','evidence_ids':[0]};c.update(changes)
        return {'claims':[c],'features':{'command_access_relation':[]},'character':'technical'}

    def test_unresolved_and_invalid_evidence_do_not_earn_coverage(self):
        for changes in ({'node':'n9'}, {'evidence_ids':[8]}):
            r=assess(self.source(),'Blue.',self.extracted(**changes))
            self.assertFalse(r['facts'][0]['correct']);self.assertEqual(r['coverage_correct'],0)

    def test_partial_identity_and_explicit_unknown_are_separate(self):
        r=assess(self.source(),'Blue.',self.extracted())
        self.assertTrue(r['facts'][0]['correct']);self.assertEqual(r['coverage_correct'],1)
        unknown=assess(self.source(),'No shape.',self.extracted(dimension='shape',status='unidentified',value=None))
        self.assertTrue(unknown['facts'][0]['correct']);self.assertEqual(unknown['coverage_correct'],0)
        wrong=assess(self.source(),'Square.',self.extracted(dimension='shape',value='square'))
        self.assertFalse(wrong['facts'][0]['correct'])

    def test_missing_command_prediction_cannot_support_command_claim(self):
        r=assess(self.source(),'Under k0 blue.',self.extracted(scope='command',command='k0'))
        self.assertFalse(r['facts'][0]['correct']);self.assertEqual(r['missing_relation_command_claims'],1)


if __name__=='__main__':unittest.main()
