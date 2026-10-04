import copy
import json
import unittest
import torch
from scripts.neutral_functional_reports_v3 import complete_record
from scripts.assess_functional_prose_v3 import assess, expected_process
import neutral_functional_reports as v1


class CompleteAttentionInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.states=torch.load(v1.MODEL_ROOT/'states.pt',weights_only=True)
        cls.sample=next(s for s in json.loads((v1.MODEL_ROOT/'samples.json').read_text()) if s['row']==2)

    def test_fields_match_archived_forecasts_and_output_has_no_selection(self):
        source=complete_record(self.sample,self.states,'neutral')
        state=self.states[self.sample['visual_seed']][self.sample['physical_owner']]
        output=next(n for n in source['predicted_current'] if n['node']==source['output_node'])
        self.assertNotIn('selected_position',output)
        self.assertTrue(all('recovery_by_delay' not in o for o in output['objects']))
        # row 2 ordering is physical 1, output, physical 0.
        for physical,name in [(0,'n2'),(1,'n0')]:
            node=next(n for n in source['predicted_current'] if n['node']==name)
            self.assertEqual(list(node['selection_distribution'].values()),[round(float(x),5) for x in state['modeled_allocation'][2,physical]])
            for p,obj in enumerate(node['objects']):
                self.assertEqual(list(obj['recovery_by_delay'].values()),[round(float(x),5) for x in state['modeled_access'][2,:,physical,p]])
        self.assertEqual(len(expected_process(source)),18)

    def test_controls_preserve_current_state_and_restore_names(self):
        source=complete_record(self.sample,self.states,'neutral')
        remapped=complete_record(self.sample,self.states,'remapped')
        self.assertEqual(v1.remap(remapped,{'q7':'n0','q2':'n1','q9':'n2'}),source)
        self.assertEqual(complete_record(self.sample,self.states,'restored'),source)
        swap=complete_record(self.sample,self.states,'model_swap')
        self.assertEqual(swap['predicted_current'],source['predicted_current'])
        self.assertEqual(swap['observed_history'],source['observed_history'])
        for old,new in zip(source['predicted_by_command'],swap['predicted_by_command']):
            a={n['node']:n for n in old['nodes']};b={n['node']:n for n in new['nodes']}
            self.assertEqual(a['n0']['selection_distribution'],b['n2']['selection_distribution'])
            self.assertEqual(a['n2']['selection_distribution'],b['n0']['selection_distribution'])
        missing=complete_record(self.sample,self.states,'missing_relation')
        self.assertIsNone(missing['observed_history']);self.assertIsNone(missing['predicted_by_command'])
        self.assertEqual(missing['predicted_current'],source['predicted_current'])
        self.assertEqual(len(expected_process(missing)),10)

    def test_scoring_rejects_readout_selection_and_unsupported_trends(self):
        source=complete_record(self.sample,self.states,'neutral')
        key=next(k for k in expected_process(source) if k[4]=='selection' and k[0]=='current')
        claim=dict(zip(('scope','command','node','position','dimension','value'),key));claim['evidence_ids']=[0]
        data={'claims':[],'process_claims':[claim],'features':{'command_access_relation':[]},'character':'technical'}
        correct=assess(source,'Selected.',data)
        self.assertEqual(correct['process_coverage_correct'],1)
        for changes in ({'node':'output'}, {'evidence_ids':[9]}, {'position':'p0'}, {'scope':'command','command':'k9'}):
            wrong=copy.deepcopy(data);wrong['process_claims'][0].update(changes)
            result=assess(source,'Selected.',wrong)
            self.assertFalse(result['facts'][0]['correct']);self.assertEqual(result['process_coverage_correct'],0)
        trend=next(k for k in expected_process(source) if k[4]=='recovery_trend')
        data['process_claims']=[dict(zip(('scope','command','node','position','dimension','value'),trend),evidence_ids=[0])]
        self.assertEqual(assess(source,'Declines.',data)['process_coverage_correct'],1)
        data['process_claims'][0]['value']='invented'
        self.assertFalse(assess(source,'Invented.',data)['facts'][0]['correct'])


if __name__=='__main__':unittest.main()
