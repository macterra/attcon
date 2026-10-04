import unittest
import torch
from attcon.functional_interface import record, remap, without_relation
from attcon.predictive_attention import Forecast


class FunctionalInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.visual=torch.zeros(1,2,4,8)
        self.visual[...,0]=1;self.visual[...,5]=1
        self.current=Forecast(torch.nn.functional.one_hot(torch.tensor([[1,2]]),4).float(),
            torch.ones(1,3,2,4)*.8,
            torch.nn.functional.one_hot(torch.tensor([[[0,2],[1,2],[2,2],[3,2]]]),4).float())
        self.q=torch.ones(1,4,2,4)*.8
        self.observed=torch.zeros(1,2,20);self.observed[:,:,-4]=1
        self.anticipated=torch.zeros(1,1,20);self.anticipated[:,:,-1]=1

    def test_readout_has_content_but_no_separate_attention(self):
        p=record(self.visual,self.current,self.q,self.observed,0,order=(2,1,0))
        nodes={n['node']:n for n in p['predicted_current']}
        self.assertEqual(p['output_node'],'n0')
        self.assertNotIn('selection_distribution',nodes['n0'])
        self.assertNotIn('recovery_by_delay',nodes['n0']['objects'][0])
        self.assertEqual(nodes['n0']['objects'][0]['color_distribution'],nodes['n2']['objects'][0]['color_distribution'])
        self.assertEqual(nodes['n2']['selected_position'],'p1')
        for trial in p['predicted_by_command']:
            ns={n['node']:n for n in trial['nodes']}
            self.assertEqual(ns['n0']['objects'],ns['n2']['objects'])

    def test_counterbalanced_identifiers_and_commands_are_reversible(self):
        p=record(self.visual,self.current,self.q,self.observed,0)
        nodes={'n0':'q7','n1':'q2','n2':'q9'};cmds={'k0':'m7','k1':'m2','k2':'m9','k3':'m4'}
        changed=remap(p,nodes,cmds)
        self.assertEqual(remap(changed,{v:k for k,v in nodes.items()},{v:k for k,v in cmds.items()}),p)
        self.assertEqual(p['output_node'],'n2')
        with self.assertRaises(ValueError):remap(p,{'n0':'q','n1':'q','n2':'r'})

    def test_anticipated_events_are_never_presented_as_observations(self):
        p=record(self.visual,self.current,self.q,self.observed,0,
                 anticipated=self.anticipated,modeled_step=3)
        self.assertEqual(len(p['observed_history']),2)
        self.assertEqual([x['command'] for x in p['anticipated_history']],['k3'])
        self.assertEqual(p['modeled_step'],3)
        removed=without_relation(p)
        self.assertIsNone(removed['observed_history'])
        self.assertIsNone(removed['anticipated_history'])
        self.assertIsNone(removed['predicted_by_command'])
        self.assertEqual(removed['predicted_current'],p['predicted_current'])


if __name__=='__main__':unittest.main()
