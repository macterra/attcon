from pathlib import Path
import sys,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from attcon.prospective import make_splits,ProspectiveAgent,root_events
from attcon.prospective_reporting import *
from attcon.regulation import weight_fingerprint

class ProspectiveReportingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(3)
        self.data=make_splits(5,'routing')['report_fit'].subset(torch.arange(216))
        self.agent=ProspectiveAgent().eval().requires_grad_(False)
        self.states=self.agent.advance(root_events(self.data))

    def test_quality_pairs_preserve_all_other_root_factors(self):
        data=self.data.subset(torch.randperm(len(self.data)))
        low,high=quality_pairs(data)
        for name in ('group','value','condition','cost'):
            self.assertTrue(torch.equal(getattr(data,name)[low],getattr(data,name)[high]))
        self.assertTrue((data.quality[low]<data.quality[high]).all())

    def test_quality_directions_leave_answer_weights_invariant(self):
        directions,_=quality_directions(self.agent,self.states,self.data,3)
        for direction in directions.values():self.assertLess((self.agent.answer.weight@direction).abs().max().item(),1e-5)

    def test_report_fitting_isolated_and_equal_capacity(self):
        before=weight_fingerprint(self.agent);features=report_features(self.agent,self.states,self.data);labels=labels_for(self.data)
        counts=[]
        for feature in features.values():
            model,_=fit_report(feature,labels,feature,labels,4,steps=2)
            counts.append(sum(p.numel() for p in model.parameters()))
        self.assertEqual(counts,[4590]*3);self.assertEqual(before,weight_fingerprint(self.agent))

    def test_perfect_joint_report_has_perfect_components(self):
        labels=labels_for(self.data)
        self.assertTrue(all(value==1 for value in scores(labels,labels).values()))
if __name__=='__main__':unittest.main()
