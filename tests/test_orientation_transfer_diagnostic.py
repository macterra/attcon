import importlib.util
from pathlib import Path
import unittest
import torch

spec=importlib.util.spec_from_file_location('orientation_probe',
    Path(__file__).resolve().parents[1]/'scripts/orientation_transfer_diagnostic_v1.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


class OrientationTransferTests(unittest.TestCase):
    def test_zero_contrast_contains_no_orientation_information(self):
        patches=probe.stimuli(451,count=12)
        self.assertTrue(torch.equal(patches[:,0,0],patches[:,0,1]))
        control=probe.edge_control(patches)
        readout=probe.fit(control)
        scores,_=probe.assess(readout,control,torch.arange(12)[None])
        self.assertEqual(scores['accuracy_by_contrast'][0],.5)

    def test_signal_preserves_orientation_at_nonzero_contrast(self):
        development=probe.edge_control(probe.stimuli(462,count=16))
        evaluation=probe.edge_control(probe.stimuli(473,count=16))
        score,_=probe.assess(probe.fit(development),evaluation,torch.arange(16)[None])
        self.assertEqual(score['accuracy_by_contrast'][-1],1.)

    def test_counterfactual_invariant_features_cannot_classify_orientation(self):
        torch.manual_seed(57)
        features=torch.randn(12,6,1,5).expand(-1,-1,2,-1)
        scores,predictions=probe.assess(probe.fit(features),features,torch.arange(12)[None])
        self.assertEqual(scores['accuracy_by_contrast'],[.5]*6)
        self.assertTrue(torch.equal(predictions,torch.zeros_like(predictions)))


if __name__=='__main__':unittest.main()
