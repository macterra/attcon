"""Controls catch scene leakage and confounding in the diagnostic readout."""
import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location('physical_probe',
    Path(__file__).resolve().parents[1]/'scripts/physical_source_localization_v1.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class PhysicalSourceDiagnosticTests(unittest.TestCase):
    def test_scene_level_confound_does_not_predict_within_scene_variation(self):
        torch.manual_seed(42)
        scene = torch.randn(12,1,1,1,3).expand(-1,6,2,4,-1)
        targets = torch.randn(12,6,2,4,2)
        fit = probe.fit_readout(scene,targets)
        samples = torch.arange(12)[None]
        score,prediction = probe.evaluate_readout(fit,scene,targets,samples)
        self.assertEqual(score['r2'],[0.,0.])
        self.assertTrue(torch.equal(prediction,torch.zeros_like(prediction)))

    def test_measured_varying_feature_outperforms_invariant_feature(self):
        torch.manual_seed(54)
        feature = torch.randn(20,6,2,4,1,dtype=torch.float64)
        targets = torch.cat((feature,2*feature),-1)
        fit = probe.fit_readout(feature[:10],targets[:10])
        score,_ = probe.evaluate_readout(fit,feature[10:],targets[10:],torch.arange(10)[None])
        self.assertTrue(all(value>.99 for value in score['r2']))

    def test_readout_scaling_uses_development_only(self):
        torch.manual_seed(65)
        feature = torch.randn(8,6,2,4,2,dtype=torch.float64)
        fit = probe.fit_readout(feature,feature)
        original = {key:value.clone() for key,value in fit.items()}
        probe.evaluate_readout(fit,feature*100,feature,torch.arange(8)[None])
        for key in fit: self.assertTrue(torch.equal(fit[key],original[key]))

    def test_blur_preserves_constant_pixels_and_reduces_edge_energy(self):
        constant = torch.full((1,2,4,3,16,16),.5)
        self.assertTrue(torch.allclose(probe.transform(constant,1,True),constant))
        checker = (torch.arange(16)[:,None]+torch.arange(16)[None,:])%2
        pixels = checker.float().expand(1,2,4,3,16,16).clone()
        before = probe.physical_targets(pixels)[...,1]
        after = probe.physical_targets(probe.transform(pixels,1,True))[...,1]
        self.assertTrue(bool((after<before).all()))


if __name__=='__main__': unittest.main()
