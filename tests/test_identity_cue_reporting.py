import importlib.util,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('identity_report',ROOT/'scripts/audit_identity_cue_reporting.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from attcon.prospective_reporting import report_features

class IdentityCueTests(unittest.TestCase):
    def test_equal_confidence_different_answers_remain_distinguishable(self):
        torch.set_num_threads(1)
        answer=torch.nn.Linear(6,6)
        with torch.no_grad():answer.weight.copy_(torch.eye(6));answer.bias.zero_()
        agent=SimpleNamespace(answer=answer)
        states=torch.eye(6)[:2]*5
        data=SimpleNamespace(quality=torch.tensor([.55,.55]))
        flawed=report_features(agent,states,data)['cue']
        corrected=module.identity_cue_features(agent,states,data)
        self.assertTrue(torch.allclose(flawed[0],flawed[1]))
        self.assertFalse(torch.equal(corrected[0],corrected[1]))
        self.assertEqual(corrected[:,:6].argmax(-1).tolist(),[0,1])
        self.assertTrue(torch.equal(corrected[:,6],data.quality))
        self.assertEqual(corrected[:,7:].abs().sum().item(),0)
if __name__=='__main__':unittest.main()
