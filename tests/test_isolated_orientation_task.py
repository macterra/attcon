import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from isolated_orientation_task_v1 import stimuli,render,fit_classifier,predict,fit_calibrator,probability,metrics



import unittest

class IsolatedOrientationTests(unittest.TestCase):

    def test_nuisances_and_zero_do_not_depend_on_label(self):
        patches,labels,n=stimuli(87,32)
        swapped=render(1-labels,**n)
        assert torch.equal(patches[:,0],swapped[:,0])
        assert not torch.equal(patches[:,-1],swapped[:,-1])
        assert labels.sum()==16
        assert not torch.equal(n['noise'][:,0],n['noise'][:,1])
        a=stimuli(10087,32)
        assert not torch.equal(patches,a[0])


    def test_single_image_prediction_and_fit_split_isolation(self):
        g=torch.Generator().manual_seed(12)
        dev=torch.randn(32,6,4,generator=g)
        labels=torch.arange(32)%2
        r=fit_classifier(dev,labels)
        changed=dev.clone();changed[:,:-1]=999
        r2=fit_classifier(changed,labels)
        assert all(torch.equal(r[k],r2[k]) for k in r)
        evaluation=torch.randn(10,6,4,generator=g)
        alone=predict(r,evaluation[3,2])
        assert torch.allclose(alone,predict(r,evaluation)[3,2],atol=1e-14,rtol=0)
        batch=evaluation.clone();batch[:3]=1000;batch[4:]=-1000
        assert torch.allclose(alone,predict(r,batch)[3,2],atol=1e-14,rtol=0)
        assert all(torch.equal(r[k],r2[k]) for k in r)


    def test_calibration_isolation_and_scoring_fixture(self):
        scores=torch.tensor([[-1.,-.1],[.1,1.],[-.2,.2],[.2,-.2]],dtype=torch.float64)
        labels=torch.tensor([0,1,0,1])
        fit=fit_calibrator(scores,labels)
        before={k:v.clone() for k,v in fit.items()}
        probability(fit,torch.tensor([1000.,0.]))
        assert all(torch.equal(fit[k],before[k]) for k in fit)
        assert float(fit['baseline'])==.75
        fixture=metrics(torch.tensor([.2,.8,1.,0.],dtype=torch.float64),torch.tensor([0.,1.,1.,0.]))
        assert abs(fixture['brier']-.02)<1e-14
        assert abs(fixture['ece']-.1)<1e-14
        assert fixture['reliability'][0]['count']==1
        assert fixture['reliability'][1]['count']==1
        assert fixture['reliability'][4]['count']==2
        assert not bool(torch.tensor(0.)>0)  # prescribed horizontal tie


    def test_physical_isolated_control_and_invariant_feature_null(self):
        from orientation_transfer_diagnostic_v1 import edge_control
        development,labels,_=stimuli(453,64)
        evaluation,targets,_=stimuli(10453,64)
        readout=fit_classifier(edge_control(development),labels)
        answer=(predict(readout,edge_control(evaluation))[:,-1]>0).long()
        assert float((answer==targets).double().mean())>=.90
        invariant=torch.ones(64,6,3)
        tied=predict(fit_classifier(invariant,labels),invariant)
        assert torch.equal(tied,torch.zeros_like(tied))
        assert float(((tied>0).long()==targets[:,None]).double().mean())==.5

    def test_whole_base_bootstrap_preserves_contrasts(self):
        units=torch.arange(12).reshape(2,6)
        indices=torch.tensor([[1,1],[0,1]])
        sampled=units[indices]
        assert sampled.shape==(2,2,6)
        assert torch.equal(sampled[0,0],units[1])
        assert torch.equal(sampled[0,1],units[1])

if __name__=='__main__':unittest.main()
