import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from sensory_task_adapter_v1 import fit_classifier,predict,metrics,bootstrap
from isolated_orientation_task_v1 import fit_calibrator,probability,stimuli,render


class SensoryAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_all_training_conditions_and_reproducible_single_image_inference(self):
        g=torch.Generator().manual_seed(25)
        features=torch.randn(16,6,3,generator=g)
        labels=torch.arange(16)%2
        before=torch.random.get_rng_state().clone()
        first=fit_classifier(features,labels,123,steps=20)
        second=fit_classifier(features,labels,123,steps=20)
        assert torch.equal(before,torch.random.get_rng_state())
        assert all(torch.equal(first['state_dict'][k],second['state_dict'][k]) for k in first['state_dict'])
        assert torch.equal(first['mean'],features.flatten(0,1).mean(0))
        changed=features.clone();changed[:,0]+=3
        different=fit_classifier(changed,labels,123,steps=20)
        assert not torch.equal(first['mean'],different['mean'])
        assert not torch.equal(first['state_dict']['0.weight'],different['state_dict']['0.weight'])
        sample=features[3,2].clone()
        batch=features.clone();batch[:3]=999;batch[4:]=-999
        assert torch.allclose(predict(first,sample),predict(first,batch)[3,2],atol=1e-6,rtol=1e-6)
        assert first['parameter_count']==161

    def test_zero_scale_zero_ties_and_separate_null_probabilities(self):
        x=torch.ones(16,6,2);labels=torch.arange(16)%2
        fit=fit_classifier(x,labels,18,steps=10)
        assert torch.equal(fit['scale'],torch.ones(2))
        scores=predict(fit,x)
        cal=fit_calibrator(scores,labels)
        null=torch.stack((scores[:,0],scores[:,0]),1)
        p=probability(cal,null)
        assert torch.equal(p[:,0],p[:,1])
        assert float(((null>0).long()==torch.arange(2)[None]).double().mean())==.5
        assert not bool(torch.tensor(0.)>0)

    def test_ten_bin_fixture_and_conditional_cancellation(self):
        prob=torch.tensor([.1,.9,0.,1.],dtype=torch.float64)
        correct=torch.tensor([0.,1.,0.,1.],dtype=torch.float64)
        m=metrics(prob,correct)
        assert abs(m['brier']-.005)<1e-14
        assert abs(m['ece']-.05)<1e-14
        assert m['reliability'][0]['count']==1 and m['reliability'][1]['count']==1
        assert m['reliability'][9]['count']==2
        # Same prediction has perfect pooled calibration but wrong conditional rates.
        p=torch.full((8,2),.75,dtype=torch.float64)
        y=torch.tensor([[1.,1.]]*4+[[0.,1.]]*4,dtype=torch.float64)
        assert metrics(p,y)['ece']==0
        assert metrics(p[:,0],y[:,0])['ece']==.25
        assert metrics(p[:,1],y[:,1])['ece']==.25

    def test_bootstrap_keeps_bases_and_condition_specific_baselines(self):
        p=torch.tensor([[.1,.8],[.8,.6],[.4,.9]],dtype=torch.float64)
        y=torch.tensor([[0.,1.],[1.,0.],[1.,1.]],dtype=torch.float64)
        baseline=torch.tensor([.4,.8],dtype=torch.float64)
        indices=torch.tensor([[0,0,2],[2,1,1],[1,0,2]])
        b=bootstrap(p,y,baseline,indices)
        residual=[];eces=[];differences=[]
        for sample in indices:
            residual.append((p[sample]-y[sample]).mean(0))
            eces.append(torch.tensor([metrics(p[sample,c],y[sample,c])['ece'] for c in range(2)],dtype=torch.float64))
            differences.append(((p[sample]-y[sample]).square()-(baseline-y[sample]).square()).mean(0))
        for key,values in [('signed_residual_95_by_contrast',residual),('ece_95_by_contrast',eces),('brier_vs_conditional_constant_95_by_contrast',differences)]:
            expected=torch.quantile(torch.stack(values),torch.tensor([.025,.975],dtype=torch.float64),dim=0).T
            assert torch.allclose(torch.tensor(b[key],dtype=torch.float64),expected,atol=1e-14,rtol=0)

    def test_stimulus_labels_do_not_drive_nuisance_or_null(self):
        x,labels,n=stimuli(123987,16)
        changed=render(1-labels,**n)
        assert torch.equal(x[:,0],changed[:,0])
        assert not torch.equal(x[:,-1],changed[:,-1])
        assert not torch.equal(x,stimuli(133987,16)[0])


if __name__=='__main__':unittest.main()
