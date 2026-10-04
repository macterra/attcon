import unittest
import torch
from attcon.control_probes import probe_commands,probe
from attcon.functional_controls import advance_world,last_forecast
from attcon.predictive_attention import PredictiveAttention


class ControlProbeTests(unittest.TestCase):
    def setUp(self):
        g=torch.Generator().manual_seed(73)
        self.phase=torch.randint(4,(32,2),generator=g)
        self.direction=torch.randint(2,(32,2),generator=g)*2-1
        self.random=torch.randint(4,(32,8),generator=g)
        self.offsets=torch.randint(1,4,(32,8),generator=g)
        self.q=torch.rand(32,2,4,generator=g)

    def test_agreement_worlds_have_identical_measurements_and_recovery(self):
        for old in (0,1):
            commands=probe_commands(self.phase,self.direction,old,self.random,self.offsets,'agree')
            phase_a,phase_b=self.phase.clone(),self.phase.clone()
            q_a,q_b=self.q.clone(),self.q.clone()
            for t in range(8):
                phase_a,q_a,x_a,_=advance_world(phase_a,self.direction,q_a,torch.full((32,),-1),commands[:,t])
                phase_b,q_b,x_b,_=advance_world(phase_b,self.direction,q_b,torch.full((32,),old),commands[:,t])
                self.assertTrue(torch.equal(x_a,x_b));self.assertTrue(torch.equal(q_a,q_b))
                self.assertTrue(torch.equal(phase_a,phase_b))

    def test_contradictory_commands_never_match_automatic_destination(self):
        for old in (0,1):
            commands=probe_commands(self.phase,self.direction,old,self.random,self.offsets,'contradict')
            phase=self.phase.clone()
            for t in range(8):
                phase=(phase+self.direction)%4
                self.assertTrue(bool((commands[:,t]!=phase[:,old]).all()))
            self.assertTrue(torch.equal(probe_commands(self.phase,self.direction,old,self.random,self.offsets,'random'),self.random))

    def test_identical_inputs_preserve_model_states_despite_different_counterfactual_truth(self):
        torch.manual_seed(83);model=PredictiveAttention();model.eval()
        _,_,x,_=advance_world(self.phase,self.direction,self.q,torch.full((32,),-1),self.random[:,0])
        seq,h=model(x[:,None]);current=last_forecast(seq)
        commands=probe_commands(self.phase,self.direction,0,self.random,self.offsets,'agree')
        _,a=probe(model,current,h,self.phase,self.direction,self.q,commands,-1)
        _,b=probe(model,current,h,self.phase,self.direction,self.q,commands,0)
        self.assertTrue(torch.equal(a['observations'],b['observations']))
        for t in (1,2,4,8):
            for k in ('modeled_allocation','modeled_access','modeled_effects','hidden','predicted_recovery'):
                self.assertTrue(torch.equal(a['windows'][t][k],b['windows'][t][k]),k)
            self.assertFalse(torch.equal(a['windows'][t]['target_effects'],b['windows'][t]['target_effects']))


if __name__=='__main__':unittest.main()
