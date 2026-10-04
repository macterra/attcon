import unittest
import torch
from attcon.functional_controls import (CONDITIONS, simulate_controls, effect_table,
    advance_world, physical_recovery, matched_revision, control_mask, true_control_mask)
from attcon.predictive_attention import PredictiveAttention, Forecast


class ThreeWayControlTests(unittest.TestCase):
    def test_paired_physics_share_commands_and_quality_without_equality_shortcut(self):
        episodes=[simulate_controls(79,24,12,c) for c in CONDITIONS]
        for other in episodes[1:]:
            self.assertTrue(torch.equal(other.commands,episodes[0].commands))
            self.assertTrue(torch.equal(other.replay,episodes[0].replay))
            self.assertTrue(torch.equal(other.direction,episodes[0].direction))
            qa=episodes[0].observations[...,8:16].reshape(24,12,2,4).sum(-1)
            qb=other.observations[...,8:16].reshape(24,12,2,4).sum(-1)
            self.assertTrue(torch.equal(qa,qb))
        independent=episodes[-1]
        self.assertTrue(bool((independent.allocation[:,:,0]!=independent.allocation[:,:,1]).any()))
        self.assertTrue(torch.equal(independent.next_allocation[:,:,0],independent.next_allocation[:,:,3]))
        for i,e in enumerate(episodes[:2]):
            # Non-controlled channel is independent of all commands; the controlled one is not.
            self.assertTrue(torch.equal(e.next_allocation[:,:,0,1-i],e.next_allocation[:,:,3,1-i]))
            self.assertFalse(torch.equal(e.next_allocation[:,:,0,i],e.next_allocation[:,:,3,i]))

    def test_mixed_training_balance_and_no_control_mask(self):
        e=simulate_controls(83,24)
        self.assertEqual([int((e.controlled==c).sum()) for c in CONDITIONS],[8,8,8])
        truth=Forecast(e.allocation[:,-1],e.access[:,-1],e.next_allocation[:,-1])
        self.assertTrue(torch.equal(control_mask(truth),true_control_mask(e.controlled)))
        self.assertEqual(e.observations.shape[-1],20)

    def test_physical_transition_is_pure_and_decoupled_commands_cannot_change_access(self):
        e=simulate_controls(83,12,12,-1)
        q=e.access[:,-1,0].clone();replay=e.replay.clone()
        outcomes=[]
        for c in range(4):
            next_replay,next_q,x,_=advance_world(e.replay,e.direction,q,e.controlled,torch.full((12,),c))
            outcomes.append(next_q)
            self.assertTrue(torch.equal(next_replay,(e.replay+e.direction)%4))
            self.assertEqual(x.shape,(12,20))
        self.assertTrue(all(torch.equal(outcomes[0],v) for v in outcomes))
        self.assertTrue(torch.equal(q,e.access[:,-1,0]));self.assertTrue(torch.equal(replay,e.replay))
        targets=physical_recovery(q,e.next_allocation[:,-1])
        self.assertTrue(torch.equal(targets[:,0],outcomes[0]))

    def test_revision_routes_use_same_probe_time_and_full_target(self):
        torch.manual_seed(5);model=PredictiveAttention();model.eval()
        e=simulate_controls(87,6,12,0)
        commands=torch.tensor([[0,1,2,3]]).expand(6,-1)
        result,trace=matched_revision(model,e,1,commands,windows=(1,2,4))
        replay=e.replay.clone();q=e.access[:,-1,0].clone();owner=torch.ones(6,dtype=torch.long)
        for t in range(1,5):
            replay,q,_,_=advance_world(replay,e.direction,q,owner,commands[:,t-1])
            if t not in trace['windows']:continue
            row=trace['windows'][t];target=effect_table((replay+e.direction)%4,owner)
            self.assertTrue(torch.equal(row['target_effects'],target))
            expected=physical_recovery(q,target)
            self.assertTrue(torch.equal(row['target_recovery'],expected))
            metric=next(r['metrics'] for r in result['windows'] if r['observations']==t)
            for prefix in ('no_feedback','feedback'):
                error=float((row[prefix+'_recovery']-expected).abs().double().mean())
                self.assertAlmostEqual(metric[prefix+'_recovery_mae'],error,places=14)
        self.assertEqual(trace['observed'].shape,trace['anticipated'].shape)
        self.assertEqual(trace['commands'].shape,(6,4))
        _,disconnected=matched_revision(model,e,-1,commands,windows=(1,2,4))
        for key in ('initial_allocation','initial_access','initial_effects','initial_hidden','initial_physical_recovery','initial_replay'):
            self.assertTrue(torch.equal(trace[key],disconnected[key]))


if __name__=='__main__':unittest.main()
