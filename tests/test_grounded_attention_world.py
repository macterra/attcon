import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from grounded_attention_world_v1 import OrientationWorld,OrientationMemory,execute,evaluator_correctness,counterfactual_branches


class GroundedWorldTests(unittest.TestCase):
    def test_world_labels_are_not_forced_balanced(self):
        labels=[OrientationWorld.sample(seed).labels for seed in range(10)]
        assert any(int(x.sum())!=2 for x in labels)
        assert torch.equal(OrientationWorld.sample(115).labels,OrientationWorld.sample(115).labels)

    def test_static_memory_does_not_invent_decay_or_rewrite_content(self):
        memory=OrientationMemory.empty().observe(2,2.4)
        content=memory.uncalibrated_orientation_coordinates().clone()
        for _ in range(10):memory=memory.advance()
        assert torch.equal(content,memory.uncalibrated_orientation_coordinates())
        assert memory.answer(2)==1 and memory.last_update[2]==0 and memory.time==10
        assert not memory.observed[0] and memory.answer(0)==0
        assert torch.equal(memory.uncalibrated_orientation_coordinates()[0],torch.tensor([.5,.5],dtype=torch.float64))

    def test_delivery_mapping_and_label_boundary(self):
        world=OrientationWorld.sample(112).remap([2,3,0,1])
        memory=OrientationMemory.empty()
        post,observation=execute(world,memory,0,torch.zeros(3,16,16),lambda image:1.3)
        assert observation.delivered_location==2 and post.observed.sum()==1
        assert post.observed[2] and not memory.observed.any()
        fields=observation.model_input()
        assert set(fields)=={'command','delivered_location','sensory_logit','time'}
        assert 'correctness' not in fields and 'contrast' not in fields and 'label' not in fields
        assert evaluator_correctness(world,post)[2]==(world.labels[2]==1)
        remapped=world.remap([1,0,3,2])
        assert torch.equal(world.labels,remapped.labels)
        assert torch.equal(world.image(2,torch.zeros(3,16,16)),remapped.image(2,torch.zeros(3,16,16)))
        newer,obs=execute(remapped,memory,0,torch.zeros(3,16,16),lambda image:1.3)
        assert obs.delivered_location==1 and newer.observed[1]

    def test_exact_replay_refresh_can_worsen_and_branches_do_not_leak(self):
        world=OrientationWorld.sample(129).remap([0,1,2,3])
        label=int(world.labels[0]);correct_logit=2. if label else -2.
        memory=OrientationMemory.empty().observe(0,correct_logit)
        noises=torch.randn(2,4,3,16,16,generator=torch.Generator().manual_seed(11))*.015
        sensor=lambda image:-correct_logit
        first=counterfactual_branches(world,memory,noises,sensor)
        second=counterfactual_branches(world,memory,noises,sensor)
        assert evaluator_correctness(world,memory)[0]
        assert not first[0][0]['evaluator_correctness'][0]
        assert memory.time==0 and memory.observed.sum()==1
        for command in range(4):
            branch=first[0][command]
            assert branch['memory'].time==1
            assert torch.equal(branch['memory'].logits,second[0][command]['memory'].logits)
            assert torch.equal(branch['evaluator_correctness'],second[0][command]['evaluator_correctness'])
            assert 'evaluator_correctness' not in branch['observation'].model_input()
        assert not torch.equal(world.image(0,noises[0,0]),world.image(0,noises[1,0]))


if __name__=='__main__':unittest.main()
