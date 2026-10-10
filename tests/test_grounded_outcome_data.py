import unittest
import torch
from grounded_outcome_data_v1 import world_batch, render_locations, collect, record_state, remap_data
from grounded_attention_world_v1 import OrientationWorld


def fake_sensor(pixels):
    value=pixels.mean((-3,-2,-1))-.7
    return value[...,None].expand(*value.shape,64).clone(), value


class GroundedOutcomeDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_batch_renderer_matches_single_deliveries(self):
        worlds=world_batch(1301,0,4)
        locations=torch.tensor([[0,1,3],[2,0,1],[3,1,2],[2,3,0]])
        noise=.015*torch.randn(4,3,3,16,16,generator=torch.Generator().manual_seed(91))
        pixels=render_locations(worlds,locations,noise)
        for row in range(4):
            world=OrientationWorld(**{key:value[row] for key,value in worlds.items()})
            for step in range(3):
                assert torch.equal(pixels[row,step],world.image(int(locations[row,step]),noise[row,step]))


    def test_fresh_collection_truth_is_separate_branches_are_executed_and_replay_exact(self):
        saved={}
        def archive(name,value):saved[name]=value
        state=torch.random.get_rng_state().clone()
        data,world=collect(1301,0,fake_sensor,archive,count=8,repeats=2)
        assert torch.equal(state,torch.random.get_rng_state())
        replay,_=collect(1301,0,fake_sensor,lambda name,value:None,count=8,repeats=2)
        for key in data:
            if isinstance(data[key],torch.Tensor):assert torch.equal(data[key],replay[key])
        assert data['sensory'].shape==(8,4,4,66)
        assert not data['history'][:,0].any() and not data['sensory'][:,0].any()
        assert torch.equal(data['current_labels'][:,0],world['labels']==0)
        for s,time in enumerate(data['snapshots']):
            branch=saved[f'branch-{time}-decoded']
            for row in range(8):
                for r in range(2):
                    for cmd in range(4):
                        loc=int(branch['delivered'][row,r,cmd])
                        cur=data['current_labels'][row,s].clone()
                        cur[loc]=(branch['logits'][row,r,cmd]>0).long()==world['labels'][row,loc]
                        assert torch.equal(cur,data['after_labels'][row,s,r,cmd])
                        assert data['refresh_labels'][row,s,r,loc]==cur[loc]
        memory=data['sensory'][:,2].clone();route=data['history'][:,2].clone()
        original=record_state(memory,route,world['labels'])
        changed=record_state(memory,route,1-world['labels'])
        assert torch.equal(original[0],changed[0]) and torch.equal(original[1],changed[1])
        assert torch.equal(original[2],~changed[2])


    def test_remap_observations_update_rows_and_restore_both_sources(self):
        data,world=collect(1301,2,fake_sensor,lambda n,v:None,count=64,repeats=1)
        result=remap_data(1301,data,world,fake_sensor,lambda n,v:None)
        final=result['history'][:,-1].argmax(-1)
        assert torch.equal(final,(world['command_mapping']+1)%4)
        assert torch.equal(result['restored']['sensory'][:,0],data['sensory'][:,-1])
        assert torch.equal(result['restored']['history'][:,0],data['history'][:,-1])
        for step in range(4):
            assert torch.equal(result['history'][:,step+1,step].argmax(-1),result['delivery_labels'][:,0,step])
