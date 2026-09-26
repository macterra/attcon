from pathlib import Path
import sys,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from attcon.prospective import make_splits,ProspectiveAgent,states_for,action_values
from attcon.prospective_exploration import environmental_rewards,experienced_loss,exploratory_actions,train_exploration

class ExplorationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.data=make_splits(7,'routing')['train'].subset(torch.arange(12))

    def test_reward_simulator_charges_inspection_and_decline(self):
        actions=torch.tensor([6,7,8]*4)
        rewards=environmental_rewards(actions,self.data)
        self.assertTrue(torch.allclose(rewards[actions==6],torch.full_like(rewards[actions==6],.3)))
        self.assertTrue(torch.equal(rewards[actions>=7],-self.data.cost[actions>=7]))

    def test_unchosen_and_inactive_rewards_do_not_train_answers(self):
        logits=torch.zeros(3,6,requires_grad=True)
        values=torch.zeros(3,9,requires_grad=True)
        actions=torch.tensor([1,6,2]);rewards=torch.tensor([1.,.3,0.]);active=torch.tensor([True,True,False])
        loss,count=experienced_loss(logits,values,actions,rewards,torch.zeros(3),active)
        loss.backward()
        self.assertEqual(count,1)
        self.assertEqual(logits.grad[1:].abs().sum().item(),0)
        self.assertLess(logits.grad[0,1].item(),0)

    def test_exploration_never_selects_masked_actions(self):
        values=torch.zeros(512,9);values[:,7:]=-1e9
        actions=exploratory_actions(values,1.,torch.Generator().manual_seed(2))
        self.assertTrue((actions<7).all())

    def test_from_scratch_reward_learning_changes_weights(self):
        validation=make_splits(7,'routing')['validation'].subset(torch.arange(108))
        model,record=train_exploration(self.data,validation,7,'state',updates=3)
        self.assertNotEqual(record['initial_sha256'],record['selected_sha256'])
        self.assertGreater(record['experience']['answers'],0)
        self.assertGreater(record['experience']['inspections'],0)
        self.assertEqual(record['episodes_sampled'],1536)
if __name__=='__main__':unittest.main()
