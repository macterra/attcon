from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from attcon.active_inspection import make_splits, AcquisitionAgent, rollout
from attcon.acquisition_learning import train_controller


class AcquisitionLearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        data = make_splits(12)
        self.fit = data['train'].subset(torch.arange(54))
        self.validation = data['validation'].subset(torch.arange(54))

    def test_reward_training_changes_controller_and_selects_validation_epoch(self):
        model, record = train_controller(self.fit, self.validation, 'state', 2, epochs=2)
        self.assertNotEqual(record['initial_sha256'], record['selected_sha256'])
        self.assertEqual(record['selected_epoch'], 2)
        self.assertEqual(record['updates'], 2)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_restoring_initial_state_restores_closed_loop_trajectory(self):
        torch.manual_seed(5)
        model = AcquisitionAgent()
        from attcon.active_inspection import initial_events
        state = model.advance(initial_events(self.validation))
        before = rollout(model, self.validation, initial_state=state)
        delta = torch.randn_like(state) * .1
        after = rollout(model, self.validation, initial_state=state + delta - delta)
        for key in ('return', 'answer', 'inspections', 'visited'):
            self.assertTrue(torch.equal(before[key], after[key]))

if __name__ == '__main__':
    unittest.main()
