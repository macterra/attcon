import unittest
import torch
from attcon.functional_decisions import (
    DecisionWorld, answer_query, execute_decision, plan_command)
from attcon.predictive_attention import Forecast


def visual(count=1):
    out = torch.zeros(count, 4, 8)
    out[..., 2] = 1
    out[..., 5] = 1
    return out


class ObservationModel:
    """Deliberately exposes acquisition rather than the plan to check the boundary."""
    def __call__(self, observed, hidden):
        self.observed = observed.clone()
        hidden.add_(1)  # A bad consumer must still not mutate the harness input.
        a = observed[..., :8].reshape(-1, 1, 2, 4)
        q = observed[..., 8:16].reshape(-1, 1, 2, 4)
        access = q[:, :, None].expand(-1, -1, 3, -1, -1).clone()
        effects = a[:, :, None].expand(-1, -1, 4, -1, -1).clone()
        return Forecast(a, access, effects), hidden


class FunctionalDecisionTests(unittest.TestCase):
    def test_plan_follows_changed_forecast_and_ties_have_declared_rule(self):
        q = torch.zeros(1, 4, 4)
        q[0, 3, 1] = .8
        command, _ = plan_command(visual(), q, torch.tensor([1]))
        self.assertEqual(command.item(), 3)
        changed, _ = plan_command(visual(), q.flip(1), torch.tensor([1]))
        self.assertEqual(changed.item(), 0)
        tied, _ = plan_command(visual(), torch.full_like(q, .8), torch.tensor([1]))
        self.assertEqual(tied.item(), 0)

    def test_abstention_intervention_preserves_dominant_content(self):
        query = torch.tensor([0])
        base = answer_query(visual(), torch.full((1, 4), .8), query)
        attenuated = answer_query(visual(), torch.full((1, 4), .2), query)
        self.assertEqual(base['response'].tolist(), [[2, 1]])
        self.assertEqual(attenuated['response'].tolist(), [[-1, -1]])
        self.assertTrue(torch.equal(base['labels'], attenuated['labels']))
        # Both attributes must clear the threshold; one known attribute is insufficient.
        one = visual(); one[..., 4:] = .25
        self.assertFalse(answer_query(one, torch.ones(1, 4), query)['answered'].item())

    def test_execution_observes_real_world_without_model_or_world_input_mutation(self):
        world = DecisionWorld(torch.tensor([[0, 1]]), torch.ones(1, 2, dtype=torch.long),
                              torch.zeros(1, 2, 4), torch.tensor([-1]))
        hidden = torch.zeros(1, 1, 4)
        original = [x.clone() for x in (world.replay, world.recovery, hidden)]
        model = ObservationModel()
        updated, _, trace = execute_decision(model, hidden, world, torch.tensor([3]))
        # Disconnected world actually visits positions 1 and 2, despite command 3.
        self.assertEqual(trace['allocation'].argmax(-1).tolist(), [[1, 2]])
        self.assertTrue(torch.equal(model.observed[:, 0], trace['observed']))
        self.assertEqual(updated.access[0, 0, 0, 3].item(), 0)
        response = answer_query(visual(), updated.access[:, 0, 0], torch.tensor([3]))
        self.assertFalse(response['answered'].item())
        for old, now in zip(original, (world.replay, world.recovery, hidden)):
            self.assertTrue(torch.equal(old, now))
        # Answering uses the model even if it is wrong about physical recovery.
        wrong = answer_query(visual(), torch.full((1, 4), .9), torch.tensor([3]))
        self.assertTrue(wrong['answered'].item())
        self.assertEqual(trace['physical_recovery'][0, 0, 3].item(), 0)


if __name__ == '__main__':
    unittest.main()
