import unittest
import torch
from attcon.predictive_attention import Forecast, PredictiveAttention, simulate, prediction_loss


class PredictiveAttentionTests(unittest.TestCase):
    def test_simulator_reproducibility_and_process(self):
        a, b = simulate(31, 16), simulate(31, 16)
        self.assertTrue(torch.equal(a.observations, b.observations))
        self.assertTrue(torch.equal(a.objects, b.objects))
        self.assertTrue(torch.allclose(a.access[:, :, 1], a.access[:, :, 0] * .75))
        for row in range(16):
            channel = a.controlled[row]
            self.assertTrue(torch.equal(a.allocation[row, :, channel].argmax(-1), a.commands[row]))
            self.assertTrue(torch.equal(a.next_allocation[row, :, :, channel].argmax(-1), torch.arange(4).expand(12, -1)))

    def test_policy_uses_effect_model_and_restores(self):
        e = simulate(12, 32)
        f = Forecast(e.allocation[:, 5], e.access[:, 5], e.next_allocation[:, 5])
        query = torch.arange(32) % 4
        original = f.command(e.controlled, query)
        self.assertTrue(torch.equal(original, query))
        changed = f.intervene('effects', f.effects.roll(1, dims=1))
        self.assertTrue(torch.equal(changed.command(e.controlled, query), (query + 1) % 4))
        self.assertTrue(torch.equal(changed.access, f.access))
        restored = changed.intervene('effects', f.effects)
        self.assertTrue(torch.equal(restored.command(e.controlled, query), original))
        self.assertTrue(torch.equal(f.controllability().argmax(-1), e.controlled))

    def test_intervention_validation_and_nonmutation(self):
        e = simulate(11, 8)
        f = Forecast(e.allocation[:, 3], e.access[:, 3], e.next_allocation[:, 3])
        changed = f.intervene('access', torch.zeros_like(f.access))
        self.assertGreater(f.access.sum().item(), 0)
        self.assertEqual(changed.access.sum().item(), 0)
        with self.assertRaises(ValueError):
            f.intervene('effects', torch.zeros_like(f.effects))
        with self.assertRaises(ValueError):
            f.intervene('access', torch.full_like(f.access, float('nan')))

    def test_prediction_training_and_streaming(self):
        torch.manual_seed(2)
        e = simulate(7, 4, 6)
        model = PredictiveAttention()
        f, h = model(e.observations)
        streamed, sh = model(e.observations[:, :3])
        streamed, sh = model(e.observations[:, 3:], sh)
        self.assertTrue(torch.allclose(f.access[:, 3:], streamed.access, atol=1e-6))
        self.assertTrue(torch.allclose(h, sh, atol=1e-6))
        prediction_loss(f, e).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
