import unittest
import torch
from attcon.bound_content import BoundState, VisualEncoder, bind, scenes
from attcon.predictive_attention import Forecast, simulate


class BoundContentTests(unittest.TestCase):
    def test_render_and_encoder_distributions(self):
        a, color, shape = scenes(77, 3)
        b, _, _ = scenes(77, 3)
        self.assertEqual(a.shape, (3, 2, 4, 3, 16, 16))
        self.assertTrue(torch.equal(a, b))
        prediction = VisualEncoder()(a)
        self.assertTrue(torch.allclose(prediction[..., :4].sum(-1), torch.ones(3, 2, 4)))
        self.assertTrue(torch.allclose(prediction[..., 4:].sum(-1), torch.ones(3, 2, 4)))

    def test_content_policy_binding_is_causal_and_restorable(self):
        e = simulate(32, 16)
        f = Forecast(e.allocation[:, 7], e.access[:, 7], e.next_allocation[:, 7])
        visual = torch.cat((torch.eye(4), torch.eye(4)), -1).expand(16, 2, 4, 8).clone()
        state = BoundState(f, visual, torch.eye(4).expand(16, 2, 4, 4).clone(), torch.ones(16, 2, 4, dtype=torch.bool))
        query = torch.arange(16) % 4
        command = state.content_policy(e.controlled, query, query)
        self.assertTrue(torch.equal(command, query))
        altered = state.replace_binding(state.binding.roll(1, -2))
        self.assertTrue(torch.equal(altered.content_policy(e.controlled, query, query), (query + 1) % 4))
        self.assertTrue(torch.equal(altered.visual, state.visual))
        self.assertTrue(torch.equal(altered.attention.effects, state.attention.effects))
        restored = altered.replace_binding(state.binding)
        self.assertTrue(torch.equal(restored.content_policy(e.controlled, query, query), command))

    def test_unobserved_content_is_not_leaked(self):
        patches, _, _ = scenes(88, 2)
        e = simulate(10, 2)
        f = Forecast(e.allocation[:, 0], e.access[:, 0], e.next_allocation[:, 0])
        state = bind(VisualEncoder(), patches, f, e.allocation[:, :1])
        self.assertTrue(torch.all(state.visual[~state.remembered] == .25))
        self.assertEqual(int(state.remembered.sum()), 4)

    def test_selective_interventions_do_not_mutate_source(self):
        e = simulate(2, 2)
        f = Forecast(e.allocation[:, 7], e.access[:, 7], e.next_allocation[:, 7])
        patches, _, _ = scenes(2, 2)
        state = bind(VisualEncoder(), patches, f, e.allocation[:, :8])
        changed = state.replace_visual(state.visual.roll(1, -2))
        self.assertTrue(torch.equal(changed.binding, state.binding))
        self.assertTrue(torch.equal(changed.attention.access, state.attention.access))
        with self.assertRaises(ValueError):
            state.replace_binding(torch.zeros_like(state.binding))
