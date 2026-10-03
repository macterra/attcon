import json
import sys
import unittest
from pathlib import Path
import torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'scripts'))
import extract_bound_prose_v10 as v10
import extract_bound_prose_v11 as v11
from self_coupled_gates import decide
from self_coupled_reports import gate, states, label
from specificity_reports import build_states, render

CONFIG = json.loads((REPO/'configs/bound_content/self_coupled_v1.json').read_text())
CHECKPOINT = REPO/'audits/bound_content/confirmation_v2/seed1301.pt'


@unittest.skipUnless(CHECKPOINT.exists(), 'checkpoints not present')
class SelfCoupledStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, _, _ = build_states(CONFIG['pairs'][0], CONFIG)
        cls.states = states(cls.base)

    def certainty(self, state):
        return state.contents[..., :4].max(-1).values

    def test_coupled_certainty_tracks_forecast_access(self):
        access = self.base.attention.access[:, 0]
        expected = access*self.base.contents[..., :4].max(-1).values+(1-access)*.25
        self.assertTrue(torch.allclose(self.certainty(self.states['coupled']), expected, atol=1e-5))

    def test_decoupled_uses_same_weights_permuted_and_same_attention(self):
        coupled, decoupled = self.states['coupled'], self.states['decoupled_matched']
        for field in ('allocation', 'access', 'effects'):
            self.assertTrue(torch.equal(getattr(coupled.attention, field), getattr(decoupled.attention, field)))
        access = self.base.attention.access[:, 0]
        a, b = torch.sort(access.flatten(1), -1).values, torch.sort(access.roll(1, -1).flatten(1), -1).values
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.allclose(coupled.visual, decoupled.visual))

    def test_model_only_access_intervention_moves_content(self):
        intervened = self.states['coupled_access_intervention']
        self.assertTrue(torch.equal(intervened.attention.access, self.base.attention.access.flip(-1)))
        expected = gate(self.base.replace_attention(intervened.attention), intervened.attention.access[:, 0])
        self.assertTrue(torch.allclose(intervened.visual, expected.visual))

    def test_unobserved_objects_stay_uniform(self):
        unseen = ~self.base.remembered
        if unseen.any():
            self.assertTrue(torch.allclose(self.states['coupled'].visual[unseen], torch.full_like(self.states['coupled'].visual[unseen], .25)))

    def test_rendering_labels(self):
        self.assertEqual(label('coupled_opaque'), 'opaque'); self.assertEqual(label('decoupled_matched'), 'model')
        _, _, prompt = render(self.states['coupled'], 0, 'model', CONFIG)
        self.assertIn('Selection probabilities\nmodel current attention', prompt)
        self.assertNotIn('depend', prompt.split('\n[')[0])


class ExtractorV11Tests(unittest.TestCase):
    def test_v10_instruction_preserved_with_one_addition(self):
        self.assertEqual(v11.INSTRUCTION.replace(v11.ADDITION, ''), v10.INSTRUCTION)
        self.assertEqual(v11.MODEL, v10.MODEL)
        self.assertEqual(v11.SCHEMA['properties']['claims'], v10.SCHEMA['properties']['claims'])
        self.assertEqual(set(v11.STRUCTURE)-set(v10.structure_props), {'self_coupled_access'})

    def test_self_coupled_fixture_rule(self):
        hit = {'parsed': {'structure_evidence': {'self_coupled_access': [0]}}}
        miss = {'parsed': {'structure_evidence': {'self_coupled_access': []}}}
        self.assertTrue(v11.assess_self_coupled('One sentence.', True, hit))
        self.assertFalse(v11.assess_self_coupled('One sentence.', False, hit))
        self.assertTrue(v11.assess_self_coupled('One sentence.', False, miss))
        self.assertFalse(v11.assess_self_coupled('One sentence.', True, {}))


def case(seed, episode, condition, hit, complete=True):
    checks = [{'field': f, 'correct': True} for f in ('color', 'shape', 'focal')]
    return {'id': f'{seed}_{episode}_{condition}', 'seed': seed, 'episode': episode, 'condition': condition, 'complete': complete,
            'structure': {'self_coupled_access': hit}, 'character': 'mixed', 'checks': checks, 'content_covered': 1,
            'known_objects': 1, 'unresolved_claims': [], 'invalid_evidence': []}


def study(decoupled_hits, complete=True):
    cases = []
    for k in range(24):
        seed, episode = (1301, 1311, 1321)[k//8], k % 8
        cases += [case(seed, episode, 'coupled', True), case(seed, episode, 'decoupled_matched', k < decoupled_hits, complete)]
    return cases


class SelfCoupledGateTests(unittest.TestCase):
    def test_verdicts(self):
        self.assertEqual(decide(study(12))['verdict'], 'self_coupling_specificity_supported')
        self.assertEqual(decide(study(20))['verdict'], 'self_coupling_specificity_not_supported')
        self.assertEqual(decide(study(0, complete=False))['verdict'], 'incomplete')
        bad = study(0)
        for c in bad:
            if c['condition'] == 'coupled': c['checks'][0]['correct'] = False
        self.assertEqual(decide(bad)['verdict'], 'uninterpretable')

    def test_untested_field_does_not_fail(self):
        out = decide(study(12))
        self.assertIsNone(out['interpretability']['fields']['command_next']['accuracy'])
        self.assertTrue(out['interpretability']['gates']['command_next_accuracy'])


if __name__ == '__main__': unittest.main()
