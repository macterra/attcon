import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from specificity_gates import decide, mcnemar_one_sided


def case(seed, episode, condition, hit, complete=True):
    structure = {'object_linked_access': hit, 'graded_or_temporal_access': hit, 'focal_background_contrast': True, 'agency_relation': True}
    checks = [{'field': f, 'correct': True} for f in ('color', 'shape', 'focal', 'most_recoverable', 'access_trend', 'under_own_control', 'command_next')]
    return {'id': f'{seed}_{episode}_{condition}', 'seed': seed, 'episode': episode, 'condition': condition, 'complete': complete,
            'structure': structure, 'character': 'mixed', 'checks': checks, 'content_covered': 1, 'known_objects': 1,
            'unresolved_claims': [], 'invalid_evidence': []}


def study(external_hits, complete=True):
    cases = []
    for k in range(24):
        seed, episode = (1301, 1311, 1321)[k//8], k % 8
        cases += [case(seed, episode, 'model', True), case(seed, episode, 'external', k < external_hits, complete)]
    return cases


class SpecificityGateTests(unittest.TestCase):
    def test_exact_one_sided_mcnemar(self):
        self.assertAlmostEqual(mcnemar_one_sided(6, 0), 1/64)
        self.assertAlmostEqual(mcnemar_one_sided(0, 0), 1.)
        self.assertAlmostEqual(mcnemar_one_sided(1, 1), .75)

    def test_large_paired_drop_supports_specificity(self):
        out = decide(study(external_hits=12))
        self.assertEqual(out['verdict'], 'specificity_supported')
        self.assertAlmostEqual(out['primary']['difference'], .5)

    def test_small_drop_is_not_supported(self):
        self.assertEqual(decide(study(external_hits=20))['verdict'], 'specificity_not_supported')

    def test_incomplete_pairs_block_a_verdict(self):
        self.assertEqual(decide(study(external_hits=0, complete=False))['verdict'], 'incomplete')

    def test_failed_model_fidelity_is_uninterpretable(self):
        cases = study(external_hits=0)
        for c in cases:
            if c['condition'] == 'model': c['checks'][0]['correct'] = False
        self.assertEqual(decide(cases)['verdict'], 'uninterpretable')


if __name__ == '__main__': unittest.main()
