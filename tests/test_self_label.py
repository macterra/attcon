import json
import sys
import unittest
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'scripts'))
from self_access_table_reports import counterfactual_table, conditions, with_table, TABLE_GLOSSARY, CAMERA_GLOSSARY, CAMERA_KEY
from self_label_gates import decide
from specificity_reports import build_states

CONFIG = json.loads((REPO/'configs/bound_content/self_label_v1.json').read_text())
CHECKPOINT = REPO/'audits/bound_content/confirmation_v2/seed1301.pt'


@unittest.skipUnless(CHECKPOINT.exists(), 'checkpoints not present')
class CameraRenderingTests(unittest.TestCase):
    def test_external_table_changes_only_key_and_glossary_sentence(self):
        pair = CONFIG['pairs'][0]
        base, _, _ = build_states(pair, CONFIG); table = counterfactual_table(pair, CONFIG); conds = conditions(base, table)
        own = with_table(*conds['coupled_table'], 0, CONFIG)
        cam = with_table(*conds['external_table'], 0, CONFIG, camera=True)
        self.assertEqual(own[0], cam[0])  # identical scoring source
        for v_own, v_cam in zip(own[1], cam[1]):
            self.assertEqual(v_own['own_content_by_command'], v_cam[CAMERA_KEY])
            self.assertNotIn('own_content_by_command', v_cam)
        self.assertEqual(own[2].split('\n[{')[0].replace(TABLE_GLOSSARY, CAMERA_GLOSSARY), cam[2].split('\n[{')[0])


class ExtractorV13Tests(unittest.TestCase):
    def test_v12_preserved_with_one_addition(self):
        import extract_bound_prose_v12 as v12, extract_bound_prose_v13 as v13
        self.assertEqual(v13.INSTRUCTION.replace(v13.ADDITION, ''), v12.INSTRUCTION)
        self.assertEqual(v13.SCHEMA['properties']['claims'], v12.SCHEMA['properties']['claims'])
        self.assertEqual(set(v13.STRUCTURE)-set(v12.SCHEMA['properties']['structure_evidence']['properties']), {'stated_content_dependence'})


def case(seed, episode, condition, self_flag, dep_flag):
    checks = [{'field': f, 'correct': True} for f in ('color', 'shape', 'focal')]
    return {'id': f'{seed}_{episode}_{condition}', 'seed': seed, 'episode': episode, 'condition': condition, 'complete': True,
            'structure': {'self_coupled_access': self_flag, 'stated_content_dependence': dep_flag}, 'character': 'mixed',
            'checks': checks, 'content_covered': 1, 'known_objects': 1, 'unresolved_claims': [], 'invalid_evidence': []}


def study(independent_self, external_dep):
    cases = []
    for k in range(48):
        seed, episode = (1301, 1311, 1321)[k//16], k % 16
        cases += [case(seed, episode, 'coupled_table', True, True), case(seed, episode, 'independent_table', k < independent_self, False),
                  case(seed, episode, 'external_table', False, k < external_dep)]
    return cases


class SelfLabelGateTests(unittest.TestCase):
    def test_verdicts(self):
        self.assertEqual(decide(study(40, 48))['verdict'], 'self_access_reporting_not_replicated')
        self.assertEqual(decide(study(5, 46))['verdict'], 'replicated_attribution_follows_label')
        self.assertEqual(decide(study(5, 10))['verdict'], 'replicated_and_self_specific_beyond_label')


if __name__ == '__main__': unittest.main()


class ExtractorV14Tests(unittest.TestCase):
    def test_v13_preserved_with_one_clause(self):
        import extract_bound_prose_v13 as v13, extract_bound_prose_v14 as v14
        self.assertEqual(v14.INSTRUCTION.replace(v14.CLAUSE, ''), v13.INSTRUCTION)
        self.assertEqual(v14.SCHEMA, v13.SCHEMA)
