import json
import re
import sys
import unittest
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'scripts'))
from specificity_reports import build_states, render

V5 = REPO/'audits/bound_content/language_confirmation_v5/requests.json'
CONFIG = json.loads((REPO/'configs/bound_content/specificity_v1.json').read_text())
BANNED = re.compile(r'attention|attend|select|recover|access|control|redirect|focus|aware', re.I)


def values(record):
    """All leaf values in order, ignoring key names."""
    if isinstance(record, dict): return [v for x in record.values() for v in values(x)]
    if isinstance(record, list): return [v for x in record for v in values(x)]
    return [record]


@unittest.skipUnless(V5.exists(), 'archived v5 requests not present')
class SpecificityRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v5 = json.loads((REPO/'configs/bound_content/language_confirmation_v5.json').read_text())
        pair = v5['pairs'][0]
        cls.base, _, _ = build_states(pair, v5)
        cls.v5 = {r['id']: r for r in json.loads(V5.read_text())}
        cls.config = dict(CONFIG, question=v5['question'], prose_instruction=v5['prose_instruction'])
        cls.seed = pair['visual_seed']

    def test_model_and_visual_only_reproduce_v5_prompts_exactly(self):
        for i in range(2):
            for condition in ('model', 'visual_only'):
                source, presented, prompt = render(self.base, i, condition, self.config)
                archived = self.v5[f'{self.seed}_{i}_{condition}']
                self.assertEqual(prompt, archived['input'])
                self.assertEqual(source, archived['source'])
                self.assertEqual(presented, source)

    def test_variants_preserve_values_and_canonical_source(self):
        model_source, _, model_prompt = render(self.base, 0, 'model', self.config)
        for condition in ('analyst', 'external', 'opaque'):
            source, presented, prompt = render(self.base, 0, condition, self.config)
            self.assertEqual(source, model_source)
            self.assertEqual(values(presented), values(model_source))
            self.assertNotEqual(prompt, model_prompt)

    def test_analyst_changes_only_framing(self):
        _, presented, prompt = render(self.base, 0, 'analyst', self.config)
        _, model_presented, model_prompt = render(self.base, 0, 'model', self.config)
        self.assertEqual(presented, model_presented)
        head, model_head = prompt.split('\nUse at most')[0], model_prompt.split('\nUse at most')[0]
        self.assertEqual(head, model_head)
        self.assertIn('third person', prompt)

    def test_opaque_prompt_has_no_attention_access_or_control_vocabulary(self):
        _, presented, prompt = render(self.base, 0, 'opaque', self.config)
        self.assertIsNone(BANNED.search(json.dumps([list(v.keys()) for v in presented])))
        text = prompt.split('\n[')[0]
        self.assertIsNone(BANNED.search(text), BANNED.search(text))

    def test_external_assigns_access_to_outside_camera(self):
        _, presented, prompt = render(self.base, 0, 'external', self.config)
        keys = set(presented[0]) | set(presented[0]['objects'][0]) | set(presented[0]['derived_indexes'])
        for key in keys:
            self.assertIsNone(re.search(r'selection|recoverab|^command_predictions$|selected', key), key)
        self.assertIn('outside camera', prompt)
        self.assertIn("under this system's control", prompt)


if __name__ == '__main__': unittest.main()
