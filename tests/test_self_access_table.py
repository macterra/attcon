import json
import sys
import unittest
from pathlib import Path
import torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'scripts'))
from self_access_table_reports import counterfactual_table, conditions, with_table, TABLE_GLOSSARY
from self_access_table_gates import decide
from specificity_reports import build_states
from tests.test_self_coupled import case

CONFIG = json.loads((REPO/'configs/bound_content/self_access_table_v1.json').read_text())
CHECKPOINT = REPO/'audits/bound_content/confirmation_v2/seed1301.pt'


@unittest.skipUnless(CHECKPOINT.exists(), 'checkpoints not present')
class TableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pair = CONFIG['pairs'][0]
        cls.base, _, _ = build_states(pair, CONFIG)
        cls.table = counterfactual_table(pair, CONFIG)
        cls.conds = conditions(cls.base, cls.table)

    def test_table_varies_by_command_in_controlled_view_only(self):
        spread = self.table.max(1).values-self.table.min(1).values  # episode, view, slot
        self.assertGreater(spread.max(-1).values.max(-1).values.min().item(), .2)
        self.assertLess(spread.max(-1).values.min(-1).values.max().item(), .2)

    def test_flat_tables_and_independence(self):
        for name in ('independent_table', 'coupled_table_swapped'):
            t = self.conds[name][1]
            self.assertTrue(torch.allclose(t, t[:, :1].expand_as(t)))
        self.assertTrue(torch.equal(self.conds['coupled_table'][0].visual, self.conds['coupled_table_swapped'][0].visual))
        self.assertTrue(torch.equal(self.conds['coupled_table'][0].visual, self.conds['coupled_no_table'][0].visual))
        self.assertFalse(torch.allclose(self.conds['coupled_table'][0].visual, self.conds['independent_table'][0].visual))

    def test_rendering_adds_table_and_one_glossary_sentence(self):
        state, table = self.conds['coupled_table']
        source, presented, prompt = with_table(state, table, 0, CONFIG)
        plain = with_table(*self.conds['coupled_no_table'], 0, CONFIG)[2]
        self.assertEqual(prompt.count(TABLE_GLOSSARY), 1)
        self.assertEqual(prompt.split('\n[{')[0].replace(TABLE_GLOSSARY, ''), plain.split('\n[{')[0])
        self.assertIn('own_content_by_command', source[0]); self.assertEqual(source, presented)
        self.assertAlmostEqual(source[0]['own_content_by_command']['left']['upper'], round(float(table[0, 3, 0, 0]), 5))


def study(other, hits):
    cases = []
    for k in range(24):
        seed, episode = (1301, 1311, 1321)[k//8], k % 8
        cases += [case(seed, episode, 'coupled_table', True), case(seed, episode, other, k < hits)]
    return cases


class TableGateTests(unittest.TestCase):
    def test_primary_verdicts(self):
        self.assertEqual(decide(study('independent_table', 12))['verdict'], 'self_access_reporting_supported')
        self.assertEqual(decide(study('independent_table', 20))['verdict'], 'self_access_reporting_not_supported')


if __name__ == '__main__': unittest.main()


class ExtractorV12Tests(unittest.TestCase):
    def test_v11_instruction_preserved_with_one_addition(self):
        import extract_bound_prose_v11 as v11, extract_bound_prose_v12 as v12
        self.assertEqual(v12.INSTRUCTION.replace(v12.ADDITION, ''), v11.INSTRUCTION)
        self.assertEqual(v12.SCHEMA, v11.SCHEMA); self.assertEqual(v12.MODEL, v11.MODEL)

    def test_counterfactual_rule_and_blocking(self):
        import extract_bound_prose_v12 as v12
        from self_access_table_reports import blocking_v12
        claim = {'focal': None, 'most_recoverable': True, 'access_trend': None}
        self.assertFalse(v12.assess_counterfactual('x.', {'parsed': {'claims': [claim]}}))
        self.assertTrue(v12.assess_counterfactual('x.', {'parsed': {'claims': [dict(claim, most_recoverable=None)]}}))
        results = [{'id': 'a', 'kind': 'v10_claim', 'passed': False}, {'id': 'b', 'kind': 'counterfactual', 'passed': False}]
        self.assertEqual(blocking_v12(results), ['b'])
