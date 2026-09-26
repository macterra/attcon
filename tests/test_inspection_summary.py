import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('summarize_inspection', ROOT / 'scripts/summarize_inspection.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(seed):
    gates = {'gain_over_action': True, 'gain_over_confidence': True,
             'positive_bound_over_action': True, 'positive_bound_over_confidence': True,
             'beats_never_inspect': True, 'beats_always_inspect': True, 'gain_over_null_p95': True}
    cost = {'policies': {name: {'return': value} for name, value in [('state', .9), ('action', .85), ('confidence', .84)]},
        'state_minus_baseline_intervals': {name: {'mean': .05, 'low': .01} for name in ('action', 'confidence', 'never_inspect', 'always_inspect')},
        'null_p95': .7, 'gates': gates, 'all_gates_pass': True}
    return {'audit': 'test', 'seed': seed, 'source_sha256': {'code': 'same'}, 'policy_parameters': {'state': 4161},
        'seen_choice_accuracy': .95, 'test': {'costs': {key: copy.deepcopy(cost) for key in ('0.2', '0.4', '0.6')}},
        'gates': {'task_viable': True, 'cost_0.2': True, 'cost_0.4': True, 'cost_0.6': True}}


class InspectionSummaryTests(unittest.TestCase):
    def summarize(self, runs):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, run in enumerate(runs):
                path = Path(directory) / f'{index}.json'
                path.write_text(json.dumps(run))
                paths.append(path)
            return module.summarize(paths)

    def test_duplicate_seeds_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unique seeds'):
            self.summarize([fixture(1)] * 3)

    def test_unearned_gate_pass_rejected(self):
        runs = [fixture(i) for i in range(3)]
        runs[0]['test']['costs']['0.4']['state_minus_baseline_intervals']['action']['mean'] = -.01
        with self.assertRaisesRegex(ValueError, 'cost gates'):
            self.summarize(runs)

    def test_one_failed_cost_is_retained(self):
        runs = [fixture(i) for i in range(3)]
        cost = runs[0]['test']['costs']['0.4']
        cost['state_minus_baseline_intervals']['action']['mean'] = .01
        cost['gates']['gain_over_action'] = False
        cost['all_gates_pass'] = False
        runs[0]['gates']['cost_0.4'] = False
        result = self.summarize(runs)
        self.assertEqual(result['status'], 'inspection_advantage_gates_not_met')
        self.assertEqual(result['gate_pass_counts']['cost_0.4'], 2)

    def test_source_mismatch_rejected(self):
        runs = [fixture(i) for i in range(3)]
        runs[0]['source_sha256'] = {'code': 'changed'}
        with self.assertRaisesRegex(ValueError, 'source_sha256'):
            self.summarize(runs)

if __name__ == '__main__':
    unittest.main()
