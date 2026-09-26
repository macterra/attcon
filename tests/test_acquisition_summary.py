import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('summary', ROOT / 'scripts/summarize_acquisition.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AcquisitionSummaryTests(unittest.TestCase):
    def setUp(self):
        template = json.loads((ROOT / 'audits/acquisition_seed2309.json').read_text())
        self.runs = [copy.deepcopy(template) for _ in range(3)]
        for index, run in enumerate(self.runs):
            run['seed'] = index

    def summarize(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, run in enumerate(self.runs):
                path = Path(directory) / f'{index}.json'
                path.write_text(json.dumps(run))
                paths.append(path)
            return module.summarize(paths)

    def test_failed_comparator_gate_remains_failed(self):
        self.assertFalse(self.summarize()['all_seeds_supported'])

    def test_duplicate_seed_rejected(self):
        self.runs[1]['seed'] = self.runs[0]['seed']
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.summarize()

    def test_forged_gate_rejected(self):
        self.runs[0]['test']['costs']['0.1']['gates']['gain_over_action'] = True
        with self.assertRaisesRegex(ValueError, 'unearned'):
            self.summarize()

    def test_changed_initialization_rejected(self):
        self.runs[0]['training']['action']['initial_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'unmatched'):
            self.summarize()

if __name__ == '__main__':
    unittest.main()
