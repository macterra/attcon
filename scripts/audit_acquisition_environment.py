"""Validate acquisition partitions, observable upper comparator, and reward costs."""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from attcon.active_inspection import make_splits, analytic_policy, COSTS


def main():
    torch.set_num_threads(1)
    runs = {}
    for seed in (2309, 2333, 2351):
        splits = make_splits(seed)
        groups = {name: set(data.group.tolist()) for name, data in splits.items()}
        disjoint = all(not a & b for i, a in enumerate(groups.values()) for b in list(groups.values())[i + 1:])
        data = splits['validation']
        oracle = analytic_policy(data)
        runs[str(seed)] = {'disjoint_contexts': disjoint,
            'contexts': {name: len(group) for name, group in groups.items()},
            'dataset_sha256': {name: data.fingerprint() for name, data in splits.items()},
            'validation_analytic_policy': {str(cost): {'realized_return': oracle['return'][data.cost == cost].mean().item(), 'expected_return': oracle['expected_return'][data.cost == cost].mean().item()} for cost in COSTS}}
        if not disjoint:
            raise ValueError('overlapping partitions')
    sources = ('src/attcon/active_inspection.py', 'docs/ACTIVE_INSPECTION_PROTOCOL.md', 'scripts/audit_acquisition_environment.py')
    result = {'audit': 'acquisition_environment', 'runs': runs,
        'source_sha256': {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources},
        'boundary': 'Analytic policy knows sensor probabilities and observable information conditions, not future sample correctness. Validation only; primary and stress outcomes are not evaluated in this scaffold audit.'}
    (ROOT / 'audits/acquisition_environment.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Three partition audits passed.')
if __name__ == '__main__':
    main()
