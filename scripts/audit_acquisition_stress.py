"""Evaluate frozen controllers on reserved contexts and registered stress shifts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from attcon.active_inspection import make_splits
from attcon.acquisition_learning import load_controllers, evaluate_controllers
from attcon.regulation import weight_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    runs = {}
    for path in args.artifacts:
        run = json.loads(path.read_text())
        saved, models = load_controllers(ROOT / run['checkpoint'])
        seed = run['seed']
        if saved['seed'] != seed or str(seed) in runs:
            raise ValueError('mismatched or duplicate seed')
        before = {family: weight_fingerprint(model) for family, model in models.items()}
        if any(before[name] != run['training'][name]['selected_sha256'] for name in models):
            raise ValueError('checkpoint mismatch')
        splits = make_splits(seed)
        baseline = splits['stress']
        if baseline.fingerprint() != run['dataset_sha256']['stress']:
            raise ValueError('reserved data mismatch')
        stress_groups = set(baseline.group.tolist())
        if any(stress_groups & set(data.group.tolist()) for name, data in splits.items() if name != 'stress'):
            raise ValueError('stress contexts overlap earlier partitions')
        degraded = make_splits(seed, reliability=.55)['stress']
        if any(not torch.equal(getattr(baseline, field), getattr(degraded, field)) for field in baseline.__dataclass_fields__ if field != 'sample'):
            raise ValueError('sensor shift changed non-sensor data')
        conditions = {}
        for name, data, delay, reliability in (('baseline', baseline, 1, .75), ('long_delay', baseline, 5, .75), ('degraded_sensor', degraded, 1, .55)):
            conditions[name] = {'delay': delay, 'sensor_reliability': reliability, 'dataset_sha256': data.fingerprint(),
                'realized_sensor_accuracy': (data.sample == data.value).float().mean().item(),
                'evaluation': evaluate_controllers(models, data, seed, delay, reliability)}
        if before != {name: weight_fingerprint(model) for name, model in models.items()}:
            raise ValueError('stress evaluation changed controllers')
        runs[str(seed)] = {'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'reserved_contexts': len(stress_groups), 'disjoint_contexts': True, 'conditions': conditions}
    sources = ('scripts/audit_acquisition_stress.py', 'docs/ACTIVE_INSPECTION_PROTOCOL.md', 'src/attcon/acquisition_learning.py', 'src/attcon/active_inspection.py')
    result = {'audit': 'recurrent_acquisition_stress', 'runs': runs,
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        'boundary': 'Reserved contexts, no refitting or threshold changes. Sensor degradation changes an unobserved generative probability; the analytic comparator knows the new probability but frozen controllers do not. Stress is diagnostic, not new independent-system replication.'}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'out': str(args.out), 'seeds': list(runs)}))
if __name__ == '__main__':
    main()
