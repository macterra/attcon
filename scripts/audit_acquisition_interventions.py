"""Audit recurrent-history dependence without changing environment rewards."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from attcon.active_inspection import make_splits
from attcon.acquisition_learning import load_controllers
from attcon.acquisition_interventions import intervene
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
        if str(seed) in runs or saved['seed'] != seed:
            raise ValueError('duplicate/mismatched seed')
        splits = make_splits(seed)
        if any(data.fingerprint() != run['dataset_sha256'][name] for name, data in splits.items()):
            raise ValueError('dataset fingerprint mismatch')
        if weight_fingerprint(models['state']) != run['training']['state']['selected_sha256']:
            raise ValueError('checkpoint mismatch')
        results = intervene(models['state'], splits['report_fit'], splits['test'], seed + 9100)
        runs[str(seed)] = {'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'results': results}
    sources = ('scripts/audit_acquisition_interventions.py', 'src/attcon/acquisition_interventions.py')
    result = {'audit': 'recurrent_acquisition_interventions', 'runs': runs,
        'all_contrasts_valid': all(run['results']['all_contrasts_valid'] for run in runs.values()),
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        'boundary': 'Environmental correctness and inspection costs remain defined after synthetic state changes. Return changes measure policy consequences, not the accuracy of an introspective access label. Availability directions are diagnostic fits only, never controller training targets.'}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'out': str(args.out), 'valid': result['all_contrasts_valid']}))
if __name__ == '__main__':
    main()
