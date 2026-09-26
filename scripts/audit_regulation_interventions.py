"""Compare report families under the registered state interventions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from attcon.regulation import load_checkpoint, mixed_states, weight_fingerprint
from attcon.regulation_interventions import report_interventions
from attcon.report_sufficiency import make_sufficiency_splits, batch_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    runs = {}
    for path in args.artifacts:
        run = json.loads(path.read_text())
        seed = run['settings']['seed']
        if str(seed) in runs or run['settings']['recipe'] != 'variable' or run['settings']['architecture'] != 'gru':
            raise ValueError('requires unique variable GRU runs')
        saved, agent, config, readouts = load_checkpoint(ROOT / run['checkpoint'])
        fingerprint = weight_fingerprint(agent)
        if fingerprint != run['training']['final_sha256']:
            raise ValueError('checkpoint mismatch')
        splits = make_sufficiency_splits(seed, 512)
        states = {}
        for name in ('report_fit', 'test'):
            if batch_fingerprint(splits[name]) != run['dataset_sha256'][name]:
                raise ValueError('dataset mismatch')
            states[name], _ = mixed_states(agent, splits[name], seed + saved['mixed_delay_offsets'][name])
        results = report_interventions(saved, agent, config, readouts, states['report_fit'], splits['report_fit'], states['test'], splits['test'], seed + 5000)
        if weight_fingerprint(agent) != fingerprint:
            raise ValueError('agent mutated')
        runs[str(seed)] = {'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'results': results}
    sources = ('scripts/audit_regulation_interventions.py', 'src/attcon/regulation_interventions.py', 'src/attcon/report_erasure.py')
    result = {'audit': 'regulation_report_interventions', 'runs': runs,
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        'valid_choice_null_contrasts': all(v['valid'] for r in runs.values() for v in r['results']['choice_null_transplants'].values()),
        'boundary': 'Fitted report heads on frozen state. Choice-preserving perturbations are off-distribution and have no ground-truth access label; sensitivity does not establish introspection or beneficial control.'}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'out': str(args.out), 'valid': result['valid_choice_null_contrasts']}))
if __name__ == '__main__':
    main()
