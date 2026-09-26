"""Train all registered recurrent acquisition comparators for one seed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from attcon.active_inspection import make_splits
from attcon.acquisition_learning import train_controller, evaluate_controllers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    splits = make_splits(args.seed)
    models, training = {}, {}
    for family in ('state', 'action', 'confidence'):
        models[family], training[family] = train_controller(splits['train'], splits['validation'], family, args.seed)
    if len({v['initial_sha256'] for v in training.values()}) != 1 or len({v['parameters'] for v in training.values()}) != 1:
        raise ValueError('unmatched initialization or parameters')
    result = evaluate_controllers(models, splits['test'], args.seed)
    checkpoint = ROOT / 'outputs/acquisition' / (args.out.stem + '.pt')
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    fingerprints = {name: data.fingerprint() for name, data in splits.items()}
    torch.save({'seed': args.seed, 'models': {name: model.state_dict() for name, model in models.items()},
        'training': training, 'dataset_sha256': fingerprints}, checkpoint)
    sources = ('src/attcon/active_inspection.py', 'src/attcon/acquisition_learning.py', 'scripts/train_acquisition.py', 'docs/ACTIVE_INSPECTION_PROTOCOL.md')
    artifact = {'audit': 'recurrent_acquisition_v1', 'seed': args.seed, 'training': training,
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        'dataset_sha256': fingerprints, 'checkpoint': str(checkpoint.relative_to(ROOT)), 'test': result,
        'status': 'bounded_recurrent_acquisition_advantage' if result['all_gates_pass'] else 'acquisition_advantage_gates_not_met',
        'boundary': 'Representations and inspection values train from answer rewards and replayed transitions, with no reporting/access supervision. This is fitted control, not on-policy discovery of exploration or evidence of conscious experience. Stage 8 remains unchanged.'}
    args.out.write_text(json.dumps(artifact, indent=2) + '\n')
    print(json.dumps({'seed': args.seed, 'status': artifact['status']}))
if __name__ == '__main__':
    main()
