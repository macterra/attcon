"""Test reward-trained inspection policy sensitivity at fixed choice logits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from attcon.inspection import policy_switches
from attcon.regulation import WIDTH, Readout, load_checkpoint, mixed_states, weight_fingerprint
from attcon.regulation_interventions import paired_indices, choice_null_directions, projection_transplant
from attcon.report_sufficiency import make_sufficiency_splits, batch_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = {}
    for path in args.artifacts:
        run = json.loads(path.read_text())
        seed = run['seed']
        if str(seed) in results:
            raise ValueError('duplicate seeds')
        saved_policy = torch.load(ROOT / run['checkpoint'], weights_only=True)
        saved, agent, config, _ = load_checkpoint(ROOT / saved_policy['source_agent_checkpoint'])
        if saved_policy['seed'] != seed or saved['settings']['seed'] != seed or weight_fingerprint(agent) != run['agent_sha256'] or saved_policy['agent_sha256'] != run['agent_sha256']:
            raise ValueError('policy/agent checkpoint mismatch')
        models = {}
        for name, weights in saved_policy['models'].items():
            model = Readout(torch.zeros(2, WIDTH), classes=1, hidden=32)
            model.load_state_dict(weights)
            models[name] = model.eval().requires_grad_(False)
        splits = make_sufficiency_splits(seed, 512)
        states = {}
        for name in ('report_fit', 'test'):
            if batch_fingerprint(splits[name]) != run['dataset_sha256'][name]:
                raise ValueError('dataset mismatch')
            states[name], _ = mixed_states(agent, splits[name], seed + saved['mixed_delay_offsets'][name])
        directions, metadata = choice_null_directions(agent.choice.weight, states['report_fit'], splits['report_fit'], seed + 5000)
        seen, unseen = paired_indices(splits['test'])
        recipient, donor = states['test'][seen], states['test'][unseen]
        reference = projection_transplant(recipient, donor, directions['access'])
        interventions = {}
        for kind, direction in directions.items():
            delta = reference if kind == 'access' else projection_transplant(recipient, donor, direction, reference)
            changed = recipient + delta
            switches = policy_switches(models, agent, recipient, changed)
            residual = (agent.choice(changed) - agent.choice(recipient)).abs().max().item()
            invariant = all(switches[name]['max_feature_residual'] <= 1e-5 and all(c['switch_rate'] == 0 for c in switches[name]['costs'].values()) for name in ('action', 'confidence'))
            restored = policy_switches(models, agent, recipient, changed - delta)
            restoration_valid = all(all(c['switch_rate'] == 0 for c in value['costs'].values()) for value in restored.values())
            interventions[kind] = {'valid': residual <= 1e-5 and invariant and restoration_valid,
                'max_logit_residual': residual, 'mean_delta_norm': delta.norm(dim=1).mean().item(),
                'policies': switches, 'restoration': restored}
        results[str(seed)] = {'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'choice_null_fit': metadata, 'interventions': interventions,
            'state_switch_advantage_over_random': {cost: interventions['access']['policies']['state']['costs'][cost]['switch_rate'] - interventions['random']['policies']['state']['costs'][cost]['switch_rate'] for cost in ('0.2', '0.4', '0.6')}}
    sources = ('scripts/audit_inspection_interventions.py', 'src/attcon/inspection.py', 'src/attcon/regulation_interventions.py')
    result = {'audit': 'inspection_choice_preserving_interventions', 'runs': results,
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        'all_contrasts_valid': all(v['valid'] for run in results.values() for v in run['interventions'].values()),
        'boundary': 'Synthetic off-distribution interventions have no ground-truth reward correction. Switching measures sensitivity, not benefit. Reward-trained policies are external and one-step, not native recurrent attention regulation.'}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'out': str(args.out), 'valid': result['all_contrasts_valid']}))
if __name__ == '__main__':
    main()
