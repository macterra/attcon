"""Registered reward-only inspection fitting and held-out evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from attcon.inspection import select_policy, policy_features, immediate_rewards, evaluate_policies
from attcon.regulation import load_checkpoint, mixed_states, weight_fingerprint
from attcon.report_delay import insert_delay
from attcon.report_sufficiency import make_sufficiency_splits, batch_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run = json.loads(args.artifact.read_text())
    seed = run['settings']['seed']
    if run['settings']['recipe'] != 'variable' or run['settings']['architecture'] != 'gru':
        raise ValueError('requires variable-delay GRU')
    saved, agent, config, _ = load_checkpoint(ROOT / run['checkpoint'])
    initial = weight_fingerprint(agent)
    if initial != run['training']['final_sha256']:
        raise ValueError('checkpoint fingerprint mismatch')
    splits = make_sufficiency_splits(seed, 512)
    states, features, rewards = {}, {}, {}
    for name in ('report_fit', 'validation', 'test'):
        if batch_fingerprint(splits[name]) != run['dataset_sha256'][name]:
            raise ValueError('dataset fingerprint mismatch')
        states[name], _ = mixed_states(agent, splits[name], seed + saved['mixed_delay_offsets'][name])
        features[name] = policy_features(agent, states[name])
        rewards[name] = immediate_rewards(agent, states[name], splits[name].value)
    models, selections = {}, {}
    for name in ('state', 'action', 'confidence'):
        models[name], selections[name] = select_policy(features['report_fit'][name], rewards['report_fit'], features['validation'][name], rewards['validation'], seed=seed + 6000)
    null_models, null_selections = [], []
    for index in range(20):
        null_seed = seed + 7000 + index
        order = torch.randperm(len(rewards['report_fit']), generator=torch.Generator().manual_seed(null_seed))
        model, selected = select_policy(features['report_fit']['state'], rewards['report_fit'][order], features['validation']['state'], rewards['validation'], seed=null_seed)
        null_models.append(model)
        null_selections.append(selected)
        if (index + 1) % 5 == 0:
            print(f'inspection {seed}: nulls {index + 1}/20', flush=True)
    test = evaluate_policies(models, features['test'], rewards['test'], splits['test'], seed, null_models)
    with torch.no_grad():
        delayed = agent.state(insert_delay(splits['test'], 9).events)
    delay_rewards = immediate_rewards(agent, delayed, splits['test'].value)
    delay9 = evaluate_policies(models, policy_features(agent, delayed), delay_rewards, splits['test'], seed, null_models)
    seen_accuracy = rewards['test'][splits['test'].seen].mean().item()
    gates = {'task_viable': seen_accuracy >= 0.85,
        **{f'cost_{cost}': value['all_gates_pass'] for cost, value in test['costs'].items()}}
    if weight_fingerprint(agent) != initial:
        raise ValueError('agent mutated during policy fitting')
    checkpoint = ROOT / 'outputs/regulation' / (args.out.stem + '.pt')
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'models': {name: model.state_dict() for name, model in models.items()}, 'seed': seed,
        'agent_sha256': initial, 'source_agent_checkpoint': run['checkpoint'], 'selections': selections}, checkpoint)
    sources = ('src/attcon/inspection.py', 'scripts/train_inspection.py', 'src/attcon/regulation.py', 'docs/REGULATION_PROTOCOL.md')
    result = {'audit': 'reward_trained_inspection_v1', 'seed': seed, 'source': str(args.artifact),
        'source_artifact_sha256': hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        'checkpoint': str(checkpoint.relative_to(ROOT)), 'agent_sha256': initial,
        'dataset_sha256': run['dataset_sha256'], 'policy_parameters': {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
        'selections': selections, 'null_selections': null_selections, 'seen_choice_accuracy': seen_accuracy,
        'test': test, 'delay9': delay9, 'gates': gates,
        'status': 'bounded_inspection_advantage' if all(gates.values()) else 'inspection_advantage_gates_not_met',
        'boundary': 'External offline one-step policies use environmental correctness rewards, no report or access labels. Historical availability is used only for evaluation breakdowns. Context intervals are conditional and pointwise. Delay 9 is extrapolation. Synthetic task only; no endogenous recurrent attention or Stage 8 upgrade.'}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'seed': seed, 'status': result['status'], 'gates': gates}))
if __name__ == '__main__':
    main()
