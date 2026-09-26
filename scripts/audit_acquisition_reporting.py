"""Fit matched state/action reporters on frozen reward-trained controllers."""
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
from attcon.acquisition_reporting import measurement_data, fit_report, report_scores
from attcon.regulation import weight_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run = json.loads(args.artifact.read_text())
    seed = run['seed']
    saved, controllers = load_controllers(ROOT / run['checkpoint'])
    if saved['seed'] != seed:
        raise ValueError('checkpoint seed mismatch')
    agent = controllers['state']
    fingerprint = weight_fingerprint(agent)
    if fingerprint != run['training']['state']['selected_sha256']:
        raise ValueError('checkpoint weights mismatch')
    splits = make_splits(seed)
    if any(data.fingerprint() != run['dataset_sha256'][name] for name, data in splits.items()):
        raise ValueError('dataset mismatch')
    batches = {name: measurement_data(agent, splits[name]) for name in ('report_fit', 'validation', 'test')}
    models, selections, reports = {}, {}, {}
    for family in ('state', 'action'):
        models[family], selections[family] = fit_report(batches['report_fit'][0][family], batches['report_fit'][1], batches['validation'][0][family], batches['validation'][1], seed + 10000)
        with torch.no_grad():
            predicted = models[family](batches['test'][0][family]).argmax(-1)
        reports[family] = {'all_stages': report_scores(predicted, batches['test'][1]),
            'visited_stages': report_scores(predicted, batches['test'][1], batches['test'][2])}
    nulls, null_selections = [], []
    labels = batches['report_fit'][1]
    for index in range(10):
        null_seed = seed + 11000 + index
        order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(null_seed))
        model, selection = fit_report(batches['report_fit'][0]['state'], labels[order], batches['validation'][0]['state'], batches['validation'][1], null_seed)
        with torch.no_grad():
            nulls.append(report_scores(model(batches['test'][0]['state']).argmax(-1), batches['test'][1])['balanced_accuracy'])
        null_selections.append(selection)
        print(f'reporting {seed}: null {index + 1}/10', flush=True)
    null_p95 = torch.tensor(nulls).quantile(.95).item()
    primary = reports['state']['all_stages']
    gates = {'verified_accuracy': primary['verified_value_accuracy'] >= .90,
        'unverified_accuracy': primary['unverified_accuracy'] >= .90,
        'gain_over_action': primary['balanced_accuracy'] - reports['action']['all_stages']['balanced_accuracy'] >= .02,
        'gain_over_null_p95': primary['balanced_accuracy'] - null_p95 >= .10}
    if weight_fingerprint(agent) != fingerprint:
        raise ValueError('report fitting mutated controller')
    checkpoint = ROOT / 'outputs/acquisition' / (args.out.stem + '.pt')
    torch.save({'models': {name: model.state_dict() for name, model in models.items()}, 'seed': seed, 'agent_sha256': fingerprint}, checkpoint)
    sources = ('src/attcon/acquisition_reporting.py', 'scripts/audit_acquisition_reporting.py', 'docs/ACTIVE_INSPECTION_PROTOCOL.md')
    result = {'audit': 'acquisition_verified_reporting', 'seed': seed, 'source': str(args.artifact),
        'source_artifact_sha256': hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        'checkpoint': str(checkpoint.relative_to(ROOT)), 'agent_sha256': fingerprint,
        'parameters': {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
        'selections': selections, 'reports': reports, 'null_balanced_accuracies': nulls, 'null_p95': null_p95,
        'null_selections': null_selections, 'gates': gates, 'all_gates_pass': all(gates.values()),
        'boundary': 'Verified refers to the environmental source, not internal or subjective access. External supervised readouts fit after reward learning. All-stage metrics include counterfactual inspection histories; visited-stage metrics describe the actual selected controller.'}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'seed': seed, 'gates': gates}))
if __name__ == '__main__':
    main()
