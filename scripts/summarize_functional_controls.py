"""Post-hoc exports of the frozen controls assessment; never rescore or retrain."""
import csv
import gzip
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from attcon.predictive_attention import Forecast, PredictiveAttention
from attcon.functional_controls import CONDITIONS, WINDOWS, control_mask, true_control_mask
from run_functional_controls import ROOT, SEEDS, check_sources, digest, save_trace


def csv_export(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    manifest = json.loads((ROOT/'manifest.json').read_text())
    check_sources(manifest)
    records, static_rows, revision_rows, failures, cases = {}, [], [], [], []
    supplements = {}
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    colors = ('#0072B2', '#D55E00', '#009E73')
    labels = {0:'buffer 0', 1:'buffer 1', -1:'neither'}
    for seed, color in zip(SEEDS, colors):
        path = ROOT/f'seed{seed}.json'; record = json.loads(path.read_text())
        assert record['status']=='completed' and record['seed']==seed
        assert record['source_sha256']==manifest['source_sha256']
        cp = ROOT/f'seed{seed}.pt'; archive = ROOT/f'seed{seed}_trace.pt.gz'
        assert digest(cp)==record['checkpoint_sha256']
        assert digest(archive)==record['trace_sha256']
        with gzip.open(archive, 'rb') as stream:
            trace = torch.load(io.BytesIO(stream.read()), weights_only=True)
        model = PredictiveAttention()
        model.load_state_dict(torch.load(cp, weights_only=True)['state_dict']); model.eval()
        tail_archive = {}
        assessment = record['assessment']; records[seed] = assessment
        for condition in CONDITIONS:
            static = assessment['static'][str(condition)]
            static_rows.append({'seed':seed, 'condition':condition, **static['metrics'],
                                'all_gates_pass':static['all_gates_pass']})
            # Original static archive exposes final modeled heads explicitly. This
            # supplement names all five scored observations without altering it.
            source = trace['static'][condition]
            seq, _ = model(source['observations'])
            tail = Forecast(*(value[:, 7:].clone().contiguous() for value in
                              (seq.allocation, seq.access, seq.effects)))
            recomputed = {
                'allocation_accuracy':float((tail.allocation.argmax(-1)==source['allocation'][:,7:].argmax(-1)).double().mean()),
                'effect_accuracy':float((tail.effects.argmax(-1)==source['physical_effects'][:,7:].argmax(-1)).double().mean()),
                'three_way_control_accuracy':float((control_mask(tail)==true_control_mask(torch.full((512,),condition))[:,None]).all(-1).double().mean()),
                'recovery_mae':float((tail.access-source['access'][:,7:]).abs().double().mean()),
            }
            for name, value in recomputed.items(): assert value==static['metrics'][name], (seed, condition, name)
            assert torch.equal(tail.effects[:,-1],source['current_effects'])
            tail_archive[condition] = {'observation_numbers':torch.arange(8,13),
                'modeled_allocation':tail.allocation, 'modeled_access':tail.access,
                'modeled_effects':tail.effects}
        supplement = ROOT/f'seed{seed}_static_tail.pt.gz'
        if not supplement.exists(): save_trace(tail_archive, supplement)
        with gzip.open(supplement, 'rb') as stream:
            saved = torch.load(io.BytesIO(stream.read()), weights_only=True)
        for c in CONDITIONS:
            for key in tail_archive[c]: assert torch.equal(saved[c][key],tail_archive[c][key])
        supplements[seed] = {'kind':'post_hoc_named_static_forecast_export',
            'checkpoint_sha256':digest(cp), 'original_record_sha256':digest(path),
            'original_trace_sha256':digest(archive), 'file':str(supplement),
            'sha256':digest(supplement), 'original_scores_unchanged':True}
        for old_index, old in enumerate(CONDITIONS):
            for new_index, new in enumerate(CONDITIONS):
                route = f'{old}->{new}'; result = assessment['revision'][route]
                route_trace = trace['revision'][route]
                for gate, passed in result['final_gates'].items():
                    if not passed: failures.append({'seed':seed,'route':route,'gate':gate,
                        'value':result['windows'][-1]['metrics'].get(gate, result.get(gate)),
                        'registered_minimum':.99 if gate.endswith('accuracy') else None})
                for window in result['windows']:
                    revision_rows.append({'seed':seed,'old_condition':old,'new_condition':new,
                        'observations':window['observations'], **window['metrics']})
                ax = axes[old_index, new_index]
                values = [w['metrics']['feedback_control_accuracy'] for w in result['windows']]
                ax.plot(WINDOWS, values, 'o-', color=color, label=str(seed), markersize=3)
                if seed==SEEDS[0]:
                    ax.set_title(f'{labels[old]} → {labels[new]}')
                    ax.axhline(.99, color='gray', linestyle=':', linewidth=1)
                    ax.set_xscale('log', base=2); ax.set_xticks(WINDOWS, labels=WINDOWS)
                    ax.set_ylim(-.03,1.03); ax.grid(alpha=.15)
                final = route_trace['windows'][16]
                forecast = Forecast(torch.empty(0),torch.empty(0),final['feedback_effects'])
                tv = forecast.controllability(); mask = control_mask(forecast)
                truth = true_control_mask(torch.full((512,), new))
                wrong = (mask!=truth).any(-1).nonzero().flatten().tolist()
                for episode in wrong:
                    cases.append({'seed':seed,'route':route,'episode':episode,
                        'buffer0_TV':float(tv[episode,0]),'buffer1_TV':float(tv[episode,1]),
                        'buffer0_classified_controlled':bool(mask[episode,0]),
                        'buffer1_classified_controlled':bool(mask[episode,1]),
                        'expected_buffer0_controlled':bool(truth[episode,0]),
                        'expected_buffer1_controlled':bool(truth[episode,1]),
                        'threshold':.75})
    for ax in axes[-1]: ax.set_xlabel('Actual post-switch observations')
    for ax in axes[:,0]: ax.set_ylabel('Correct new control mask (episodes)')
    axes[0,0].legend(title='Model seed', loc='lower right', fontsize=8)
    fig.suptitle('Feedback revision: all nine routes, each fixed final model\nDotted line: registered 99% final-window minimum; intermediate windows descriptive')
    fig.tight_layout(rect=(0,0,1,.94))
    for extension in ('png','svg'): fig.savefig(ROOT/f'control_revision.{extension}', dpi=180)
    plt.close(fig)
    # Optional columns occur only for no-control static cases.
    fields = list(dict.fromkeys(key for row in static_rows for key in row))
    static_rows = [{key:row.get(key,'') for key in fields} for row in static_rows]
    csv_export(ROOT/'static_metrics.csv', static_rows)
    csv_export(ROOT/'revision_metrics.csv', revision_rows)
    if cases: csv_export(ROOT/'final_control_mismatches.csv', cases)
    result = {'kind':'post_hoc_aggregate_of_frozen_engineering_assessment',
        'all_gates_pass':all(a['all_gates_pass'] for a in records.values()),
        'models_passing':sum(a['all_gates_pass'] for a in records.values()),
        'models_total':len(records), 'static_cases_passing':sum(s['all_gates_pass'] for a in records.values() for s in a['static'].values()),
        'static_cases_total':9, 'revision_routes_passing':sum(r['all_gates_pass'] for a in records.values() for r in a['revision'].values()),
        'revision_routes_total':27, 'failed_gates':failures,
        'post_hoc_mismatches':len(cases), 'supplements':supplements,
        'source_sha256':digest(Path(__file__)), 'no_new_language_reports':True,
        'no_retraining_or_gate_changes':True}
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('supplements',)},indent=2))


if __name__=='__main__': main()
