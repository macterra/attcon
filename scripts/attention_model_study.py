"""Registered attention-model report study: train, evaluate, or replay one seed."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from attcon.attention_model_reporting import (SEEDS, FAMILIES, SPLITS, STEPS, INTERVENTION_STEP,
    Reporter, capture, make_agent, make_contexts, fingerprint, fit_reporter,
    scores, per_case, scene_bootstrap, fidelity_gates, renderer, parse_rendered, render_transition)

SOURCES = ('src/attcon/attention_model_reporting.py', 'src/attcon/models.py', 'src/attcon/data.py',
           'scripts/attention_model_study.py', 'docs/ATTENTION_MODEL_STATE_SPEC.md',
           'docs/ATTENTION_MODEL_PHENOMENOLOGY_PROTOCOL.md')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value


def selected(trace, step=INTERVENTION_STEP):
    return {k: trace[k][:, step] for k in ('m', 'belief', 'preference', 'physical', 'attention')}


def prediction_at(pred, step=INTERVENTION_STEP):
    return {k: v[:, step] for k, v in pred.items()}


def intervals(pred, truth, seed):
    c = per_case(pred, truth)
    result = {key: scene_bootstrap(c[key], seed) for key in ('preference', 'exact_map', 'exact_report')}
    # Cluster-resample scenes; recompute both cell recalls within each draw.
    correct, positive = c['belief_correct'], c['positive']
    correct = correct.reshape(len(correct), -1)
    positive = positive.reshape(len(positive), -1)
    counts = torch.stack(((correct & positive).sum(1), positive.sum(1),
                          (correct & ~positive).sum(1), (~positive).sum(1)), 1).float()
    idx = torch.randint(len(counts), (1000, len(counts)), generator=torch.Generator().manual_seed(8100 + seed))
    totals = counts[idx].sum(1)
    balanced = .5 * (totals[:, 0] / totals[:, 1].clamp_min(1) + totals[:, 2] / totals[:, 3].clamp_min(1))
    result['balanced_belief'] = [float(torch.quantile(balanced, .025)), float(torch.quantile(balanced, .975))]
    return result


def correspondence(pred, trace):
    pred_shift = pred['preference'][:, 1:] != pred['preference'][:, :-1]
    true_shift = trace['preference'][:, 1:] != trace['preference'][:, :-1]
    pred_persist = pred['belief'][:, 1:] & pred['belief'][:, :-1]
    true_persist = trace['belief'][:, 1:] & trace['belief'][:, :-1]
    disagree = trace['belief'] != trace['physical']
    return {'preference_shift_agreement': (pred_shift == true_shift).float().mean().item(),
            'shift_positive_count': int(true_shift.sum()), 'transition_count': true_shift.numel(),
            'belief_persistence_agreement': (pred_persist == true_persist).float().mean().item(),
            'persistence_positive_count': int(true_persist.sum()),
            'model_physical_disagreement_count': int(disagree.sum()),
            'disagreement_report_accuracy': (pred['belief'][disagree] == trace['belief'][disagree]).float().mean().item() if disagree.any() else None,
            'boundary': 'Operational relations only; the report vocabulary and relational definitions are authored.'}


def evaluation(agent, reporters, contexts, seed):
    trace = capture(agent, contexts)
    predictions = {name: reporter.report(trace['features'][name]) for name, reporter in reporters.items()}
    primary, cases = {}, {'ordinary': {'context_ids': contexts.ids, 'cues': contexts.cues,
                                      'truth': {k: trace[k] for k in ('m', 'belief', 'preference', 'physical', 'attention')},
                                      'predictions': predictions}}
    for name, pred in predictions.items():
        metric = scores(pred, trace)
        primary[name] = {'metrics': metric, 'gates': fidelity_gates(metric), 'ci95': intervals(pred, trace, seed),
                         'correspondence': correspondence(pred, trace),
                         'by_step': [scores(prediction_at(pred, t), selected(trace, t)) for t in range(STEPS)],
                         'by_regime': {regime: scores({k: v[mask] for k, v in pred.items()},
                                                      {k: trace[k][mask] for k in ('belief', 'preference')})
                                       for regime, mask in {'switch': contexts.cues[:, 0] != contexts.cues[:, -1],
                                                           'fixed': contexts.cues[:, 0] == contexts.cues[:, -1]}.items()}}
    baseline = selected(trace)
    baseline_predictions = {name: prediction_at(pred) for name, pred in predictions.items()}
    n = len(contexts.ids)
    donor_index = torch.arange(n).roll(1)
    donor = baseline['m'][donor_index]
    flip = baseline['m'].clone()
    q = torch.arange(n) % 25
    flip[torch.arange(n), q] = 1 - flip[torch.arange(n), q]
    overrides = {'donor': donor, 'flip': flip, 'erase': torch.full_like(donor, .5)}
    interventions = {}
    for kind, override in overrides.items():
        changed = capture(agent, contexts, override)
        truth = selected(changed)
        predictions_changed = {name: reporter.report(changed['features'][name][:, INTERVENTION_STEP])
                               for name, reporter in reporters.items()}
        # Exact replay of pre-intervention context after removing the override.
        restored = capture(agent, contexts)
        restored_ok = all(torch.equal(trace[k], restored[k]) for k in ('m', 'attention', 'belief', 'preference'))
        model_restore = all(torch.equal(predictions[name][k], reporters[name].report(restored['features'][name])[k])
                            for name in FAMILIES for k in ('belief', 'preference'))
        invariant = all(torch.equal(trace[k][:, INTERVENTION_STEP], changed[k][:, INTERVENTION_STEP])
                        for k in ('hidden', 'physical')) and torch.equal(trace['attention'][:, :3], changed['attention'][:, :3])
        per_reporter = {}
        for name, pred in predictions_changed.items():
            both = per_case(baseline_predictions[name], baseline)['exact_report'] & per_case(pred, truth)['exact_report']
            unchanged = truth['belief'] == baseline['belief']
            stable = pred['belief'] == baseline_predictions[name]['belief']
            per_reporter[name] = {'metrics': scores(pred, truth),
                                  'paired_exact_accuracy': both.float().mean().item(),
                                  'paired_exact_ci95': scene_bootstrap(both, seed),
                                  'unchanged_belief_preservation': stable[unchanged].float().mean().item() if unchanged.any() else None,
                                  'preference_change_rate': (pred['preference'] != baseline_predictions[name]['preference']).float().mean().item(),
                                  'belief_change_rate': (pred['belief'] != baseline_predictions[name]['belief']).float().mean().item()}
        delta = (truth['attention'] - baseline['attention']).abs().mean().item()
        interventions[kind] = {'prior_trace_and_hidden_invariant': invariant, 'restoration_exact': restored_ok and model_restore,
                                'max_override_roundtrip_error': float((truth['m'] - override).abs().max()),
                                'mean_absolute_attention_change': delta,
                                'physical_next_choice_switch_rate': (truth['attention'].argmax(-1) != baseline['attention'].argmax(-1)).float().mean().item(),
                                'model_preference_change_rate': (truth['preference'] != baseline['preference']).float().mean().item(),
                                'reporters': per_reporter}
        cases[kind] = {'truth': truth, 'predictions': predictions_changed, 'donor_indexes': donor_index if kind == 'donor' else None,
                       'flipped_cells': q if kind == 'flip' else None}
    # Information-loss controls scored against the recipient's original model.
    m_features = trace['features']['state']
    state_reporter = reporters['state']
    null_inputs = {'shuffled': m_features[donor_index], 'zero': torch.zeros_like(m_features),
                   'constant': torch.cat((torch.full_like(m_features[..., :25], .5), torch.zeros_like(m_features[..., 25:])), -1)}
    nulls = {}
    for name, feature in null_inputs.items():
        pred = state_reporter.report(feature)
        supplied_m = feature[..., :25]
        supplied_truth = {'belief': supplied_m >= .5, 'preference': agent.policy_self_model_head(supplied_m).argmax(-1)}
        nulls[name] = {'against_original': scores(pred, trace), 'against_supplied': scores(pred, supplied_truth)}
        cases['null_' + name] = {'predictions': pred, 'supplied_truth': supplied_truth}
    outside = {}
    for name, reporter in reporters.items():
        feature = trace['features'][name] if name in ('state', 'hidden') else trace['features'][name][donor_index]
        pred = reporter.report(feature)
        outside[name] = {'metrics_against_fixed_model': scores(pred, trace),
                         'report_preserved': float(((pred['belief'] == predictions[name]['belief']).all(-1) &
                                                    (pred['preference'] == predictions[name]['preference'])).float().mean())}
        cases['outside_' + name] = {'predictions': pred}
    roundtrip = True
    # Check every ordinary state report, including errors, not just selected examples.
    pred = predictions['state']
    for b, f in zip(pred['belief'].flatten(0, 1), pred['preference'].flatten()):
        parsed = parse_rendered(renderer(b, f))
        roundtrip = roundtrip and torch.equal(parsed['belief'], b) and parsed['preference'] == int(f)
    exact = per_case(pred, trace)['exact_report']
    error = torch.where(~exact.flatten())[0]
    indexes = [0] + ([int(error[0])] if len(error) and int(error[0]) != 0 else [])
    examples = []
    for index in indexes:
        scene, step = divmod(index, STEPS)
        examples.append({'selection': 'first_case' if index == 0 else 'first_error', 'scene': scene, 'step': step,
                         'model_report': renderer(trace['belief'][scene, step], trace['preference'][scene, step]),
                         'learned_report': renderer(pred['belief'][scene, step], pred['preference'][scene, step]),
                         'physical_inspected': torch.where(trace['physical'][scene, step])[0].tolist(),
                         'actual_next_cell': int(trace['attention'][scene, step].argmax()),
                         'illustrative_first_person': 'My model of attention favors cell ' + str(int(pred['preference'][scene, step])) + '.',
                         'wording_is_authored_not_independent_phenomenology': True})
    # One trace includes all successive model states and learned/physical reports.
    example_trace = [{'step': t, 'model': renderer(trace['belief'][0, t], trace['preference'][0, t]),
                      'learned': renderer(pred['belief'][0, t], pred['preference'][0, t]),
                      'physical_inspected': torch.where(trace['physical'][0, t])[0].tolist(),
                      'next_cell': int(trace['attention'][0, t].argmax()),
                      'learned_transition_report': render_transition(pred['belief'][0,t-1], pred['preference'][0,t-1], pred['belief'][0,t], pred['preference'][0,t]) if t else None,
                      'preference_shift': bool(trace['preference'][0, t] != trace['preference'][0, t-1]) if t else None}
                     for t in range(STEPS)]
    cor = primary['state']['correspondence']
    gates = {'primary': all(primary['state']['gates'].values()),
             'paired_donor': interventions['donor']['reporters']['state']['paired_exact_accuracy'] >= .90,
             'paired_flip': interventions['flip']['reporters']['state']['paired_exact_accuracy'] >= .90,
             'paired_erase': interventions['erase']['reporters']['state']['paired_exact_accuracy'] >= .90,
             'flip_preservation': interventions['flip']['reporters']['state']['unchanged_belief_preservation'] >= .95,
             'causal_identity': interventions['donor']['mean_absolute_attention_change'] > 1e-6,
             'isolation_restoration': all(v['prior_trace_and_hidden_invariant'] and v['restoration_exact'] for v in interventions.values()),
             'outside_invariance': outside['state']['report_preserved'] == 1.,
             'shift_agreement': cor['preference_shift_agreement'] >= .95,
             'persistence_agreement': cor['belief_persistence_agreement'] >= .95,
             'model_disagreement': cor['disagreement_report_accuracy'] is not None and cor['disagreement_report_accuracy'] >= .95,
             'renderer_roundtrip': roundtrip}
    return {'primary': primary, 'interventions': interventions, 'information_loss': nulls, 'outside_information': outside,
            'gates': gates, 'full_fidelity_supported': all(v for k,v in gates.items() if k not in ('shift_agreement', 'persistence_agreement', 'model_disagreement')),
            'structural_correspondences_supported': all(gates[k] for k in ('shift_agreement', 'persistence_agreement', 'model_disagreement')),
            'theory_verdict': 'underdetermined: supervised field decoding and authored relations; missing phenomenological mechanisms',
            'examples': examples, 'first_episode_trace': example_trace}, plain(cases)


def run(seed, replay=False, smoke=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    outdir = ROOT / ('outputs/attention_model_smoke' if smoke else 'audits/attention_model')
    checkpoint = ROOT / ('outputs/attention_model_smoke' if smoke else 'outputs/attention_model') / f'seed{seed}.pt'
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if replay:
        saved = torch.load(checkpoint, weights_only=False, map_location='cpu')
        config, agent_state = saved['config'], saved['agent']
        source_meta = saved['source_checkpoint']
    else:
        source = ROOT / f'outputs/stage7_content_memory_v3_seed{seed}/experiment.pt'
        legacy = torch.load(source, weights_only=False, map_location='cpu')
        config, agent_state = legacy['config'], legacy['models']['recurrent']
        source_meta = {'path': str(source.relative_to(ROOT)), 'sha256': sha(source), 'training_seed': config['seed']}
        assert config['seed'] == seed
    agent = make_agent(config, agent_state)
    contexts = {name: make_contexts(agent.task_config, seed, name, 16 if smoke else None) for name in SPLITS}
    ids = [set(c.ids) for c in contexts.values()]
    assert all(not a & b for i,a in enumerate(ids) for b in ids[i+1:])
    fingerprints = {name: fingerprint(c.scene, c.cues) for name,c in contexts.items()}
    reporters, training = {}, {}
    if replay:
        assert fingerprints == saved['split_fingerprints']
        training = saved['training']
        for family in FAMILIES:
            reporters[family] = Reporter().eval().requires_grad_(False)
            reporters[family].load_state_dict(saved['reporters'][family])
    else:
        train, validation = capture(agent, contexts['fit']), capture(agent, contexts['validation'])
        for family in FAMILIES:
            reporters[family], training[family] = fit_reporter(train, validation, family, seed, steps=4 if smoke else 1000)
            print(f'{seed} {family}: selected {training[family]["selected_step"]}', flush=True)
        torch.save({'config': config, 'agent': agent_state, 'reporters': {k:v.state_dict() for k,v in reporters.items()},
                    'training': training, 'split_fingerprints': fingerprints, 'source_checkpoint': source_meta}, checkpoint)
    result, cases = evaluation(agent, reporters, contexts['test'], seed)
    result.update({'audit': 'attention_model_reporting_v1', 'seed': seed, 'smoke': smoke,
                   'source_checkpoint': source_meta, 'split_fingerprints': fingerprints, 'disjoint_partitions': True,
                   'training': training, 'checkpoint_sha256': sha(checkpoint),
                   'source_sha256': {name:sha(ROOT/name) for name in SOURCES},
                   'feedback_weight_max_abs': float(agent.policy_self_model_head.weight.abs().max()),
                   'case_counts': {k:len(v.ids) for k,v in contexts.items()}})
    data = (json.dumps(cases, sort_keys=True, separators=(',', ':')) + '\n').encode()
    compressed = gzip.compress(data, mtime=0)
    result['case_file_sha256'] = hashlib.sha256(compressed).hexdigest()
    target = outdir / f'seed{seed}.json'
    if replay:
        reference = json.loads(target.read_text())
        if result != reference: raise AssertionError('replayed metrics/provenance differ from committed results')
        if compressed != (outdir / f'seed{seed}_cases.json.gz').read_bytes(): raise AssertionError('case replay differs')
        print(f'{seed}: complete metric and case replay passed', flush=True)
    else:
        target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        (outdir / f'seed{seed}_cases.json.gz').write_bytes(compressed)
        print(f'{seed}: evaluated {result["gates"]}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.seed, args.replay, args.smoke)
