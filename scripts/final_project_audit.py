"""Validate the finite final matrix, archive checkpoints, and publish claim status."""
import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from summarize_prospective import summarize, validate_control, validate_report

TASKS = ('serial', 'routing')
ARCHITECTURES = ('gru', 'rnn')
SEEDS = (2503, 2521, 2539)
ARCHIVE = 'artifacts/final_checkpoints.tar.gz'
MANIFEST = 'audits/project_completion.json'
STAGE8_SHA = 'f68143bd58094541a24b50fd44fdda1dd45818fdc9a0ba451ca8d5523d5754a7'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paths_for(role):
    if role == 'exploration':
        return [f'audits/exploration_{task}_{seed}.json' for task in TASKS for seed in SEEDS]
    prefix = {'fitted':'prospective', 'reporting':'prospective_reporting', 'identity':'identity_reporting', 'stress':'prospective_stress'}[role]
    return [f'audits/{prefix}_{task}_{arch}_{seed}.json' for task in TASKS for arch in ARCHITECTURES for seed in SEEDS]


def read_json(path):
    def invalid(token):
        raise ValueError(f'nonfinite JSON number {token}: {path}')
    return json.loads(path.read_text(), parse_constant=invalid)


def assert_close(observed, expected, label='result'):
    if isinstance(expected, dict):
        if set(observed) != set(expected):
            raise ValueError(f'{label}: mismatched keys')
        for key in expected:
            assert_close(observed[key], expected[key], label + '.' + key)
    elif isinstance(expected, list):
        if len(observed) != len(expected):
            raise ValueError(f'{label}: mismatched lengths')
        for index, (a, b) in enumerate(zip(observed, expected)):
            assert_close(a, b, f'{label}[{index}]')
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if observed != expected:
            raise ValueError(f'{label}: mismatch')
    elif not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f'{label}: {observed} != {expected}')


def validate_sources(run):
    for path, expected in run['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('changed source: ' + path)
    for field, sha_field in (('source', 'source_artifact_sha256'), ('original_report', 'original_report_sha256')):
        if sha_field in run and digest(ROOT / run[field]) != run[sha_field]:
            raise ValueError('changed source artifact: ' + run[field])


def package_checkpoints(checkpoints):
    archive = ROOT / ARCHIVE
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as tar:
                for path in sorted(checkpoints):
                    content = (ROOT / path).read_bytes()
                    info = tarfile.TarInfo(path)
                    info.size = len(content)
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    tar.addfile(info, io.BytesIO(content))
    return {'path':ARCHIVE, 'sha256':digest(archive), 'bytes':archive.stat().st_size}


def safe_archive_payloads(archive, expected):
    """Verify an exact regular-file allowlist before any extraction occurs."""
    payloads = {}
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError('archive member set differs from manifest')
        for member in members:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts or path.parts[:2] != ('outputs','prospective'):
                raise ValueError('unsafe archive member')
            content = tar.extractfile(member).read()
            if hashlib.sha256(content).hexdigest() != expected[member.name]:
                raise ValueError('checkpoint archive checksum mismatch')
            payloads[member.name] = content
    return payloads


def restore_checkpoints(manifest):
    archive = ROOT / manifest['archive']['path']
    if digest(archive) != manifest['archive']['sha256']:
        raise ValueError('archive checksum mismatch')
    for name, content in safe_archive_payloads(archive, manifest['checkpoint_sha256']).items():
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def replay_controls(run):
    from attcon.prospective import make_splits
    from attcon.prospective_learning import load_models, evaluate
    from attcon.regulation import weight_fingerprint
    saved, models = load_models(ROOT / run['checkpoint'])
    if (saved['seed'], saved['task'], saved['architecture']) != (run['seed'], run['task'], run['architecture']):
        raise ValueError('checkpoint metadata mismatch')
    for family, model in models.items():
        if weight_fingerprint(model) != run['training'][family]['selected_sha256']:
            raise ValueError('checkpoint weight mismatch')
    splits = make_splits(run['seed'], run['task'])
    for name, data in splits.items():
        if data.fingerprint() != run['dataset_sha256'][name]:
            raise ValueError('reconstructed dataset mismatch')
    actual = evaluate(models, splits['test'], run['seed'], native=run['audit']=='prospective_exploration_v1')
    assert_close(actual, run['test'], run['checkpoint'])


def replay_reports(run):
    import torch
    from attcon.prospective import make_splits, root_events
    from attcon.prospective_learning import load_models
    from attcon.prospective_reporting import labels_for, report_features, scores, causal_audit
    from attcon.regulation import Readout
    source = read_json(ROOT / (run['source'] if run['audit']=='prospective_reporting_coupling' else read_json(ROOT / run['original_report'])['source']))
    _, controllers = load_models(ROOT / source['checkpoint'])
    agent = controllers['state']
    splits = make_splits(run['seed'], run['task'])
    with torch.no_grad():
        state = agent.advance(root_events(splits['test']))
    labels = labels_for(splits['test'])
    saved = torch.load(ROOT / run['checkpoint'], map_location='cpu', weights_only=True)
    if saved['agent_sha256'] != run['agent_sha256']:
        raise ValueError('report checkpoint agent mismatch')
    weights = saved['models'] if 'models' in saved else {'identity':saved['model']}
    models = {}
    for name, value in weights.items():
        model = Readout(torch.zeros(2,128), classes=14, hidden=32)
        model.load_state_dict(value)
        models[name] = model.eval().requires_grad_(False)
    if run['audit']=='prospective_reporting_coupling':
        features = report_features(agent, state, splits['test'])
        for name, model in models.items():
            actual = scores(model(features[name]).argmax(-1), labels)
            assert_close(actual, run['reports'][name], 'report.' + name)
        with torch.no_grad():
            fit_state = agent.advance(root_events(splits['report_fit']))
        causal = causal_audit(agent, models, fit_state, splits['report_fit'], state, splits['test'], run['seed']+17000)
        assert_close(causal, run['causal'], 'causal')
    else:
        from audit_identity_cue_reporting import identity_cue_features
        actual = scores(models['identity'](identity_cue_features(agent,state,splits['test'])).argmax(-1),labels)
        assert_close(actual, run['identity_cue_report'], 'identity-corrected report')


def replay_stress(run):
    from attcon.prospective import make_splits
    from attcon.prospective_learning import load_models, evaluate
    source = read_json(ROOT / run['source'])
    _, models = load_models(ROOT / source['checkpoint'])
    for name, condition in run['conditions'].items():
        data = make_splits(run['seed'], run['task'], degradation=.15 if name=='misspecified' else 0)['stress']
        if data.fingerprint() != condition['dataset_sha256']:
            raise ValueError('stress data mismatch')
        actual = evaluate(models,data,run['seed'],condition['delay'])
        assert_close(actual,condition['evaluation'],'stress.'+name)


def build(replay=False, package=False):
    import torch
    import numpy
    torch.set_num_threads(1)
    artifacts = {}
    by_role = {}
    checkpoints = {}
    for role in ('fitted','exploration','reporting','identity','stress'):
        paths = paths_for(role)
        runs = [read_json(ROOT / path) for path in paths]
        expected_cells = {(task,arch,seed) for task in TASKS for arch in (('gru',) if role=='exploration' else ARCHITECTURES) for seed in SEEDS}
        if {(r['task'],r['architecture'],r['seed']) for r in runs} != expected_cells:
            raise ValueError('incomplete registered matrix: ' + role)
        for path, run in zip(paths,runs):
            validate_sources(run)
            artifacts[path] = digest(ROOT / path)
            if 'checkpoint' in run:
                checkpoints[run['checkpoint']] = digest(ROOT / run['checkpoint'])
            if role in ('fitted','exploration'):
                validate_control(run)
                record_key = 'selected_epoch' if role=='fitted' else 'selected_update'
                candidate_key = 'epoch' if role=='fitted' else 'update'
                for record in run['training'].values():
                    best = max(record['candidates'],key=lambda c:c['validation']['return'])
                    if record[record_key] != best[candidate_key]:
                        raise ValueError('checkpoint was not selected by validation')
                if replay: replay_controls(run)
            elif role=='reporting':
                validate_report(run)
                if replay: replay_reports(run)
            elif role=='identity':
                original = read_json(ROOT / run['original_report'])
                gain = original['reports']['state']['mean_quality_source_accuracy'] - run['identity_cue_report']['mean_quality_source_accuracy']
                if abs(gain-run['state_minus_identity_cue_gain'])>1e-8 or (gain>=.02)!=run['descriptive_margin_pass']:
                    raise ValueError('inconsistent corrected report comparison')
                if run['parameters'] != original['parameters']['state']:
                    raise ValueError('unequal corrected reporter capacity')
                if replay: replay_reports(run)
            else:
                if not run['disjoint_reserved_contexts'] or run['reserved_context_count']!=64:
                    raise ValueError('invalid stress partition')
                if set(run['conditions']) != {'baseline','long_delay','misspecified'}:
                    raise ValueError('incomplete stress conditions')
                source = read_json(ROOT / run['source'])
                for condition in run['conditions'].values():
                    validate_control({**source,'test':condition['evaluation']})
                if replay: replay_stress(run)
            print(f'verified {role}: {run["task"]}/{run["architecture"]}/{run["seed"]}',flush=True)
        by_role[role] = runs
    environment = read_json(ROOT / 'audits/prospective_environment.json')
    validate_sources(environment)
    for run in by_role['fitted'] + by_role['exploration']:
        expected = environment['runs'][f'{run["task"]}_{run["seed"]}']['dataset_sha256']
        if run['dataset_sha256'] != expected:
            raise ValueError('data differ from registered environment')
    artifacts['audits/prospective_environment.json'] = digest(ROOT / 'audits/prospective_environment.json')
    if digest(ROOT/'audits/stage8_convergence_current.json') != STAGE8_SHA:
        raise ValueError('prior Stage 8 artifact changed')
    fitted = summarize([ROOT / p for p in paths_for('fitted')])
    exploration = summarize([ROOT / p for p in paths_for('exploration')])
    reporting = summarize([ROOT / p for p in paths_for('reporting')])
    promotion = any(all(fitted['groups'][task+'_'+arch]['all_seeds_supported'] for arch in ARCHITECTURES) for task in TASKS)
    corrected = {}
    for task in TASKS:
        for arch in ARCHITECTURES:
            group = [r for r in by_role['identity'] if r['task']==task and r['architecture']==arch]
            gains = [r['state_minus_identity_cue_gain'] for r in group]
            corrected[task+'_'+arch] = {'descriptive_margin_pass_count':sum(r['descriptive_margin_pass'] for r in group),
                'state_gain_min':min(gains),'state_gain_max':max(gains),'post_registration_diagnostic':True}
    stress_groups = {}
    for task in TASKS:
        for arch in ARCHITECTURES:
            group = [r for r in by_role['stress'] if r['task']==task and r['architecture']==arch]
            conditions = {}
            for condition in ('baseline','long_delay','misspecified'):
                costs = {}
                for cost in ('0.1','0.25','0.4'):
                    values = [r['conditions'][condition]['evaluation']['costs'][cost] for r in group]
                    returns = [v['policies']['state']['return'] for v in values]
                    costs[cost] = {'state_return_min':min(returns),'state_return_max':max(returns),
                        'all_gate_pass_count':sum(v['all_gates_pass'] for v in values),
                        'fresh_viability_pass_count':sum(v['gates']['fresh_accuracy'] for v in values),
                        'final_observation_viability_pass_count':sum(v['gates']['forced_final_accuracy'] for v in values)}
                conditions[condition] = costs
            stress_groups[task+'_'+arch] = conditions
    archive = package_checkpoints(checkpoints) if package else {'path':ARCHIVE,'sha256':digest(ROOT/ARCHIVE),'bytes':(ROOT/ARCHIVE).stat().st_size}
    safe_archive_payloads(ROOT/ARCHIVE, checkpoints)
    result = {'audit':'project_completion_v1','bounded_evaluation_complete':True,
        'strong_access_monitoring_claim_supported':False,'completion_scope':'Finite reproducible research prototype and final evaluation; a positive access-monitoring or consciousness finding is not a completion condition.',
        'registered_fitted_controllers':36,'reward_only_controllers':12,'frozen_state_systems_reported':12,
        'checkpoint_files':len(checkpoints),'artifact_sha256':artifacts,'checkpoint_sha256':checkpoints,'archive':archive,
        'verification_source_sha256':{path:digest(ROOT/path) for path in ('scripts/final_project_audit.py','scripts/reproduce_final.py','scripts/summarize_prospective.py','requirements-final.txt')},
        'environment':{'python':sys.version.split()[0],'torch':str(torch.__version__),'numpy':numpy.__version__,'torch_threads':1},
        'fitted_control_groups':fitted['groups'],'reward_only_groups':exploration['groups'],
        'registered_reporting_groups':reporting['groups'],'corrected_report_comparisons':corrected,'stress_groups':stress_groups,
        'independent_content_promotion_prerequisite_met':promotion,
        'convergence_disposition':'Promotion prerequisite met; additional convergence work required before claiming completion of that branch.' if promotion else 'Promotion closed as prerequisite not met. Prior unforced-convergence failures retained; no positive convergence claim.',
        'stage8_status':'not_met','stage8_sha256':STAGE8_SHA,'stage8_recomputed':False,
        'original_reporting_comparator_confounded':True,'correction_document':'docs/COMPLETION_CORRECTIONS.md',
        'all_checkpoint_results_replayed':bool(replay),
        'limitations':['Synthetic finite tasks and small recurrent models; both tasks share the six-value generator, so this is not domain generalization.','Fitted and reward-only objectives/budgets differ; no algorithm superiority claim.','Reports label environmental source/quality, not subjective or necessarily current internal access.','Original report comparator omitted identity; corrected tests reuse data and are diagnostic.','Cue comparator has exact retention of an observed cue; this is a strong control.','Context intervals are pointwise and conditional; no population-level certainty.','Sensor stress hides reliability changes from learned policies while analytic policies know them.']}
    if promotion:
        result['bounded_evaluation_complete'] = False
        result['strong_access_monitoring_claim_supported'] = False
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',action='store_true',help='Build deterministic checkpoint archive.')
    p.add_argument('--restore',action='store_true',help='Verify and restore the committed checkpoint archive.')
    p.add_argument('--replay',action='store_true',help='Recompute all controller, report, causal, and stress metrics.')
    p.add_argument('--verify',action='store_true',help='Compare against the committed manifest without rewriting it.')
    a=p.parse_args()
    if a.restore:
        restore_checkpoints(read_json(ROOT/MANIFEST))
    result=build(replay=a.replay,package=a.package)
    if a.verify:
        expected=read_json(ROOT/MANIFEST)
        # Replay is an execution property; environment versions are separately recorded.
        for name in ('artifact_sha256','checkpoint_sha256','archive','verification_source_sha256','fitted_control_groups','reward_only_groups','registered_reporting_groups','corrected_report_comparisons','stress_groups','stage8_sha256','bounded_evaluation_complete'):
            assert_close(result[name],expected[name],name)
        print('Final manifest and requested replay checks verified.')
    else:
        (ROOT/MANIFEST).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'bounded_evaluation_complete':result['bounded_evaluation_complete'],'promotion_prerequisite':result['independent_content_promotion_prerequisite_met'],'artifacts':len(result['artifact_sha256']),'checkpoints':len(result['checkpoint_sha256'])}))
if __name__=='__main__':main()
