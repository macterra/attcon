"""Fixed-budget learned three-way controls; retain every model and outcome."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import torch
from attcon.predictive_attention import PredictiveAttention, prediction_loss
from attcon.functional_controls import CONDITIONS, WINDOWS, simulate_controls, assess_static, matched_revision

ROOT = Path('audits/functional_controls_v1')
SEEDS = (2011, 2021, 2031)
UPDATES, BATCH, TRAIN_STEPS = 1600, 192, 16
COUNT, HISTORY = 512, 12
SOURCES = ('src/attcon/functional_controls.py', 'src/attcon/predictive_attention.py',
           'scripts/run_functional_controls.py', 'docs/FUNCTIONAL_CONTROLS_PROTOCOL.md')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    path = ROOT/'manifest.json'
    if path.exists():
        return json.loads(path.read_text())
    ROOT.mkdir(parents=True, exist_ok=True)
    sources = {p:Path(p).read_text() for p in SOURCES}
    (ROOT/'source_code.json').write_text(json.dumps(sources, indent=2)+'\n')
    manifest = {'kind':'engineering_development', 'seeds':SEEDS, 'updates':UPDATES,
        'batch':BATCH, 'train_steps':TRAIN_STEPS, 'evaluation_count':COUNT,
        'initial_history':HISTORY, 'windows':WINDOWS, 'conditions':CONDITIONS,
        'training_seed_rule':'seed*100000 + update (zero based)',
        'evaluation_seed_rule':'730000000 + seed', 'post_command_seed_rule':'740000000 + seed',
        'torch_version':torch.__version__, 'python_version':platform.python_version(),
        'source_sha256':{p:hashlib.sha256(s.encode()).hexdigest() for p,s in sources.items()},
        'no_retries':True, 'checkpoint_selection':'fixed final update only'}
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def check_sources(manifest):
    for p,h in manifest['source_sha256'].items():
        assert digest(p)==h,p


@torch.no_grad()
def assess_model(model, seed):
    model.eval()
    episodes = {condition:simulate_controls(730000000+seed, COUNT, HISTORY, condition)
                for condition in CONDITIONS}
    static, static_trace = {}, {}
    for condition,episode in episodes.items():
        static[condition], static_trace[condition] = assess_static(model,episode)
    generator = torch.Generator().manual_seed(740000000+seed)
    commands = torch.randint(4, (COUNT,max(WINDOWS)), generator=generator)
    revision, revision_trace = {}, {}
    for old,episode in episodes.items():
        for new in CONDITIONS:
            key = f'{old}->{new}'
            revision[key], revision_trace[key] = matched_revision(model,episode,new,commands)
    initial_keys = ('initial_allocation','initial_access','initial_effects','initial_hidden',
                    'initial_physical_recovery','initial_replay')
    initial_preservation = all(torch.equal(revision_trace[f'{old}->{old}'][k],
        revision_trace[f'{old}->{new}'][k]) for old in CONDITIONS for new in CONDITIONS
        for k in initial_keys)
    for old in CONDITIONS:
        unchanged = revision[f'{old}->{old}']['windows'][-1]['metrics']
        unchanged_gain = unchanged['feedback_effect_accuracy']-unchanged['no_feedback_effect_accuracy']
        for new in CONDITIONS:
            if new==old:continue
            record = revision[f'{old}->{new}']
            m = record['windows'][-1]['metrics']
            effect_gain = m['feedback_effect_accuracy']-m['no_feedback_effect_accuracy']
            record['effect_gain_over_no_change'] = effect_gain-unchanged_gain
            record['final_gates']['effect_gain_over_no_change'] = record['effect_gain_over_no_change']>=.30
            record['all_gates_pass'] = all(record['final_gates'].values())
    intervention_keys = ('allocation_preserved','access_preserved','hidden_preserved',
                         'restoration_exact','control_mask_swapped_exact')
    gates = {'initial_world_change_preservation':initial_preservation,'static_minima':all(r['all_gates_pass'] for r in static.values()),
        'intervention_preservation':all(all(r['intervention'][k] for k in intervention_keys) for r in static.values()),
        'matched_revision_minima':all(r['all_gates_pass'] for r in revision.values())}
    return {'static':static,'revision':revision,'gates':gates,'all_gates_pass':all(gates.values())}, {
        'static':static_trace,'revision':revision_trace}


def save_trace(trace,path):
    # Lossless tensor archive with a deterministic gzip header; replay checks tensors.
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed:
            torch.save(trace,compressed)


def train(seed,manifest):
    check_sources(manifest)
    path = ROOT/f'seed{seed}.json'
    checkpoint,trace_path = ROOT/f'seed{seed}.pt',ROOT/f'seed{seed}_trace.pt.gz'
    if any(p.exists() for p in (path,checkpoint,trace_path)):
        raise SystemExit('refusing to retry or overwrite model attempt')
    path.write_text(json.dumps({'seed':seed,'status':'attempt_reserved'})+'\n')
    run_commit = subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()
    torch.manual_seed(seed)
    model = PredictiveAttention()
    optimizer = torch.optim.Adam(model.parameters(),lr=.003)
    history=[]
    for update in range(UPDATES):
        episode = simulate_controls(seed*100000+update,BATCH,TRAIN_STEPS)
        forecast,_ = model(episode.observations)
        loss = prediction_loss(forecast,episode)
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if (update+1)%400==0:
            entry={'update':update+1,'loss':float(loss.detach())};history.append(entry)
            print(seed,entry,flush=True)
    torch.save({'state_dict':model.state_dict(),'seed':seed,'updates':UPDATES},checkpoint)
    assessment,trace = assess_model(model,seed)
    save_trace(trace,trace_path)
    record={'seed':seed,'status':'completed','run_commit':run_commit,'training_history':history,
        'assessment':assessment,'checkpoint_sha256':digest(checkpoint),'trace_sha256':digest(trace_path),
        'source_sha256':manifest['source_sha256']}
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'seed':seed,'gates':assessment['gates'],
        'all_gates_pass':assessment['all_gates_pass']}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('prepare','train'),required=True)
    p.add_argument('--seed',type=int,choices=SEEDS);args=p.parse_args()
    torch.set_num_threads(2);manifest=prepare()
    if args.stage=='prepare':print('Prepared fixed-budget three-way control protocol')
    elif args.seed is None:raise SystemExit('--seed required')
    else:train(args.seed,manifest)


if __name__=='__main__':main()
