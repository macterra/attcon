"""Synthetic orientation information, without participant or appearance labels."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import torch
from attcon.bound_content import VisualEncoder

ROOT = Path('audits/orientation_transfer_diagnostic_v1')
SEEDS = (1301,1311,1321)
COUNT, RIDGE, BOOTSTRAPS = 64,.01,256
LUMINANCE = ((38.2,38.2),(39.,37.3),(39.5,36.5),(40.,36.2),(40.2,36.),(41.1,35.8))
SOURCES = ('scripts/orientation_transfer_diagnostic_v1.py',
    'docs/ORIENTATION_TRANSFER_DIAGNOSTIC_V1_PROTOCOL.md',
    'tests/test_orientation_transfer_diagnostic.py','src/attcon/bound_content.py')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint(seed):
    return Path(f'audits/bound_content/confirmation_v2/seed{seed}.pt')


def stimuli(seed, count=COUNT):
    generator=torch.Generator().manual_seed(seed)
    yy,xx=torch.meshgrid(torch.arange(16),torch.arange(16),indexing='ij')
    cx=7.5+2*torch.rand(count,1,1,generator=generator)-1
    cy=7.5+2*torch.rand(count,1,1,generator=generator)-1
    r=3.7+1.5*torch.rand(count,1,1,generator=generator)
    dx,dy=xx[None]-cx,yy[None]-cy
    circle=dx.square()+dy.square()<=r.square()
    square=(dx.abs()<=r)&(dy.abs()<=r)
    mask=torch.where((torch.arange(count)%2==0)[:,None,None],circle,square)
    phase=2*torch.pi*torch.rand(count,1,1,generator=generator)
    texture=torch.stack((torch.sin(yy[None]*torch.pi/2+phase)>=0,
                         torch.sin(xx[None]*torch.pi/2+phase)>=0),1)
    noise=.015*torch.randn(count,1,1,3,16,16,generator=generator)
    levels=torch.tensor(LUMINANCE)/50
    target=torch.where(texture[:,None],levels[None,:,None,0,None,None],
                       levels[None,:,None,1,None,None])
    image=torch.where(mask[:,None,None],target,torch.tensor(35.5/50))
    patches=(image[:,:,:,None].expand(-1,-1,-1,3,-1,-1)+noise).clamp(0,1)
    assert torch.equal(patches[:,0,0],patches[:,0,1])
    return patches


def centered(x):
    x=x.double()
    return x-x.mean(2,keepdim=True)


def edge_control(patches):
    x=(patches[..., :,1:]-patches[..., :,:-1]).square().mean((-3,-2,-1))
    y=(patches[..., 1:,:]-patches[..., :-1,:]).square().mean((-3,-2,-1))
    return (x-y)[...,None]


def fit(features):
    x=centered(features)[:,-1].flatten(0,1)
    y=torch.tensor([-1.,1.],dtype=torch.float64).repeat(len(features))[:,None]
    scale=x.square().mean(0).sqrt()
    scale=torch.where(scale>0,scale,torch.ones_like(scale))
    x=x/scale
    weights=torch.linalg.solve(x.T@x/len(x)+RIDGE*torch.eye(x.shape[-1],dtype=x.dtype),x.T@y/len(x))
    return {'scale':scale,'weights':weights}


def assess(readout,features,indices):
    scores=(centered(features)/readout['scale']@readout['weights'])[...,0]
    labels=torch.arange(2)[None,None]
    correct=(scores>0).long()==labels
    per_unit=correct.double().mean(2)
    intervals=torch.quantile(per_unit[indices].mean(1),
        torch.tensor([.025,.975],dtype=torch.float64),dim=0).T
    assert torch.equal(features[:,0,0],features[:,0,1])
    assert float(per_unit[:,0].mean())==.5
    return {'accuracy_by_contrast':per_unit.mean(0).tolist(),
            'conditional_base_bootstrap_95':intervals.tolist()},scores


@torch.no_grad()
def extract(encoder,patches):
    leading=patches.shape[:-3]
    flat=patches.reshape(-1,3,16,16)
    hidden=encoder.net[:-1](flat).reshape(*leading,64)
    logits=encoder.net[-1](hidden)
    categories=encoder(patches)
    color,shape=categories[...,:4],categories[...,4:]
    confidence=torch.stack((color.max(-1).values,shape.max(-1).values,
        -(color*color.clamp_min(1e-30).log()).sum(-1),
        -(shape*shape.clamp_min(1e-30).log()).sum(-1)),-1)
    return {'visual_hidden':hidden,'visual_logits':logits,
        'category_probabilities':categories,'category_confidence':confidence,
        'physical_directional_edge':edge_control(patches)}


@torch.no_grad()
def build():
    trace,results={},{}
    for seed in SEEDS:
        encoder=VisualEncoder().eval()
        encoder.load_state_dict(torch.load(checkpoint(seed),weights_only=True)['state_dict'])
        inputs=[stimuli(890000000+seed+10000*split) for split in (0,1)]
        development,evaluation=[extract(encoder,p) for p in inputs]
        indices=torch.randint(COUNT,(BOOTSTRAPS,COUNT),
            generator=torch.Generator().manual_seed(900000000+seed))
        readouts,scores,assessments={},{},{}
        for key,features in development.items():
            readout=fit(features)
            assessment,prediction=assess(readout,evaluation[key],indices)
            readouts[key],scores[key],assessments[key]=readout,prediction,assessment
        gates={'zero_contrast_identical_inputs':torch.equal(inputs[1][:,0,0],inputs[1][:,0,1]),
            'all_zero_contrast_chance':all(v['accuracy_by_contrast'][0]==.5 for v in assessments.values()),
            'physical_control_high_contrast':assessments['physical_directional_edge']['accuracy_by_contrast'][-1]>=.90,
            'hidden_readout_high_contrast':assessments['visual_hidden']['accuracy_by_contrast'][-1]>=.90}
        results[str(seed)]={'scores':assessments,'gates':gates,
            'synthetic_sensory_feasibility_pass':all(gates.values()),
            'native_color_argmax_counts':torch.bincount(evaluation['category_probabilities'][...,:4].argmax(-1).flatten(),minlength=4).tolist()}
        trace[str(seed)]={'development_patches':inputs[0],'evaluation_patches':inputs[1],
            'development_features':development,'evaluation_features':evaluation,
            'readouts':readouts,'scores':scores,'bootstrap_indices':indices,
            'generated_orientation_labels':torch.arange(2).expand(COUNT,6,2).clone()}
    return trace,{'kind':'synthetic_orientation_ancillary_diagnostic',
        'models':results,'all_synthetic_sensory_gates_pass':all(v['synthetic_sensory_feasibility_pass'] for v in results.values()),
        'participant_rows_tested':0,'empirical_stimulus_transfer_ready':False,
        'attention_source_tested':False,'goal_achieved':False,'new_reports':0}


def check_manifest():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for p,h in {**manifest['source_sha256'],**manifest['checkpoint_sha256']}.items():
        assert digest(p)==h,p
    snapshots=json.loads((ROOT/'source_snapshots.json').read_text())
    for p,h in manifest['source_sha256'].items():
        assert hashlib.sha256(snapshots[p].encode()).hexdigest()==h,p


def compare(before,after,path='root'):
    if isinstance(before,torch.Tensor):assert torch.equal(before,after),path
    elif isinstance(before,dict):
        assert before.keys()==after.keys(),path
        for key in before:compare(before[key],after[key],path+'/'+str(key))
    else:assert before==after,path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=('prepare','run','verify'))
    args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=='prepare':
        if (ROOT/'manifest.json').exists():raise SystemExit('refusing overwrite')
        ROOT.mkdir(parents=True,exist_ok=True)
        manifest={'kind':'synthetic_ancillary_development_registration','seeds':SEEDS,'count_per_split':COUNT,
            'luminance_pairs':LUMINANCE,'ridge':RIDGE,'bootstraps':BOOTSTRAPS,
            'native_stimuli_verified':False,'human_ratings_used':False,
            'source_sha256':{p:digest(p) for p in SOURCES},
            'checkpoint_sha256':{str(checkpoint(seed)):digest(checkpoint(seed)) for seed in SEEDS},
            'torch_version':torch.__version__,'threads':2,'single_attempt':True}
        (ROOT/'source_snapshots.json').write_text(json.dumps({p:Path(p).read_text() for p in SOURCES},indent=2)+'\n')
        (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('Synthetic sensory diagnostic registered; no model outputs');return
    check_manifest()
    if args.stage=='run':
        if (ROOT/'attempt.json').exists():raise SystemExit('refusing retry')
        (ROOT/'attempt.json').write_text(json.dumps({'status':'reserved','no_retry':True})+'\n')
        trace,results=build()
        with (ROOT/'states.pt.gz').open('wb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as stream:torch.save(trace,stream)
        (ROOT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        (ROOT/'completion.json').write_text(json.dumps({'status':'completed',
            'states_sha256':digest(ROOT/'states.pt.gz'),'results_sha256':digest(ROOT/'results.json'),
            'goal_achieved':False},indent=2)+'\n');print(json.dumps(results,indent=2))
    else:
        completion=json.loads((ROOT/'completion.json').read_text())
        assert digest(ROOT/'states.pt.gz')==completion['states_sha256']
        assert digest(ROOT/'results.json')==completion['results_sha256']
        trace,results=build()
        with gzip.open(ROOT/'states.pt.gz','rb') as stream:compare(torch.load(stream,weights_only=True),trace)
        assert json.loads((ROOT/'results.json').read_text())==results
        print('Exact synthetic input/feature/readout/result replay and provenance passed')


if __name__=='__main__':main()
