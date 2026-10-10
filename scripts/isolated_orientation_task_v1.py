"""Isolated synthetic orientation task and held-out correctness calibration."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import torch
from attcon.bound_content import VisualEncoder
from orientation_transfer_diagnostic_v1 import SEEDS, LUMINANCE, checkpoint, extract, digest, compare

ROOT = Path('audits/isolated_orientation_task_v1')
COUNTS = (256,256,512)
RIDGE, CAL_L2, CAL_STEPS, CAL_TOL, BOOTSTRAPS = .01,.001,100,1e-8,256
SOURCES = ('scripts/isolated_orientation_task_v1.py',
 'scripts/orientation_transfer_diagnostic_v1.py', 'src/attcon/bound_content.py',
 'docs/ISOLATED_ORIENTATION_TASK_V1_PROTOCOL.md','tests/test_isolated_orientation_task.py')


def stimuli(seed, count):
    assert count%2==0
    nuisance=torch.Generator().manual_seed(seed)
    labels=torch.arange(count)%2
    labels=labels[torch.randperm(count,generator=torch.Generator().manual_seed(seed+1000000))]
    yy,xx=torch.meshgrid(torch.arange(16),torch.arange(16),indexing='ij')
    cx=7.5+2*torch.rand(count,1,1,generator=nuisance)-1
    cy=7.5+2*torch.rand(count,1,1,generator=nuisance)-1
    radius=3.7+1.5*torch.rand(count,1,1,generator=nuisance)
    dx,dy=xx[None]-cx,yy[None]-cy
    circle=dx.square()+dy.square()<=radius.square()
    square=(dx.abs()<=radius)&(dy.abs()<=radius)
    shape=torch.randint(2,(count,1,1),generator=nuisance)
    mask=torch.where(shape==0,circle,square)
    phase=2*torch.pi*torch.rand(count,1,1,generator=nuisance)
    noise=.015*torch.randn(count,6,3,16,16,generator=nuisance)
    return render(labels,mask,phase,noise),labels,{'mask':mask,'phase':phase,'noise':noise}


def render(labels,mask,phase,noise):
    yy,xx=torch.meshgrid(torch.arange(16),torch.arange(16),indexing='ij')
    axis=torch.where(labels[:,None,None]==0,yy[None],xx[None])
    texture=torch.sin(axis*torch.pi/2+phase)>=0
    levels=torch.tensor(LUMINANCE)/50
    target=torch.where(texture[:,None],levels[None,:,0,None,None],levels[None,:,1,None,None])
    image=torch.where(mask[:,None],target,torch.tensor(35.5/50))
    return (image[:,:,None].expand(-1,-1,3,-1,-1)+noise).clamp(0,1)


def fit_classifier(features,labels):
    x=features[:,-1].double()
    mean=x.mean(0)
    scale=(x-mean).square().mean(0).sqrt()
    scale=torch.where(scale>0,scale,torch.ones_like(scale))
    x=(x-mean)/scale
    y=labels.double()*2-1
    intercept=y.mean()
    weights=torch.linalg.solve(x.T@x/len(x)+RIDGE*torch.eye(x.shape[-1],dtype=x.dtype),x.T@(y-intercept)/len(x))
    return {'mean':mean,'scale':scale,'weights':weights,'intercept':intercept}


def predict(readout,features):
    return (features.double()-readout['mean'])/readout['scale']@readout['weights']+readout['intercept']


def fit_calibrator(scores,labels):
    margin=scores.abs().flatten().double()
    correct=((scores>0).long()==labels[:,None]).double().flatten()
    mean=margin.mean();scale=margin.std(unbiased=False)
    if float(scale)==0: scale=torch.ones_like(scale)
    x=torch.stack((torch.ones_like(margin),(margin-mean)/scale),1)
    # Fixed convex penalized logistic objective, deterministic Newton iterations.
    beta=torch.zeros(2,dtype=torch.float64)
    penalty=CAL_L2*torch.eye(2,dtype=torch.float64)
    for _ in range(CAL_STEPS):
        p=(x@beta).sigmoid()
        grad=x.T@(p-correct)/len(x)+penalty@beta
        hessian=x.T@((p*(1-p))[:,None]*x)/len(x)+penalty
        beta=beta-torch.linalg.solve(hessian,grad)
    gradient=x.T@((x@beta).sigmoid()-correct)/len(x)+penalty@beta
    gradient_max=gradient.abs().max()
    objective=(torch.nn.functional.softplus(x@beta)-correct*(x@beta)).mean()+CAL_L2/2*beta.square().sum()
    assert torch.isfinite(beta).all() and float(gradient_max)<=CAL_TOL
    return {'mean':mean,'scale':scale,'beta':beta,'baseline':correct.mean(),
        'gradient_max':gradient_max,'objective':objective}


def probability(calibrator,scores):
    margin=scores.abs().double()
    return (calibrator['beta'][0]+calibrator['beta'][1]*(margin-calibrator['mean'])/calibrator['scale']).sigmoid()


def metrics(prob,correct):
    prob,correct=prob.flatten().double(),correct.flatten().double()
    bins=torch.bucketize(prob,torch.tensor([.2,.4,.6,.8],dtype=torch.float64),right=True)
    reliability=[];ece=0.
    for i in range(5):
        keep=bins==i;n=int(keep.sum())
        if n:
            predicted=float(prob[keep].mean());observed=float(correct[keep].mean())
            ece+=n/len(prob)*abs(predicted-observed)
        else:predicted=observed=None
        reliability.append({'bin':i,'count':n,'mean_probability':predicted,'accuracy':observed})
    return {'accuracy':float(correct.mean()),'brier':float((prob-correct).square().mean()),'ece':ece,'reliability':reliability}


@torch.no_grad()
def build():
    trace,results={},{}
    for seed in SEEDS:
        encoder=VisualEncoder().eval()
        encoder.load_state_dict(torch.load(checkpoint(seed),weights_only=True)['state_dict'])
        splits=[stimuli(910000000+seed+10000*split,count) for split,count in enumerate(COUNTS)]
        features=[extract(encoder,item[0]) for item in splits]
        indices=torch.randint(COUNTS[-1],(BOOTSTRAPS,COUNTS[-1]),generator=torch.Generator().manual_seed(920000000+seed))
        null_patches=splits[2][0][:,0,None].expand(-1,2,-1,-1,-1).clone()
        null_features=extract(encoder,null_patches)
        saved,assessments={},{}
        for key in features[0]:
            readout=fit_classifier(features[0][key],splits[0][1])
            scores=[predict(readout,x[key]) for x in features]
            calibrator=fit_calibrator(scores[1],splits[1][1])
            prob=probability(calibrator,scores[2])
            correct=((scores[2]>0).long()==splits[2][1][:,None]).double()
            assessment=metrics(prob,correct)
            assessment['by_contrast']=[metrics(prob[:,c],correct[:,c]) for c in range(6)]
            assessment['calibration_mean_baseline']=float(calibrator['baseline'])
            assessment['baseline_brier']=float((calibrator['baseline']-correct).square().mean())
            assessment['accuracy_bootstrap_95_by_contrast']=torch.quantile(correct[indices].mean(1),torch.tensor([.025,.975],dtype=torch.float64),dim=0).T.tolist()
            differences=((prob-correct).square()-(calibrator['baseline']-correct).square()).mean(1)
            assessment['brier_difference_bootstrap_95']=torch.quantile(differences[indices].mean(1),torch.tensor([.025,.975],dtype=torch.float64)).tolist()
            # Separate null duplication: inputs/feature rows identical, opposite labels.
            nullscore=predict(readout,null_features[key])
            assert torch.equal(null_features[key][:,0],null_features[key][:,1])
            assert torch.equal(nullscore[:,0],nullscore[:,1])
            nullcorrect=(nullscore>0).long()==torch.arange(2)[None]
            assessment['duplicate_zero_accuracy']=float(nullcorrect.double().mean())
            assessments[key]=assessment
            saved[key]={'readout':readout,'calibrator':calibrator,'scores':scores,'probabilities':prob,'correct':correct,'null_scores':nullscore}
        hidden=assessments['visual_hidden'];edge=assessments['physical_directional_edge']
        gates={'hidden_high_contrast':hidden['by_contrast'][-1]['accuracy']>=.90,
            'physical_high_contrast':edge['by_contrast'][-1]['accuracy']>=.90,
            'hidden_zero_gross_chance':abs(hidden['by_contrast'][0]['accuracy']-.5)<=.075,
            'physical_zero_gross_chance':abs(edge['by_contrast'][0]['accuracy']-.5)<=.075,
            'all_duplicate_null_chance':all(a['duplicate_zero_accuracy']==.5 for a in assessments.values()),
            'hidden_pooled_ece':hidden['ece']<=.10,
            'hidden_brier_vs_calibration_baseline':hidden['brier']<=hidden['baseline_brier']}
        results[str(seed)]={'sources':assessments,'gates':gates,'engineering_gates_pass':all(gates.values())}
        trace[str(seed)]={'splits':splits,'features':features,'instruments':saved,'bootstrap_indices':indices,'null_patches':null_patches,'null_features':null_features}
    return trace,{'kind':'isolated_synthetic_orientation_and_correctness_instrument',
        'models':results,'all_engineering_gates_pass':all(v['engineering_gates_pass'] for v in results.values()),
        'goal_achieved':False,'attention_source_tested':False,'participant_rows_tested':0,'new_reports':0}


def check_manifest():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for p,h in {**manifest['source_sha256'],**manifest['checkpoint_sha256']}.items():assert digest(p)==h,p
    for p,source in json.loads((ROOT/'source_snapshots.json').read_text()).items():
        assert hashlib.sha256(source.encode()).hexdigest()==manifest['source_sha256'][p],p


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=('prepare','run','verify'))
    args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=='prepare':
        if (ROOT/'manifest.json').exists():raise SystemExit('refusing overwrite')
        ROOT.mkdir(parents=True,exist_ok=True)
        manifest={'kind':'synthetic_development_registration','seeds':SEEDS,'split_counts':COUNTS,
            'ridge':RIDGE,'calibration_l2':CAL_L2,'calibration_newton_steps':CAL_STEPS,'calibration_gradient_tolerance':CAL_TOL,'bootstraps':BOOTSTRAPS,
            'source_sha256':{p:digest(p) for p in SOURCES},
            'checkpoint_sha256':{str(checkpoint(s)):digest(checkpoint(s)) for s in SEEDS},
            'threads':2,'torch_version':torch.__version__,'single_attempt':True,'human_targets_used':False}
        (ROOT/'source_snapshots.json').write_text(json.dumps({p:Path(p).read_text() for p in SOURCES},indent=2)+'\n')
        (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('Registered before model outputs');return
    check_manifest()
    if args.stage=='run':
        if (ROOT/'attempt.json').exists():raise SystemExit('refusing retry')
        (ROOT/'attempt.json').write_text(json.dumps({'status':'reserved','no_retry':True})+'\n')
        trace,results=build()
        archive_hashes={}
        for seed,state in trace.items():
            filename=f'states-{seed}.pt.gz'
            with (ROOT/filename).open('wb') as raw:
                with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as stream:torch.save(state,stream)
            archive_hashes[filename]=digest(ROOT/filename)
        (ROOT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        (ROOT/'completion.json').write_text(json.dumps({'status':'completed','goal_achieved':False,
            'archive_sha256':archive_hashes,'results_sha256':digest(ROOT/'results.json')},indent=2)+'\n')
        print(json.dumps({s:r['gates'] for s,r in results['models'].items()},indent=2))
    else:
        completion=json.loads((ROOT/'completion.json').read_text())
        assert digest(ROOT/'results.json')==completion['results_sha256']
        for file,h in completion['archive_sha256'].items():assert digest(ROOT/file)==h
        trace,results=build()
        for seed,state in trace.items():
            with gzip.open(ROOT/f'states-{seed}.pt.gz','rb') as stream:compare(torch.load(stream,weights_only=True),state)
        assert json.loads((ROOT/'results.json').read_text())==results
        print('Exact isolated inputs/features/readouts/calibration/bootstrap/result replay passed')


if __name__=='__main__':main()
