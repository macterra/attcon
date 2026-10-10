"""All-condition sensory task development; no appearance or attention targets."""
import argparse
import gzip
import json
from pathlib import Path
import torch
from attcon.bound_content import VisualEncoder
from orientation_transfer_diagnostic_v1 import SEEDS, checkpoint, extract, digest
from isolated_orientation_task_v1 import stimuli, fit_calibrator, probability, predict as legacy_predict
from verify_isolated_orientation_task_v1 import compare

ROOT=Path('audits/sensory_task_adapter_v1')
COUNTS=(1024,256,512)
STEPS, WIDTH, RATE, DECAY, BOOTSTRAPS=1500,32,.003,.001,256
SOURCES=('scripts/sensory_task_adapter_v1.py','scripts/isolated_orientation_task_v1.py',
 'scripts/orientation_transfer_diagnostic_v1.py','scripts/verify_isolated_orientation_task_v1.py',
 'src/attcon/bound_content.py','docs/SENSORY_TASK_ADAPTER_V1_PROTOCOL.md','tests/test_sensory_task_adapter.py')


def network(dim):return torch.nn.Sequential(torch.nn.Linear(dim,WIDTH),torch.nn.ReLU(),torch.nn.Linear(WIDTH,1))


def fit_classifier(features,labels,seed,steps=STEPS):
    x=features.reshape(-1,features.shape[-1]).float()
    mean=x.mean(0);scale=(x-mean).square().mean(0).sqrt()
    scale=torch.where(scale>1e-6,scale,torch.ones_like(scale))
    x=(x-mean)/scale
    y=labels[:,None].expand(-1,features.shape[1]).reshape(-1).float()
    with torch.random.fork_rng():
        torch.manual_seed(seed);model=network(x.shape[-1])
    optimizer=torch.optim.Adam(model.parameters(),lr=RATE,betas=(.9,.999),eps=1e-8,
        weight_decay=DECAY,amsgrad=False,foreach=False)
    loss_trace=[]
    with torch.enable_grad():
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x)[:,0],y)
            loss.backward();optimizer.step()
            if step%500==0:loss_trace.append([step,float(loss.detach())])
    with torch.no_grad():final_loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x)[:,0],y)
    assert torch.isfinite(final_loss)
    return {'mean':mean,'scale':scale,'state_dict':{k:v.detach().clone() for k,v in model.state_dict().items()},
        'final_bce':final_loss.detach(),'loss_trace':loss_trace,
        'parameter_count':sum(p.numel() for p in model.parameters()),'steps':steps,'seed':seed}


@torch.no_grad()
def predict(readout,features):
    with torch.random.fork_rng():model=network(features.shape[-1])
    model.load_state_dict(readout['state_dict']);model.eval()
    return model((features.float()-readout['mean'])/readout['scale'])[...,0].double()


def metrics(prob,correct):
    prob,correct=prob.flatten().double(),correct.flatten().double()
    bins=torch.bucketize(prob.contiguous(),torch.arange(1,10,dtype=torch.float64)/10,right=True)
    reliability=[];ece=0.
    for i in range(10):
        keep=bins==i;n=int(keep.sum())
        predicted=float(prob[keep].mean()) if n else None
        observed=float(correct[keep].mean()) if n else None
        if n:ece+=n/len(prob)*abs(predicted-observed)
        reliability.append({'bin':i,'count':n,'mean_probability':predicted,'accuracy':observed})
    return {'count':len(prob),'accuracy':float(correct.mean()),'mean_probability':float(prob.mean()),
        'signed_residual':float((prob-correct).mean()),'brier':float((prob-correct).square().mean()),'ece':ece,'reliability':reliability}


def intervals(values):return torch.quantile(values,torch.tensor([.025,.975],dtype=torch.float64),dim=0).movedim(0,-1).tolist()


def bootstrap(prob,correct,baseline_by_contrast,indices):
    bins=torch.bucketize(prob.contiguous(),torch.arange(1,10,dtype=torch.float64)/10,right=True)
    contributions=torch.nn.functional.one_hot(bins,10)*(prob-correct)[...,None]
    ece=contributions[indices].sum(1).abs().sum(-1)/len(prob)
    difference=(prob-correct).square()-(baseline_by_contrast-correct).square()
    return {'accuracy_95_by_contrast':intervals(correct[indices].mean(1)),
        'signed_residual_95_by_contrast':intervals((prob-correct)[indices].mean(1)),
        'ece_95_by_contrast':intervals(ece),
        'brier_vs_conditional_constant_95_by_contrast':intervals(difference[indices].mean(1))}


@torch.no_grad()
def build():
    traces,models={},{}
    for seed in SEEDS:
        encoder=VisualEncoder().eval()
        encoder.load_state_dict(torch.load(checkpoint(seed),weights_only=True)['state_dict'])
        splits=[stimuli(930000000+seed+10000*split,count) for split,count in enumerate(COUNTS)]
        features=[extract(encoder,s[0]) for s in splits]
        null_patches=splits[2][0][:,0,None].expand(-1,2,-1,-1,-1).clone()
        null_features=extract(encoder,null_patches)
        indices=torch.randint(COUNTS[2],(BOOTSTRAPS,COUNTS[2]),generator=torch.Generator().manual_seed(950000000+seed))
        legacy_path=Path(f'audits/isolated_orientation_task_v1/states-{seed}.pt.gz')
        with gzip.open(legacy_path,'rb') as stream:legacy=torch.load(stream,weights_only=True)['instruments']
        instruments,assessments={},{}
        for key in features[0]:
            readout=fit_classifier(features[0][key],splits[0][1],940000000+seed)
            scores=[predict(readout,f[key]) for f in features]
            calibrator=fit_calibrator(scores[1],splits[1][1])
            prob=probability(calibrator,scores[2])
            correct=((scores[2]>0).long()==splits[2][1][:,None]).double()
            calibration_correct=((scores[1]>0).long()==splits[1][1][:,None]).double()
            conditional_baseline=calibration_correct.mean(0)
            assessment=metrics(prob,correct)
            assessment['by_contrast']=[metrics(prob[:,c],correct[:,c]) for c in range(6)]
            for c,summary in enumerate(assessment['by_contrast']):
                summary['calibration_mean_baseline']=float(conditional_baseline[c])
                summary['constant_brier']=float((conditional_baseline[c]-correct[:,c]).square().mean())
                summary['true_orientation']={str(label):metrics(prob[splits[2][1]==label,c],correct[splits[2][1]==label,c]) for label in (0,1)}
                summary['predicted_orientation']={}
                for label in (0,1):
                    keep=(scores[2][:,c]>0).long()==label
                    summary['predicted_orientation'][str(label)]=metrics(prob[keep,c],correct[keep,c]) if keep.any() else {'count':0}
            assessment['bootstrap']=bootstrap(prob,correct,conditional_baseline,indices)
            assessment['calibration_mean_baseline']=float(calibrator['baseline'])
            assessment['baseline_brier']=float((calibrator['baseline']-correct).square().mean())
            delta=((prob-correct).square()-(calibrator['baseline']-correct).square()).mean(1)
            assessment['pooled_brier_difference_95']=intervals(delta[indices].mean(1))
            assessment['parameter_count']=readout['parameter_count']
            assessment['final_training_bce']=float(readout['final_bce'])
            legacy_scores=legacy_predict(legacy[key]['readout'],features[2][key])
            legacy_prob=probability(legacy[key]['calibrator'],legacy_scores)
            legacy_correct=((legacy_scores>0).long()==splits[2][1][:,None]).double()
            assessment['frozen_legacy_comparator']={'pooled':metrics(legacy_prob,legacy_correct),
                'by_contrast':[metrics(legacy_prob[:,c],legacy_correct[:,c]) for c in range(6)]}
            null_scores=predict(readout,null_features[key]);null_prob=probability(calibrator,null_scores)
            assert torch.equal(null_features[key][:,0],null_features[key][:,1])
            assert torch.equal(null_scores[:,0],null_scores[:,1])
            assert torch.equal(null_prob[:,0],null_prob[:,1])
            assessment['duplicate_null_accuracy']=float(((null_scores>0).long()==torch.arange(2)[None]).double().mean())
            instruments[key]={'readout':readout,'scores':scores,'calibrator':calibrator,
                'probabilities':prob,'correct':correct,'conditional_baseline':conditional_baseline,
                'null_scores':null_scores,'null_probabilities':null_prob,
                'legacy_scores':legacy_scores,'legacy_probabilities':legacy_prob,'legacy_correct':legacy_correct}
            assessments[key]=assessment
        hidden=assessments['visual_hidden'];physical=assessments['physical_directional_edge']
        gates={'hidden_high_contrast':hidden['by_contrast'][-1]['accuracy']>=.90,
            'physical_high_contrast':physical['by_contrast'][-1]['accuracy']>=.90,
            'hidden_zero_gross_chance':abs(hidden['by_contrast'][0]['accuracy']-.5)<=.075,
            'physical_zero_gross_chance':abs(physical['by_contrast'][0]['accuracy']-.5)<=.075,
            'all_duplicate_null_chance':all(a['duplicate_null_accuracy']==.5 for a in assessments.values()),
            'hidden_all_contrast_ece':all(a['ece']<=.10 for a in hidden['by_contrast']),
            'hidden_all_contrast_signed_residual':all(abs(a['signed_residual'])<=.075 for a in hidden['by_contrast']),
            'hidden_pooled_brier':hidden['brier']<=hidden['baseline_brier']}
        models[str(seed)]={'sources':assessments,'gates':gates,'engineering_gates_pass':all(gates.values())}
        traces[str(seed)]={'splits':splits,'features':features,'null_patches':null_patches,
            'null_features':null_features,'instruments':instruments,'bootstrap_indices':indices}
    return traces,{'kind':'all_condition_synthetic_sensory_task_development','models':models,
        'all_engineering_gates_pass':all(m['engineering_gates_pass'] for m in models.values()),
        'goal_achieved':False,'attention_source_tested':False,'participant_rows_tested':0,'new_reports':0}


def check_manifest():
    m=json.loads((ROOT/'manifest.json').read_text())
    for p,h in {**m['source_sha256'],**m['checkpoint_sha256'],**m['legacy_archive_sha256']}.items():assert digest(p)==h,p
    for p,s in json.loads((ROOT/'source_snapshots.json').read_text()).items():
        import hashlib
        assert hashlib.sha256(s.encode()).hexdigest()==m['source_sha256'][p],p


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=('prepare','run','verify'))
    args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=='prepare':
        if (ROOT/'manifest.json').exists():raise SystemExit('refusing overwrite')
        ROOT.mkdir(parents=True,exist_ok=True)
        manifest={'kind':'synthetic_task_development_registration','seeds':SEEDS,'counts':COUNTS,
            'hidden_width':WIDTH,'steps':STEPS,'rate':RATE,'weight_decay':DECAY,'bootstrap_count':BOOTSTRAPS,
            'source_sha256':{p:digest(p) for p in SOURCES},
            'checkpoint_sha256':{str(checkpoint(s)):digest(checkpoint(s)) for s in SEEDS},
            'legacy_archive_sha256':{f'audits/isolated_orientation_task_v1/states-{s}.pt.gz':digest(f'audits/isolated_orientation_task_v1/states-{s}.pt.gz') for s in SEEDS},
            'torch_version':torch.__version__,'threads':2,'single_attempt':True,'human_targets_used':False}
        (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (ROOT/'source_snapshots.json').write_text(json.dumps({p:Path(p).read_text() for p in SOURCES},indent=2)+'\n')
        print('Registered before model outputs');return
    check_manifest()
    if args.stage=='run':
        if (ROOT/'attempt.json').exists():raise SystemExit('refusing retry')
        (ROOT/'attempt.json').write_text(json.dumps({'status':'reserved','no_retry':True})+'\n')
        traces,results=build();hashes={}
        for seed,state in traces.items():
            filename=f'states-{seed}.pt.gz'
            with (ROOT/filename).open('wb') as raw:
                with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as stream:torch.save(state,stream)
            hashes[filename]=digest(ROOT/filename)
        (ROOT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        (ROOT/'completion.json').write_text(json.dumps({'status':'completed','archive_sha256':hashes,
            'results_sha256':digest(ROOT/'results.json'),'goal_achieved':False},indent=2)+'\n')
        print(json.dumps({s:m['gates'] for s,m in results['models'].items()},indent=2))
    else:
        completion=json.loads((ROOT/'completion.json').read_text())
        for p,h in completion['archive_sha256'].items():assert digest(ROOT/p)==h
        assert digest(ROOT/'results.json')==completion['results_sha256']
        traces,results=build()
        for seed,state in traces.items():
            with gzip.open(ROOT/f'states-{seed}.pt.gz','rb') as stream:compare(torch.load(stream,weights_only=True),state)
        assert json.loads((ROOT/'results.json').read_text())==results
        receipt={'exact_reconstruction_pass':True,'goal_achieved':False,'completion_sha256':digest(ROOT/'completion.json')}
        (ROOT/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print('Exact task training, calibration, predictions and assessments replay passed')


if __name__=='__main__':main()
