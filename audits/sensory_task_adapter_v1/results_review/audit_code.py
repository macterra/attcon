import gzip, hashlib, json, math, subprocess, sys, time
from pathlib import Path
import torch
from attcon.bound_content import VisualEncoder

ROOT=Path('audits/sensory_task_adapter_v1')
torch.set_num_threads(2)
Q=torch.tensor([.025,.975],dtype=torch.float64)
max_errors={}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def equivalent(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype,path
        if a.is_floating_point():
            err=float((a-b).abs().max()) if a.numel() else 0.
            max_errors[path.split('/')[0]]=max(max_errors.get(path.split('/')[0],0),err)
            assert torch.allclose(a,b,atol=2e-12,rtol=2e-12), (path,err)
        else:assert torch.equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),(path,a.keys(),b.keys())
        for k in a:equivalent(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equivalent(x,y,path+'/'+str(i))
    elif isinstance(a,float):
        assert math.isclose(a,b,rel_tol=2e-12,abs_tol=2e-12),(path,a,b)
    else:assert a==b,(path,a,b)

def generate(seed,count):
    rng=torch.Generator().manual_seed(seed)
    labels=(torch.arange(count)%2)[torch.randperm(count,generator=torch.Generator().manual_seed(seed+1000000))]
    row,col=torch.meshgrid(torch.arange(16),torch.arange(16),indexing='ij')
    cx=7.5+2*torch.rand(count,1,1,generator=rng)-1
    cy=7.5+2*torch.rand(count,1,1,generator=rng)-1
    radius=3.7+1.5*torch.rand(count,1,1,generator=rng)
    x,y=col[None]-cx,row[None]-cy
    round_mask=x.square()+y.square()<=radius.square()
    square_mask=(x.abs()<=radius)&(y.abs()<=radius)
    kind=torch.randint(2,(count,1,1),generator=rng)
    mask=torch.where(kind==0,round_mask,square_mask)
    phase=2*torch.pi*torch.rand(count,1,1,generator=rng)
    noise=.015*torch.randn(count,6,3,16,16,generator=rng)
    coord=torch.where(labels[:,None,None]==0,row[None],col[None])
    bright=torch.sin(coord*torch.pi/2+phase)>=0
    levels=torch.tensor(((38.2,38.2),(39.,37.3),(39.5,36.5),(40.,36.2),(40.2,36.),(41.1,35.8)))/50
    target=torch.where(bright[:,None],levels[None,:,0,None,None],levels[None,:,1,None,None])
    base=torch.where(mask[:,None],target,torch.tensor(35.5/50))
    patches=(base[:,:,None].expand(-1,-1,3,-1,-1)+noise).clamp(0,1)
    return patches,labels,dict(mask=mask,phase=phase,noise=noise)

@torch.no_grad()
def features(encoder,p):
    leading=p.shape[:-3]
    hidden=encoder.net[:-1](p.reshape(-1,3,16,16)).reshape(*leading,64)
    logits=encoder.net[-1](hidden)
    # Separate full encoder pass replicates production batch numerical behavior.
    full=encoder.net(p.reshape(-1,3,16,16)).reshape(*leading,8)
    c,s=full[...,:4].softmax(-1),full[...,4:].softmax(-1)
    category=torch.cat((c,s),-1)
    conf=torch.stack((c.max(-1).values,s.max(-1).values,-(c*c.clamp_min(1e-30).log()).sum(-1),-(s*s.clamp_min(1e-30).log()).sum(-1)),-1)
    dx=(p[..., :,1:]-p[..., :,:-1]).square().mean((-3,-2,-1))
    dy=(p[..., 1:,:]-p[..., :-1,:]).square().mean((-3,-2,-1))
    return dict(visual_hidden=hidden,visual_logits=logits,category_probabilities=category,category_confidence=conf,physical_directional_edge=(dx-dy)[...,None])

def create(dim):return torch.nn.Sequential(torch.nn.Linear(dim,32),torch.nn.ReLU(),torch.nn.Linear(32,1))

def train(x,labels,seed):
    flat=x.reshape(-1,x.shape[-1]).float()
    mean=flat.mean(0)
    scale=(flat-mean).square().mean(0).sqrt()
    scale=torch.where(scale>1e-6,scale,torch.ones_like(scale))
    norm=(flat-mean)/scale
    y=labels[:,None].expand(-1,6).reshape(-1).float()
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        net=create(flat.shape[-1])
    optim=torch.optim.Adam(net.parameters(),lr=.003,betas=(.9,.999),eps=1e-8,weight_decay=.001,amsgrad=False,foreach=False)
    trace=[]
    for step in range(1500):
        optim.zero_grad(set_to_none=True)
        logits=net(norm)[:,0]
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)
        loss.backward()
        optim.step()
        if step%500==0:trace.append([step,float(loss.detach())])
    with torch.no_grad():final=torch.nn.functional.binary_cross_entropy_with_logits(net(norm)[:,0],y)
    return dict(mean=mean,scale=scale,state_dict={k:v.detach().clone() for k,v in net.state_dict().items()},final_bce=final.detach(),loss_trace=trace,parameter_count=sum(p.numel() for p in net.parameters()),steps=1500,seed=seed)

@torch.no_grad()
def scores(readout,x):
    with torch.random.fork_rng():net=create(x.shape[-1])
    net.load_state_dict(readout['state_dict'])
    net.eval()
    return net((x.float()-readout['mean'])/readout['scale'])[...,0].double()

def calibrate(logit,labels):
    m=logit.abs().flatten().double()
    target=((logit>0).long()==labels[:,None]).double().flatten()
    mean=m.mean();scale=m.std(unbiased=False)
    if float(scale)==0:scale=torch.ones_like(scale)
    design=torch.stack((torch.ones_like(m),(m-mean)/scale),1)
    b=torch.zeros(2,dtype=torch.float64);penalty=.001*torch.eye(2,dtype=torch.float64)
    for iteration in range(100):
        p=(design@b).sigmoid()
        grad=design.T@(p-target)/len(target)+penalty@b
        h=design.T@((p*(1-p))[:,None]*design)/len(target)+penalty
        b-=torch.linalg.solve(h,grad)
    final=(design.T@((design@b).sigmoid()-target)/len(target)+penalty@b).abs().max()
    objective=(torch.nn.functional.softplus(design@b)-target*(design@b)).mean()+.001/2*b.square().sum()
    assert final<=1e-8
    return dict(mean=mean,scale=scale,beta=b,baseline=target.mean(),gradient_max=final,objective=objective)

def probabilities(cal,logit):return (cal['beta'][0]+cal['beta'][1]*(logit.abs().double()-cal['mean'])/cal['scale']).sigmoid()

def stats(p,y):
    p=p.flatten().double();y=y.flatten().double()
    assigned=torch.bucketize(p.contiguous(),torch.arange(1,10,dtype=torch.float64)/10,right=True)
    table=[];ece=0.
    for bin in range(10):
        select=assigned==bin;n=int(select.sum())
        pm=float(p[select].mean()) if n else None
        acc=float(y[select].mean()) if n else None
        if n:ece+=(n/len(p))*abs(pm-acc)
        table.append(dict(bin=bin,count=n,mean_probability=pm,accuracy=acc))
    return dict(count=len(p),accuracy=float(y.mean()),mean_probability=float(p.mean()),signed_residual=float((p-y).mean()),brier=float((p-y).square().mean()),ece=ece,reliability=table)

def ci(x):return torch.quantile(x,Q,dim=0).movedim(0,-1).tolist()

def assess(p,y,logit,labels,baseline,global_baseline,index):
    result=stats(p,y)
    result['by_contrast']=[]
    for c in range(6):
        item=stats(p[:,c],y[:,c])
        item['calibration_mean_baseline']=float(baseline[c])
        item['constant_brier']=float((baseline[c]-y[:,c]).square().mean())
        item['true_orientation']={str(l):stats(p[labels==l,c],y[labels==l,c]) for l in (0,1)}
        item['predicted_orientation']={}
        for l in (0,1):
            keep=(logit[:,c]>0).long()==l
            item['predicted_orientation'][str(l)]=stats(p[keep,c],y[keep,c]) if keep.any() else {'count':0}
        result['by_contrast'].append(item)
    # Independently calculate bootstrapped ECE from explicit selected samples,
    # rather than the study's signed one-hot contribution shortcut.
    boot_ece=[]
    for selected in index:
        boot_ece.append([stats(p[selected,c],y[selected,c])['ece'] for c in range(6)])
    diff=(p-y).square()-(baseline-y).square()
    result['bootstrap']=dict(accuracy_95_by_contrast=ci(y[index].mean(1)),signed_residual_95_by_contrast=ci((p-y)[index].mean(1)),ece_95_by_contrast=ci(torch.tensor(boot_ece,dtype=torch.float64)),brier_vs_conditional_constant_95_by_contrast=ci(diff[index].mean(1)))
    result['calibration_mean_baseline']=float(global_baseline)
    result['baseline_brier']=float((global_baseline-y).square().mean())
    delta=((p-y).square()-(global_baseline-y).square()).mean(1)
    result['pooled_brier_difference_95']=ci(delta[index].mean(1))
    return result

def main():
    started=time.time()
    manifest=json.loads((ROOT/'manifest.json').read_text());completion=json.loads((ROOT/'completion.json').read_text())
    for group in ('source_sha256','checkpoint_sha256','legacy_archive_sha256'):
        for path,h in manifest[group].items():assert sha(path)==h,path
    for path,source in json.loads((ROOT/'source_snapshots.json').read_text()).items():
        assert hashlib.sha256(source.encode()).hexdigest()==manifest['source_sha256'][path]
        frozen=subprocess.check_output(['git','show','7f80599dc42e4f672d096c1d01f28d6e697ec3c7:'+path],cwd='/tmp/attcon-review-milestones-vvjs166i')
        assert hashlib.sha256(frozen).hexdigest()==manifest['source_sha256'][path],path
    for path,h in completion['archive_sha256'].items():assert sha(ROOT/path)==h,path
    assert sha(ROOT/'results.json')==completion['results_sha256']
    predecessor=Path('audits/isolated_orientation_task_v1')
    old_completion=json.loads((predecessor/'completion.json').read_text())
    assert sha(predecessor/'results.json')==old_completion['results_sha256']
    for path,h in old_completion['archive_sha256'].items():assert sha(predecessor/path)==h
    execution=json.loads((ROOT/'run_execution.json').read_text())
    assert execution['saved_tip_before_execution']=='7f80599dc42e4f672d096c1d01f28d6e697ec3c7' and execution['exit_code']==0
    assert sha(ROOT/'run_execution.log')==execution['log_sha256']
    report=json.loads((ROOT/'results.json').read_text())
    summaries={}
    for seed in (1301,1311,1321):
        print('independent reconstruction',seed,flush=True)
        with gzip.open(ROOT/f'states-{seed}.pt.gz','rb') as stream:saved=torch.load(stream,weights_only=True)
        with gzip.open(predecessor/f'states-{seed}.pt.gz','rb') as stream:old=torch.load(stream,weights_only=True)['instruments']
        encoder=VisualEncoder().eval()
        encoder.load_state_dict(torch.load(f'audits/bound_content/confirmation_v2/seed{seed}.pt',weights_only=True)['state_dict'])
        splits=[generate(930000000+seed+10000*split,n) for split,n in enumerate((1024,256,512))]
        equivalent(saved['splits'],splits,f'{seed}/splits')
        feats=[features(encoder,x[0]) for x in splits]
        equivalent(saved['features'],feats,f'{seed}/features')
        null=splits[2][0][:,0,None].expand(-1,2,-1,-1,-1).clone();nf=features(encoder,null)
        equivalent(saved['null_patches'],null,f'{seed}/null_patches');equivalent(saved['null_features'],nf,f'{seed}/null_features')
        indices=torch.randint(512,(256,512),generator=torch.Generator().manual_seed(950000000+seed))
        equivalent(saved['bootstrap_indices'],indices,f'{seed}/bootstrap')
        assessments={}
        for source in feats[0]:
            fit=train(feats[0][source],splits[0][1],940000000+seed)
            predictions=[scores(fit,f[source]) for f in feats]
            cal=calibrate(predictions[1],splits[1][1])
            p=probabilities(cal,predictions[2]);y=((predictions[2]>0).long()==splits[2][1][:,None]).double()
            cy=((predictions[1]>0).long()==splits[1][1][:,None]).double();base=cy.mean(0)
            ns=scores(fit,nf[source]);np=probabilities(cal,ns)
            assert torch.equal(nf[source][:,0],nf[source][:,1]) and torch.equal(ns[:,0],ns[:,1]) and torch.equal(np[:,0],np[:,1])
            oldfit=old[source]['readout'];os=(feats[2][source].double()-oldfit['mean'])/oldfit['scale']@oldfit['weights']+oldfit['intercept']
            op=probabilities(old[source]['calibrator'],os);oy=((os>0).long()==splits[2][1][:,None]).double()
            instruments=dict(readout=fit,scores=predictions,calibrator=cal,probabilities=p,correct=y,conditional_baseline=base,null_scores=ns,null_probabilities=np,legacy_scores=os,legacy_probabilities=op,legacy_correct=oy)
            equivalent(saved['instruments'][source],instruments,f'{seed}/instrument/{source}')
            assessment=assess(p,y,predictions[2],splits[2][1],base,cal['baseline'],indices)
            assessment['parameter_count']=fit['parameter_count'];assessment['final_training_bce']=float(fit['final_bce'])
            assessment['frozen_legacy_comparator']=dict(pooled=stats(op,oy),by_contrast=[stats(op[:,c],oy[:,c]) for c in range(6)])
            assessment['duplicate_null_accuracy']=float(((ns>0).long()==torch.arange(2)[None]).double().mean())
            equivalent(report['models'][str(seed)]['sources'][source],assessment,f'{seed}/assessment/{source}')
            assessments[source]=assessment
            print(' verified',source,flush=True)
        hidden=assessments['visual_hidden'];physical=assessments['physical_directional_edge']
        gates=dict(hidden_high_contrast=hidden['by_contrast'][-1]['accuracy']>=.90,physical_high_contrast=physical['by_contrast'][-1]['accuracy']>=.90,hidden_zero_gross_chance=abs(hidden['by_contrast'][0]['accuracy']-.5)<=.075,physical_zero_gross_chance=abs(physical['by_contrast'][0]['accuracy']-.5)<=.075,all_duplicate_null_chance=all(a['duplicate_null_accuracy']==.5 for a in assessments.values()),hidden_all_contrast_ece=all(a['ece']<=.10 for a in hidden['by_contrast']),hidden_all_contrast_signed_residual=all(abs(a['signed_residual'])<=.075 for a in hidden['by_contrast']),hidden_pooled_brier=hidden['brier']<=hidden['baseline_brier'])
        equivalent(report['models'][str(seed)]['gates'],gates,f'{seed}/gates')
        assert report['models'][str(seed)]['engineering_gates_pass']==all(gates.values())
        summaries[str(seed)]={'gates':gates,'hidden_accuracy':[x['accuracy'] for x in hidden['by_contrast']],'hidden_ece':[x['ece'] for x in hidden['by_contrast']],'hidden_residual':[x['signed_residual'] for x in hidden['by_contrast']],'hidden_pooled_brier_difference':hidden['brier']-hidden['baseline_brier'],'hidden_legacy_accuracy':[x['accuracy'] for x in hidden['frozen_legacy_comparator']['by_contrast']]}
    assert report['all_engineering_gates_pass']==all(all(v['gates'].values()) for v in summaries.values())
    assert report['goal_achieved'] is False and report['attention_source_tested'] is False and report['participant_rows_tested']==0 and report['new_reports']==0
    out=dict(independent_reconstruction_pass=True,source_and_checkpoint_and_legacy_hashes_pass=True,freeze_commit_sources_pass=True,predecessor_completion_hashes_pass=True,max_tensor_error_by_model=max_errors,summaries=summaries,elapsed_seconds=time.time()-started,completion_sha256=sha(ROOT/'completion.json'),manifest_sha256=sha(ROOT/'manifest.json'),audit_script_sha256=sha(__file__))
    Path('/tmp/sensory_adapter_audit_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
