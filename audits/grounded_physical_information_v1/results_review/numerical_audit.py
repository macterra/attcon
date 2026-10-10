"""Post-output independent archive arithmetic; imports no production module.

One invocation only; no sensor/network inference, no new collection, no retraining.
Independent numpy image/statistic arithmetic and unnormalized ridge normal equations.
"""
import datetime, gzip, hashlib, json, traceback
from pathlib import Path
import numpy as np
import torch

ROOT=Path('audits/grounded_physical_information_v1')
OUT=ROOT/'results_review'
CHECKS=[]
MODELS={}
INVENTORY={}
ERRORS=[]
torch.set_num_threads(2)
np.seterr(divide='ignore',invalid='ignore')

def plain(x):
    if isinstance(x,torch.Tensor): return plain(x.detach().cpu().numpy())
    if isinstance(x,np.ndarray): return plain(x.tolist())
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(float,np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,np.bool_): return bool(x)
    return x

def check(name,condition,**detail):
    CHECKS.append({'name':name,'passed':bool(condition),**plain(detail)})
    if not condition: print('CHECK FAILURE',name,plain(detail),flush=True)

def a(x): return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)

def close(name,x,y,atol=1e-10,rtol=1e-8):
    xx=np.asarray(a(x),dtype=float); yy=np.asarray(a(y),dtype=float)
    same=xx.shape==yy.shape
    delta=float(np.nanmax(np.abs(xx-yy))) if same and xx.size else 0.
    check(name,same and np.allclose(xx,yy,atol=atol,rtol=rtol,equal_nan=True),maximum_absolute_difference=delta,atol=atol,rtol=rtol)

def exact(name,x,y): check(name,torch.equal(x,y) if isinstance(x,torch.Tensor) else np.array_equal(x,y))

def load(name):
    with gzip.open(ROOT/(name+'.pt.gz'),'rb') as f: return torch.load(f,weights_only=True)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def target(pixels):
    x=a(pixels).astype(np.float64)
    flat=x.reshape(*x.shape[:-3],-1)
    sd=np.std(flat,axis=-1,ddof=0)
    dh=np.diff(x,axis=-1).reshape(*x.shape[:-3],-1)
    dv=np.diff(x,axis=-2).reshape(*x.shape[:-3],-1)
    rms=np.sqrt((np.einsum('...i,...i->...',dh,dh)/dh.shape[-1]+np.einsum('...i,...i->...',dv,dv)/dv.shape[-1])/2)
    return np.stack([sd,rms],axis=-1)

def independent_fit(tx,ty,ta,cx,cy,alphas):
    tx=torch.as_tensor(tx,dtype=torch.float64); ty=torch.as_tensor(ty,dtype=torch.float64)
    cx=torch.as_tensor(cx,dtype=torch.float64); cy=torch.as_tensor(cy,dtype=torch.float64)
    # All centers/scales are estimated solely on the training rows.
    xm=tx.mean(0); ym=ty.mean(0)
    raw_x=torch.sqrt(torch.sum((tx-xm)**2,dim=0)/tx.shape[0])
    raw_y=torch.sqrt(torch.sum((ty-ym)**2,dim=0)/ty.shape[0])
    xs=raw_x.clone(); xs[xs<=1e-6]=1
    ys=raw_y.clone(); ys[ys<=1e-6]=1
    z=(tx-xm)/xs; yc=(ty-ym)/ys
    answer=torch.as_tensor(ta,dtype=torch.float64); am=answer.mean()
    rhs=torch.column_stack([yc,answer-am]); cv=(cx-xm)/xs; ct=(cy-ym)/ys
    # Unnormalized equivalent normal equation: X'X + n alpha I.
    normal=z.T@z; cross=z.T@rhs
    candidates=[]; losses=[]
    for alpha in alphas:
        w=torch.linalg.solve(normal+len(z)*alpha*torch.eye(z.shape[1],dtype=torch.float64),cross)
        candidates.append(w)
        err=cv@w[:,:2]-ct
        losses.append(float(torch.sum(err*err)/err.numel()))
    ix=int(np.argmin(losses))
    return {'feature_mean':xm,'feature_scale':xs,'target_mean':ym,'target_scale':ys,'answer_mean':am,'weights':candidates[ix],'alpha':alphas[ix],'losses':losses,'raw_feature_rms':raw_x,'raw_target_rms':raw_y,'train_rows':len(tx),'cal_rows':len(cx)}

def decode(fits,x):
    x=torch.as_tensor(x,dtype=torch.float64); n=len(x)
    phys=[]; scores=[]; base=[]
    for loc in range(4):
        f=fits[0] if len(fits)==1 else fits[loc]
        raw=((x[:,loc]-f['feature_mean'])/f['feature_scale'])@f['weights']
        phys.append(raw[:,:2]*f['target_scale']+f['target_mean'])
        scores.append(raw[:,2]+f['answer_mean'])
        base.append(f['target_mean'].expand(n,2))
    score=torch.stack(scores,dim=1)
    return {'physical':torch.stack(phys,dim=1),'answer_score':score,'answer':(score>.5).long(),'train_mean_baseline':torch.stack(base,dim=1)}

def compare_prediction(name,p,q):
    for key in p:
        if key=='answer': exact(name+'/'+key,p[key],q[key])
        else: close(name+'/'+key,p[key],q[key])

def interval(x):
    x=np.asarray(x); shape=x.shape[1:]; flat=x.reshape(len(x),-1)
    b=[]; counts=[]
    for col in flat.T:
        v=col[np.isfinite(col)]; counts.append(len(v))
        b.append(np.quantile(v,[.025,.975]).tolist() if len(v) else [np.nan,np.nan])
    return np.asarray(b).reshape(*shape,2),np.asarray(counts).reshape(shape)

def group(y,p,base,answer,truth,mask,world_weights):
    sq=(p-y)**2*mask[...,None]; ab=np.abs(p-y)*mask[...,None]; en=(base-y)**2*mask[...,None]
    cr=(answer==truth)*mask; n=mask.sum(0)
    def stats(sq,ab,en,n,cr):
        total=n.sum(-1)
        local={'r2_vs_train_mean':1-sq/en,'mae':ab/n[...,None],'rmse':np.sqrt(sq/n[...,None]),'stored_answer_fidelity':cr/n}
        pooled={'r2_vs_train_mean':1-sq.sum(-2)/en.sum(-2),'mae':ab.sum(-2)/total[...,None],'rmse':np.sqrt(sq.sum(-2)/total[...,None]),'stored_answer_fidelity':cr.sum(-1)/total}
        macro={k:np.nanmean(v,axis=-1 if k=='stored_answer_fidelity' else -2) for k,v in local.items()}
        return pooled,local,macro
    point=stats(sq.sum(0),ab.sum(0),en.sum(0),n,cr.sum(0))
    def weighted(v): return (world_weights@v.reshape(len(v),-1)).reshape(len(world_weights),*v.shape[1:])
    boot=stats(weighted(sq),weighted(ab),weighted(en),weighted(mask),weighted(cr))
    means=y[mask].mean(0) if mask.any() else np.full(2,np.nan)
    var=y[mask].var(0,ddof=0) if mask.any() else np.full(2,np.nan)
    return {'count':int(mask.sum()),'by_location':n,'worlds':int(mask.any(1).sum()),'mean':means,'var':var,'point':point,'boot':boot}

def compare_group(name,g,saved):
    check(name+'/counts',g['count']==saved['seen_rows'] and g['worlds']==saved['worlds_with_seen_rows'] and g['by_location'].tolist()==saved['seen_rows_by_location'])
    close(name+'/target_mean',g['mean'],[np.nan if x is None else x for x in saved['target_mean']])
    close(name+'/target_variance',g['var'],[np.nan if x is None else x for x in saved['target_population_variance']])
    for k,label in enumerate(['pooled','per_location','macro_location']):
        for metric,v in g['point'][k].items():
            s=saved[label][metric]; vv=np.asarray(s['value'],dtype=float)
            # Fidelity is serialized as singleton output for pooled/location entries.
            close(name+'/'+label+'/'+metric,np.asarray(v).reshape(-1),vv.reshape(-1))
            bounds,counts=interval(g['boot'][k][metric])
            # Missing intervals are null per output in production JSON.
            # Expand null to two undefined endpoints only for comparison.
            def expand_missing(v):
                if v is None: return [np.nan,np.nan]
                if isinstance(v,list): return [expand_missing(x) for x in v]
                return v
            sb=np.asarray(expand_missing(s['conditional_world_bootstrap_95']),dtype=float)
            close(name+'/'+label+'/'+metric+'/CI',bounds.reshape(-1),sb.reshape(-1))
            close(name+'/'+label+'/'+metric+'/valid',counts.reshape(-1),np.asarray(s['valid_bootstrap_counts']).reshape(-1),atol=0,rtol=0)


def review_model(seed,manifest,results):
    prefix=f'model-{seed}'
    states=[]; worlds=[]; views=[]; targets=[]; forecasts=[]
    split_inventory=[]
    for split,count in enumerate([512,256,512]):
        pre=f'{prefix}/split-{split}/'; st=load(pre+'snapshot'); w=load(pre+'world_truth'); d=load(pre+'decoded'); px=load(pre+'pixels'); no=load(pre+'noise'); fs=load(pre+'forecasts'); vf=load(pre+'features')
        check(pre+'shapes',px.shape==(count,8,3,16,16) and no.shape==px.shape and d['commands'].shape==(count,8))
        exact(pre+'actual_delivery',d['delivered'],torch.gather(w['command_mapping'],1,d['commands']))
        exact(pre+'mapping_is_permutation',w['command_mapping'].sort(1).values,torch.arange(4).expand(count,4))
        ws=torch.arange(1300000000+seed*10000+split*1000,1300000000+seed*10000+split*1000+count)
        exact(pre+'world_rng_namespace',st['world_seeds'],ws)
        exact(pre+'command_rng',d['commands'],torch.randint(4,(count,8),generator=torch.Generator().manual_seed(1400000000+seed*10000+split*1000)))
        exact(pre+'noise_rng',no,.015*torch.randn(count,8,3,16,16,generator=torch.Generator().manual_seed(1500000000+seed*10000+split*1000)))
        memory=torch.zeros(count,4,66); history=torch.zeros(count,4,4); last=torch.full((count,4),-1); lp=torch.zeros(count,4,3,16,16)
        # Independent last-occurrence queries; no production update loop.
        for i in range(count):
            for loc in range(4):
                occurrences=torch.where(d['delivered'][i]==loc)[0]
                if len(occurrences):
                    step=int(occurrences[-1]); last[i,loc]=step; lp[i,loc]=px[i,step]
                    memory[i,loc]=torch.cat([d['hidden'][i,step],d['raw_logits'][i,step].reshape(1),torch.ones(1)])
            for cmd in range(4):
                occurrences=torch.where(d['commands'][i]==cmd)[0]
                if len(occurrences): history[i,cmd,int(d['delivered'][i,int(occurrences[-1])])]=1
        exact(pre+'latest_memory',memory,st['sensory'][:,0]); exact(pre+'actual_history',history,st['history'][:,0]); exact(pre+'last_step',last,st['last_step']); exact(pre+'last_pixels',lp,st['last_pixels']); exact(pre+'seen',last>=0,st['seen']); exact(pre+'aux_logit_sign',(memory[...,64]>0).long(),st['stored_answers'])
        check(pre+'unseen_placeholders',not torch.any(memory[~st['seen']]) and not torch.any(lp[~st['seen']]))
        y=target(lp); close(pre+'pooled_RGB_SD_and_neighbor_RMS',y,st['physical_targets'],atol=1e-14,rtol=1e-12)
        check(pre+'finite',all(torch.isfinite(t).all() for t in [px,no,d['hidden'],d['raw_logits'],st['sensory']]))
        primary=fs['structured']; joint=fs['joint']; mem=memory
        expected={'risk_hidden':primary['risk_hidden'][:,0],'joint_hidden':joint['joint_hidden'][:,0,None].repeat(1,4,1),'sensory_hidden':mem[...,:64],'outcomes_only':torch.stack([primary['current'][:,0],primary['refresh'][:,0]],-1),'rawlogit_outcomes':torch.stack([mem[...,64],primary['current'][:,0],primary['refresh'][:,0]],-1),'route_hidden':primary['route_hidden'][:,0,None].repeat(1,4,1),'observed_only':mem[...,65:66]}
        for name,x in expected.items(): exact(pre+'view_export/'+name,x,vf[name])
        close(pre+'structured_after_composition',primary['after'],primary['delivery']*primary['refresh'].unsqueeze(-2)+(1-primary['delivery'])*primary['current'].unsqueeze(-2),atol=0,rtol=0)
        states.append(st); worlds.append(w); targets.append(y); forecasts.append(fs); views.append(vf)
        split_inventory.append({'split':split,'worlds':count,'seen_rows':int(st['seen'].sum()),'unseen_rows':int((~st['seen']).sum()),'zero_contrast_seen':int((st['seen'] & (w['contrasts']==0)).sum()),'low_contrast_seen':int((st['seen'] & (w['contrasts']==1)).sum())})
    all_ws=torch.cat([s['world_seeds'] for s in states]); check(prefix+'/split_nonoverlap',len(all_ws.unique())==1280)
    archived=load(prefix+'/decoders'); savedpred=load(prefix+'/evaluation_predictions'); samples=load(prefix+'/bootstrap_world_indices')
    exact(prefix+'/bootstrap_rng',samples,torch.randint(512,(256,512),generator=torch.Generator().manual_seed(1600000000+seed*10000)))
    ww=np.stack([np.bincount(row,minlength=512) for row in a(samples)]).astype(float)
    model={'split_inventory':split_inventory,'readouts':{},'paired_error_comparisons':{},'correlations':{},'interventions':{}}
    ownfits={}; ownpred={}; seen=[a(s['seen']) for s in states]
    for name in manifest['view_dimensions']:
        local=name in manifest['shared_local_views']; fits=[]
        for i,loc in enumerate([None] if local else range(4)):
            tx=a(views[0][name])[seen[0]] if local else a(views[0][name])[:,loc][seen[0][:,loc]]
            cx=a(views[1][name])[seen[1]] if local else a(views[1][name])[:,loc][seen[1][:,loc]]
            ty=targets[0][seen[0]] if local else targets[0][:,loc][seen[0][:,loc]]
            cy=targets[1][seen[1]] if local else targets[1][:,loc][seen[1][:,loc]]
            ta=a(states[0]['stored_answers'])[seen[0]] if local else a(states[0]['stored_answers'])[:,loc][seen[0][:,loc]]
            f=independent_fit(tx,ty,ta,cx,cy,manifest['alphas']); saved=archived[name]['fits'][i]; tag=f'{prefix}/fit/{name}/{i}'
            for k in ['feature_mean','feature_scale','target_mean','target_scale','answer_mean','weights']: close(tag+'/'+k,f[k],saved[k])
            close(tag+'/all_candidate_calibration_losses',f['losses'],saved['calibration_standardized_physical_mse'])
            check(tag+'/physical_calibration_only_alpha',f['alpha']==saved['alpha'] and saved['alpha_candidates']==manifest['alphas'])
            check(tag+'/fitting_row_counts',f['train_rows']==saved['train_seen_rows'] and f['cal_rows']==saved['calibration_seen_rows'])
            fits.append(f)
        ownfits[name]=fits; p=decode(fits,views[2][name]); ownpred[name]=p
        compare_prediction(prefix+'/test_prediction/'+name,p,savedpred[name])
        masks={'all_seen':seen[2]}
        for c in range(6): masks[f'per_contrast/{c}']=seen[2] & (a(worlds[2]['contrasts'])==c)
        for c in range(2): masks[f'per_orientation/{c}']=seen[2] & (a(worlds[2]['labels'])==c)
        bins=np.searchsorted([.2,.4,.6,.8],a(forecasts[2]['structured']['current'][:,0]).astype(float),side='right')
        for c in range(5): masks[f'current_success_bins/{c}']=seen[2] & (bins==c)
        groups={}
        for label,mask in masks.items():
            g=group(targets[2],a(p['physical']),a(p['train_mean_baseline']),a(p['answer']),a(states[2]['stored_answers']),mask,ww)
            sr=results['readouts'][name]['assessment']
            for part in label.split('/'): sr=sr[part]
            compare_group(prefix+'/'+name+'/'+label,g,sr)
            groups[label]={'seen_rows':g['count'],'by_location':g['by_location'],'pooled':g['point'][0],'per_location':g['point'][1],'macro_location':g['point'][2],'target_mean':g['mean'],'target_variance':g['var'],'pooled_CI':{k:interval(v)[0] for k,v in g['boot'][0].items()}}
        parameter_count=sum(f['weights'].numel()+3 for f in fits)
        check(prefix+'/'+name+'/budget',parameter_count==archived[name]['parameter_count']==results['readouts'][name]['parameter_count'])
        model['readouts'][name]={'dimensions':manifest['view_dimensions'][name],'decoder_count':len(fits),'parameters':parameter_count,'fits':[{'alpha':f['alpha'],'all_calibration_losses':f['losses'],'train_rows':f['train_rows'],'calibration_rows':f['cal_rows'],'features_using_RMS_floor':int((f['raw_feature_rms']<=1e-6).sum()),'targets_using_RMS_floor':int((f['raw_target_rms']<=1e-6).sum())} for f in fits],'groups':groups}
    y=targets[2]; m=seen[2]; ref=(a(ownpred['risk_hidden']['physical'])-y)**2
    for name in ['outcomes_only','rawlogit_outcomes','sensory_hidden']:
        delta=((a(ownpred[name]['physical'])-y)**2-ref)*m[...,None]
        world_error=delta.sum(1); boot=(ww@world_error)/(ww@m.sum(1))[:,None]; point=delta.sum((0,1))/m.sum(); bounds,counts=interval(boot)
        sr=results['risk_associations']['paired_error_comparisons'][name]
        close(prefix+'/paired/'+name+'/point',point,sr['other_minus_risk_hidden_mse']); close(prefix+'/paired/'+name+'/CI',bounds,sr['conditional_world_bootstrap_95'])
        model['paired_error_comparisons'][name]={'other_minus_risk_MSE':point,'paired_world_CI':bounds,'bootstrap_counts':counts}
    r=np.stack([a(forecasts[2]['structured']['current'][:,0]),a(forecasts[2]['structured']['refresh'][:,0]),np.abs(a(states[2]['sensory'][:,0,:,64]))],axis=-1).astype(float)
    for label,mask in [('all_seen',m)]+[(f'per_contrast/{c}',m & (a(worlds[2]['contrasts'])==c)) for c in range(6)]+[(f'per_orientation/{c}',m & (a(worlds[2]['labels'])==c)) for c in range(2)]:
        corr=np.corrcoef(np.column_stack([r[mask],y[mask]]),rowvar=False)[:3,3:] if mask.sum()>=2 else None
        sr=results['risk_associations']
        for part in label.split('/'): sr=sr[part]
        if corr is not None: close(prefix+'/correlation/'+label,corr,np.asarray(sr['pearson'],dtype=float))
        model['correlations'][label]={'count':int(mask.sum()),'current_refresh_abslogit_vs_physical':corr}
    trace=load(prefix+'/interventions'); original=trace['original']; changed=trace['readdressed']; restored=trace['restored']; patched=trace['patched_forecasts']
    exact(prefix+'/intervention/original_sensory',trace['original_sensory'],states[2]['sensory']); exact(prefix+'/intervention/history_fixed',trace['history'],states[2]['history']); exact(prefix+'/intervention/native_rotation',trace['readdressed_sensory'],states[2]['sensory'].roll(1,-2)); exact(prefix+'/intervention/inverse_memory',trace['restored_sensory'],states[2]['sensory'])
    for k,v in original.items():
        exact(prefix+'/intervention/original/'+k,v,forecasts[2]['structured'][k]); exact(prefix+'/intervention/inverse_forecast/'+k,restored[k],v)
        if k not in ['delivery','after']: exact(prefix+'/intervention/patch_invariant/'+k,patched[k],v)
    exact(prefix+'/intervention/memory_route_unchanged',changed['delivery'],original['delivery'])
    for k in ['risk_hidden','current','refresh','current_logits','refresh_logits']:
        expectation=original[k].roll(1,-2 if k=='risk_hidden' else -1); err=(changed[k]-expectation).abs()
        check(prefix+'/intervention/source_tol/'+k,torch.all(err<=1e-6+1e-5*expectation.abs()),max_error=float(err.max()))
        model['interventions'][k]={'maximum_absolute_permutation_error':float(err.max())}
    exact(prefix+'/intervention/patch_delivery',patched['delivery'],original['delivery'].roll(1,-2)); exact(prefix+'/intervention/patch_inverse_delivery',trace['restored_patch_delivery'],original['delivery']); exact(prefix+'/intervention/patch_inverse_after',trace['restored_patch_after'],original['after'])
    exact(prefix+'/intervention/post_command_composition',patched['after'],patched['delivery']*patched['refresh'].unsqueeze(-2)+(1-patched['delivery'])*patched['current'].unsqueeze(-2))
    for name in manifest['shared_local_views']:
        x=trace['readdressed_local_features'][name].double(); expected=views[2][name].roll(1,1).double(); delta=(x-expected).abs(); f=archived[name]['fits'][0]
        check(prefix+'/intervention/feature_tol/'+name,torch.all(delta<=1e-6+1e-5*expected.abs()))
        # Bound is computed independently with archived exact coefficients;
        # own-fit predictions are also checked above and below.
        bound=torch.einsum('nld,dk->nlk',delta/f['feature_scale'],f['weights'][:,:2].abs())*f['target_scale']
        actual=(trace['readdressed_local_decoded'][name]['physical']-trace['original_local_decoded'][name]['physical'].roll(1,1)).abs()
        saved=trace['decode_permutation_error_bounds'][name]
        close(prefix+'/intervention/bound/'+name,bound,saved['propagated_bound'],atol=1e-18,rtol=1e-10); exact(prefix+'/intervention/feature_delta/'+name,delta,saved['absolute_feature_difference']); exact(prefix+'/intervention/output_delta/'+name,actual,saved['absolute_output_difference'])
        check(prefix+'/intervention/linear_allowance/'+name,torch.all(actual<=bound+1e-12),max_output_error=float(actual.max()),max_bound=float(bound.max()))
        for k,v in trace['original_local_decoded'][name].items(): exact(prefix+'/intervention/inverse_readout/'+name+'/'+k,v,trace['restored_local_decoded'][name][k]); exact(prefix+'/intervention/patch_readout_invariant/'+name+'/'+k,v,trace['patch_local_decoded'][name][k])
        compare_prediction(prefix+'/intervention/own_decode/'+name,decode(ownfits[name],x),trace['readdressed_local_decoded'][name])
        model['interventions'][name+'_decoder']={'max_feature_error':float(delta.max()),'max_output_error':float(actual.max()),'max_propagated_bound':float(bound.max()),'minimum_allowance_minus_error':float((bound+1e-12-actual).min())}
    print('Independent archive audit completed',seed,flush=True)
    return model

def main():
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest=json.loads((ROOT/'manifest.json').read_text()); completion=json.loads((ROOT/'completion.json').read_text()); results=json.loads((ROOT/'results.json').read_text()); snapshots=json.loads((ROOT/'source_snapshots.json').read_text())
    for section in ['source_sha256','frozen_input_sha256','design_review_sha256']:
        INVENTORY[section]={}
        for path,expected in manifest[section].items():
            actual=sha(path); INVENTORY[section][path]={'expected':expected,'actual':actual,'matches':actual==expected}; check('hash/'+path,actual==expected)
    INVENTORY['archive_files']={}
    for path,entry in completion['archive_files'].items():
        actual=sha(path); size=Path(path).stat().st_size; INVENTORY['archive_files'][path]={'expected':entry['sha256'],'actual':actual,'bytes':size,'matches':actual==entry['sha256']}; check('archive_hash/'+path,actual==entry['sha256'] and size==entry['bytes'] and size<100000000)
    for path,text in snapshots.items(): check('source_snapshot/'+path,hashlib.sha256(text.encode()).hexdigest()==manifest['source_sha256'][path])
    check('source_snapshot_keys',set(snapshots)==set(manifest['source_sha256']))
    check('manifest_receipt_hash',sha(ROOT/'manifest.json')==completion['manifest_sha256']); check('results_receipt_hash',sha(ROOT/'results.json')==completion['results_sha256']); check('completed_once',completion['status']=='single_registered_attempt_completed' and not (ROOT/'failure.json').exists())
    check('bounded_nonappearance_flags',results['appearance_tested'] is False and results['new_reports']==0 and results['participant_data_used'] is False and results['source_retraining'] is False and results['goal_achieved'] is False)
    for seed in manifest['visual_seeds']:
        try: MODELS[str(seed)]=review_model(seed,manifest,results['models'][str(seed)])
        except Exception as error:
            msg=traceback.format_exc(); ERRORS.append({'model':seed,'error':str(error),'traceback':msg}); print(msg,flush=True)
    report={'kind':'independent_post_output_archive_numerical_audit','started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed' if not ERRORS and all(c['passed'] for c in CHECKS) and len(MODELS)==3 else 'failures_preserved','production_module_imports':False,'fresh_production_attempts':0,'neural_reconstruction':False,'checks':CHECKS,'check_count':len(CHECKS),'failed_checks':[c for c in CHECKS if not c['passed']],'execution_errors':ERRORS,'hash_inventory':INVENTORY,'models':MODELS}
    with (OUT/'numerical_audit.json').open('x') as stream: json.dump(plain(report),stream,indent=2,allow_nan=False); stream.write('\n')
    print('Audit disposition:',report['status'],'checks',len(CHECKS),'failures',len(report['failed_checks']),'errors',len(ERRORS),flush=True)

if __name__=='__main__': main()
