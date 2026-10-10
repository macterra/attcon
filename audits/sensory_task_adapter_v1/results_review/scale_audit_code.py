import gzip,json
from pathlib import Path
import torch
torch.set_num_threads(2)
out={}
for seed in (1301,1311,1321):
    with gzip.open(f'audits/sensory_task_adapter_v1/states-{seed}.pt.gz','rb') as stream:state=torch.load(stream,weights_only=True)
    out[str(seed)]={}
    for source,images in state['features'][0].items():
        x=images.reshape(-1,images.shape[-1]).float()
        rms=(x-x.mean(0)).square().mean(0).sqrt()
        high=x.double()
        double_rms=(high-high.mean(0)).square().mean(0).sqrt()
        span=high.max(0).values-high.min(0).values
        fit=state['instruments'][source]['readout'];scores=state['instruments'][source]['scores'][2]
        out[str(seed)][source]={
            'dimension':x.shape[-1],'rms_floor_count':int((rms<=1e-6).sum()),'float32_rms':rms.tolist(),
            'float32_rms_max':float(rms.max()),'double_rms':double_rms.tolist(),'double_rms_max':float(double_rms.max()),
            'nonconstant_columns':int((span>0).sum()),'column_range_max':float(span.max()),
            'float32_rms_zero_but_double_positive_columns':int(((rms==0)&(double_rms>0)).sum()),
            'parameter_count':fit['parameter_count'],'loss_trace':fit['loss_trace'],'final_training_bce':float(fit['final_bce']),
            'evaluation_score_min':float(scores.min()),'evaluation_score_max':float(scores.max()),
            'evaluation_vertical_counts_by_contrast':(scores>0).sum(0).tolist()}
Path('/tmp/sensory_adapter_scale_audit_results.json').write_text(json.dumps(out,indent=2)+'\n')
print('Scale/optimization diagnostics archived; no refit or rescore')
