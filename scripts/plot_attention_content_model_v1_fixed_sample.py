"""Plot one fixed archived fitting sample without invoking any trained module."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch

SEEDS=(1301,1311,1321)
WORLD=0
DECISION=12
SCHEDULE='current'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fit_dir',type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--summary-manifest',required=True,type=Path)
    args=parser.parse_args()
    output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    names=('fixed_sample.json','fixed_sample.png','fixed_sample.pdf','manifest.json')
    if any((output/name).exists() for name in names):
        raise FileExistsError('fixed-sample outputs are exclusive')
    receipt_path=args.fit_dir/'receipt.json'
    receipt_bytes=receipt_path.read_bytes(); receipt=json.loads(receipt_bytes)
    if receipt['phase']!='fit' or receipt['status']!='completed':
        raise ValueError('only completed retained fit is permitted')
    summary_bytes=args.summary_manifest.read_bytes()
    summary_manifest=json.loads(summary_bytes)
    if summary_manifest['receipt_sha256']!=digest(receipt_bytes):
        raise ValueError('summary manifest belongs to a different fitting receipt')
    sample={'schema':'attention_content_model_v1_fixed_fit_field_sample_v1',
        'selection':{'phase':'fit','world_index':WORLD,'decision_time':DECISION,
            'donor_schedule':SCHEDULE,'source':'A','encoder_seeds':list(SEEDS),
            'roles':['red','blue'],'selection_based_on_observed_difference':False},
        'status':{'fields_are_numerical_estimates':True,
            'independent_reconstruction':'incomplete; initial reference failed, correction pending',
            'physical_truth_native_content_and_manner_qualification':'pending',
            'reports_generated':False,'fitting_world_is_independent_evidence':False},
        'receipt_sha256':digest(receipt_bytes),'summary_manifest_sha256':digest(summary_bytes),
        'summary_raw_sha256':summary_manifest['raw_sha256'],
        'source_sha256':digest(Path(__file__).read_bytes()),'seeds':{}}
    for seed in SEEDS:
        path=args.fit_dir/f'states-{seed}.pt.gz'
        compressed=path.read_bytes(); archive_sha=digest(compressed)
        matches=[key for key in receipt['artifact_sha256'] if Path(key).name==path.name]
        if len(matches)!=1 or receipt['artifact_sha256'][matches[0]]!=archive_sha:
            raise ValueError('archive/receipt hash binding failed')
        with gzip.GzipFile(fileobj=io.BytesIO(compressed),mode='rb') as stream:
            bundle=torch.load(stream,map_location='cpu',weights_only=True)
        evaluation=bundle['evaluation']; shared=evaluation['tensors']['shared']
        retained=evaluation['tensors']['sources']['A']
        if evaluation['metadata']['split']!='fit' or evaluation['metadata']['encoder_seed']!=seed:
            raise ValueError('archive metadata mismatch')
        if int(shared['world_indices'][WORLD])!=WORLD:
            raise ValueError('fixed world index does not match archived row')
        address=int(shared['command_addresses'][WORLD,DECISION-1])
        baseline=retained['selected_current_fields'][WORLD,DECISION-1]
        actual=shared['actual_completed_fields'][WORLD,DECISION,address]
        donor=retained['donors'][SCHEDULE]['selected_current_fields'][WORLD,DECISION-12]
        for value in (baseline,actual,donor):
            if value.shape!=(2,4,4) or not torch.isfinite(value).all():
                raise ValueError('unexpected archived fixed-sample field')
        sample['seeds'][str(seed)]={'archive_path':str(path.resolve()),
            'receipt_artifact_key':matches[0],'archive_sha256':archive_sha,
            'address_index':address,'address':f'l{address}',
            'previous_command_index':DECISION-1,
            'previous_command_symbol':int(shared['command_symbols'][WORLD,DECISION-1]),
            'evaluator_truth_cells':shared['evaluator_cells'][WORLD,address].tolist(),
            'baseline_A_current_selected_fields':baseline.tolist(),
            'actual_W12_previous_address_fields':actual.tolist(),
            'donor_patched_A_current_selected_fields':donor.tolist(),
            'archive_keys':{
                'baseline':'evaluation.tensors.sources.A.selected_current_fields[0,11]',
                'actual':f'evaluation.tensors.shared.actual_completed_fields[0,12,{address}]',
                'donor':'evaluation.tensors.sources.A.donors.current.selected_current_fields[0,0]'}}
        print(json.dumps({'event':'fixed_archived_sample_extracted','encoder_seed':seed,
                          'world_index':WORLD,'decision_time':DECISION,'address_index':address}),flush=True)
        del bundle,compressed
    write_json(output/'fixed_sample.json',sample)
    fig,axes=plt.subplots(6,3,figsize=(9.7,13.2),layout='constrained')
    keys=('baseline_A_current_selected_fields','actual_W12_previous_address_fields',
          'donor_patched_A_current_selected_fields')
    titles=('Baseline A_current\n(source estimate)','Actual W12\n(archived H(W))',
            'Donor-patched A_current\n(current schedule; source estimate)')
    image=None
    for encoder_row,seed in enumerate(SEEDS):
        entry=sample['seeds'][str(seed)]
        for role_index,role in enumerate(('red','blue')):
            row=encoder_row*2+role_index
            for column,key in enumerate(keys):
                axis=axes[row,column]
                values=entry[key][role_index]
                image=axis.imshow(values,cmap='viridis',norm=Normalize(0,1),origin='upper')
                axis.set_xticks(range(4)); axis.set_yticks(range(4))
                axis.tick_params(labelsize=8)
                if row==0:
                    axis.set_title(titles[column],fontsize=10)
                if column==0:
                    axis.set_ylabel(f'Encoder {seed} · {role}\n{entry["address"]} · grid row',fontsize=9)
                if row==5:
                    axis.set_xlabel('Grid column',fontsize=9)
                for y in range(4):
                    for x in range(4):
                        value=values[y][x]
                        axis.text(x,y,f'{value:.3f}',ha='center',va='center',fontsize=7,
                                  color='black' if value>.55 else 'white')
    fig.suptitle('Fixed fitting sample: world 0 · decision 12\n'
                 'Archived numerical fields for both roles; no sample selection by outcome',fontsize=12)
    colorbar=fig.colorbar(image,ax=axes,shrink=.65,pad=.02)
    colorbar.set_label('Conditional role localization probability\n(shared scale; not calibrated presence)',fontsize=9)
    fig.supxlabel('Fitting diagnostics. Physical-truth, native-content and manner qualification pending.\n'
                  'Independent reconstruction incomplete. No restoration image or experience claim.',fontsize=9)
    for name,format_name in (('fixed_sample.png','png'),('fixed_sample.pdf','pdf')):
        with (output/name).open('xb') as stream:
            fig.savefig(stream,format=format_name,dpi=220,
                        metadata={'Title':'Fixed archived fitting attention-model fields'} if format_name=='pdf' else None)
    plt.close(fig)
    if receipt_path.read_bytes()!=receipt_bytes:
        raise ValueError('fitting receipt changed during plotting')
    manifest={'schema':'attention_content_model_v1_fixed_fit_field_visualization_manifest_v1',
        'status':'completed_single_fixed_sample_render','source_sha256':sample['source_sha256'],
        'fit_receipt_sha256':sample['receipt_sha256'],
        'summary_manifest_sha256':sample['summary_manifest_sha256'],
        'summary_raw_sha256':sample['summary_raw_sha256'],
        'selection':sample['selection'],'qualification_status':sample['status'],
        'input_archive_sha256':{seed:entry['archive_sha256'] for seed,entry in sample['seeds'].items()},
        'output_sha256':{name:digest((output/name).read_bytes()) for name in names[:3]},
        'no_model_source_forward_renderer_optimizer_world_generation_or_reporter':True}
    write_json(output/'manifest.json',manifest)
    print(json.dumps({'event':'completed_single_fixed_sample_render','output_sha256':manifest['output_sha256']}),flush=True)


if __name__=='__main__':
    main()
