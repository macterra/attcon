"""Companion traversal repair; leaves the frozen study and first verifier unchanged."""
import gzip
import json
from pathlib import Path
import torch
import isolated_orientation_task_v1 as study


def compare(before,after,path='root'):
    if isinstance(before,torch.Tensor):
        assert isinstance(after,torch.Tensor) and torch.equal(before,after),path
    elif isinstance(before,dict):
        assert before.keys()==after.keys(),path
        for key in before:compare(before[key],after[key],path+'/'+str(key))
    elif isinstance(before,(list,tuple)):
        assert type(before)==type(after) and len(before)==len(after),path
        for i,(left,right) in enumerate(zip(before,after)):compare(left,right,path+'/'+str(i))
    else:assert before==after,path


if __name__=='__main__':
    torch.set_num_threads(2)
    study.check_manifest()
    completion=json.loads((study.ROOT/'completion.json').read_text())
    assert study.digest(study.ROOT/'results.json')==completion['results_sha256']
    for file,h in completion['archive_sha256'].items():assert study.digest(study.ROOT/file)==h
    trace,results=study.build()
    for seed,state in trace.items():
        with gzip.open(study.ROOT/f'states-{seed}.pt.gz','rb') as stream:
            compare(torch.load(stream,weights_only=True),state)
    assert json.loads((study.ROOT/'results.json').read_text())==results
    receipt={'exact_reconstruction_pass':True,'frozen_study_modified':False,'results_replaced':False,
        'original_verifier_failure':'Imported comparator lacked list/tuple traversal; tensor list equality raised RuntimeError: Boolean value of Tensor with more than one value is ambiguous',
        'repair':'Separate recursive verifier adds list/tuple traversal; frozen original and outputs retained',
        'companion_source_sha256':study.digest(__file__),'completion_sha256':study.digest(study.ROOT/'completion.json'),
        'goal_achieved':False}
    (study.ROOT/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Exact tensor/fit/calibration/bootstrap/result reconstruction passed with companion traversal repair')
