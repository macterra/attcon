"""Post-evaluation descriptive error audit; no new gates or model selection."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from attcon.attention_model_reporting import SEEDS, renderer


def main():
    torch.set_num_threads(1)
    runs={}
    for seed in SEEDS:
        source=ROOT/f'audits/attention_model/seed{seed}_cases.json.gz'
        case=json.loads(gzip.decompress(source.read_bytes()))
        ordinary=case['ordinary']
        m=torch.tensor(ordinary['truth']['m'])
        belief=torch.tensor(ordinary['truth']['belief'])
        preferred=torch.tensor(ordinary['truth']['preference'])
        physical=torch.tensor(ordinary['truth']['physical'])
        pred=torch.tensor(ordinary['predictions']['state']['belief'])
        pred_preferred=torch.tensor(ordinary['predictions']['state']['preference'])
        errors=belief!=pred
        margin=(m-.5).abs()
        shift=preferred[:,1:]!=preferred[:,:-1]
        pred_shift=pred_preferred[:,1:]!=pred_preferred[:,:-1]
        pred_persist=pred[:,1:] & pred[:,:-1]
        persist=belief[:,1:] & belief[:,:-1]
        exact=(~errors).all(-1)&(pred_preferred==preferred)
        mismatch=(belief!=physical).any(-1)
        eligible=torch.where((exact&mismatch).flatten())[0]
        example=None
        if len(eligible):
            scene,step=divmod(int(eligible[0]),6)
            example={'selection':'first exact learned report with model/physical disagreement (post-evaluation illustration)',
                     'scene':scene,'step':step,'context_id':ordinary['context_ids'][scene],
                     'model_and_learned_report':renderer(belief[scene,step],preferred[scene,step]),
                     'physically_inspected':torch.where(physical[scene,step])[0].tolist()}
        runs[str(seed)]={'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                        'belief_cell_error_count':int(errors.sum()),'belief_cell_count':errors.numel(),
                        'error_fraction_within_0_1_of_threshold':float((margin[errors]<=.1).double().mean()),
                        'median_error_distance_from_threshold':float(margin[errors].median()),
                        'model_positive_fraction':float(belief.double().mean()),
                        'always_no_shift_accuracy':float((~shift).double().mean()),
                        'shift_positive_recall':float(pred_shift[shift].double().mean()),
                        'persistence_positive_recall':float(pred_persist[persist].double().mean()),
                        'disagreement_all_negative_accuracy':float((~belief[belief!=physical]).double().mean()),
                        'always_no_persistence_accuracy':float((~persist).double().mean()),
                        'all_negative_map_exact_accuracy':float((~belief.any(-1)).double().mean()),
                        'preferred_cell_counts':torch.bincount(preferred.flatten(),minlength=25).tolist(),
                        'example':example}
    result={'audit':'attention_model_post_evaluation_errors',
            'boundary':'Descriptive diagnosis after evaluation; no thresholds, training choices, or confirmatory results changed.',
            'runs':runs,'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/'audits/attention_model/errors.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(runs,indent=2))

if __name__=='__main__':main()
