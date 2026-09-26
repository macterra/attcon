"""Package and verify the three-seed attention-model reporting evaluation."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from attcon.attention_model_reporting import SEEDS
from attention_model_study import SOURCES

ARCHIVE = ROOT / 'artifacts/attention_model_checkpoints.tar.gz'
SUMMARY = ROOT / 'audits/attention_model/summary.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_paths():
    return [f'outputs/attention_model/seed{s}.pt' for s in SEEDS]


def package():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w', format=tarfile.USTAR_FORMAT) as archive:
        for name in checkpoint_paths():
            content = (ROOT / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(content), 0o644, 0
            archive.addfile(info, io.BytesIO(content))
    ARCHIVE.write_bytes(gzip.compress(buffer.getvalue(), mtime=0))


def restore(reference):
    if sha(ARCHIVE) != reference['archive_sha256']:
        raise ValueError('checkpoint archive hash mismatch')
    expected = reference['checkpoint_sha256']
    if set(expected) != set(checkpoint_paths()):
        raise ValueError('unexpected checkpoint allowlist')
    contents = {}
    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        for member in archive.getmembers():
            if member.name not in expected or member.name in contents or not member.isfile():
                raise ValueError('unexpected archive member')
            content = archive.extractfile(member).read()
            if hashlib.sha256(content).hexdigest() != expected[member.name]:
                raise ValueError('checkpoint hash mismatch')
            contents[member.name] = content
    if set(contents) != set(expected):
        raise ValueError('missing archive members')
    # Validate every member before writing; never overwrite a mismatched local file.
    for name, content in contents.items():
        path = ROOT / name
        if path.exists() and sha(path) != expected[name]:
            raise ValueError(f'local checkpoint differs: {name}')
    for name, content in contents.items():
        path = ROOT / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)


def build_summary():
    results = []
    artifacts = {}
    for seed in SEEDS:
        path = ROOT / f'audits/attention_model/seed{seed}.json'
        data = json.loads(path.read_text())
        assert data['seed'] == seed and not data['smoke']
        assert data['case_counts'] == {'fit': 512, 'validation': 128, 'test': 256}
        assert data['source_checkpoint']['training_seed'] == seed
        assert data['disjoint_partitions']
        assert data['checkpoint_sha256'] == sha(ROOT / f'outputs/attention_model/seed{seed}.pt')
        assert data['source_sha256'] == {name:sha(ROOT/name) for name in SOURCES}
        cases = ROOT / f'audits/attention_model/seed{seed}_cases.json.gz'
        assert sha(cases) == data['case_file_sha256']
        artifacts[str(path.relative_to(ROOT))] = sha(path)
        artifacts[str(cases.relative_to(ROOT))] = sha(cases)
        results.append(data)
    metrics = ('preference_accuracy', 'balanced_belief_accuracy', 'exact_map_accuracy', 'exact_report_accuracy')
    ranges = {name:{metric:{'min':min(d['primary'][name]['metrics'][metric] for d in results),
                            'max':max(d['primary'][name]['metrics'][metric] for d in results)} for metric in metrics}
              for name in results[0]['primary']}
    gates = {gate:sum(d['gates'][gate] for d in results) for gate in results[0]['gates']}
    interventions = {kind:{
        'paired_state_report_range':[min(d['interventions'][kind]['reporters']['state']['paired_exact_accuracy'] for d in results),
                                     max(d['interventions'][kind]['reporters']['state']['paired_exact_accuracy'] for d in results)],
        'hard_choice_switch_range':[min(d['interventions'][kind]['physical_next_choice_switch_rate'] for d in results),
                                    max(d['interventions'][kind]['physical_next_choice_switch_rate'] for d in results)]}
        for kind in ('donor','flip','erase')}
    return {'audit':'attention_model_reporting_completion_v1', 'seeds':list(SEEDS),
            'registered_matrix_executed': True, 'fitted_reporters':15, 'frozen_controllers':3,
            'test_scenes_per_seed':256, 'ordinary_snapshots_per_seed':1536,
            'all_seed_primary_fidelity': all(d['gates']['primary'] for d in results),
            'all_seed_full_fidelity': all(d['full_fidelity_supported'] for d in results),
            'all_seed_operational_correspondences':all(d['structural_correspondences_supported'] for d in results),
            'independent_phenomenological_correspondence_established':False,
            'source_of_qualia_hypothesis_established':False,
            'theory_verdict':'underdetermined; learned attention-model reporting evaluated, broader phenomenological mechanisms absent and expressive relations authored',
            'gate_pass_counts':gates,'report_ranges':ranges,'interventions':interventions,
            'archive':str(ARCHIVE.relative_to(ROOT)), 'archive_sha256':sha(ARCHIVE),
            'checkpoint_sha256':{name:sha(ROOT/name) for name in checkpoint_paths()},
            'artifact_sha256':artifacts,
            'source_sha256':{name:sha(ROOT/name) for name in (*SOURCES,'scripts/summarize_attention_model.py')},
            'boundary':'Completed finite evaluation is not theoretical success. No human phenomenology validation, native consciousness report, or proof of qualia is claimed.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',action='store_true')
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    if args.package and args.verify: parser.error('choose package or verify')
    if not (args.package or args.verify): parser.error('choose package or verify')
    if args.package: package()
    if args.verify:
        reference=json.loads(SUMMARY.read_text())
        for name,value in reference['source_sha256'].items():
            if sha(ROOT/name)!=value: raise ValueError(f'source changed: {name}')
        restore(reference)
    summary=build_summary()
    if args.replay:
        for seed in SEEDS:
            subprocess.run([sys.executable,str(ROOT/'scripts/attention_model_study.py'),'--seed',str(seed),'--replay'],check=True,cwd=ROOT)
    if args.verify:
        if summary!=reference: raise ValueError('summary differs from recomputed artifacts')
        print('Archive, provenance, artifacts, and summary verified.' + (' All metrics/cases replayed.' if args.replay else ''))
    else:
        SUMMARY.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:summary[k] for k in ('gate_pass_counts','report_ranges','interventions')},indent=2))

if __name__=='__main__':main()
