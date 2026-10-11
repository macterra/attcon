"""Exact, stdlib-only provenance for the one time-zero reference correction.

Original registration, source closure, native inputs, fit and first failure stay
immutable. This module never imports torch, models, collectors or renderers.
Creation/validation is not authorized merely by preparing this source.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_SCHEMA = 'attention_content_model_v1_time_zero_reference_amendment'
APPROVAL_SCOPE = 'attention_content_model_v1_exact_reference_amendment_preexecution_review'
REPORT_SCHEMA = 'attention_content_model_v1_corrected_reference_phase_reconstruction'
PASS_STATUS = 'passed_corrected_reference_phase_reconstruction'
FAIL_STATUS = 'failed_first_corrected_reference_phase_reconstruction'
ORIGINAL_VERIFIER = 'scripts/verify_attention_content_model_v1.py'
CORRECTED_VERIFIER = 'scripts/verify_attention_content_model_v1_corrected.py'
CORRECTED_RUNNER = 'scripts/run_attention_content_model_v1_corrected.py'
NEW_CLOSURE = (
    CORRECTED_VERIFIER, CORRECTED_RUNNER,
    'scripts/attention_content_model_v1_reference.py',
    'tests/test_attention_content_model_v1_reference.py',
    'audits/attention_content_model_v1/reference_amendment/design.md',
)
REGISTRATION_PATH = 'audits/attention_content_model_v1/registration.json'
REGISTRATION_SHA = '01b545d498ba5a23e521ed4838ec4c403be44272e333cb3ab3c5b696a363d312'
REGISTRATION_COMMIT = '304a052e537115e04b641e4fca3b3c3136fbd739'
FIT_COMMIT = '28406d63c053b810c0fd19348cad84b147f03690'
FIRST_REVIEW_COMMIT = '393eb124449634c0b72597231d6f3cd54ab3645b'
ORIGINAL_VERIFIER_SHA = '11b821bea207f4ef8250fa589148777df9d600bb3dffd01ef600cf4f78a423ab'
FIRST_DIR = 'audits/attention_content_model_v1/fit_reconstruction'
FIRST_RAW_SHA = '75559ba487323ab1a59b0f6ce3ebec23876895b198c59ece6755812f567cb37e'
FIRST_CHECK_COUNT = 27616
FIRST_TRANSPORT_HASHES = {
    FIRST_DIR+'/first_report_manifest.json': 'ac79fcc1ba6e4588c6c2a49275b5673322edc31b668957b28c48832af3143b20',
    FIRST_DIR+'/first_report.json.gz': '7901896463b8a7619f49210f52af5f614b8d809a9eec4944d10fda602a6e52c5',
    FIRST_DIR+'/first_cli_execution.json': 'b8b0a3398a1da0532c54cb4e82c124e1943dc499aa848239d7effb501c442c2a',
    FIRST_DIR+'/first_cli.log': '7486bcddff5a11b72507a39eadbf9788773fc7be7669c6e1fadf9ae6aca57426',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(path):
    path = Path(path)
    resolved = (ROOT/path).resolve() if not path.is_absolute() else path.resolve()
    require(resolved.is_relative_to(ROOT), 'reference artifact outside repository')
    name = resolved.relative_to(ROOT).as_posix()
    if not path.is_absolute():
        require(path.as_posix() == name, 'noncanonical reference artifact path')
    return name


def json_file(path):
    return json.loads((ROOT/canonical(path)).read_text())


def full_commit(value):
    require(isinstance(value, str) and len(value) == 40
            and all(c in '0123456789abcdef' for c in value), 'full lowercase preservation commit required')


def bind(commit, hashes):
    """Stream exact Git blobs; no mutations, pointers or arbitrary hash acceptance."""
    full_commit(commit)
    for name, expected in hashes.items():
        require(canonical(name) == name, 'noncanonical committed reference path')
        require(isinstance(expected,str) and len(expected)==64
                and all(c in '0123456789abcdef' for c in expected), 'invalid expected SHA-256')
        require(digest(ROOT/name) == expected, 'current reference binding failed: '+name)
        process = subprocess.Popen(['git','-C',str(ROOT),'show',commit+':'+name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        h = hashlib.sha256()
        for chunk in iter(lambda: process.stdout.read(1024*1024), b''):
            h.update(chunk)
        error = process.stderr.read().decode(errors='replace')
        status = process.wait()
        require(status == 0 and h.hexdigest() == expected,
                'committed reference binding failed: '+name+' '+error)


def ancestor(older, newer):
    full_commit(older); full_commit(newer)
    result = subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',older,newer],capture_output=True)
    require(result.returncode == 0, 'reference preservation ancestry failed')


def inventory_hash(checks):
    return hashlib.sha256(json.dumps([row['name'] for row in checks],separators=(',',':')).encode()).hexdigest()


def evidence_hash(checks):
    return hashlib.sha256(json.dumps(checks,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def original_prefix_evidence(checks, context):
    """Exact complete first rows; supplementary amendment rows stay separate."""
    expected = first_failed_report()['checks']
    require(len(expected)==context['immutable']['first_failed_check_count']
            and evidence_hash(expected)==context['immutable']['first_failed_evidence_sha256'],
            'original first-prefix evidence identity changed')
    original = [row for row in checks if not row['name'].startswith('reference_amendment/')]
    mismatches = []
    for index, row in enumerate(expected):
        actual = original[index] if index < len(original) else None
        try:
            equal = actual is not None and json.dumps(actual,sort_keys=True,separators=(',',':'),allow_nan=False) == json.dumps(
                row,sort_keys=True,separators=(',',':'),allow_nan=False)
        except (TypeError,ValueError,OverflowError):
            equal = False
        if not equal:
            mismatches.append(dict(index=index,name=row['name'],original=row,corrected=actual))
    return not mismatches, mismatches


def first_failed_report():
    bind(FIRST_REVIEW_COMMIT, FIRST_TRANSPORT_HASHES)
    manifest = json_file(FIRST_DIR+'/first_report_manifest.json')
    packed = ROOT/(FIRST_DIR+'/first_report.json.gz')
    raw = gzip.decompress(packed.read_bytes())
    require(manifest['raw_sha256'] == FIRST_RAW_SHA == hashlib.sha256(raw).hexdigest()
            and manifest['raw_bytes'] == len(raw)
            and manifest['compressed_sha256'] == digest(packed)
            and manifest['compressed_bytes'] == packed.stat().st_size,
            'first failed report transport identity changed')
    report = json.loads(raw)
    require(report['schema'] == 'attention_content_model_v1_independent_phase_reconstruction'
            and report['phase'] == 'fit' and report['status'] == 'failed_first_phase_reconstruction'
            and report['verifier_sha256'] == ORIGINAL_VERIFIER_SHA
            and report['check_count'] == len(report['checks']) == FIRST_CHECK_COUNT
            and all(row['passed'] is True for row in report['checks'])
            and report['errors'] == ['verification_exception'] and report['error_count'] == 1
            and 'initial state is exactly observed V without a command' in report['exception'],
            'first failed report identity/status changed')
    commands = report['command_bindings']
    require(canonical(commands['registration']) == REGISTRATION_PATH
            and commands['registration_commit'] == REGISTRATION_COMMIT
            and commands['phase_commit'] == FIT_COMMIT,
            'first failed report used different registration/fit')
    return report


def immutable_context():
    """JSON/byte bindings only, including all original native/fit archive bytes."""
    from run_attention_content_model_v1 import validate_registration, native_acceptance, receipt
    require(digest(ROOT/REGISTRATION_PATH) == REGISTRATION_SHA, 'original registration changed')
    registration = json_file(REGISTRATION_PATH)
    validate_registration(registration)
    bind(REGISTRATION_COMMIT,{REGISTRATION_PATH:REGISTRATION_SHA,
                             **registration['source_hashes'], **registration['input_hashes']})
    require(registration['source_hashes'][ORIGINAL_VERIFIER] == ORIGINAL_VERIFIER_SHA,
            'registered original verifier changed')
    require(native_acceptance() == registration['input_hashes'], 'accepted original native inputs changed')
    base = ROOT/registration['output_dir']
    fit = receipt(base,'fit',FIT_COMMIT,REGISTRATION_SHA)
    claim_path = canonical(base/'fit'/'claim.json')
    claim = json_file(claim_path)
    bind(FIT_COMMIT,{claim_path:digest(ROOT/claim_path)})
    require(fit['registration_commit'] == REGISTRATION_COMMIT
            and fit['retry_performed'] is False and fit['no_reporter_calls'] is True
            and fit['semantic_qualification_asserted'] is False
            and claim['phase']=='fit' and claim['registration_sha256']==REGISTRATION_SHA
            and claim['registration_commit']==REGISTRATION_COMMIT and claim['retry_allowed'] is False,
            'original first fit receipt/claim changed')
    ancestor(registration['source_commit'],REGISTRATION_COMMIT)
    ancestor(REGISTRATION_COMMIT,FIT_COMMIT); ancestor(FIT_COMMIT,FIRST_REVIEW_COMMIT)
    first = first_failed_report()
    return dict(registration_path=REGISTRATION_PATH,registration_sha256=REGISTRATION_SHA,
        registration_commit=REGISTRATION_COMMIT,original_source_commit=registration['source_commit'],
        original_source_hashes=registration['source_hashes'],native_input_hashes=registration['input_hashes'],
        fit_commit=FIT_COMMIT,fit_receipt_path=canonical(base/'fit'/'receipt.json'),
        fit_receipt_sha256=digest(base/'fit'/'receipt.json'),fit_artifact_sha256=fit['artifact_sha256'],
        fit_claim_path=claim_path,fit_claim_sha256=digest(ROOT/claim_path),
        first_failed_review_commit=FIRST_REVIEW_COMMIT,first_failed_transport_sha256=FIRST_TRANSPORT_HASHES,
        first_failed_report_raw_sha256=FIRST_RAW_SHA,first_failed_check_count=FIRST_CHECK_COUNT,
        first_failed_inventory_sha256=inventory_hash(first['checks']),
        first_failed_evidence_sha256=evidence_hash(first['checks']))


def create_amendment(source_commit, approval_path, amendment_path):
    immutable = immutable_context()
    source_hashes = {name:digest(ROOT/name) for name in NEW_CLOSURE}
    require(digest(__file__)==source_hashes['scripts/attention_content_model_v1_reference.py'],
            'executing reference helper differs from new closure')
    approval_path = canonical(approval_path)
    require(approval_path not in NEW_CLOSURE and approval_path != REGISTRATION_PATH,
            'approval path collides with source/registration')
    approval = json_file(approval_path)
    require(approval['scope']==APPROVAL_SCOPE and approval['approved'] is True
            and approval['approved_source_sha256']==source_hashes
            and approval['original_registration_sha256']==REGISTRATION_SHA
            and approval['first_failed_report_raw_sha256']==FIRST_RAW_SHA,
            'reference approval does not bind exact new bytes/original context')
    approval_hash = digest(ROOT/approval_path)
    bind(source_commit,{**source_hashes,approval_path:approval_hash})
    ancestor(FIRST_REVIEW_COMMIT,source_commit)
    value = dict(schema=AMENDMENT_SCHEMA,version=1,
        created_at_utc=datetime.now(timezone.utc).isoformat(),reference_source_commit=source_commit,
        source_hashes=source_hashes,approval_path=approval_path,approval_sha256=approval_hash,
        immutable=immutable,method_change='Coherent W0=V in noncandidate time-zero decoy only',
        numerical_tolerances=dict(atol=1e-6,rtol=1e-5,discrete='exact'),
        original_scientific_registration_unchanged=True,no_fit_retry=True,no_reporter=True,
        presentation_qualified=False,qualia_established=False,
        shared_model_and_inherited_hypothesis_context=True,independent_model_family=False,human_endorsement=False)
    name = canonical(amendment_path)
    require(name not in source_hashes and name not in immutable['original_source_hashes']
            and name != approval_path and name != REGISTRATION_PATH,
            'amendment path collides with immutable evidence')
    # Fully serialize before the exclusive first creation; no overwrites.
    serialized = json.dumps(value,indent=2,allow_nan=False)+'\n'
    with (ROOT/name).open('x',encoding='utf-8') as stream:
        stream.write(serialized)
    return value


def validate_amendment(path, commit, consumer, registration_path, registration_commit, fit_commit):
    name = canonical(path)
    bind(commit,{name:digest(ROOT/name)})
    value = json_file(name)
    require(value['schema']==AMENDMENT_SCHEMA and value['version']==1, 'wrong reference amendment schema')
    expected_context = immutable_context()
    require(value['immutable']==expected_context, 'reference amendment changes original immutable context')
    require(canonical(registration_path)==REGISTRATION_PATH and registration_commit==REGISTRATION_COMMIT
            and fit_commit==FIT_COMMIT, 'amendment used with different registration/fit')
    require(set(value['source_hashes'])==set(NEW_CLOSURE), 'incomplete or changed reference closure')
    require(digest(__file__)==value['source_hashes']['scripts/attention_content_model_v1_reference.py'],
            'executing reference helper differs from approved exact bytes')
    require(consumer in (CORRECTED_VERIFIER,CORRECTED_RUNNER,'scripts/attention_content_model_v1_reference.py'),
            'unapproved reference consumer')
    approval_path = value['approval_path']
    require(canonical(approval_path)==approval_path and approval_path not in NEW_CLOSURE,
            'invalid reference approval path')
    approval = json_file(approval_path)
    require(approval['scope']==APPROVAL_SCOPE and approval['approved'] is True
            and approval['approved_source_sha256']==value['source_hashes']
            and approval['original_registration_sha256']==REGISTRATION_SHA
            and approval['first_failed_report_raw_sha256']==FIRST_RAW_SHA,
            'reference amendment lacks exact independent approval')
    bind(value['reference_source_commit'],{**value['source_hashes'],approval_path:value['approval_sha256']})
    bind(commit,{**value['source_hashes'],approval_path:value['approval_sha256']})
    ancestor(FIRST_REVIEW_COMMIT,value['reference_source_commit'])
    ancestor(value['reference_source_commit'],commit)
    require(value['method_change']=='Coherent W0=V in noncandidate time-zero decoy only'
            and value['numerical_tolerances']==dict(atol=1e-6,rtol=1e-5,discrete='exact')
            and value['original_scientific_registration_unchanged'] is True
            and value['no_fit_retry'] is True and value['no_reporter'] is True
            and value['presentation_qualified'] is False and value['qualia_established'] is False,
            'amendment changes method/scientific constraints')
    return dict(amendment_path=name,amendment_sha256=digest(ROOT/name),amendment_commit=commit,
        reference_source_commit=value['reference_source_commit'],
        corrected_verifier_sha256=value['source_hashes'][CORRECTED_VERIFIER],
        corrected_runner_sha256=value['source_hashes'][CORRECTED_RUNNER],
        registered_verifier_sha256=ORIGINAL_VERIFIER_SHA,immutable=expected_context)


def report_identity_matches(report, context, phase):
    """Only the single exact reviewed correction; full pass remains mandatory."""
    provenance = report.get('reference_provenance',{})
    return (report.get('schema')==REPORT_SCHEMA and report.get('version')==1
        and report.get('phase')==phase and report.get('status')==PASS_STATUS
        and report.get('verifier_sha256')==context['corrected_verifier_sha256']
        and report.get('registered_verifier_sha256')==ORIGINAL_VERIFIER_SHA
        and provenance==context
        and report.get('hard_invariants_pass') is True and report.get('error_count')==0
        and report.get('errors')==[] and 'exception' not in report and 'serialization_error' not in report
        and 'reference_evidence_exception' not in report
        and report.get('check_count')==len(report.get('checks',[]))>0
        and all(row.get('passed') is True for row in report['checks'])
        and report.get('no_fit_replay') is True and report.get('no_optimizer') is True
        and report.get('no_reporter') is True and report.get('qualia_established') is False
        and report.get('presentation_qualified') is False and report.get('retry_or_repair_performed') is False)


def validate_corrected_fit(manifest_path, report_commit, context):
    """First corrected fit audit must be committed before dev/heldout generation."""
    name = canonical(manifest_path)
    manifest = json_file(name)
    packed = canonical(manifest['published_path'])
    require(packed.endswith('.json.gz'), 'corrected fit report must use lossless gzip transport')
    bind(report_commit,{name:digest(ROOT/name),packed:manifest['compressed_sha256']})
    raw = gzip.decompress((ROOT/packed).read_bytes())
    require(len(raw)==manifest['raw_bytes'] and hashlib.sha256(raw).hexdigest()==manifest['raw_sha256']
            and (ROOT/packed).stat().st_size==manifest['compressed_bytes'], 'corrected fit raw transport changed')
    report = json.loads(raw)
    require(report_identity_matches(report,context,'fit'), 'corrected fit full audit did not pass')
    reproduced,mismatches=original_prefix_evidence(report['checks'],context)
    require(reproduced and not mismatches, 'corrected fit changed original first-prefix evidence')
    require(manifest['schema']=='attention_content_model_v1_lossless_first_corrected_reconstruction_transport'
            and manifest['phase']=='fit' and manifest['status']==PASS_STATUS
            and manifest['first_phase_preservation_commit']==FIT_COMMIT
            and manifest['verifier_sha256']==context['corrected_verifier_sha256']
            and manifest['check_count']==report['check_count'] and manifest['error_count']==0
            and manifest['first_report_unchanged'] is True and manifest['retry_performed'] is False,
            'corrected fit transport metadata differs from full first report')
    commands = report['command_bindings']
    require(canonical(commands['registration'])==REGISTRATION_PATH
            and commands['registration_commit']==REGISTRATION_COMMIT and commands['phase_commit']==FIT_COMMIT
            and commands['fit_commit'] is None
            and report['fitted_artifact_sha256']==context['immutable']['fit_artifact_sha256']
            and set(report['models'])=={'1301','1311','1321'}
            and report['original_first_inventory_reproduced'] is True,
            'corrected fit audit used different fitted bytes or incomplete inventory')
    ancestor(context['amendment_commit'],report_commit)
    return dict(manifest_path=name,manifest_sha256=digest(ROOT/name),report_commit=report_commit,
        raw_sha256=manifest['raw_sha256'],published_path=packed,compressed_sha256=manifest['compressed_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--create',action='store_true',required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--approval',type=Path,required=True)
    parser.add_argument('--amendment',type=Path,required=True)
    args = parser.parse_args()
    create_amendment(args.source_commit,args.approval,args.amendment)
    print('Created exclusive exact reference amendment; no models, data or fitting.')


if __name__ == '__main__':
    main()
