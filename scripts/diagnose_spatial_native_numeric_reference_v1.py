"""One committed, read-only arithmetic diagnosis of the preserved failed review.

Never trains, renders, retries the study, changes tolerances, or calls a Reporter.
The original verifier is imported only to reproduce its reference arithmetic;
independent explicit linear equations supply separately labelled comparisons.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'audits/spatial_native_task_v1'
SEEDS = (1301, 1311, 1321)
ATOL, RTOL = 1e-6, 1e-5
EXPECTED_RESULTS = '614e74ae74ccf1d553f6fc6e0ec318f057a94ed8'
EXPECTED_FIRST_REVIEW = '432146ae8f1420b13bacdbb8f00518570d34b95f'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def committed_sha(commit, relative):
    p = subprocess.Popen(['git', '-C', str(ROOT), 'show', f'{commit}:{relative}'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    h = hashlib.sha256()
    for chunk in iter(lambda: p.stdout.read(1024 * 1024), b''):
        h.update(chunk)
    error = p.stderr.read().decode(errors='replace')
    if p.wait() != 0:
        raise ValueError(f'committed binding failed: {commit}:{relative}: {error}')
    return h.hexdigest()


def bind(report, relative, expected, commit):
    actual, preserved = sha(ROOT / relative), committed_sha(commit, relative)
    report.setdefault('bindings', []).append(dict(path=relative, expected=expected,
        actual=actual, committed=preserved, commit=commit))
    if actual != expected or preserved != expected:
        raise ValueError(f'byte binding mismatch: {relative}')


def comparison(a, b, exact=False):
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    same = a.shape == b.shape
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not same or not finite:
        return dict(passed=False, actual_shape=list(a.shape), expected_shape=list(b.shape), finite=finite)
    delta = (a.double() - b.double()).abs()
    tolerance = ATOL + RTOL * b.double().abs()
    violations = delta > tolerance
    return dict(passed=bool(torch.equal(a, b) and a.dtype == b.dtype) if exact else bool((~violations).all()),
        actual_shape=list(a.shape), expected_shape=list(b.shape), actual_dtype=str(a.dtype), expected_dtype=str(b.dtype),
        finite=finite, exact=exact, atol=0. if exact else ATOL, rtol=0. if exact else RTOL,
        maximum_absolute_difference=float(delta.max()) if delta.numel() else 0.,
        unequal_elements=int((a != b).sum()), tolerance_violations=int(violations.sum()),
        compared_elements=a.numel())


def tensor_layout(x):
    return dict(shape=list(x.shape), stride=list(x.stride()), dtype=str(x.dtype),
                contiguous=x.is_contiguous(), device=str(x.device))


def head_input(maps, fit):
    stats = fit['fit_only_standardizer']
    mean = torch.tensor(stats['mean'], dtype=torch.float32)[:, None, None]
    scale = torch.tensor(stats['scale'], dtype=torch.float32)[:, None, None]
    return ((maps - mean) / scale).movedim(-3, -1)


def head_equation(x, fit, mode):
    w = fit['weights']
    if x.dtype == torch.float64:
        w = {k: v.double() for k, v in w.items()}
    if mode == 'linear':
        hidden = torch.nn.functional.linear(x, w['net.0.weight'], w['net.0.bias']).clamp_min(0)
        return torch.nn.functional.linear(hidden, w['net.2.weight'], w['net.2.bias'])
    hidden = (x @ w['net.0.weight'].T + w['net.0.bias']).clamp_min(0)
    return hidden @ w['net.2.weight'].T + w['net.2.bias']


def head_variant(x, fit, mode, itemwise=False, flatten=False):
    if itemwise:
        result = torch.stack([head_equation(x[w, i], fit, mode) for w in range(64) for i in range(4)])
        return result.reshape(64, 4, 4, 4, 3).movedim(-1, -3)
    if flatten:
        return head_equation(x.reshape(-1, 24), fit, mode).reshape(64, 4, 4, 4, 3).movedim(-1, -3)
    return head_equation(x, fit, mode).movedim(-1, -3)


def fields_equation(logits, kernel=False):
    scores = (logits[..., 1:, :, :].double() - logits[..., :1, :, :].double()).flatten(-2)
    if kernel:
        uniform = (scores == scores[..., :1]).all(-1, keepdim=True)
        value = torch.where(uniform, torch.full_like(scores, 1/16), torch.softmax(scores, -1))
    else:
        numerator = (scores - scores.max(-1, keepdim=True).values).exp()
        value = numerator / numerator.sum(-1, keepdim=True)
    return value.reshape(*logits.shape[:-3], 2, 4, 4).to(logits.dtype)


def policy(fields, cells):
    flat = fields.flatten(-2)
    idx = flat.argmax(-1)
    selected = torch.stack((idx // 4, idx % 4), -1)
    ties = (flat == flat.max(-1, keepdim=True).values).sum(-1)
    columns = (fields.double() * torch.arange(4, dtype=torch.float64)).sum((-2, -1))
    delta = columns[..., 0] - columns[..., 1]
    coarse = torch.where(delta.abs() <= 1e-6, 0, torch.where(delta < 0, -1, 1))
    return dict(selected_cells=selected, tie_counts=ties, fine_correct=(selected == cells).all(-1), coarse_codes=coarse)


def identical_input_symmetry(maps, logits, fields):
    sites = maps.flatten(-2).movedim(-2, -1)
    same = (sites.unsqueeze(-2) == sites.unsqueeze(-3)).all(-1)
    same &= ~torch.eye(16, dtype=torch.bool)
    result = dict(identical_input_ordered_pairs=int(same.sum()))
    for label, value in [('logits', logits), ('fields', fields)]:
        rows = value.flatten(-2).movedim(-2, -1)
        unequal = (rows.unsqueeze(-2) != rows.unsqueeze(-3)).any(-1) & same
        difference = (rows.unsqueeze(-2).double() - rows.unsqueeze(-3).double()).abs().max(-1).values
        result[label + '_violating_pairs'] = int(unequal.sum())
        result[label + '_maximum_difference'] = float(difference[same].max()) if same.any() else 0.
    return result


def head_case(maps, retained, trace, reference):
    fit, cells = trace['native_head_fit'], trace['diagnostic_worlds']['evaluator_cells']
    x = head_input(maps, fit)
    values = {'original_reference': reference.head_reference(maps, fit),
        'matched_item_linear': head_variant(x, fit, 'linear', itemwise=True),
        'item_matmul_plus_bias': head_variant(x, fit, 'matmul', itemwise=True),
        'batch_linear': head_variant(x, fit, 'linear'),
        'flattened_linear': head_variant(x, fit, 'linear', flatten=True),
        'float64_equation_on_float32_input': head_variant(x.double(), fit, 'matmul')}
    stats = fit['fit_only_standardizer']
    full64 = ((maps.double() - torch.tensor(stats['mean'], dtype=torch.float64)[:, None, None]) /
               torch.tensor(stats['scale'], dtype=torch.float64)[:, None, None]).movedim(-3, -1)
    values['float64_full_coordinates'] = head_variant(full64, fit, 'matmul')
    details, reconstructed = {}, {}
    for name, logits in values.items():
        fields = fields_equation(logits, kernel=name == 'matched_item_linear')
        p = policy(fields, cells)
        details[name] = dict(logits_vs_archived=comparison(retained['local_logits'], logits),
            fields_vs_archived=comparison(retained['role_fields'], fields),
            logits_exact_archived=comparison(retained['local_logits'], logits, exact=True),
            fields_exact_archived=comparison(retained['role_fields'], fields, exact=True),
            policy_vs_archived={k: comparison(retained[k], v, exact=True) for k, v in p.items()},
            symmetry=identical_input_symmetry(maps, logits, fields),
            logits_vs_float64_equation=comparison(logits, values['float64_equation_on_float32_input']))
        reconstructed[name] = dict(local_logits=logits, role_fields=fields, **p)
    # Separate softmax arithmetic from linear-kernel arithmetic using equal logits.
    manual = fields_equation(values['matched_item_linear'])
    details['manual_normalization_on_matched_item_logits'] = dict(fields=comparison(retained['role_fields'], manual),
        fields_exact=comparison(retained['role_fields'], manual, exact=True),
        policy={k: comparison(retained[k], v, exact=True) for k, v in policy(manual, cells).items()})
    return dict(input_layout=tensor_layout(x), item_input_layout=tensor_layout(x[0, 0]),
                variants=details), reconstructed


def controls_equation(maps, trace, linear=False, higher_precision=False):
    encoder, orientation = trace['frozen_source_weights']['encoder'], trace['frozen_source_weights']['orientation_head']
    stats = trace['frozen_orientation_standardizer']
    x = maps.reshape(-1, 384)
    if higher_precision:
        x = x.double()
        encoder = {k: v.double() for k, v in encoder.items()}
        orientation = {k: v.double() for k, v in orientation.items()}
        stats = {k: v.double() for k, v in stats.items()}
    def layer(z, weight, bias):
        return torch.nn.functional.linear(z, weight, bias) if linear else z @ weight.T + bias
    hidden = layer(x, encoder['net.7.weight'], encoder['net.7.bias']).clamp_min(0)
    logits = layer(hidden, encoder['net.9.weight'], encoder['net.9.bias'])
    standardized = (hidden - stats['mean']) / stats['scale']
    ori_hidden = layer(standardized, orientation['0.weight'], orientation['0.bias']).clamp_min(0)
    ori = layer(ori_hidden, orientation['2.weight'], orientation['2.bias'])[:, 0]
    leading = maps.shape[:-3]
    return dict(category_logits=logits.reshape(*leading, 8), orientation_hidden=hidden.reshape(*leading, 64),
        orientation_standardized_hidden=standardized.reshape(*leading, 64), orientation_logits=ori.reshape(leading))


def witnesses(actual, old, matched, high, exact, limit=3):
    difference = (actual.double() - old.double()).abs()
    mask = actual != old if exact else difference > ATOL + RTOL * old.double().abs()
    output = []
    for idx in mask.nonzero()[:limit]:
        index = tuple(idx.tolist())
        output.append(dict(index=list(index), actual=actual[index].item(), original_reference=old[index].item(),
            matched_linear=matched[index].item(), float64_equation=high[index].item(),
            actual_minus_reference=float(actual[index].double() - old[index].double()),
            allowed_original_tolerance=0. if exact else float(ATOL + RTOL * old[index].double().abs())))
    return output


def profile_operation(name, operation, layouts):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
        with torch.profiler.record_function(name):
            operation()
    events = []
    for event in prof.key_averages(group_by_input_shape=True):
        if event.key.startswith('aten::'):
            events.append(dict(operator=event.key, input_shapes=event.input_shapes, count=event.count,
                self_cpu_time_us=event.self_cpu_time_total))
    return dict(name=name, layouts=layouts, cpu_operator_events=events,
                limitation='Profiler identifies CPU ATen dispatch and shapes, not vendor microkernel assembly.')


def verify_metrics_reproduced(original, measured):
    keys = ('actual_shape', 'expected_shape', 'actual_dtype', 'expected_dtype', 'finite',
            'maximum_absolute_difference', 'unequal_elements', 'tolerance_violations', 'compared_elements')
    return all(original[k] == measured[k] for k in keys)


def run(args, report):
    if args.results_commit != EXPECTED_RESULTS or args.first_review_commit != EXPECTED_FIRST_REVIEW:
        raise ValueError('wrong preserved first-results/first-review commits')
    for relative in ('scripts/diagnose_spatial_native_numeric_reference_v1.py',
                     'audits/spatial_native_task_v1/numeric_diagnosis/design.md'):
        bind(report, relative, sha(ROOT / relative), args.diagnosis_commit)
    manifest_path = 'audits/spatial_native_task_v1/numerical_review/metadata_manifest.json'
    bind(report, manifest_path, sha(ROOT / manifest_path), args.first_review_commit)
    manifest = json.loads((ROOT / manifest_path).read_text())
    bind(report, manifest['published_path'], manifest['compressed_sha256'], args.first_review_commit)
    raw = gzip.decompress((ROOT / manifest['published_path']).read_bytes())
    if hashlib.sha256(raw).hexdigest() != manifest['raw_sha256']:
        raise ValueError('first metadata transport is not byte-identical')
    first = json.loads(raw)
    failures = [c for c in first['checks'] if not c['passed']]
    if first['status'] != 'failed_first_verification' or len(failures) != 106 or first['check_count'] != 206070:
        raise ValueError('unexpected first findings')
    bind(report, 'scripts/verify_spatial_native_task_v1.py', first['verifier_sha256'], args.first_review_commit)
    for relative in ('audits/spatial_native_task_v1/numerical_review/review.md',
                     'audits/spatial_native_task_v1/numerical_review/first_run.log'):
        bind(report, relative, sha(ROOT / relative), args.first_review_commit)
    registration_path = 'audits/spatial_native_task_v1/registration.json'
    bind(report, registration_path, sha(ROOT / registration_path), args.results_commit)
    registration = json.loads((ROOT / registration_path).read_text())
    for relative, expected in registration['source_sha256'].items():
        bind(report, relative, expected, args.results_commit)
    completion_path = 'audits/spatial_native_task_v1/run/completion.json'
    bind(report, completion_path, sha(ROOT / completion_path), args.results_commit)
    completion = json.loads((ROOT / completion_path).read_text())
    for relative, expected in completion['archive_sha256'].items():
        bind(report, relative, expected, args.results_commit)
    bind(report, 'audits/spatial_native_task_v1/run/results.json', completion['results_sha256'], args.results_commit)
    report['first_findings'] = dict(status=first['status'], error_count=106, check_count=206070,
        raw_sha256=manifest['raw_sha256'], unchanged=True)
    global torch
    import torch
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    spec = importlib.util.spec_from_file_location('preserved_failed_reference', ROOT / 'scripts/verify_spatial_native_task_v1.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    reference.torch = torch
    cpu_model = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                      if line.startswith('model name')), None)
    report['environment'] = dict(torch_version=torch.__version__, python_version=platform.python_version(),
        platform=platform.platform(), cpu_model=cpu_model, threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(), torch_build_config=torch.__config__.show(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(), mkldnn_enabled=torch.backends.mkldnn.enabled)
    report['models'], report['failed_check_diagnoses'], report['profiles'] = {}, [], []
    with torch.no_grad():
        for seed in SEEDS:
            with gzip.open(BASE / 'run' / f'states-{seed}.pt.gz', 'rb') as stream:
                trace = torch.load(stream, weights_only=True, map_location='cpu')
            native = trace['native_execution']
            control_inputs = {'original_controls': (trace['diagnostic_V'], native['original_native_controls']),
                              'fit_controls': (trace['fit_V'], trace['fit_native_controls'])}
            head_outputs, control_outputs, model_cases = {}, {}, {}
            for band in (1, 2, 4):
                primary = native['primary'][f'T{band}']
                reconstructed_map = reference.block_reference(trace['diagnostic_V'], band)
                if not torch.equal(reconstructed_map, primary['actual_W']):
                    raise ValueError(f'{seed}/T{band} W differs from preserved reference primitive')
                for label, maps, row in [(f'T{band}', reconstructed_map, primary),
                    (f'synthetic_site_permutation/T{band}', reconstructed_map.flatten(-2)[..., list(reference.SITE_ROLL)].reshape_as(reconstructed_map),
                     native['controls']['synthetic_site_permutation'][f'T{band}'])]:
                    if not torch.equal(maps, row['actual_W']):
                        raise ValueError(f'{seed}/{label} saved W differs from reconstructed map')
                    details, outputs = head_case(maps, row, trace, reference)
                    model_cases[label] = details
                    head_outputs[label] = outputs
                    control_inputs[label] = (maps, row['native_controls'])
            for label, (maps, retained) in control_inputs.items():
                old = reference.controls_reference(maps, trace)
                matched = controls_equation(maps, trace, linear=True)
                high = controls_equation(maps, trace, higher_precision=True)
                control_outputs[label] = dict(original_reference=old, matched=matched, high=high, retained=retained)
                model_cases.setdefault(label, {})['control_comparisons'] = {
                    key: dict(original_reference_vs_archived=comparison(retained[key], old[key]),
                              matched_linear_exact_archived=comparison(retained[key], matched[key], exact=True),
                              original_reference_vs_float64=comparison(old[key], high[key]),
                              matched_linear_vs_float64=comparison(matched[key], high[key])) for key in matched}
            if seed == 1301:
                # Fixed CPU dispatch schedule: same saved T4 inputs, item and batch paths,
                # plus the already flattened saved original control path.
                maps = native['primary']['T4']['actual_W']
                x = head_input(maps, trace['native_head_fit'])
                for label, mode, itemwise, flatten in [('head_batch_matmul', 'matmul', False, False),
                    ('head_batch_linear', 'linear', False, False), ('head_item_matmul', 'matmul', True, False),
                    ('head_item_linear', 'linear', True, False), ('head_flattened_linear', 'linear', False, True)]:
                    operation = (lambda mode=mode: head_equation(x[0, 0], trace['native_head_fit'], mode)) if itemwise else (
                        lambda mode=mode, flatten=flatten: head_variant(x, trace['native_head_fit'], mode, flatten=flatten))
                    report['profiles'].append(profile_operation(label, operation,
                        dict(batch_input=tensor_layout(x), effective_input=tensor_layout(x[0, 0] if itemwise else x.reshape(-1, 24) if flatten else x))))
                for label, linear in [('controls_matmul_plus_bias', False), ('controls_flattened_linear', True)]:
                    report['profiles'].append(profile_operation(label,
                        lambda linear=linear: controls_equation(trace['diagnostic_V'], trace, linear=linear),
                        dict(effective_input=tensor_layout(trace['diagnostic_V'].reshape(-1, 384)))))
            for failure in failures:
                if not failure['name'].startswith(str(seed) + '/'):
                    continue
                parts = failure['name'].split('/')[1:]
                if parts[0] in ('original_controls', 'fit_controls'):
                    label, quantity = parts
                    c = control_outputs[label]
                    actual, old, matched, high = (c['retained'][quantity], c['original_reference'][quantity], c['matched'][quantity], c['high'][quantity])
                    category, explanation = 'control_linear_kernel_roundoff', 'Matched flattened F.linear versus separate matmul then bias.'
                elif 'original_path' in parts or 'original_path_delta' in parts:
                    location = next(i for i, part in enumerate(parts) if part.startswith('original_path'))
                    label, quantity = '/'.join(parts[:location]), parts[-1]
                    c = control_outputs[label]
                    actual, old, matched, high = (c['retained'][quantity], c['original_reference'][quantity], c['matched'][quantity], c['high'][quantity])
                    if parts[location] == 'original_path_delta':
                        row = native['primary'][label] if label.startswith('T') else native['controls']['synthetic_site_permutation'][label.split('/')[-1]]
                        base = control_outputs['original_controls']
                        actual = row['control_changes_from_original'][quantity]
                        old = old - base['original_reference'][quantity]
                        matched = matched - base['matched'][quantity]
                        high = high - base['high'][quantity]
                        explanation = 'F.linear endpoint differences plus baseline subtraction cross fixed delta tolerance; no changed threshold.'
                    else:
                        explanation = 'Matched flattened F.linear versus separate matmul then bias.'
                    category = 'control_linear_kernel_roundoff'
                else:
                    label, quantity = '/'.join(parts[:-1]), parts[-1]
                    if label.startswith('address_permutation/'):
                        case = label.split('/')[-1]
                        row = native['controls']['address_permutation'][case]
                    elif label.startswith('synthetic_site_permutation/'):
                        case = label
                        row = native['controls']['synthetic_site_permutation'][label.split('/')[-1]]
                    else:
                        case, row = label, native['primary'][label]
                    outputs = head_outputs[case]
                    actual, old, matched, high = (row[quantity], outputs['original_reference'][quantity],
                        outputs['matched_item_linear'][quantity], outputs['float64_equation_on_float32_input'][quantity])
                    category = 'head_arithmetic_shape_discrete_policy'
                    explanation = 'Archived exact policy compared to reference perturbation; matched itemwise F.linear and explicit kernel normalization checked without tie tolerance.'
                measured = comparison(actual, old, exact=failure['exact'])
                matched_comparison = comparison(actual, matched, exact=True)
                reproduced = verify_metrics_reproduced(failure, measured) and not measured['passed']
                explained = reproduced and matched_comparison['passed']
                diagnosis = dict(name=failure['name'], category=category, explanation=explanation,
                    original_check=failure, reproduced_reference_comparison=measured,
                    first_failure_metrics_reproduced=reproduced, matched_linear_exact_comparison=matched_comparison,
                    higher_precision_comparison=comparison(actual, high), explained=explained,
                    unresolved_reason=None if explained else 'First failure metrics did not reproduce or shape-matched F.linear did not match archived value exactly.',
                    witnesses=witnesses(actual, old, matched, high, failure['exact']))
                if category == 'head_arithmetic_shape_discrete_policy':
                    diagnosis['reference_symmetry'] = model_cases[case]['variants']['original_reference']['symmetry']
                    diagnosis['matched_symmetry'] = model_cases[case]['variants']['matched_item_linear']['symmetry']
                    diagnosis['matched_logits_and_fields_exact'] = all(model_cases[case]['variants']['matched_item_linear'][key]['passed']
                        for key in ('logits_exact_archived', 'fields_exact_archived'))
                    diagnosis['explained'] &= diagnosis['matched_logits_and_fields_exact']
                elif 'original_path_delta' in parts:
                    endpoints = control_outputs[label]
                    baseline = control_outputs['original_controls']
                    diagnosis['endpoint_matches'] = dict(changed=comparison(endpoints['retained'][quantity], endpoints['matched'][quantity], exact=True),
                        baseline=comparison(baseline['retained'][quantity], baseline['matched'][quantity], exact=True))
                    diagnosis['explained'] &= all(v['passed'] for v in diagnosis['endpoint_matches'].values())
                    for witness in diagnosis['witnesses']:
                        index = tuple(witness['index'])
                        witness['endpoint_roundoff'] = dict(
                            changed_actual=float(endpoints['retained'][quantity][index]), changed_reference=float(endpoints['original_reference'][quantity][index]),
                            baseline_actual=float(baseline['retained'][quantity][index]), baseline_reference=float(baseline['original_reference'][quantity][index]))
                report['failed_check_diagnoses'].append(diagnosis)
            report['models'][str(seed)] = model_cases
            print(json.dumps(dict(seed=seed, diagnosed_failures=len(report['failed_check_diagnoses']),
                explained_failures=sum(row['explained'] for row in report['failed_check_diagnoses']))), flush=True)
    verdict = next(c for c in failures if c['name'] == 'results/all_software_invariants_pass')
    report['failed_check_diagnoses'].append(dict(name=verdict['name'], category='derived_first_verdict',
        original_check=verdict, explained=verdict['reported'] is True and verdict['reconstruction_errors_before_verdict'] == 105,
        explanation='Original verdict fails by design because the first verifier retained 105 previous discrepancies; diagnosis does not replace that verdict.'))
    if sorted(row['name'] for row in report['failed_check_diagnoses']) != sorted(row['name'] for row in failures):
        raise ValueError('first failure inventory not covered exactly')
    report['explained_count'] = sum(row['explained'] for row in report['failed_check_diagnoses'])
    report['unresolved'] = [row['name'] for row in report['failed_check_diagnoses'] if not row['explained']]
    report['status'] = 'diagnosis_complete_all_first_failures_explained' if not report['unresolved'] else 'diagnosis_complete_with_unresolved'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-commit', required=True)
    p.add_argument('--first-review-commit', required=True)
    p.add_argument('--diagnosis-commit', required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--ack-committed-diagnosis', action='store_true', required=True)
    args = p.parse_args()
    with args.report.open('x', encoding='utf-8') as stream:
        report = dict(schema='spatial_native_numeric_reference_v1_arithmetic_diagnosis',
            created_at_utc=datetime.now(timezone.utc).isoformat(), reviewer='/root/spatial_native_numerical_execution_review',
            shared_model_and_inherited_hypothesis_context=True, independent_model_family=False, human_endorsement=False,
            development_only=True, arithmetic_diagnosis_only=True, passing_revalidation=False,
            original_verification_status='failed_first_verification', no_new_worlds=True, no_fit_replay=True,
            no_source_or_tolerance_changes=True, presentation_qualified=False, qualia_established=False, goal_achieved=False,
            results_commit=args.results_commit, first_review_commit=args.first_review_commit, diagnosis_commit=args.diagnosis_commit)
        try:
            run(args, report)
        except BaseException:
            report.update(status='failed_first_diagnosis', exception=traceback.format_exc())
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        print(json.dumps({k: report.get(k) for k in ('status', 'explained_count', 'unresolved', 'exception')}), flush=True)
    if report['status'] == 'failed_first_diagnosis':
        sys.exit(1)


if __name__ == '__main__':
    main()
