"""Independent fit/dev/heldout reconstruction, pending preexecution review.

Never renders, samples worlds, trains, creates an optimizer, or calls a Reporter.
Do not execute before complete source review/registration and preservation of
the actual first phase outputs. Missing reconstruction evidence fails closed.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
SEEDS, SOURCES, BANDS = (1301, 1311, 1321), ('A', 'ordinary', 'analytic'), (1, 2, 4)
MAPPINGS = tuple(permutations(BANDS))
ATOL, RTOL = 1e-6, 1e-5
INTEGRATION_READY = True  # Schema integrated; committed exact-source approval is separately mandatory.
NATIVE_PASS_COMMIT='c84d3dabc0e3ca8b2d9898b8554b032556c0acff'
REQUIRED_CLOSURE = (
    'scripts/verify_attention_content_model_v1.py',
    'scripts/run_attention_content_model_v1.py',
    'scripts/attention_content_model_v1.py',
    'scripts/attention_content_model_v1_data.py',
    'scripts/attention_content_model_v1_experiment.py',
    'scripts/attention_content_form_primitives_v1.py',
    'scripts/spatial_native_task_v1.py',
    'docs/ATTENTION_CONTENT_MODEL_V1_DESIGN.md',
    'docs/ATTENTION_CONTENT_MODEL_V1_PERMISSION_SCHEMA.md',
    'src/attcon/__init__.py', 'src/attcon/bound_content.py',
    'src/attcon/data.py', 'src/attcon/models.py', 'src/attcon/predictive_attention.py',
)
sys.dont_write_bytecode = True


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(f'{value.dtype}:{tuple(value.shape)}:'.encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


class Audit:
    def __init__(self):
        self.checks, self.errors = [], []

    def check(self, name, condition, **evidence):
        passed = bool(condition)
        self.checks.append(dict(name=name, passed=passed, **evidence))
        if not passed:
            self.errors.append(name)

    def compare(self, name, actual, expected, exact=False):
        a = torch.as_tensor(actual,dtype=torch.float64) if type(actual) is float else torch.as_tensor(actual)
        b = torch.as_tensor(expected,dtype=torch.float64) if type(expected) is float else torch.as_tensor(expected)
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        same = a.shape == b.shape
        delta = (a.double() - b.double()).abs() if same and finite else None
        within = delta <= ATOL + RTOL * b.double().abs() if delta is not None else None
        passed = bool(torch.equal(a, b) and a.dtype == b.dtype) if exact and same and finite else (
            bool(within.all()) if not exact and within is not None else False)
        self.check(name, passed, actual_shape=list(a.shape), expected_shape=list(b.shape),
            actual_dtype=str(a.dtype), expected_dtype=str(b.dtype), finite=finite, exact=exact,
            atol=0. if exact else ATOL, rtol=0. if exact else RTOL,
            maximum_absolute_difference=float(delta.max()) if delta is not None and delta.numel() else 0. if delta is not None else None,
            unequal_elements=int((a != b).sum()) if delta is not None else None,
            tolerance_violations=int((~within).sum()) if within is not None else None,
            compared_elements=a.numel())

    def tree(self, name, actual, expected):
        if isinstance(expected, dict):
            self.check(name + '/keys', set(actual) == set(expected), actual=sorted(actual), expected=sorted(expected))
            for key in expected:
                if key in actual:
                    self.tree(name + '/' + key, actual[key], expected[key])
        elif isinstance(expected, (tuple,list)):
            self.check(name+'/length',len(actual)==len(expected),actual=len(actual),expected=len(expected))
            for index,(left,right) in enumerate(zip(actual,expected)):
                self.tree(name+'/'+str(index),left,right)
        elif isinstance(expected, (str, bool, int)) or expected is None:
            self.check(name, actual == expected, actual=actual, expected=expected)
        else:
            self.compare(name, actual, expected)


def finite_tree(audit, value, name):
    if isinstance(value, torch.Tensor):
        audit.check(name + '/finite_cpu_detached', bool(torch.isfinite(value).all())
                    and value.device.type == 'cpu' and not value.requires_grad,
                    shape=list(value.shape), dtype=str(value.dtype))
    elif isinstance(value, dict):
        for key, child in value.items():
            finite_tree(audit, child, name + '/' + str(key))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            finite_tree(audit, child, name + '/' + str(index))
    elif isinstance(value, float):
        audit.check(name + '/finite_scalar', math.isfinite(value))


def committed_bindings(audit, checkout, commit, bindings, name):
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('preservation commits must be full lowercase hex')
    for relative, expected in bindings.items():
        path = (ROOT / relative).resolve()
        valid = path.is_relative_to(ROOT) and path.is_file()
        actual = digest(path) if valid else None
        p = subprocess.Popen(['git', '-C', str(checkout), 'show', f'{commit}:{relative}'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        h = hashlib.sha256()
        for chunk in iter(lambda: p.stdout.read(1024 * 1024), b''):
            h.update(chunk)
        error, status = p.stderr.read().decode(errors='replace'), p.wait()
        audit.check(name + '/' + relative, valid and actual == expected and status == 0 and h.hexdigest() == expected,
            commit=commit, expected_sha256=expected, actual_sha256=actual,
            committed_sha256=h.hexdigest(), git_exit=status, git_error=error)


def normalize(value, normalizer, double=False):
    dtype = torch.float64 if double else torch.float32
    mean, scale = (torch.tensor(normalizer[k], dtype=dtype)[:, None, None] for k in ('mean', 'scale'))
    return (value.to(dtype) - mean) / scale


def normalizer_reference(v):
    rows = v.double().movedim(-3, -1).reshape(-1, 24)
    mean = rows.mean(0)
    scale = ((rows - mean).square().mean(0)).sqrt()
    scale = torch.where(scale < 1e-6, 1., scale)
    return dict(mean=mean.tolist(), scale=scale.tolist())


def block_reference(v, bandwidth):
    if bandwidth == 1:
        return v.clone()
    result = torch.empty_like(v)
    for row in range(0, 4, bandwidth):
        for col in range(0, 4, bandwidth):
            average = (v[..., row:row+bandwidth, col:col+bandwidth].double() /
                       (bandwidth * bandwidth)).sum((-2, -1)).to(v.dtype)
            result[..., row:row+bandwidth, col:col+bandwidth] = average[..., None, None]
    return result


def family_reference(v):
    return torch.stack([block_reference(v, band) for band in BANDS], -4)


def schedule_reference(seed, phase):
    split = ('fit', 'dev', 'heldout').index(phase)
    mapping_index = torch.empty(96, 4, dtype=torch.long)
    for address in range(4):
        generator = torch.Generator().manual_seed(330000000 + seed*1000 + split*100000 + address)
        mapping_index[:, address] = torch.arange(6).repeat_interleave(16)[torch.randperm(96, generator=generator)]
    addresses, symbols = torch.empty(96, 15, dtype=torch.long), torch.empty(96, 15, dtype=torch.long)
    roles = torch.full((96, 15), -1, dtype=torch.long)
    for world in range(96):
        for address in range(4):
            generator = torch.Generator().manual_seed(340000000 + seed*1000 + split*100000 + world*4 + address)
            order = torch.randperm(3, generator=generator)
            for round_index in range(3):
                addresses[world, 4*round_index+address] = address
                symbols[world, 4*round_index+address] = order[round_index]
        for j in range(3):
            addresses[world, 12+j], symbols[world, 12+j], roles[world, 12+j] = (world+j)%4, (world+j)%3, (world+j)%2
    return dict(evaluator_mapping_indices=mapping_index, evaluator_mappings=torch.tensor(MAPPINGS)[mapping_index],
                command_addresses=addresses, command_symbols=symbols, query_roles=roles)


def layout_reference():
    cells = torch.empty(96, 4, 2, 2, dtype=torch.long)
    for world in range(96):
        for item in range(4):
            code, side = world ^ ((item*32) | (item%2)), (world ^ ((item*32) | (item%2))) & 1
            cells[world, item, 0] = torch.tensor(((code >> 2) & 3, ((code >> 1) & 1) + 2*side))
            cells[world, item, 1] = torch.tensor(((((code >> 3) ^ (code >> 5)) & 1) + 2*((code >> 4) & 1),
                (((code >> 1) ^ (code >> 4) ^ (code >> 6)) & 1) + 2*(1-side)))
    return cells


def source_reference(pixels, weights):
    x = pixels.reshape(-1, 3, 16, 16)
    for layer in (0, 3):
        x = torch.nn.functional.conv2d(x, weights[f'net.{layer}.weight'], weights[f'net.{layer}.bias'], padding=1)
        x = torch.nn.functional.max_pool2d(x.clamp_min(0), 2)
    return x.reshape(96, 4, 24, 4, 4)


def actual_reference(episode, mapping_indices=None, transitions=15):
    v, family = episode['V'], family_reference(episode['V'])
    mappings = torch.tensor(MAPPINGS)[mapping_indices] if mapping_indices is not None else episode['evaluator_mappings']
    rows = torch.arange(96)
    completed, forks = [v.clone()], []
    for time in range(transitions):
        address = episode['command_addresses'][:, time]
        alternatives = []
        for symbol in range(3):
            band = mappings[rows, address, symbol]
            index = torch.where(band == 1, 0, torch.where(band == 2, 1, 2))
            state = completed[-1].clone()
            state[rows, address] = family[rows, address, index]
            alternatives.append(state)
        alternatives = torch.stack(alternatives, 1)
        forks.append(alternatives)
        completed.append(alternatives[rows, episode['command_symbols'][:, time]].clone())
    return torch.stack(completed, 1), torch.stack(forks, 1), family


def head_reference(maps, native):
    fit = native['native_head_fit']
    x = normalize(maps, fit['fit_only_standardizer']).movedim(-3, -1)
    params = fit['weights']
    outputs = []
    # Item shape [4,4,24] preserves the reviewed native CPU dispatch.
    for item in x.reshape(-1, 4, 4, 24):
        hidden = (item @ params['net.0.weight'].T + params['net.0.bias']).clamp_min(0)
        outputs.append(hidden @ params['net.2.weight'].T + params['net.2.bias'])
    return torch.stack(outputs).reshape(*maps.shape[:-3], 4, 4, 3).movedim(-1, -3)


def fields_reference(logits):
    score = (logits[..., 1:, :, :].double() - logits[..., :1, :, :].double()).flatten(-2)
    numerator = (score - score.max(-1, keepdim=True).values).exp()
    return (numerator / numerator.sum(-1, keepdim=True)).reshape(*logits.shape[:-3], 2, 4, 4).to(logits.dtype)


def field_metrics(estimated, actual, cells):
    flat = estimated.flatten(-2)
    selected, actual_selected = flat.argmax(-1), actual.flatten(-2).argmax(-1)
    truth = cells[..., 0]*4 + cells[..., 1]
    columns = (estimated.double() * torch.arange(4, dtype=torch.float64)).sum((-2, -1))
    actual_columns = (actual.double() * torch.arange(4, dtype=torch.float64)).sum((-2, -1))
    def order(value):
        delta = value[..., 0] - value[..., 1]
        return torch.where(delta.abs() <= 1e-6, 0, torch.where(delta < 0, 1, -1))
    maximum = flat.max(-1).values
    second = flat.sort(-1).values[..., -2]
    truth_order = torch.where(cells[..., 0, 1] < cells[..., 1, 1], 1, -1)
    return dict(field_l1=(estimated-actual).abs().mean((-3,-2,-1)), cell_disagreement=selected != actual_selected,
        estimated_cell_correct=selected == truth, actual_cell_correct=actual_selected == truth,
        coarse_disagreement=order(columns) != order(actual_columns), estimated_coarse_correct=order(columns) == truth_order,
        actual_coarse_correct=order(actual_columns) == truth_order, expected_columns=columns,
        entropy=-(flat.double()*flat.double().clamp_min(1e-300).log()).sum(-1), maxima=maximum,
        top_two_margin=maximum-second, max_tie_count=(flat == maximum[..., None]).sum(-1))


def coarse_reference(fields):
    columns=(fields.double()*torch.arange(4,dtype=torch.float64)).sum((-2,-1))
    delta=columns[...,0]-columns[...,1]
    return torch.where(delta.abs()<=1e-6,0,torch.where(delta<0,1,-1))


def field_scalars_reference(fields):
    flat=fields.flatten(-2);top=flat.sort(-1).values[...,-2:]
    return dict(entropy=-(flat.double()*flat.double().clamp_min(1e-300).log()).sum(-1),
        maxima=top[...,-1],top_two_margin=top[...,-1]-top[...,-2],
        expected_columns=(fields.double()*torch.arange(4,dtype=torch.float64)).sum((-2,-1)))


def mse(predicted, actual, normalizer):
    return ((predicted.double()-actual.double()) /
            torch.tensor(normalizer['scale'], dtype=torch.float64)[:, None, None]).square().mean((-3,-2,-1))


def decoder_reference(v, hidden, symbols, weights, normalizer):
    normalized = normalize(v, normalizer)
    features = torch.cat((normalized.flatten(1), hidden, torch.nn.functional.one_hot(symbols, 3).float()), 1)
    middle = torch.nn.functional.linear(features, weights['decoder.0.weight'], weights['decoder.0.bias']).clamp_min(0)
    residual = torch.nn.functional.linear(middle, weights['decoder.2.weight'], weights['decoder.2.bias']).reshape(-1,24,4,4)
    scale, mean = (torch.tensor(normalizer[k], dtype=torch.float32)[:,None,None] for k in ('scale','mean'))
    return (normalized+residual)*scale+mean


def gru_reference(hidden, v, completed, symbols, weights, normalizer):
    features = torch.cat((normalize(v,normalizer).flatten(1), normalize(completed,normalizer).flatten(1),
                          torch.nn.functional.one_hot(symbols,3).float()),1)
    incoming = torch.nn.functional.linear(features,weights['gru.weight_ih'],weights['gru.bias_ih'])
    recurrent = torch.nn.functional.linear(hidden,weights['gru.weight_hh'],weights['gru.bias_hh'])
    ir, iz, inn = incoming.chunk(3,1)
    hr, hz, hn = recurrent.chunk(3,1)
    reset, update = (ir+hr).sigmoid(), (iz+hz).sigmoid()
    candidate = (inn+reset*hn).tanh()
    return candidate+update*(hidden-candidate)


def posterior_update(survivors, seen, v, completed, address, symbols):
    rows = torch.arange(96)
    family = family_reference(v[rows,address])
    indices = torch.tensor([[BANDS.index(mapping[int(symbol)]) for mapping in MAPPINGS] for symbol in symbols])
    alternatives = family[rows[:,None],indices]
    target=completed[rows,address,None]
    # Registered float32 elimination predicate, independently expanded.
    delta = (alternatives-target).abs()
    matches = (delta <= ATOL+RTOL*target.abs()).flatten(2).all(-1)
    result, revealed = survivors.clone(), seen.clone()
    result[rows,address] &= matches
    revealed[rows,address,symbols] = True
    if not result.any(-1).all():
        raise ValueError('zero-survivor independent posterior')
    return result, revealed


def posterior_decode(v, survivors, symbols):
    rows = torch.arange(v.shape[0])
    family = family_reference(v)
    indices = torch.tensor([[BANDS.index(mapping[int(symbol)]) for mapping in MAPPINGS] for symbol in symbols])
    weights = survivors.double()/survivors.sum(-1,keepdim=True)
    return (family[rows[:,None],indices].double()*weights[:,:,None,None,None]).sum(1).float()


def visibility_expected(episode, completed, time, kind):
    remap = kind == 'address_remapping'
    v, last = episode['V'], completed[:,max(0,time-1)]
    if remap:
        v,last = v.roll(1,1),last.roll(1,1)
    address = episode['command_addresses'][:,time-1] if time else None
    if remap:
        address = (address+1)%4
    return dict(kind=kind, decision_time=time,last_revealed_at=max(0,time-1),source_revealed_through=max(0,time-1),
        update_command_indices=list(range(max(0,time-1))),V_hash=tensor_hash(v),last_revealed_W_hash=tensor_hash(last),
        observed_mask_hash=tensor_hash(torch.ones(96,4,dtype=torch.bool)),
        previous_address_hash=tensor_hash(address) if time else None,
        previous_symbol_hash=tensor_hash(episode['command_symbols'][:,time-1]) if time else None,
        current_completion_visible=time==0,fork_outcomes_visible=False)


def verify_visibility(audit, name, episode, completed, calls):
    expected_counts = {'primary':15,'address_remapping':3,'donor_current':3,'donor_prospective':3,
                       'restored_current':3,'restored_prospective':3}
    if name.endswith('/A'):
        expected_counts['same_weight_alias_duplicate']=15
    for kind,count in expected_counts.items():
        selected=[row for row in calls if row['kind']==kind]
        audit.check(name+'/visibility/'+kind+'/count',len(selected)==count,actual=len(selected),expected=count)
        times=list(range(15)) if kind in ('primary','same_weight_alias_duplicate') else [12,13,14]
        for row,time in zip(selected,times):
            expected=visibility_expected(episode,completed,time,kind)
            # Future schema may add state hashes; permitted boundary keys remain mandatory.
            for key,value in expected.items():
                audit.check(name+f'/visibility/{kind}/{time}/{key}',row.get(key)==value,actual=row.get(key),expected=value)
    audit.check(name+'/visibility/known_kinds',set(row['kind'] for row in calls)==set(expected_counts))


def bootstrap_reference(metrics, indices):
    result=dict(scope='conditional fixed source/layout/mapping design; no population significance',draws=256)
    def summary(value):
        value=value.double()
        draws=value[indices].mean(-1).sort().values
        bounds=[]
        for probability in (.025,.975):
            position=255*probability; lo,hi=math.floor(position),math.ceil(position)
            bounds.append(float(draws[lo]*(hi-position)+draws[hi]*(position-lo)) if lo!=hi else float(draws[lo]))
        return dict(mean=float(value.mean()),interval=bounds)
    for metric in metrics['A']:
        for source in SOURCES:
            result[source+'/'+metric]=summary(metrics[source][metric])
        for other in ('ordinary','analytic'):
            result['A_minus_'+other+'/'+metric]=summary(metrics['A'][metric].double()-metrics[other][metric].double())
    return result


def world_metrics_reference(retained,episode):
    metrics={}
    for quantity in ('mse','field_l1'):
        suffix='selected_normalized_mse' if quantity=='mse' else 'field_l1'
        for kind,axes in [('current',(1,)),('next',(1,2))]:
            metrics[kind+'_'+quantity]=retained[kind+'_'+suffix].mean(axes)
        suffix='full_normalized_mse_by_address' if quantity=='mse' else 'full_field_l1_by_address'
        metrics['current_full_'+quantity]=retained['current_'+suffix][:,1:].mean((1,2))
        metrics['next_full_'+quantity]=retained['next_'+suffix].mean((1,2,3))
    for kind,axes in [('current',(1,)),('next',(1,2))]:
        metrics[kind+'_coarse_disagreement']=retained[kind+'_coarse_disagreement'].double().mean(axes)
    for kind,quantity in [('current','coarse'),('next','coarse'),('current','cell'),('next','cell')]:
        value=retained[kind+'_full_'+quantity+'_disagreement_by_address']
        if kind=='current':
            value=value[:,1:]
        metrics[kind+'_full_'+quantity+'_disagreement']=value.double().flatten(1).mean(1)
    metrics['policy_targeting_accuracy']=retained['policy_action_correct'].float().mean(1)
    inherited=retained['next_full_normalized_mse_by_address'].clone()
    inherited[torch.arange(96)[:,None,None],torch.arange(15)[None,:,None],torch.arange(3)[None,None,:],episode['command_addresses'][:,:,None]]=0
    metrics['next_inherited_mse']=inherited.sum((1,2,3))/135
    for role,role_name in enumerate(('red','blue')):
        for quantity in ('cell_disagreement','estimated_cell_correct','actual_cell_correct'):
            for kind in ('current','next'):
                metrics[kind+'_'+role_name+'_'+quantity]=retained[kind+'_'+quantity][...,role].double().flatten(1).mean(1)
        for kind in ('current','next'):
            value=retained[kind+'_full_cell_disagreement_by_address'][...,role]
            if kind=='current':
                value=value[:,1:]
            metrics[kind+'_full_'+role_name+'_cell_disagreement']=value.double().flatten(1).mean(1)
        mask=episode['query_roles'][:,12:]==role
        metrics['policy_'+role_name+'_accuracy']=(retained['policy_action_correct'].double()*mask).sum(1)/mask.sum(1)
    for kind in ('current','next'):
        metrics[kind+'_cell_disagreement']=retained[kind+'_cell_disagreement'].double().flatten(1).mean(1)
    return metrics


def donor_summaries_reference(retained):
    summaries={}
    for schedule,ledger in retained['donors'].items():
        summary=dict(all_predetermined_pairs=288,changed_only_subset_used=False,
            coincident_address_pairs=int(ledger['coincident_schedule_addresses'].sum()))
        for key,value in ledger.items():
            if key.endswith('_changed') or key.endswith('_disagreement_with_fixed_W'):
                summary[key+'_fraction_all_pairs']=float(value.double().mean())
            elif '_l1_' in key or key.endswith('_mse_to_fixed_W_by_address'):
                summary[key+'_mean_all_pairs']=float(value.double().mean())
            elif '_full_' in key and key.endswith('_change'):
                summary[key+'_mean_abs_all_pairs']=float(value.double().abs().mean())
            if key.endswith('_disagreement_with_fixed_W') and 'cell_' in key:
                for role,role_name in enumerate(('red','blue')):
                    summary[key+'_'+role_name+'_fraction']=float(value[...,role].double().mean())
        summaries[schedule]=summary
    return summaries


def verify_episode(audit, seed, phase, episode, native):
    name=f'{seed}/{phase}/episode'
    audit.check(name+'/identity',episode['encoder_seed']==seed and episode['split']==phase)
    audit.check(name+'/namespace',episode['world_namespace']==310000000+seed*1000+('fit','dev','heldout').index(phase)*100000)
    audit.compare(name+'/indices',episode['world_indices'],torch.arange(96),exact=True)
    audit.compare(name+'/cells',episode['evaluator_cells'],layout_reference(),exact=True)
    for key,value in schedule_reference(seed,phase).items():
        audit.compare(name+'/'+key,episode[key],value,exact=True)
    audit.check(name+'/pixels_shape_range',episode['pixels'].shape==(96,4,3,16,16) and episode['pixels'].dtype==torch.float32
                and bool(((episode['pixels']>=0)&(episode['pixels']<=1)).all()))
    audit.compare(name+'/frozen_V',episode['V'],source_reference(episode['pixels'],native['frozen_source_weights']['encoder']))
    completed,forks,family=actual_reference(episode)
    for key,value in [('actual_completed_w',completed),('actual_forks',forks),('actual_operator_family',family)]:
        if key in episode:
            audit.compare(name+'/'+key,episode[key],value,exact=True)
    audit.check(name+'/compact_schema',episode.get('actual_reconstruction_schema')=='attention_content_model_v1_savedV_address_replacement')
    audit.check(name+'/compact_status',episode.get('actual_compaction_status')=='exact_reconstruction_bound')
    audit.compare(name+'/compact_checks',episode['actual_reconstruction_exact'],torch.ones(96,61,dtype=torch.bool),exact=True)
    audit.tree(name+'/compact_recipe',episode['actual_reconstruction_recipe'],dict(initial_completed='V',
        fork='copy completed[t]; replace command_addresses[t] by actual_operator_family at mapped symbol bandwidth',
        common_completed='fork[t,command_symbols[t]]',terminal_completed='common_completed after command14; archive only'))
    return completed,forks,family


def verify_fit(audit,seed,episode,fit,completed):
    name=f'{seed}/fit'
    audit.tree(name+'/normalizer',fit['normalizer'],normalizer_reference(episode['V']))
    audit.check(name+'/source_set',set(fit['checkpoints'])=={'A','ordinary'})
    audit.check(name+'/update_budget',fit['completed_updates_per_source']==300)
    audit.check(name+'/training_log_order',[row['source'] for row in fit['training_log']]==['A']*300+['ordinary']*300)
    for source in ('A','ordinary'):
        weights=fit['checkpoints'][source]
        audit.check(name+'/'+source+'/parameter_count',sum(v.numel() for v in weights.values())==197328)
        records=[row for row in fit['training_log'] if row['source']==source]
        audit.check(name+'/'+source+'/log_count',len(records)==300)
        for index,row in enumerate(records):
            audit.check(name+f'/{source}/update/{index}/index',row['update_index']==index and row['completed_updates']==index+1)
            audit.compare(name+f'/{source}/update/{index}/loss',row['loss'],.5*row['current_loss']+.5*row['next_loss'])
            audit.check(name+f'/{source}/update/{index}/finite_nonnegative',all(math.isfinite(row[k]) and row[k]>=0
                for k in ('loss','current_loss','next_loss','preclip_gradient_norm')))
        optimizer=fit['optimizer_terminal_states'][source]
        audit.check(name+'/'+source+'/optimizer_groups',len(optimizer['param_groups'])==1)
        group=optimizer['param_groups'][0]
        for key,value in dict(lr=.003,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False).items():
            audit.check(name+'/'+source+'/adam/'+key,group[key]==value)
        audit.check(name+'/'+source+'/optimizer_state_count',len(optimizer['state'])==len(weights)==8)
        for parameter,state in optimizer['state'].items():
            audit.compare(name+f'/{source}/optimizer/{parameter}/step',state['step'],300.)
            audit.check(name+f'/{source}/optimizer/{parameter}/keys',set(state)=={'step','exp_avg','exp_avg_sq'})
            expected_shape=list(weights.values())[group['params'].index(parameter)].shape
            for moment in ('exp_avg','exp_avg_sq'):
                audit.check(name+f'/{source}/optimizer/{parameter}/{moment}_shape',state[moment].shape==expected_shape
                    and state[moment].dtype==torch.float32)
            audit.check(name+f'/{source}/optimizer/{parameter}/squared_moment_nonnegative',bool((state['exp_avg_sq']>=0).all()))
        calls=fit['fit_visibility'][source]
        audit.check(name+'/'+source+'/visibility_count',len(calls)==15)
        for time,row in enumerate(calls):
            for key,value in visibility_expected(episode,completed,time,'primary').items():
                audit.check(name+f'/{source}/visibility/{time}/{key}',row.get(key)==value)
    audit.tree(name+'/visibility_reconstruction',fit['visibility_reconstruction'],
        dict(updates_per_source=300,reset_state_each_update=True,decision_indices=list(range(15)),revealed_at=list(range(1,15)),
            source_input_manifest='identical permission/observation manifests to first update; fitted h differs',
            fit_trajectory_independently_proven=False,trajectory_reconstruction_requires_deterministic_optimizer_replay=True))
    for source in ('A','ordinary'):
        initial=fit['initial_checkpoints'][source]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(320000000+seed+(1000000 if source=='ordinary' else 0))
            layers=(torch.nn.GRUCell(771,48),torch.nn.Linear(435,96),torch.nn.Linear(96,384))
        expected={**{'gru.'+k:v for k,v in layers[0].state_dict().items()},
            **{'decoder.0.'+k:v for k,v in layers[1].state_dict().items()},
            **{'decoder.2.'+k:v for k,v in layers[2].state_dict().items()}}
        for parameter,value in expected.items():
            audit.compare(name+'/'+source+'/initial/'+parameter,initial[parameter],value,exact=True)
            audit.check(name+'/'+source+'/terminal_shape/'+parameter,fit['checkpoints'][source][parameter].shape==value.shape
                and fit['checkpoints'][source][parameter].dtype==torch.float32)
        hidden=torch.zeros(96,4,48); rows=torch.arange(96); current_losses=[]; next_losses=[]
        _,forks,_=actual_reference(episode)
        for time in range(15):
            if time:
                previous=episode['command_addresses'][:,time-1]; symbol=episode['command_symbols'][:,time-1]
                predicted=decoder_reference(episode['V'][rows,previous],hidden[rows,previous],symbol,initial,fit['normalizer'])
                current_losses.append(mse(predicted,completed[rows,time,previous],fit['normalizer']).mean())
            address=episode['command_addresses'][:,time]
            predicted=torch.stack([decoder_reference(episode['V'][rows,address],hidden[rows,address],
                torch.full((96,),symbol,dtype=torch.long),initial,fit['normalizer']) for symbol in range(3)],1)
            target=forks[rows[:,None],time,torch.arange(3)[None,:],address[:,None]]
            next_losses.append(mse(predicted,target,fit['normalizer']).mean())
            if time:
                hidden=hidden.clone();hidden[rows,previous]=gru_reference(hidden[rows,previous],episode['V'][rows,previous],
                    completed[rows,time,previous],symbol,initial,fit['normalizer'])
        first=[row for row in fit['training_log'] if row['source']==source][0]
        for metric,value in dict(current_loss=torch.stack(current_losses).mean(),next_loss=torch.stack(next_losses).mean()).items():
            audit.compare(name+'/'+source+'/first_update/'+metric,first[metric],value)
        from attention_content_model_v1 import LearnedContentModel,FitNormalizer,ObservationBatch,CommandBatch,RevealedFeedback
        core=LearnedContentModel(FitNormalizer(tuple(fit['normalizer']['mean']),tuple(fit['normalizer']['scale'])),0,source)
        core.load_state_dict(initial);core.eval();state=core.initial_state(96);manifests=[]
        for time in range(15):
            command=CommandBatch(episode['command_addresses'][:,time-1],episode['command_symbols'][:,time-1]) if time else None
            observation=ObservationBatch(episode['V'],completed[:,max(0,time-1)],time,max(0,time-1),command,torch.ones(96,4,dtype=torch.bool))
            manifests.append(observation_manifest_reference(observation,state,episode['command_addresses'][:,time]))
            if time:
                state=core.update(state,episode['V'],RevealedFeedback(completed[:,time],command,time,time-1,time))
        audit.tree(name+'/'+source+'/production_core_initial_permission_manifests',fit['fit_visibility'][source],manifests)
    return dict(optimizer_trajectory_proved=False,recorded_updates_per_source=300,
        recorded_losses={source:dict(first=records[0]['loss'],minimum=min(row['loss'] for row in records),
            last_pre_final_update=records[-1]['loss']) for source in ('A','ordinary')
            for records in [[row for row in fit['training_log'] if row['source']==source]]})


def controls_reference(maps,native):
    encoder=native['frozen_source_weights']['encoder'];orientation=native['frozen_source_weights']['orientation_head']
    logits_list,probability_list,orientation_list=[],[],[]
    # Production records each individual item through a [1,384] suffix.
    for item in maps.reshape(-1,384):
        hidden=torch.nn.functional.linear(item[None],encoder['net.7.weight'],encoder['net.7.bias']).clamp_min(0)
        logits=torch.nn.functional.linear(hidden,encoder['net.9.weight'],encoder['net.9.bias'])
        stats=native['frozen_orientation_standardizer']
        coordinates=(hidden-stats['mean'])/stats['scale']
        middle=torch.nn.functional.linear(coordinates,orientation['0.weight'],orientation['0.bias']).clamp_min(0)
        ori=torch.nn.functional.linear(middle,orientation['2.weight'],orientation['2.bias'])[:,0]
        logits_list.append(logits[0]);orientation_list.append(ori[0])
        probability_list.append(torch.cat((logits[:,:4].softmax(-1),logits[:,4:].softmax(-1)),-1)[0])
    leading=maps.shape[:-3]
    return dict(category_logits=torch.stack(logits_list).reshape(*leading,8),
        category_probabilities=torch.stack(probability_list).reshape(*leading,8),
        orientation_logits=torch.stack(orientation_list).reshape(leading))


def action_cells(fields,roles):
    return fields[torch.arange(96),roles].flatten(-2).argmax(-1)


def verify_actual_controls(audit,name,episode,shared,metadata,native,completed,family):
    rows=torch.arange(96);provenance=[]
    for time in (12,13,14):
        address=episode['command_addresses'][:,time];roles=episode['query_roles'][:,time]
        saved=completed[:,time];sham=saved.clone();sham[rows,address]=episode['V'][rows,address]
        changed=saved.clone();changed[rows,address]=family[rows,address,1]
        maps=dict(sham=sham,endogenous=changed,exogenous=changed.clone(),restored=saved.clone(),address_remapped=saved.roll(1,1))
        for label,value in maps.items():
            logits=head_reference(value,native);fields=fields_reference(logits)
            emitted=shared['actual_controls'][label+'_role_fields'][:,time-12]
            audit.compare(name+f'/actual_control/{time}/{label}/logits',shared['actual_controls'][label+'_local_logits'][:,time-12],logits)
            audit.compare(name+f'/actual_control/{time}/{label}/fields',emitted,fields)
            query=(address+1)%4 if label=='address_remapped' else address
            audit.compare(name+f'/actual_control/{time}/{label}/action',shared['actual_controls'][label+'_action_cell'][:,time-12],
                action_cells(emitted[rows,query],roles),exact=True)
        for left,right in [('endogenous','exogenous')]:
            for suffix in ('role_fields','local_logits','action_cell'):
                audit.compare(name+f'/actual_control/{time}/driver/{suffix}',shared['actual_controls'][left+'_'+suffix][:,time-12],
                    shared['actual_controls'][right+'_'+suffix][:,time-12],exact=True)
        audit.compare(name+f'/actual_control/{time}/restore_fields',shared['actual_controls']['restored_role_fields'][:,time-12],
            shared['actual_completed_fields'][:,time],exact=True)
        audit.compare(name+f'/actual_control/{time}/restore_logits',shared['actual_controls']['restored_local_logits'][:,time-12],
            shared['actual_completed_logits'][:,time],exact=True)
        audit.compare(name+f'/actual_control/{time}/remapped_fields',shared['actual_controls']['address_remapped_role_fields'][:,time-12],
            shared['actual_controls']['restored_role_fields'][:,time-12].roll(1,1),exact=True)
        audit.compare(name+f'/actual_control/{time}/remapped_actions',shared['actual_controls']['address_remapped_action_cell'][:,time-12],
            shared['actual_controls']['restored_action_cell'][:,time-12],exact=True)
        for world in range(96):
            index=int(address[world]);physical=('l0','l1','l2','l3')[index]
            provenance.append(dict(time=time,world_index=world,address=physical,
                saved_w_hashes=[tensor_hash(saved[world,a]) for a in range(4)],
                changed_w_hashes=[tensor_hash(changed[world,a]) for a in range(4)],
                drivers=['endogenous','exogenous'],restoration='exact_saved_snapshot'))
    # JSON conversion represents physical address tuples as lists.
    audit.tree(name+'/actual_control_provenance',metadata['actual_control_provenance'],
        json.loads(json.dumps(provenance)))
    for key,value in controls_reference(completed,native).items():
        audit.compare(name+'/W_derived_controls/'+key,shared['W_derived_controls'][key],value)


def donor_reference(audit,name,episode,metadata):
    rows=torch.arange(96);prefixes={};expected=[]
    for time in (12,13,14):
        for schedule in ('current','prospective'):
            address=episode['command_addresses'][:,time-1 if schedule=='current' else time]
            mapping=episode['evaluator_mapping_indices'].clone();mapping[rows,address]=(mapping[rows,address]+1)%6
            completed,_,_=actual_reference(episode,mapping,transitions=time)
            prefixes[(time,schedule)]=(address,completed)
            expected.append(dict(time=time,schedule=schedule,patched_addresses=address.tolist(),
                executor_command_indices=list(range(time)),permitted_feedback_command_indices=list(range(time-1)),
                concealed_completion_hash=tensor_hash(completed[:,time]),pre_reveal_mask=[index<time for index in range(time+1)],
                post_reveal_mask_not_used_for_donor_state=[True]*(time+1),mapping_indices=mapping.tolist(),
                updates=[dict(command_index=index,completed_at=index+1,
                    address_hash=tensor_hash(episode['command_addresses'][:,index]),
                    symbol_hash=tensor_hash(episode['command_symbols'][:,index]),completed_map_hash=tensor_hash(completed[:,index+1]))
                    for index in range(time-1)]))
    audit.tree(name+'/donor_provenance',metadata['donor_provenance'],expected)
    audit.check(name+'/donor_executor_count',96*sum(row['time'] for row in expected)==7488)
    audit.check(name+'/donor_feedback_count_per_source',96*sum(row['time']-1 for row in expected)==6912)
    return prefixes


def assemble_reference(episode,completed,time,current_selected,next_selected):
    rows=torch.arange(96);address=episode['command_addresses'][:,time]
    current=completed[:,max(0,time-1)].clone()
    if time:
        current[rows,episode['command_addresses'][:,time-1]]=current_selected
    future=current[:,None].expand(-1,3,-1,-1,-1,-1).clone()
    future[rows[:,None],torch.arange(3)[None,:],address[:,None]]=next_selected
    return current,future


def reached_reference(episode,completed,time,source,fit):
    rows=torch.arange(96);state=torch.zeros(96,4,48) if source!='analytic' else torch.ones(96,4,6,dtype=torch.bool)
    seen=torch.zeros(96,4,3,dtype=torch.bool)
    for index in range(time-1):
        address=episode['command_addresses'][:,index];symbols=episode['command_symbols'][:,index]
        if source!='analytic':
            state=state.clone();state[rows,address]=gru_reference(state[rows,address],episode['V'][rows,address],
                completed[rows,index+1,address],symbols,fit['checkpoints'][source],fit['normalizer'])
        else:
            state,seen=posterior_update(state,seen,episode['V'],completed[:,index+1],address,symbols)
    return state,seen


def verify_donors(audit,name,source,episode,retained,shared,fit,native,completed,forks,prefixes):
    rows=torch.arange(96)
    for time in (12,13,14):
        address=episode['command_addresses'][:,time];previous=episode['command_addresses'][:,time-1]
        symbols=episode['command_symbols'][:,time-1];roles=episode['query_roles'][:,time]
        base_current=retained['selected_current_maps'][:,time-1];base_next=retained['selected_next_maps'][:,time]
        original_current,original_next=assemble_reference(episode,completed,time,base_current,base_next)
        for schedule in ('current','prospective'):
            key=name+f'/donor/{schedule}/{time}';ledger=retained['donors'][schedule];j=time-12
            patched_address,donor_completed=prefixes[(time,schedule)]
            reached,_=reached_reference(episode,donor_completed,time,source,fit)
            if source!='analytic':
                saved=retained['pre_reveal_hidden'][:,time]
                donor=ledger['donor_hidden_at_patched_address'][:,j]
                audit.compare(key+'/reached_hidden',donor,reached[rows,patched_address])
            else:
                saved=retained['pre_reveal_survivors'][:,time]
                donor=ledger['donor_survivors_at_patched_address'][:,j]
                audit.compare(key+'/reached_posterior',donor,reached[rows,patched_address],exact=True)
                _,donor_seen=reached_reference(episode,donor_completed,time,source,fit)
                audit.compare(key+'/reached_revealed_symbols',ledger['donor_revealed_symbols_at_patched_address'][:,j],donor_seen[rows,patched_address],exact=True)
            patched=saved.clone();patched[rows,patched_address]=donor
            def decode(addr,symbol):
                if source=='analytic':
                    return posterior_decode(episode['V'][rows,addr],patched[rows,addr],symbol)
                return decoder_reference(episode['V'][rows,addr],patched[rows,addr],symbol,fit['checkpoints'][source],fit['normalizer'])
            emitted_current=ledger['selected_current_maps'][:,j];emitted_next=ledger['selected_next_maps'][:,j]
            audit.compare(key+'/selected_current',emitted_current,decode(previous,symbols))
            audit.compare(key+'/selected_next',emitted_next,torch.stack([decode(address,torch.full((96,),s,dtype=torch.long)) for s in range(3)],1))
            patched_current,patched_next=assemble_reference(episode,completed,time,emitted_current,emitted_next)
            fields=ledger['policy_candidate_fields'][:,j]
            audit.compare(key+'/candidate_H_of_A',fields,fields_reference(head_reference(emitted_next,native)))
            maxima=fields[rows[:,None],torch.arange(3)[None,:],roles[:,None]].flatten(-2).max(-1).values
            policy=maxima.argmax(-1);actions=shared['actual_query_action_cells'][rows,j,policy]
            audit.compare(key+'/maxima',ledger['policy_role_maxima'][:,j],maxima,exact=True)
            audit.compare(key+'/policy',ledger['policy_symbol'][:,j],policy,exact=True)
            audit.compare(key+'/action_reference',ledger['policy_action_cell_reference'][:,j],actions,exact=True)
            audit.compare(key+'/coincident_addresses',ledger['coincident_schedule_addresses'][:,j],previous==address,exact=True)
            current_emitted=ledger['selected_current_fields'][:,j]
            audit.compare(key+'/current_H_of_A',current_emitted,fields_reference(head_reference(emitted_current,native)))
            base_fields=shared['actual_completed_fields'][:,time-1].clone()
            base_fields[rows,previous]=retained['selected_current_fields'][:,time-1]
            base_future=base_fields[:,None].expand(-1,3,-1,-1,-1,-1).clone()
            base_future[rows[:,None],torch.arange(3)[None,:],address[:,None]]=retained['selected_next_fields'][:,time]
            changed_fields=base_fields.clone();changed_fields[rows,previous]=current_emitted
            changed_future=changed_fields[:,None].expand(-1,3,-1,-1,-1,-1).clone()
            changed_future[rows[:,None],torch.arange(3)[None,:],address[:,None]]=fields
            bands=episode['evaluator_mappings'][rows,address];indices=torch.where(bands==1,0,torch.where(bands==2,1,2))
            actual_next=shared['actual_family_fields'][rows[:,None],address[:,None],indices]
            actual_future=shared['actual_completed_fields'][:,time,None].expand(-1,3,-1,-1,-1,-1).clone()
            actual_future[rows[:,None],torch.arange(3)[None,:],address[:,None]]=actual_next
            for label,changed,base,actual in [('current',changed_fields,base_fields,shared['actual_completed_fields'][:,time]),
                ('next',changed_future,base_future,actual_future)]:
                comparisons={label+'_full_field_changed':~(changed==base).flatten(1).all(-1),
                    label+'_full_field_l1_change_by_address':(changed-base).abs().mean((-3,-2,-1)),
                    label+'_full_field_l1_to_fixed_W_by_address':(changed-actual).abs().mean((-3,-2,-1)),
                    label+'_full_cell_disagreement_with_fixed_W':changed.flatten(-2).argmax(-1)!=actual.flatten(-2).argmax(-1),
                    label+'_full_coarse_disagreement_with_fixed_W':coarse_reference(changed)!=coarse_reference(actual)}
                before_scalars=field_scalars_reference(base);after_scalars=field_scalars_reference(changed)
                for metric in before_scalars:
                    comparisons[label+'_full_'+metric+'_change']=after_scalars[metric]-before_scalars[metric]
                for field,expected in comparisons.items():
                    audit.compare(key+'/'+field,ledger[field][:,j],expected,exact=expected.dtype==torch.bool)
            for label,estimated,actual,truth in [('current',current_emitted,shared['actual_completed_fields'][rows,time,previous],episode['evaluator_cells'][rows,previous]),
                ('next',fields,actual_next,episode['evaluator_cells'][rows,address][:,None].expand(-1,3,-1,-1))]:
                for metric,expected in field_metrics(estimated,actual,truth).items():
                    field=label+'_factual_'+metric
                    audit.compare(key+'/'+field,ledger[field][:,j],expected,exact=expected.dtype in (torch.bool,torch.long))
            audit.compare(key+'/current_fixed_W_mse',ledger['current_full_normalized_mse_to_fixed_W_by_address'][:,j],
                mse(patched_current,completed[:,time],fit['normalizer']))
            audit.compare(key+'/next_fixed_W_mse',ledger['next_full_normalized_mse_to_fixed_W_by_address'][:,j],
                mse(patched_next,forks[:,time],fit['normalizer']))
            for field,expected in [('current_changed',~(patched_current==original_current).flatten(1).all(-1)),
                ('next_changed',~(patched_next==original_next).flatten(1).all(-1)),
                ('policy_changed',policy!=retained['policy_symbol'][:,j]),
                ('action_changed',actions!=retained['policy_action_cell'][:,j])]:
                audit.compare(key+'/'+field,ledger[field][:,j],expected,exact=True)
            for field,expected in [('restored_selected_current_maps',base_current),('restored_selected_next_maps',base_next),
                ('restored_policy_symbol',retained['policy_symbol'][:,j]),('restored_policy_candidate_fields',retained['selected_next_fields'][:,time])]:
                audit.compare(key+'/'+field,ledger[field][:,j],expected,exact=True)
            nonselected_current=(patched_current==original_current).flatten(2).all(-1)
            nonselected_next=(patched_next==original_next).flatten(3).all(-1).all(1)
            nonselected_current[rows,patched_address]=True;nonselected_next[rows,patched_address]=True
            audit.compare(key+'/nonselected_current',nonselected_current,torch.ones_like(nonselected_current),exact=True)
            audit.compare(key+'/nonselected_next',nonselected_next,torch.ones_like(nonselected_next),exact=True)


def verify_source_shapes(audit,name,source,retained):
    specifications={}
    for prefix,count,symbol_axes in [('current',14,()),('next',15,(3,))]:
        lead=(96,count,*symbol_axes)
        for field,tail in [('maps',(24,4,4)),('logits',(3,4,4)),('fields',(2,4,4))]:
            specifications['selected_'+prefix+'_'+field]=(lead+tail,torch.float32)
        specifications[prefix+'_selected_normalized_mse']=(lead,torch.float32)
        specifications[prefix+'_distance_to_each_actual_operator']=(lead+(3,),torch.float32)
        for metric in ('field_l1','coarse_disagreement','estimated_coarse_correct','actual_coarse_correct'):
            dtype=torch.float32 if metric=='field_l1' else torch.bool
            specifications[prefix+'_'+metric]=(lead,dtype)
        for metric in ('cell_disagreement','estimated_cell_correct','actual_cell_correct'):
            specifications[prefix+'_'+metric]=(lead+(2,),torch.bool)
        for metric in ('expected_columns','entropy'):
            specifications[prefix+'_'+metric]=(lead+(2,),torch.float64)
        for metric in ('maxima','top_two_margin'):
            specifications[prefix+'_'+metric]=(lead+(2,),torch.float32)
        specifications[prefix+'_max_tie_count']=(lead+(2,),torch.long)
    for prefix,lead in [('current',(96,15,4)),('next',(96,15,3))]:
        for statistic in ('mean','std','norm'):
            specifications[prefix+'_channel_'+statistic]=(lead+(24,),torch.float32)
    for prefix,shape in [('current',(96,15,4)),('next',(96,15,3,4))]:
        specifications[prefix+'_full_normalized_mse_by_address']=(shape,torch.float32)
        specifications[prefix+'_full_field_l1_by_address']=(shape,torch.float32)
        specifications[prefix+'_full_cell_disagreement_by_address']=(shape+(2,),torch.bool)
        specifications[prefix+'_full_coarse_disagreement_by_address']=(shape,torch.bool)
    for field in ('policy_symbol','policy_action_cell','policy_action_tie_count'):
        specifications[field]=((96,3),torch.long)
    specifications.update(policy_action_correct=((96,3),torch.bool),policy_role_maxima=((96,3,3),torch.float32),
        policy_actual_role_fields=((96,3,2,4,4),torch.float32))
    if source=='analytic':
        specifications.update(pre_reveal_survivors=((96,15,4,6),torch.bool),pre_reveal_revealed_symbols=((96,15,4,3),torch.bool),
            survivor_count=((96,15,4),torch.long),posterior_entropy=((96,15,4),torch.float64),
            revealed_symbol_count=((96,15,4),torch.long),terminal_state=((96,4,6),torch.bool))
        specifications['terminal_revealed_symbols']=((96,4,3),torch.bool)
    else:
        specifications.update(pre_reveal_hidden=((96,15,4,48),torch.float32),terminal_state=((96,4,48),torch.float32))
    for field,(shape,dtype) in specifications.items():
        value=retained.get(field)
        audit.check(name+'/shape_dtype/'+field,isinstance(value,torch.Tensor) and tuple(value.shape)==shape and value.dtype==dtype,
            expected_shape=list(shape),expected_dtype=str(dtype),actual_shape=list(value.shape) if isinstance(value,torch.Tensor) else None,
            actual_dtype=str(value.dtype) if isinstance(value,torch.Tensor) else None)
    expected_keys=set(specifications)|{'donors','terminal_revealed_through','terminal_update_command_indices'}
    audit.check(name+'/tensor_inventory',set(retained)==expected_keys,actual=sorted(retained),expected=sorted(expected_keys))
    audit.check(name+'/donor_schedule_inventory',set(retained['donors'])=={'current','prospective'})
    for schedule in ('current','prospective'):
        donor_specs=dict(selected_current_maps=((96,3,24,4,4),torch.float32),selected_next_maps=((96,3,3,24,4,4),torch.float32),
            policy_symbol=((96,3),torch.long),policy_role_maxima=((96,3,3),torch.float32),policy_candidate_fields=((96,3,3,2,4,4),torch.float32),
            policy_action_cell_reference=((96,3),torch.long),restored_policy_symbol=((96,3),torch.long),
            restored_policy_candidate_fields=((96,3,3,2,4,4),torch.float32),restored_selected_current_maps=((96,3,24,4,4),torch.float32),
            restored_selected_next_maps=((96,3,3,24,4,4),torch.float32))
        for flag in ('current_changed','next_changed','policy_changed','action_changed'):
            donor_specs[flag]=((96,3),torch.bool)
        donor_specs['donor_hidden_at_patched_address' if source!='analytic' else 'donor_survivors_at_patched_address']=(
            (96,3,48 if source!='analytic' else 6),torch.float32 if source!='analytic' else torch.bool)
        donor_specs.update(coincident_schedule_addresses=((96,3),torch.bool),selected_current_fields=((96,3,2,4,4),torch.float32))
        for label,lead in [('current',(96,3,4)),('next',(96,3,3,4))]:
            donor_specs[label+'_full_field_changed']=((96,3),torch.bool)
            for field in ('field_l1_change_by_address','field_l1_to_fixed_W_by_address','normalized_mse_to_fixed_W_by_address'):
                donor_specs[label+'_full_'+field]=(lead,torch.float32)
            donor_specs[label+'_full_coarse_disagreement_with_fixed_W']=(lead,torch.bool)
            donor_specs[label+'_full_cell_disagreement_with_fixed_W']=(lead+(2,),torch.bool)
            for field in ('entropy','expected_columns','maxima','top_two_margin'):
                donor_specs[label+'_full_'+field+'_change']=(lead+(2,),torch.float64 if field in ('entropy','expected_columns') else torch.float32)
        for label,lead in [('current',(96,3)),('next',(96,3,3))]:
            for field in ('field_l1','coarse_disagreement','estimated_coarse_correct','actual_coarse_correct'):
                donor_specs[label+'_factual_'+field]=(lead,torch.float32 if field=='field_l1' else torch.bool)
            for field in ('cell_disagreement','estimated_cell_correct','actual_cell_correct'):
                donor_specs[label+'_factual_'+field]=(lead+(2,),torch.bool)
            for field in ('entropy','expected_columns','maxima','top_two_margin','max_tie_count'):
                dtype=torch.long if field=='max_tie_count' else torch.float64 if field in ('entropy','expected_columns') else torch.float32
                donor_specs[label+'_factual_'+field]=(lead+(2,),dtype)
        if source=='analytic':
            donor_specs['donor_revealed_symbols_at_patched_address']=((96,3,3),torch.bool)
        audit.check(name+'/donor/'+schedule+'/tensor_inventory',set(retained['donors'][schedule])==set(donor_specs),
            actual=sorted(retained['donors'][schedule]),expected=sorted(donor_specs))
        for field,(shape,dtype) in donor_specs.items():
            value=retained['donors'][schedule].get(field)
            audit.check(name+'/donor/'+schedule+'/shape_dtype/'+field,isinstance(value,torch.Tensor) and tuple(value.shape)==shape and value.dtype==dtype,
                expected_shape=list(shape),expected_dtype=str(dtype))
        if source=='analytic':
            value=retained['donors'][schedule]['donor_revealed_symbols_at_patched_address']
            audit.check(name+'/donor/'+schedule+'/seen_shape_dtype',tuple(value.shape)==(96,3,3) and value.dtype==torch.bool)


def state_hash_reference(state):
    if hasattr(state,'hidden'):
        return dict(hidden=tensor_hash(state.hidden))
    return dict(survivors=tensor_hash(state.survivors),revealed_symbols=tensor_hash(state.revealed_symbols))


def observation_manifest_reference(obs,state,address,kind='primary'):
    return dict(kind=kind,decision_time=obs.decision_time,last_revealed_at=obs.last_revealed_at,
        source_revealed_through=state.revealed_through,update_command_indices=list(state.update_command_indices),
        source_key=state.source_key,state_hashes=state_hash_reference(state),candidate_address_hash=tensor_hash(address),
        V_hash=tensor_hash(obs.v),last_revealed_W_hash=tensor_hash(obs.last_revealed_w),observed_mask_hash=tensor_hash(obs.observed),
        previous_address_hash=tensor_hash(obs.previous_command.address) if obs.previous_command else None,
        previous_symbol_hash=tensor_hash(obs.previous_command.symbol) if obs.previous_command else None,
        current_completion_visible=obs.decision_time==0,fork_outcomes_visible=False)


def prediction_manifest_reference(obs,state,address,prediction,index,kind='primary'):
    record=observation_manifest_reference(obs,state,address,kind)
    keys=('V_hash','last_revealed_W_hash','observed_mask_hash','candidate_address_hash','previous_address_hash','previous_symbol_hash')
    return dict(**record,call_index=index,completed=True,post_call_state_hashes=record['state_hashes'],
        current_output_hash=tensor_hash(prediction.current),next_output_hash=tensor_hash(prediction.next),
        post_call_input_hashes={key:record[key] for key in keys})


def update_manifest_reference(state,v,feedback,updated,index,kind,schedule=None,time=None):
    return dict(kind=kind,source_key=state.source_key,call_index=index,pre_revealed_through=state.revealed_through,
        pre_update_command_indices=list(state.update_command_indices),pre_state_hashes=state_hash_reference(state),V_hash=tensor_hash(v),
        completed_W_hash=tensor_hash(feedback.completed_w),completed_at=feedback.completed_at,command_index=feedback.command_index,
        scored_decision_at=feedback.scored_decision_at,command_address_hash=tensor_hash(feedback.command.address),
        command_symbol_hash=tensor_hash(feedback.command.symbol),observed_mask_hash=tensor_hash(torch.ones(96,4,dtype=torch.bool)),
        pre_reveal_mask=[i<=state.revealed_through for i in range(feedback.completed_at+1)],post_reveal_mask=[True]*(feedback.completed_at+1),
        donor_schedule=schedule,donor_decision_time=time,completed=True,post_state_hashes=state_hash_reference(updated),
        post_revealed_through=updated.revealed_through,post_update_command_indices=list(updated.update_command_indices))


def decoder_probe_reference(model,episode,state,time,schedule):
    selected=episode['V'].reshape(384,24,4,4)
    latent=state.hidden.reshape(384,48) if hasattr(state,'hidden') else state.survivors.reshape(384,6)
    records=[];maps=[]
    for symbol in range(3):
        output=model.decode(selected,latent,torch.full((384,),symbol,dtype=torch.long))
        records.append(dict(kind='address_local_decoder_probe',source_key=state.source_key,decision_time=time,
            donor_schedule=schedule,symbol=symbol,state_hashes=state_hash_reference(state),V_hash=tensor_hash(episode['V']),
            addresses='all four, flattened within each world',completed=True,output_hash=tensor_hash(output),output_shape=[384,24,4,4]))
        maps.append(output.reshape(96,4,24,4,4))
    return records,torch.stack(maps,2)


def verify_production_replay(audit,name,episode,evaluation,fit,native,completed,prefixes):
    """Core input-isolation replay, explicitly distinct from independent equations.

    Only frozen archived observations/weights are used. No engine, collector,
    renderer, target-fed inference, fitting or optimizer is called.
    """
    from dataclasses import fields,replace
    import inspect
    from attention_content_model_v1 import (LearnedContentModel,AnalyticMappingSource,FitNormalizer,
        CommandBatch,ObservationBatch,RevealedFeedback,ExactAlias,NativePredictionReadout,
        patch_addressed_state,snapshot_state,restore_state)
    from spatial_native_task_v1 import LocalOccupancyHead,ChannelStandardizer
    audit.check(name+'/production_core_input_isolation/observation_fields',
        [field.name for field in fields(ObservationBatch)]==['v','last_revealed_w','decision_time','last_revealed_at','previous_command','observed'])
    audit.check(name+'/production_core_input_isolation/predict_signature',
        list(inspect.signature(LearnedContentModel.predict).parameters)==['self','observation','state','candidate_address'])
    coordinates=native['native_head_fit']['fit_only_standardizer'];head=LocalOccupancyHead()
    head.load_state_dict(native['native_head_fit']['weights']);head.eval()
    readout=NativePredictionReadout(head,ChannelStandardizer(tuple(coordinates['mean']),tuple(coordinates['scale'])))
    rows=torch.arange(96)
    for source in SOURCES:
        key=name+'/'+source+'/production_core_replay';retained=evaluation['tensors']['sources'][source]
        if source=='analytic':
            model=AnalyticMappingSource()
        else:
            model=LearnedContentModel(FitNormalizer(tuple(fit['normalizer']['mean']),tuple(fit['normalizer']['scale'])),0,source)
            model.load_state_dict(fit['checkpoints'][source]);model.eval()
        state=model.initial_state(96)
        expected_calls=[];expected_updates=[];expected_decoder_calls=[]
        before={k:tensor_hash(v) for k,v in model.state_dict().items()} if source!='analytic' else {}
        for time in range(15):
            previous=CommandBatch(episode['command_addresses'][:,time-1],episode['command_symbols'][:,time-1]) if time else None
            obs=ObservationBatch(episode['V'],completed[:,max(0,time-1)],time,max(0,time-1),previous,torch.ones(96,4,dtype=torch.bool))
            address=episode['command_addresses'][:,time]
            prediction=model.predict(obs,state,address)
            expected_calls.append(prediction_manifest_reference(obs,state,address,prediction,len(expected_calls)))
            audit.compare(key+f'/state/{time}',state.hidden if source!='analytic' else state.survivors,
                retained['pre_reveal_hidden' if source!='analytic' else 'pre_reveal_survivors'][:,time],exact=True)
            if source=='analytic':
                audit.compare(key+f'/revealed_symbols/{time}',state.revealed_symbols,retained['pre_reveal_revealed_symbols'][:,time],exact=True)
            next_selected=prediction.next[rows[:,None],torch.arange(3)[None,:],address[:,None]]
            audit.compare(key+f'/next/{time}',next_selected,retained['selected_next_maps'][:,time],exact=True)
            current_selected=prediction.current[rows,previous.address] if time else None
            if time:
                audit.compare(key+f'/current/{time}',current_selected,retained['selected_current_maps'][:,time-1],exact=True)
            current,future=assemble_reference(episode,completed,time,current_selected,next_selected)
            audit.compare(key+f'/copied_current/{time}',prediction.current,current,exact=True)
            audit.compare(key+f'/inherited_next/{time}',prediction.next,future,exact=True)
            # Perturb only addresses that cannot enter either selected decoder.
            relevant=torch.zeros(96,4,dtype=torch.bool);relevant[rows,address]=True
            if time:
                relevant[rows,previous.address]=True
            decoy_v=obs.v.clone();decoy_v[~relevant]=-decoy_v[~relevant]+.125
            if source!='analytic':
                decoy_h=state.hidden.clone();decoy_h[~relevant]+=1.
                decoy_state=replace(state,hidden=decoy_h)
            else:
                decoy_survivors=state.survivors.clone();decoy_survivors[~relevant]=True
                decoy_state=replace(state,survivors=decoy_survivors)
            decoy=model.predict(replace(obs,v=decoy_v),decoy_state,address)
            audit.compare(key+f'/decoy_selected_next/{time}',decoy.next[rows[:,None],torch.arange(3)[None,:],address[:,None]],next_selected,exact=True)
            if time:
                audit.compare(key+f'/decoy_selected_current/{time}',decoy.current[rows,previous.address],current_selected,exact=True)
            if source=='A':
                alias=ExactAlias(model).predict(obs,state,address)
                expected_calls.append(prediction_manifest_reference(obs,state,address,alias,len(expected_calls),'same_weight_alias_duplicate'))
                audit.compare(key+f'/same_object_alias_current/{time}',alias.current,prediction.current,exact=True)
                audit.compare(key+f'/same_object_alias_next/{time}',alias.next,prediction.next,exact=True)
            if time==0:
                rejected=False
                try:
                    model.predict(episode,state,address)
                except (ValueError,TypeError):
                    rejected=True
                audit.check(key+'/reject_evaluator_dictionary',rejected)
                rejected=False
                try:
                    model.predict(obs,state,address,mapping_labels=episode['evaluator_mappings'])
                except TypeError:
                    rejected=True
                audit.check(key+'/reject_extra_oracle_argument',rejected)
            if time>=12:
                role=episode['query_roles'][:,time];j=time-12
                choice=readout.choose(prediction,address,role)
                audit.compare(key+f'/primary_choice/{time}',choice.symbol,retained['policy_symbol'][:,j],exact=True)
                audit.compare(key+f'/primary_fields/{time}',choice.candidate_fields,retained['selected_next_fields'][:,time],exact=True)
                if source!='analytic' and time==12:
                    alternate='ordinary' if source=='A' else 'A'
                    relabelled=LearnedContentModel(model.normalizer,0,alternate)
                    relabelled.load_state_dict(fit['checkpoints'][source]);relabelled.eval()
                    alternate_prediction=relabelled.predict(obs,replace(state,source_key=alternate),address)
                    audit.compare(key+'/source_key_audit_only_current',alternate_prediction.current,prediction.current,exact=True)
                    audit.compare(key+'/source_key_audit_only_next',alternate_prediction.next,prediction.next,exact=True)
                moved_state=replace(state,hidden=state.hidden.roll(1,1)) if source!='analytic' else replace(
                    state,survivors=state.survivors.roll(1,1),revealed_symbols=state.revealed_symbols.roll(1,1))
                moved_obs=replace(obs,v=obs.v.roll(1,1),last_revealed_w=obs.last_revealed_w.roll(1,1),
                    previous_command=CommandBatch((previous.address+1)%4,previous.symbol),observed=obs.observed.roll(1,1))
                moved=model.predict(moved_obs,moved_state,(address+1)%4)
                expected_calls.append(prediction_manifest_reference(moved_obs,moved_state,(address+1)%4,moved,len(expected_calls),'address_remapping'))
                audit.compare(key+f'/address_current/{time}',moved.current,prediction.current.roll(1,1))
                audit.compare(key+f'/address_next/{time}',moved.next,prediction.next.roll(1,2))
                probe_records,baseline_probes=decoder_probe_reference(model,episode,state,time,'baseline')
                expected_decoder_calls.extend(probe_records)
                for schedule in ('current','prospective'):
                    donor_address,donor_completed=prefixes[(time,schedule)]
                    donor_state=model.initial_state(96)
                    for index in range(time-1):
                        command=CommandBatch(episode['command_addresses'][:,index],episode['command_symbols'][:,index])
                        feedback=RevealedFeedback(donor_completed[:,index+1],command,index+1,index,index+1)
                        donor_updated=model.update(donor_state,episode['V'],feedback)
                        expected_updates.append(update_manifest_reference(donor_state,episode['V'],feedback,donor_updated,
                            len(expected_updates),'donor_replay',schedule,time))
                        donor_state=donor_updated
                    ledger=retained['donors'][schedule];patchkey=key+f'/donor/{schedule}/{time}'
                    saved=snapshot_state(state);patched=patch_addressed_state(state,donor_state,donor_address)
                    audit.compare(patchkey+'/reached_selected_state',donor_state.hidden[rows,donor_address] if source!='analytic' else donor_state.survivors[rows,donor_address],
                        ledger['donor_hidden_at_patched_address' if source!='analytic' else 'donor_survivors_at_patched_address'][:,j],exact=True)
                    nonselected=(patched.hidden==state.hidden).flatten(2).all(-1) if source!='analytic' else (
                        (patched.survivors==state.survivors).all(-1)&(patched.revealed_symbols==state.revealed_symbols).all(-1))
                    nonselected[rows,donor_address]=True
                    audit.compare(patchkey+'/nonselected_state',nonselected,torch.ones_like(nonselected),exact=True)
                    patched_prediction=model.predict(obs,patched,address);patched_choice=readout.choose(patched_prediction,address,role)
                    reconstructed_current,reconstructed_future=assemble_reference(episode,completed,time,
                        ledger['selected_current_maps'][:,j],ledger['selected_next_maps'][:,j])
                    audit.compare(patchkey+'/full_current_copy',patched_prediction.current,reconstructed_current,exact=True)
                    audit.compare(patchkey+'/full_next_inheritance',patched_prediction.next,reconstructed_future,exact=True)
                    expected_calls.append(prediction_manifest_reference(obs,patched,address,patched_prediction,len(expected_calls),'donor_'+schedule))
                    probe_records,patched_probes=decoder_probe_reference(model,episode,patched,time,schedule)
                    expected_decoder_calls.extend(probe_records)
                    isolated=(patched_probes==baseline_probes).flatten(3).all(-1).all(-1);isolated[rows,donor_address]=True
                    audit.compare(patchkey+'/all_address_decoder_isolation',isolated,torch.ones_like(isolated),exact=True)
                    for label,actual,expected in [('current',patched_prediction.current[rows,previous.address],ledger['selected_current_maps'][:,j]),
                        ('next',patched_prediction.next[rows[:,None],torch.arange(3)[None,:],address[:,None]],ledger['selected_next_maps'][:,j]),
                        ('fields',patched_choice.candidate_fields,ledger['policy_candidate_fields'][:,j]),('symbol',patched_choice.symbol,ledger['policy_symbol'][:,j])]:
                        audit.compare(patchkey+'/'+label,actual,expected,exact=True)
                    restored=restore_state(saved);restored_prediction=model.predict(obs,restored,address)
                    expected_calls.append(prediction_manifest_reference(obs,restored,address,restored_prediction,len(expected_calls),'restored_'+schedule))
                    audit.compare(patchkey+'/restored_state',restored.hidden if source!='analytic' else restored.survivors,
                        state.hidden if source!='analytic' else state.survivors,exact=True)
                    if source=='analytic':
                        audit.compare(patchkey+'/restored_seen',restored.revealed_symbols,state.revealed_symbols,exact=True)
                    audit.compare(patchkey+'/restored_current',restored_prediction.current,prediction.current,exact=True)
                    audit.compare(patchkey+'/restored_next',restored_prediction.next,prediction.next,exact=True)
                    restored_choice=readout.choose(restored_prediction,address,role)
                    audit.compare(patchkey+'/restored_policy',restored_choice.symbol,choice.symbol,exact=True)
                    audit.compare(patchkey+'/restored_fields',restored_choice.candidate_fields,choice.candidate_fields,exact=True)
            if time:
                fb=RevealedFeedback(completed[:,time],previous,time,time-1,time)
                next_state=model.update(state,episode['V'],fb)
                expected_updates.append(update_manifest_reference(state,episode['V'],fb,next_state,len(expected_updates),'primary_feedback'))
                irrelevant=torch.ones(96,4,dtype=torch.bool);irrelevant[rows,previous.address]=False
                decoy_completion=fb.completed_w.clone();decoy_completion[irrelevant]=.25-decoy_completion[irrelevant]
                decoy_v=episode['V'].clone();decoy_v[irrelevant]=.25-decoy_v[irrelevant]
                decoy_update=model.update(state,decoy_v,replace(fb,completed_w=decoy_completion))
                audit.compare(key+f'/feedback_selected_address_isolation/{time}',
                    decoy_update.hidden if source!='analytic' else decoy_update.survivors,
                    next_state.hidden if source!='analytic' else next_state.survivors,exact=True)
                if source=='analytic':
                    audit.compare(key+f'/feedback_symbol_isolation/{time}',decoy_update.revealed_symbols,next_state.revealed_symbols,exact=True)
                unaffected=(next_state.hidden==state.hidden).all(-1) if source!='analytic' else (
                    (next_state.survivors==state.survivors).all(-1)&(next_state.revealed_symbols==state.revealed_symbols).all(-1))
                unaffected[rows,previous.address]=True
                audit.compare(key+f'/feedback_other_state_unchanged/{time}',unaffected,torch.ones_like(unaffected),exact=True)
                state=next_state
        audit.compare(key+'/terminal_state',state.hidden if source!='analytic' else state.survivors,retained['terminal_state'],exact=True)
        audit.tree(key+'/prediction_manifests',evaluation['metadata']['source_call_visibility'][source],expected_calls)
        audit.tree(key+'/update_manifests',evaluation['metadata']['source_update_visibility'][source],expected_updates)
        audit.tree(key+'/decoder_probe_manifests',evaluation['metadata']['source_decoder_visibility'][source],expected_decoder_calls)
        audit.check(key+'/exact_call_counts',len(expected_calls)==(45 if source=='A' else 30) and len(expected_updates)==86 and len(expected_decoder_calls)==27)
        audit.tree(key+'/call_inventory',evaluation['metadata']['source_call_inventory'][source],dict(
            prediction_calls=45 if source=='A' else 30,primary_update_calls=14,donor_update_calls=72,address_local_decoder_probe_calls=27))
        audit.check(key+'/terminal_timing',retained['terminal_revealed_through']==14 and tuple(retained['terminal_update_command_indices'])==tuple(range(14)))
        if source=='analytic':
            audit.compare(key+'/terminal_seen',retained['terminal_revealed_symbols'],state.revealed_symbols,exact=True)
        if source!='analytic':
            audit.check(key+'/fitted_weights_unchanged',{k:tensor_hash(v) for k,v in model.state_dict().items()}==before)


def verify_evaluation(audit,seed,phase,episode,evaluation,fit,native,completed,forks,family):
    name=f'{seed}/{phase}'
    metadata=evaluation['metadata']; shared=evaluation['tensors']['shared']; rows=torch.arange(96)
    audit.check(name+'/payload_inventory',set(evaluation)=={'tensors','metadata','software_checks'}
        and set(evaluation['tensors'])=={'shared','sources','analytic_secondary'}
        and set(evaluation['tensors']['sources'])==set(SOURCES)
        and set(evaluation['software_checks'])=={'shared','sources'} and set(evaluation['software_checks']['sources'])==set(SOURCES))
    for manifest in ('source_call_visibility','source_update_visibility','source_decoder_visibility','source_call_inventory','donor_summaries'):
        audit.check(name+'/'+manifest+'/source_inventory',set(metadata[manifest])==set(SOURCES))
    shared_shapes=dict(V=((96,4,24,4,4),torch.float32),pixels=((96,4,3,16,16),torch.float32),evaluator_cells=((96,4,2,2),torch.long),
        evaluator_mappings=((96,4,3),torch.long),evaluator_mapping_indices=((96,4),torch.long),command_addresses=((96,15),torch.long),
        command_symbols=((96,15),torch.long),query_roles=((96,15),torch.long),world_indices=((96,),torch.long),
        actual_operator_family=((96,4,3,24,4,4),torch.float32),actual_completed_logits=((96,16,4,3,4,4),torch.float32),
        actual_completed_fields=((96,16,4,2,4,4),torch.float32),actual_family_fields=((96,4,3,2,4,4),torch.float32),
        actual_query_action_cells=((96,3,3),torch.long),bootstrap_indices=((256,96),torch.long))
    for statistic in ('mean','std','norm'):
        shared_shapes['actual_completed_channel_'+statistic]=((96,16,4,24),torch.float32)
    expected=set(shared_shapes)|{'W_derived_controls','actual_controls'}
    audit.check(name+'/shared_tensor_inventory',set(shared)==expected,actual=sorted(shared),expected=sorted(expected))
    for key,(shape,dtype) in shared_shapes.items():
        audit.check(name+'/shared_shape_dtype/'+key,tuple(shared[key].shape)==shape and shared[key].dtype==dtype)
    audit.check(name+'/original_control_inventory',set(shared['W_derived_controls'])=={'category_logits','category_probabilities','orientation_logits'})
    for key,value in shared['W_derived_controls'].items():
        shape=(96,16,4) if key=='orientation_logits' else (96,16,4,8)
        audit.check(name+'/original_control_shape/'+key,tuple(value.shape)==shape and value.dtype==torch.float32)
    expected_controls={label+'_'+suffix for label in ('sham','endogenous','exogenous','restored','address_remapped')
        for suffix in ('local_logits','role_fields','action_cell')}
    audit.check(name+'/actual_control_inventory',set(shared['actual_controls'])==expected_controls)
    for key,value in shared['actual_controls'].items():
        shape=(96,3) if key.endswith('action_cell') else (96,3,4,3 if key.endswith('local_logits') else 2,4,4)
        dtype=torch.long if key.endswith('action_cell') else torch.float32
        audit.check(name+'/actual_control_shape/'+key,tuple(value.shape)==shape and value.dtype==dtype)
    secondary=evaluation['tensors']['analytic_secondary']
    audit.check(name+'/analytic_secondary_inventory',set(secondary)=={'expectation_of_H_fields','expectation_of_H_policy_symbol'})
    audit.check(name+'/analytic_secondary_fields_shape',tuple(secondary['expectation_of_H_fields'].shape)==(96,15,3,2,4,4)
        and secondary['expectation_of_H_fields'].dtype==torch.float32)
    audit.check(name+'/analytic_secondary_policy_shape',tuple(secondary['expectation_of_H_policy_symbol'].shape)==(96,3)
        and secondary['expectation_of_H_policy_symbol'].dtype==torch.long)
    audit.check(name+'/source_indices',metadata['source_indices']==list(SOURCES))
    audit.check(name+'/identity',metadata['split']==phase and metadata['encoder_seed']==seed and metadata['worlds']==96)
    for field in ('V','pixels','evaluator_cells','evaluator_mappings','evaluator_mapping_indices',
                  'command_addresses','command_symbols','query_roles','world_indices'):
        audit.compare(name+'/shared_inputs/'+field,shared[field],episode[field],exact=True)
    audit.check(name+'/namespace',metadata['world_namespace']==episode['world_namespace'])
    audit.tree(name+'/render_nuisance',metadata['render_nuisance'],episode['render_nuisance'])
    audit.check(name+'/fit_diagnostic_scope',metadata['fit_diagnostics_are_independent_evidence']==(False if phase=='fit' else None))
    audit.tree(name+'/reconstruction_recipe',metadata['reconstruction'],dict(
        actual_W='start V; replace previous issued address using actual_operator_family and evaluator mapping',
        current_source='last revealed actual W; replace previous address with selected_current_maps[t-1]',
        next_source='source current; replace scheduled address with selected_next_maps[t,s]',
        terminal_feedback='W15 archived by evaluator; no scored decision15 or source update; source terminal state through revealW14'))
    audit.tree(name+'/denominators',metadata['denominators'],dict(worlds_shared=96,common_transitions_shared=1440,
        actual_forks_shared=4320,current_selected_per_source=1344,next_selected_per_source=4320,
        full_next_item_records_per_source=17280,query_actual_actions_shared=864,
        executed_policy_duplicates_per_source=288,donor_pairs_per_source_per_schedule=288,
        donor_transitions_shared=7488,donor_feedback_events_per_source=6912,donor_policy_fork_references_per_source=576))
    audit.compare(name+'/family',shared['actual_operator_family'],family,exact=True)
    actual_logits=head_reference(completed,native)
    audit.compare(name+'/actual_logits',shared['actual_completed_logits'],actual_logits)
    audit.compare(name+'/actual_fields',shared['actual_completed_fields'],fields_reference(actual_logits))
    audit.compare(name+'/family_fields',shared['actual_family_fields'],fields_reference(head_reference(family,native)))
    for time in (12,13,14):
        address=episode['command_addresses'][:,time];role=episode['query_roles'][:,time]
        bands=episode['evaluator_mappings'][rows,address];indices=torch.where(bands==1,0,torch.where(bands==2,1,2))
        actual_fields=shared['actual_family_fields'][rows[:,None],address[:,None],indices]
        cells=actual_fields[rows[:,None],torch.arange(3)[None,:],role[:,None]].flatten(-2).argmax(-1)
        audit.compare(name+f'/actual_query_actions/{time}',shared['actual_query_action_cells'][:,time-12],cells,exact=True)
    verify_actual_controls(audit,name,episode,shared,metadata,native,completed,family)
    prefixes=donor_reference(audit,name,episode,metadata)
    for key,value in [('actual_completed_channel_mean',completed.double().mean((-2,-1))),
        ('actual_completed_channel_std',completed.double().std((-2,-1),correction=0)),
        ('actual_completed_channel_norm',completed.double().square().sum((-2,-1)).sqrt())]:
        audit.compare(name+'/'+key,shared[key],value)
    metrics={}
    for source in SOURCES:
        key=name+'/'+source; retained=evaluation['tensors']['sources'][source]
        verify_source_shapes(audit,key,source,retained)
        verify_visibility(audit,key,episode,completed,metadata['source_call_visibility'][source])
        hidden=torch.zeros(96,4,48); survivors=torch.ones(96,4,6,dtype=torch.bool); seen=torch.zeros(96,4,3,dtype=torch.bool)
        for time in range(15):
            address=episode['command_addresses'][:,time]
            if source!='analytic':
                emitted_hidden=retained['pre_reveal_hidden'][:,time]
                audit.compare(key+f'/state/{time}',emitted_hidden,hidden)
                def decode(addr,symbol):
                    return decoder_reference(episode['V'][rows,addr],emitted_hidden[rows,addr],symbol,fit['checkpoints'][source],fit['normalizer'])
            else:
                audit.compare(key+f'/posterior/{time}',retained['pre_reveal_survivors'][:,time],survivors,exact=True)
                audit.compare(key+f'/revealed_symbols/{time}',retained['pre_reveal_revealed_symbols'][:,time],seen,exact=True)
                audit.compare(key+f'/survivor_count/{time}',retained['survivor_count'][:,time],survivors.sum(-1),exact=True)
                audit.compare(key+f'/posterior_entropy/{time}',retained['posterior_entropy'][:,time],survivors.sum(-1).double().log())
                audit.compare(key+f'/revealed_symbol_count/{time}',retained['revealed_symbol_count'][:,time],seen.sum(-1),exact=True)
                weights=survivors[rows,address].double();weights/=weights.sum(-1,keepdim=True)
                derived=[]
                for symbol in range(3):
                    indices=torch.tensor([BANDS.index(mapping[symbol]) for mapping in MAPPINGS])
                    hypotheses=shared['actual_family_fields'][rows[:,None],address[:,None],indices[None,:]]
                    derived.append((hypotheses.double()*weights[:,:,None,None,None]).sum(1).float())
                expectation=torch.stack(derived,1)
                secondary=evaluation['tensors']['analytic_secondary']
                audit.compare(key+f'/expectation_of_H/{time}',secondary['expectation_of_H_fields'][:,time],expectation)
                if time>=12:
                    role=episode['query_roles'][:,time]
                    emitted=secondary['expectation_of_H_fields'][:,time]
                    maxima=emitted[rows[:,None],torch.arange(3)[None,:],role[:,None]].flatten(-2).max(-1).values
                    audit.compare(key+f'/expectation_of_H_policy/{time}',secondary['expectation_of_H_policy_symbol'][:,time-12],maxima.argmax(-1),exact=True)
                def decode(addr,symbol):
                    return posterior_decode(episode['V'][rows,addr],survivors[rows,addr],symbol)
            current=completed[:,max(0,time-1)].clone()
            current_fields=shared['actual_completed_fields'][:,max(0,time-1)].clone()
            if time:
                previous=episode['command_addresses'][:,time-1]; symbol=episode['command_symbols'][:,time-1]
                emitted=retained['selected_current_maps'][:,time-1]
                audit.compare(key+f'/current/{time}/decoder',emitted,decode(previous,symbol))
                current[rows,previous]=emitted
                logits=head_reference(emitted,native)
                audit.compare(key+f'/current/{time}/logits',retained['selected_current_logits'][:,time-1],logits)
                fields=retained['selected_current_fields'][:,time-1]
                audit.compare(key+f'/current/{time}/fields',fields,fields_reference(logits))
                current_fields[rows,previous]=fields
                audit.compare(key+f'/current/{time}/mse',retained['current_selected_normalized_mse'][:,time-1],mse(emitted,completed[rows,time,previous],fit['normalizer']))
                distances=mse(emitted[:,None],family[rows,previous],fit['normalizer'])
                audit.compare(key+f'/current/{time}/operator_distances',retained['current_distance_to_each_actual_operator'][:,time-1],distances)
                for metric,value in field_metrics(fields,shared['actual_completed_fields'][rows,time,previous],episode['evaluator_cells'][rows,previous]).items():
                    audit.compare(key+f'/current/{time}/{metric}',retained['current_'+metric][:,time-1],value,
                        exact=value.dtype in (torch.bool,torch.long))
            audit.compare(key+f'/current/{time}/full_mse',retained['current_full_normalized_mse_by_address'][:,time],mse(current,completed[:,time],fit['normalizer']))
            audit.compare(key+f'/current/{time}/full_field_l1',retained['current_full_field_l1_by_address'][:,time],
                (current_fields-shared['actual_completed_fields'][:,time]).abs().mean((-3,-2,-1)))
            audit.compare(key+f'/current/{time}/full_cells',retained['current_full_cell_disagreement_by_address'][:,time],
                current_fields.flatten(-2).argmax(-1)!=shared['actual_completed_fields'][:,time].flatten(-2).argmax(-1),exact=True)
            audit.compare(key+f'/current/{time}/full_coarse',retained['current_full_coarse_disagreement_by_address'][:,time],
                coarse_reference(current_fields)!=coarse_reference(shared['actual_completed_fields'][:,time]),exact=True)
            selected=retained['selected_next_maps'][:,time]
            predicted=torch.stack([decode(address,torch.full((96,),symbol,dtype=torch.long)) for symbol in range(3)],1)
            audit.compare(key+f'/next/{time}/decoder',selected,predicted)
            logits=head_reference(selected,native); fields=retained['selected_next_fields'][:,time]
            audit.compare(key+f'/next/{time}/logits',retained['selected_next_logits'][:,time],logits)
            audit.compare(key+f'/next/{time}/fields',fields,fields_reference(logits))
            for label,value in [('current',current),('next',selected)]:
                for metric,statistic in [('mean',value.double().mean((-2,-1))),('std',value.double().std((-2,-1),correction=0)),
                    ('norm',value.double().square().sum((-2,-1)).sqrt())]:
                    audit.compare(key+f'/{label}/{time}/channel_'+metric,retained[label+'_channel_'+metric][:,time],statistic)
            next_maps=current[:,None].expand(-1,3,-1,-1,-1,-1).clone()
            next_maps[rows[:,None],torch.arange(3)[None,:],address[:,None]]=selected
            audit.compare(key+f'/next/{time}/selected_mse',retained['next_selected_normalized_mse'][:,time],mse(selected,forks[rows[:,None],time,torch.arange(3)[None,:],address[:,None]],fit['normalizer']))
            audit.compare(key+f'/next/{time}/full_mse',retained['next_full_normalized_mse_by_address'][:,time],mse(next_maps,forks[:,time],fit['normalizer']))
            audit.compare(key+f'/next/{time}/operator_distances',retained['next_distance_to_each_actual_operator'][:,time],mse(selected[:,:,None],family[rows,address][:,None],fit['normalizer']))
            bands=episode['evaluator_mappings'][rows,address]; indices=torch.where(bands==1,0,torch.where(bands==2,1,2))
            actual_fields=shared['actual_family_fields'][rows[:,None],address[:,None],indices]
            cells=episode['evaluator_cells'][rows,address][:,None].expand(-1,3,-1,-1)
            for metric,value in field_metrics(fields,actual_fields,cells).items():
                audit.compare(key+f'/next/{time}/{metric}',retained['next_'+metric][:,time],value,exact=value.dtype in (torch.bool,torch.long))
            assembled_fields=current_fields[:,None].expand(-1,3,-1,-1,-1,-1).clone()
            assembled_fields[rows[:,None],torch.arange(3)[None,:],address[:,None]]=fields
            actual_fork_fields=shared['actual_completed_fields'][:,time,None].expand(-1,3,-1,-1,-1,-1).clone()
            actual_fork_fields[rows[:,None],torch.arange(3)[None,:],address[:,None]]=actual_fields
            audit.compare(key+f'/next/{time}/full_field_l1',retained['next_full_field_l1_by_address'][:,time],(assembled_fields-actual_fork_fields).abs().mean((-3,-2,-1)))
            audit.compare(key+f'/next/{time}/full_cells',retained['next_full_cell_disagreement_by_address'][:,time],
                assembled_fields.flatten(-2).argmax(-1)!=actual_fork_fields.flatten(-2).argmax(-1),exact=True)
            audit.compare(key+f'/next/{time}/full_coarse',retained['next_full_coarse_disagreement_by_address'][:,time],
                coarse_reference(assembled_fields)!=coarse_reference(actual_fork_fields),exact=True)
            if time>=12:
                role=episode['query_roles'][:,time]; selected_fields=fields[rows[:,None],torch.arange(3)[None,:],role[:,None]]
                maxima=selected_fields.flatten(-2).max(-1).values; choice=maxima.argmax(-1)
                audit.compare(key+f'/policy/{time}/maxima',retained['policy_role_maxima'][:,time-12],maxima,exact=True)
                audit.compare(key+f'/policy/{time}/symbol',retained['policy_symbol'][:,time-12],choice,exact=True)
                emitted_actual=actual_fields[rows,choice,role].flatten(-2)
                cell=emitted_actual.argmax(-1); ties=(emitted_actual==emitted_actual.max(-1,keepdim=True).values).sum(-1)
                audit.compare(key+f'/policy/{time}/actual_cell',retained['policy_action_cell'][:,time-12],cell,exact=True)
                audit.compare(key+f'/policy/{time}/actual_ties',retained['policy_action_tie_count'][:,time-12],ties,exact=True)
                audit.compare(key+f'/policy/{time}/actual_fields',retained['policy_actual_role_fields'][:,time-12],actual_fields[rows,choice],exact=True)
                truth=episode['evaluator_cells'][rows,address,role]
                audit.compare(key+f'/policy/{time}/actual_correct',retained['policy_action_correct'][:,time-12],cell==truth[:,0]*4+truth[:,1],exact=True)
            if time:
                previous=episode['command_addresses'][:,time-1]; symbol=episode['command_symbols'][:,time-1]
                if source!='analytic':
                    hidden=hidden.clone(); hidden[rows,previous]=gru_reference(hidden[rows,previous],episode['V'][rows,previous],completed[rows,time,previous],symbol,fit['checkpoints'][source],fit['normalizer'])
                else:
                    survivors,seen=posterior_update(survivors,seen,episode['V'],completed[:,time],previous,symbol)
        audit.compare(key+'/terminal_after_W14',retained['terminal_state'],hidden if source!='analytic' else survivors,exact=source=='analytic')
        verify_donors(audit,key,source,episode,retained,shared,fit,native,completed,forks,prefixes)
        metrics[source]=world_metrics_reference(retained,episode)
        audit.tree(key+'/donor_summaries',metadata['donor_summaries'][source],donor_summaries_reference(retained))
    expected_indices=torch.randint(96,(256,96),generator=torch.Generator().manual_seed(350000000+seed))
    audit.compare(name+'/bootstrap_indices',shared['bootstrap_indices'],expected_indices,exact=True)
    audit.tree(name+'/bootstrap',metadata['bootstrap'],bootstrap_reference(metrics,expected_indices))
    audit.tree(name+'/bootstrap_primary_metric_names',metadata['bootstrap_primary_metric_names'],list(metrics['A']))
    audit.check(name+'/semantic_negative',metadata['presentation_manner_qualified'] is False and metadata['reports_generated'] is False)
    audit.check(name+'/literal_disposition_pending',metadata['literal_content_disposition']=='pending independent fixed-dev evidence assessment')
    verify_production_replay(audit,name,episode,evaluation,fit,native,completed,prefixes)
    actual_check_names={'sham_selected_exact_v','sham_nonselected_unchanged','same_w_driver_exact',
        'same_w_driver_fields_exact','same_w_driver_action_exact','actual_restoration_exact',
        'actual_restoration_fields_exact','actual_restoration_action_exact','actual_address_fields_exact',
        'actual_address_action_exact','actual_family_reconstruction_exact','actual_compaction_reconstruction_exact'}
    source_check_names={'selected_policy_fork_exact','selected_policy_native_action_exact','confidence_choice_exact',
        'address_current_equivariance','address_next_equivariance'}
    source_check_names.update(('observation_history_exact','last_revealed_snapshot_exact','prediction_inputs_unchanged_exact',
        'prediction_state_unchanged_exact','initial_current_exact_v','current_nonselected_copy_exact','next_noncandidate_inheritance_exact'))
    for schedule in ('current','prospective'):
        source_check_names.update('restore_'+schedule+'_'+field+'_exact' for field in ('current','next','policy','policy_fields','state'))
        source_check_names.update('patch_'+schedule+'_nonselected_'+field+'_exact' for field in ('state','current','next'))
        source_check_names.update(('patch_'+schedule+'_selected_state_transfer_exact',
            'patch_'+schedule+'_nonselected_decoder_exact','patch_'+schedule+'_next_inheritance_exact'))
    for group,checks in [('shared',evaluation['software_checks']['shared'])]+list(evaluation['software_checks']['sources'].items()):
        expected=set(actual_check_names if group=='shared' else source_check_names)
        if group in ('A','ordinary'):
            expected.add('fitted_weights_unchanged')
        if group=='A':
            expected.update(('alias_current_exact','alias_next_exact'))
        audit.check(name+'/software_checks/'+group+'/inventory',set(checks)==expected,actual=sorted(checks),expected=sorted(expected))
        for check,value in checks.items():
            if check=='fitted_weights_unchanged':
                shape=(1,)
            elif check=='initial_current_exact_v':
                shape=(96,1)
            elif check=='current_nonselected_copy_exact':
                shape=(96,14)
            elif check=='actual_family_reconstruction_exact':
                shape=(96,60)
            elif check=='actual_compaction_reconstruction_exact':
                shape=(96,61)
            elif check.startswith('alias_') or check in {'observation_history_exact','last_revealed_snapshot_exact',
                'prediction_inputs_unchanged_exact','prediction_state_unchanged_exact','next_noncandidate_inheritance_exact'}:
                shape=(96,15)
            else:
                shape=(96,3)
            audit.check(name+'/recorded_software_checks/'+group+'/'+check,value.dtype==torch.bool and value.numel()>0 and bool(value.all()),
                evidence_scope='Recorded production boolean; independent reconstruction/replay checks are separately labelled.')
            audit.check(name+'/recorded_software_checks/'+group+'/'+check+'/shape',tuple(value.shape)==shape,
                actual=list(value.shape),expected=list(shape))
    audit.check(name+'/recorded_software_invariants_pass',metadata['software_invariants_pass'] is True)
    return {source:{metric:float(value.double().mean()) for metric,value in values.items()} for source,values in metrics.items()}


def bind_phase(audit,args,registration,phase,commit):
    base=ROOT/registration['output_dir']; path=base/phase/'receipt.json'
    receipt=json.loads(path.read_text())
    audit.check('receipt/'+phase+'/identity',receipt['status']=='completed' and receipt['phase']==phase
                and receipt['registration_sha256']==digest(args.registration))
    expected={str((base/phase/f'states-{seed}.pt.gz').relative_to(ROOT)) for seed in SEEDS}
    expected.add(str((base/phase/'first_run.log').relative_to(ROOT)))
    audit.check('receipt/'+phase+'/inventory',set(receipt['artifact_sha256'])==expected)
    audit.check('receipt/'+phase+'/no_retry_or_reporter',receipt['retry_performed'] is False and receipt['no_reporter_calls'] is True
        and receipt['semantic_qualification_asserted'] is False)
    claim_path=base/phase/'claim.json';claim=json.loads(claim_path.read_text())
    audit.check('receipt/'+phase+'/claim_identity',claim['phase']==phase and claim['registration_sha256']==digest(args.registration)
        and claim['registration_commit']==args.registration_commit and claim['retry_allowed'] is False)
    audit.check('receipt/'+phase+'/registered_commit',receipt['registration_commit']==args.registration_commit)
    audit.check('receipt/'+phase+'/first_claim_time',claim['started_at_utc']<receipt['completed_at_utc'])
    committed_bindings(audit,args.preservation_checkout,commit,
        {path.relative_to(ROOT).as_posix():digest(path),claim_path.relative_to(ROOT).as_posix():digest(claim_path),
         **receipt['artifact_sha256']},'preserved_phase/'+phase)
    return receipt


def ancestor(audit,checkout,older,newer,name):
    valid=all(isinstance(value,str) and len(value)==40 and all(c in '0123456789abcdef' for c in value) for value in (older,newer))
    if not valid:
        audit.check(name,False,older=older,newer=newer);return
    result=subprocess.run(['git','-C',str(checkout),'merge-base','--is-ancestor',older,newer],capture_output=True)
    audit.check(name,result.returncode==0,older=older,newer=newer,git_exit=result.returncode)


def native_acceptance(audit,args,registration):
    """Bind the actual first passing native review without inference/loading."""
    review_dir=ROOT/'audits/spatial_native_task_v1/numerical_corrected_review'
    manifest_path=review_dir/'metadata_manifest.json';manifest=json.loads(manifest_path.read_text())
    packed=ROOT/manifest['published_path']
    committed_bindings(audit,args.preservation_checkout,NATIVE_PASS_COMMIT,
        {str(manifest_path.relative_to(ROOT)):digest(manifest_path),str(packed.relative_to(ROOT)):digest(packed)},'first_native_full_pass')
    audit.check('native_acceptance/compressed_identity',digest(packed)==manifest['compressed_sha256'])
    with gzip.open(packed,'rb') as stream:
        raw=stream.read()
    audit.check('native_acceptance/raw_identity',hashlib.sha256(raw).hexdigest()==manifest['raw_sha256'])
    review=json.loads(raw)
    audit.check('native_acceptance/full_unchanged_pass',review['status']=='passed_corrected_reference_revalidation'
        and review['check_count']==206070 and review['error_count']==0 and review['core_check_inventory_matches_first_review'] is True
        and len(review['checks'])==206070 and all(row['passed'] for row in review['checks'])
        and review['reconstructed_engineering_feasibility_pass'] is True)
    audit.check('native_acceptance/original_result_identity',review['corrected_reference_provenance']['original_results_commit']==
        '614e74ae74ccf1d553f6fc6e0ec318f057a94ed8')
    bound={row['name'].removeprefix('actual_results_bytes/'):row for row in review['checks']
        if row['name'].startswith('actual_results_bytes/')}
    required=tuple(f'audits/spatial_native_task_v1/run/states-{seed}.pt.gz' for seed in SEEDS)+(
        'audits/spatial_native_task_v1/run/results.json','audits/spatial_native_task_v1/run/completion.json')
    inputs={str(manifest_path.relative_to(ROOT)):digest(manifest_path),str(packed.relative_to(ROOT)):digest(packed)}
    for path in required:
        value=digest(ROOT/path);row=bound.get(path,{})
        audit.check('native_acceptance/accepted_actual_bytes/'+path,row.get('expected_sha256')==row.get('actual_sha256')==value)
        inputs[path]=value
    audit.check('native_acceptance/complete_input_closure',registration['input_hashes']==inputs)
    for row in review['corrected_reference_provenance']['bindings']:
        committed_bindings(audit,args.preservation_checkout,row['commit'],{row['path']:row['expected_sha256']},'native_corrected_provenance')
    audit.check('native_acceptance/corrected_program',digest(ROOT/'scripts/verify_spatial_native_task_v1_corrected.py')==review['verifier_sha256'])
    results=json.loads((ROOT/'audits/spatial_native_task_v1/run/results.json').read_text())
    completion=json.loads((ROOT/'audits/spatial_native_task_v1/run/completion.json').read_text())
    audit.check('native_acceptance/completion',completion['status']=='completed' and completion['results_sha256']==
        inputs['audits/spatial_native_task_v1/run/results.json'] and results['all_engineering_feasibility_gates_pass'] is True)
    audit.check('native_acceptance/archive_inventory',all(completion['archive_sha256'].get(path)==inputs[path] for path in required[:3]))


def run(args,audit):
    audit.check('source_integration_ready',INTEGRATION_READY,
        reason='Schema integration flag only. Exact committed preexecution approval is separately required before tensor loading.')
    if args.phase!='fit' and not args.fit_commit:
        raise ValueError('dev/heldout reconstruction requires the exact preserved first fit commit')
    if args.phase=='heldout' and (not args.dev_commit or args.assessment is None):
        raise ValueError('heldout reconstruction requires preserved dev commit and explicit dev acceptance')
    registration=json.loads(args.registration.read_text())
    audit.check('registration/schema',registration['schema']=='attention_content_model_v1_phase_registration')
    review_path=registration['review_path']
    audit.check('registration/complete_verifier_closure',set(registration['source_hashes'])==set(REQUIRED_CLOSURE)|{review_path})
    audit.check('registration/exact_verifier_bytes',registration['source_hashes'].get('scripts/verify_attention_content_model_v1.py')==digest(__file__))
    audit.tree('registration/fixed_contract',{key:registration[key] for key in ('encoder_seeds','worlds_per_encoder_phase',
        'parameter_count_per_learned_source','updates_per_source','phases','source_inputs','forbidden_source_inputs',
        'phase_receipts_required_before_release','reporter_calls','no_fit_retry_or_adaptation','development_only',
        'source_semantics_qualified','presentation_qualified')},dict(encoder_seeds=list(SEEDS),worlds_per_encoder_phase=96,
        parameter_count_per_learned_source=197328,updates_per_source=300,phases=['fit','dev','heldout'],
        source_inputs=['V','issued commands','completed feedback through t-1'],
        forbidden_source_inputs=['pixels','mapping labels','bandwidth truth','renderer truth','current concealed W_t','future fork outcomes','world identifiers'],
        phase_receipts_required_before_release=True,reporter_calls=0,no_fit_retry_or_adaptation=True,development_only=True,
        source_semantics_qualified=False,presentation_qualified=False))
    committed_bindings(audit,args.preservation_checkout,registration['source_commit'],registration['source_hashes'],'approved_source_commit')
    audit.check('registration/review_path',review_path in registration['source_hashes'] and review_path not in REQUIRED_CLOSURE)
    review=json.loads((ROOT/review_path).read_text())
    expected={path:value for path,value in registration['source_hashes'].items() if path!=review_path}
    audit.check('registration/exact_complete_source_approval',review['scope']=='attention_content_model_v1_complete_preexecution_review'
        and review['approved'] is True and review['approved_source_sha256']==expected)
    committed_bindings(audit,args.preservation_checkout,args.registration_commit,
        {args.registration.resolve().relative_to(ROOT).as_posix():digest(args.registration),
         **registration['source_hashes'],**registration['input_hashes']},'preserved_registration')
    ancestor(audit,args.preservation_checkout,registration['source_commit'],args.registration_commit,'registration/source_ancestry')
    ancestor(audit,args.preservation_checkout,args.registration_commit,args.phase_commit,'phase/registration_ancestry')
    native_acceptance(audit,args,registration)
    current=bind_phase(audit,args,registration,args.phase,args.phase_commit)
    fit_receipt=current if args.phase=='fit' else bind_phase(audit,args,registration,'fit',args.fit_commit)
    base=ROOT/registration['output_dir']
    if args.phase!='fit':
        ancestor(audit,args.preservation_checkout,args.registration_commit,args.fit_commit,'fit/registration_ancestry')
        ancestor(audit,args.preservation_checkout,args.fit_commit,args.phase_commit,'phase/fit_ancestry')
        audit.check('phase/same_fit_receipt',current['fit_receipt_sha256']==digest(base/'fit'/'receipt.json')
            and current['fit_commit']==args.fit_commit)
    if args.phase=='heldout':
        dev=bind_phase(audit,args,registration,'dev',args.dev_commit)
        ancestor(audit,args.preservation_checkout,args.fit_commit,args.dev_commit,'dev/fit_ancestry')
        ancestor(audit,args.preservation_checkout,args.dev_commit,args.phase_commit,'heldout/dev_ancestry')
        assessment=json.loads(args.assessment.read_text())
        committed_bindings(audit,args.preservation_checkout,args.dev_commit,
            {args.assessment.resolve().relative_to(ROOT).as_posix():digest(args.assessment)},'preserved_dev_acceptance')
        audit.check('heldout/dev_acceptance',assessment['schema']=='attention_content_model_v1_dev_acceptance'
            and assessment['hard_invariants_pass'] is True and assessment['dev_receipt_sha256']==digest(base/'dev'/'receipt.json')
            and assessment['fitted_artifact_sha256']==fit_receipt['artifact_sha256'])
        dev_report_path=ROOT/assessment['reconstruction_report_path']
        committed_bindings(audit,args.preservation_checkout,args.dev_commit,
            {str(dev_report_path.relative_to(ROOT)):assessment['reconstruction_report_sha256']},'preserved_independent_dev_reconstruction')
        dev_report=json.loads(dev_report_path.read_text())
        audit.check('heldout/independent_dev_reconstruction',dev_report['schema']=='attention_content_model_v1_independent_phase_reconstruction'
            and dev_report['phase']=='dev' and dev_report['status']=='passed_phase_reconstruction'
            and dev_report['hard_invariants_pass'] is True and dev_report['error_count']==0 and dev_report['errors']==[]
            and 'exception' not in dev_report and dev_report['check_count']==len(dev_report['checks'])>0
            and all(row['passed'] is True for row in dev_report['checks'])
            and dev_report['dev_receipt_sha256']==digest(base/'dev'/'receipt.json')
            and dev_report['fitted_artifact_sha256']==fit_receipt['artifact_sha256']
            and dev_report['verifier_sha256']==registration['source_hashes']['scripts/verify_attention_content_model_v1.py'])
        commands=dev_report['command_bindings']
        audit.check('heldout/independent_dev_command_bindings',commands['registration_commit']==args.registration_commit
            and Path(commands['registration']).resolve()==args.registration.resolve() and commands['fit_commit']==args.fit_commit)
        ancestor(audit,args.preservation_checkout,args.fit_commit,commands['phase_commit'],'dev_report/fit_ancestry')
        ancestor(audit,args.preservation_checkout,commands['phase_commit'],args.dev_commit,'dev_report/acceptance_ancestry')
        dispositions=assessment['source_semantic_dispositions']
        audit.check('heldout/explicit_source_dispositions',set(dispositions)==set(SOURCES)
            and all(set(row)=={'literal_content_disposition','presentation_manner_qualified'}
                and row['literal_content_disposition'] in ('qualified','unqualified')
                and type(row['presentation_manner_qualified']) is bool for row in dispositions.values())
            and assessment['reporting_released'] is False)
        audit.check('heldout/assessment_receipt_binding',current['dev_commit']==args.dev_commit
            and current['dev_assessment_sha256']==digest(args.assessment)
            and dev['fit_receipt_sha256']==digest(base/'fit'/'receipt.json'))
    if audit.errors:
        raise ValueError('binding failure; refusing model/tensor reconstruction')
    global torch
    import torch
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    base=ROOT/registration['output_dir']; output={}
    with torch.no_grad():
        for seed in SEEDS:
            with gzip.open(base/args.phase/f'states-{seed}.pt.gz','rb') as f:
                bundle=torch.load(f,weights_only=True,map_location='cpu')
            with gzip.open(ROOT/f'audits/spatial_native_task_v1/run/states-{seed}.pt.gz','rb') as f:
                native=torch.load(f,weights_only=True,map_location='cpu')
            finite_tree(audit,bundle,str(seed)+'/'+args.phase+'/archive')
            completed,forks,family=verify_episode(audit,seed,args.phase,bundle['episode'],native)
            if args.phase=='fit':
                fit=bundle['fit'];output[str(seed)]=verify_fit(audit,seed,bundle['episode'],fit,completed)
                output[str(seed)]['final_fit_diagnostics']=verify_evaluation(audit,seed,'fit',bundle['episode'],bundle['evaluation'],fit,native,completed,forks,family)
            else:
                with gzip.open(base/'fit'/f'states-{seed}.pt.gz','rb') as f:
                    fit=torch.load(f,weights_only=True,map_location='cpu')['fit']
                finite_tree(audit,fit,str(seed)+'/prior_fit_archive')
                audit.check(str(seed)+'/fitted_artifact_binding',bundle['fitted_artifact_sha256']==fit_receipt['artifact_sha256'])
                output[str(seed)]=verify_evaluation(audit,seed,args.phase,bundle['episode'],bundle['evaluation'],fit,native,completed,forks,family)
            print(json.dumps(dict(seed=seed,phase=args.phase,checks=len(audit.checks),errors=len(audit.errors))),flush=True)
    return dict(models=output,fitted_artifact_sha256=fit_receipt['artifact_sha256'],
        dev_receipt_sha256=digest(base/'dev'/'receipt.json') if args.phase in ('dev','heldout') else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('fit','dev','heldout'),required=True)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--registration-commit',required=True)
    parser.add_argument('--phase-commit',required=True)
    parser.add_argument('--fit-commit')
    parser.add_argument('--dev-commit')
    parser.add_argument('--assessment',type=Path)
    parser.add_argument('--preservation-checkout',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--ack-first-phase-preserved',action='store_true',required=True)
    args=parser.parse_args()
    with args.report.open('x',encoding='utf-8') as stream:
        audit=Audit()
        report=dict(schema='attention_content_model_v1_independent_phase_reconstruction',version=1,phase=args.phase,
            created_at_utc=datetime.now(timezone.utc).isoformat(),verifier_sha256=digest(__file__),
            reviewer='/root/spatial_native_numerical_execution_review',separate_task_allocation=True,
            shared_model_and_inherited_hypothesis_context=True,independent_model_family=False,human_endorsement=False,
            no_world_generation=True,no_fit_replay=True,no_optimizer=True,no_reporter=True,development_only=True,
            presentation_qualified=False,qualia_established=False,goal_achieved=False,
            reference_tolerances=dict(atol=ATOL,rtol=RTOL,discrete_and_copy_checks='exact'),
            command_bindings=dict(registration=str(args.registration),registration_commit=args.registration_commit,
                phase_commit=args.phase_commit,fit_commit=args.fit_commit,dev_commit=args.dev_commit,
                assessment=str(args.assessment) if args.assessment else None,preservation_checkout=str(args.preservation_checkout)),
            limitations=['Recorded optimizer steps/losses/final weights do not prove a trajectory without fit replay.',
                'Permission manifests and committed exclusive claims cannot rule out unrecorded external work.',
                'Numerical comparisons use independent equations; exact policies use the same archived emitted fields.',
                'Intervals are conditional fixed-world resamples, not independent-family, human or unseen-layout evidence.',
                'Production-core replay/input isolation is separately labelled, not an independent implementation.',
                'Archived pixel/source reconstruction binds delivered inputs; no renderer replay is claimed.',
                'Controller is not Reporter; map/content/control evidence cannot establish presentation or qualia.'])
        try:
            report.update(run(args,audit))
        except BaseException:
            report['exception']=traceback.format_exc();audit.errors.append('verification_exception')
        report.update(status='passed_phase_reconstruction' if not audit.errors else 'failed_first_phase_reconstruction',
            hard_invariants_pass=not audit.errors,check_count=len(audit.checks),error_count=len(audit.errors),
            checks=audit.checks,errors=audit.errors,retry_or_repair_performed=False)
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush()
        print(json.dumps({key:report[key] for key in ('status','phase','hard_invariants_pass','check_count','error_count')}),flush=True)
    if audit.errors:
        sys.exit(1)


if __name__=='__main__':
    main()
