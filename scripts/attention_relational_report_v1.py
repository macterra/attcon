"""Prospective neutral relational export and deterministic machine-fact audit.

Importing defines stdlib functions only. Archive loading is confined to the
explicit, reviewed build_reporting_packets call after exact byte/receipt gates.
No model/head forward, rendering, fitting, Reporter or API is implemented here.
Machine propositions do not certify unlisted free prose or phenomenal truth.
"""
from copy import deepcopy
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1301,1311,1321)
WORLD_INDICES = (0,1,2,3)
DECISION_TIME = 12
SOURCE_KEYS = ('A','ordinary','analytic')
VARIANTS = ('baseline','current_donor','current_restored','prospective_donor',
            'prospective_restored','W_changed','W_restored')
W_VARIANTS = ('baseline','W_changed','W_restored')
NODES = ('n0','n1','n2','n3')
COMMANDS = ('k0','k1','k2')
ROLES = ('red','blue')
REPORT_SCHEMA = 'attention_relational_report_v1_report'
PACK_SCHEMA = 'attention_relational_report_v1_packet'
INPUT_SCHEMA = 'attention_relational_report_v1_inputs'
APPROVAL_SCOPE = 'attention_relational_report_v1_complete_source_review'
SCALAR_ABSOLUTE_TOLERANCE = 1e-6+1e-12
WORD_LIMIT = 220
REQUEST_COUNT = 552
CURRENT_MAP_BYTES = 4*24*4*4*4
CANDIDATE_MAP_BYTES = 3*CURRENT_MAP_BYTES
NARRATOR_PATH = 'audits/attention_relational_report_v1/protocol/narrator_instructions.txt'
PROTOCOL_MANIFEST_PATH = 'audits/attention_relational_report_v1/protocol/manifest.json'
STUDY_MINIMUM = {
    'scripts/attention_relational_report_v1.py','tests/test_attention_relational_report_v1.py',
    'scripts/run_attention_relational_reports_v1.py','scripts/attention_relational_reporting_pipeline_v1.py',
    'docs/ATTENTION_RELATIONAL_REPORT_PROTOCOL_V1.md',NARRATOR_PATH,PROTOCOL_MANIFEST_PATH,
    'audits/attention_relational_report_v1/protocol/prose_auditor_instructions.txt',
    'audits/attention_relational_report_v1/reference/reference.json',
    'audits/attention_relational_report_v1/reference/rater_instructions.txt',
    'docs/ATTENTION_RELATIONAL_REFERENCE_V1.md',
}
IMMUTABLE_INPUT_SHA256 = '4f6712698a1acca31d69fb2cd7f77eb7c00b0a9b0fb51ea4c192380e4274658e'


def _proposition_schema(kind,extra):
    properties=dict(kind={'const':kind},panel={'enum':['p0','p1']},
        view={'enum':['current','candidate']},command={'enum':[None,*COMMANDS]},
        node={'enum':list(NODES)},**extra)
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)


REPORT_RESPONSE_SCHEMA = dict(type='object',additionalProperties=False,
    required=['schema','text','propositions'],properties=dict(schema={'const':REPORT_SCHEMA},
        text={'type':'string'},propositions={'type':'array','items':{'anyOf':[
            _proposition_schema('peaks',dict(role={'enum':list(ROLES)},cells={
                'type':'array','minItems':1,'maxItems':16,'uniqueItems':True,'items':{
                    'type':'array','minItems':2,'maxItems':2,'items':{
                        'type':'integer','minimum':0,'maximum':3}}})),
            _proposition_schema('relation',dict(relation={'const':'red_blue_peak_relation'},
                horizontal={'type':'array','uniqueItems':True,'items':{'enum':['left_of','same_column','right_of']}},
                vertical={'type':'array','uniqueItems':True,'items':{'enum':['above','same_row','below']}},
                coincident_possible={'type':'boolean'},coincident_all={'type':'boolean'})),
            _proposition_schema('scalar',dict(role={'enum':list(ROLES)},name={'enum':[
                'peak_probability','entropy_nats','expected_row','expected_column']},value={'type':'number'})),
        ]}}))


def require(condition,message):
    if not condition:
        raise ValueError(message)


def json_bytes(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(path):
    value=Path(path)
    resolved=(ROOT/value).resolve() if not value.is_absolute() else value.resolve()
    require(resolved.is_relative_to(ROOT),'report input outside repository')
    return resolved.relative_to(ROOT).as_posix()


def committed_bytes(commit,hashes):
    require(type(commit) is str and len(commit)==40
            and all(c in '0123456789abcdef' for c in commit),'full preservation commit required')
    for name,expected in hashes.items():
        require(type(name) is str and canonical(name)==name,'noncanonical report input path')
        require(type(expected) is str and len(expected)==64
                and all(c in '0123456789abcdef' for c in expected),'invalid report input hash')
        require(digest(ROOT/name)==expected,'current report input bytes changed: '+name)
        process=subprocess.Popen(['git','-C',str(ROOT),'show',commit+':'+name],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        h=hashlib.sha256()
        for chunk in iter(lambda:process.stdout.read(1024*1024),b''):
            h.update(chunk)
        error=process.stderr.read().decode(errors='replace');status=process.wait()
        require(status==0 and h.hexdigest()==expected,'committed report input bytes changed: '+name+' '+error)


def ancestor(older,newer):
    for value in (older,newer):
        require(type(value) is str and len(value)==40
                and all(c in '0123456789abcdef' for c in value),'full report ancestry commit required')
    result=subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',older,newer],capture_output=True)
    require(result.returncode==0,'report input ancestry failed')


def _numbers(value,shape):
    if not shape:
        require(type(value) in (int,float) and math.isfinite(value),'nonfinite/non-numeric field cell')
        return [float(value)]
    require(type(value) is list and len(value)==shape[0],'field shape mismatch')
    result=[]
    for child in value:
        result.extend(_numbers(child,shape[1:]))
    return result


def _field_bytes(value,shape):
    values=_numbers(value,shape)
    require(all(0.<=v<=1. for v in values),'localization cell outside unit range')
    raw=struct.pack('<'+str(len(values))+'f',*values)
    require(list(struct.unpack('<'+str(len(values))+'f',raw))==values,
            'fields must be lossless float32 numbers; no decimal display rounding')
    for start in range(0,len(values),16):
        require(abs(math.fsum(values[start:start+16])-1.)<=SCALAR_ABSOLUTE_TOLERANCE,
                'localization distribution mass is not one within float32 roundoff')
    return raw


def _native_bytes(raw,size):
    require(type(raw) is bytes and len(raw)==size,'native addressed map byte shape mismatch')
    require(all(math.isfinite(v) for v in struct.unpack('<'+str(size//4)+'f',raw)),
            'native addressed map bytes contain nonfinite values')


@dataclass(frozen=True)
class SourcePanel:
    """Explicit whitelisted fields and native float32 maps, never whole archives."""
    source_key: str
    current_fields: list
    candidate_fields: object
    native_current_bytes: bytes
    native_candidate_bytes: object
    audit_metadata: dict


def validate_panel(panel):
    require(type(panel) is SourcePanel and panel.source_key in (*SOURCE_KEYS,'W','sensor'),
            'unknown whitelisted panel')
    current=_field_bytes(panel.current_fields,(4,2,4,4))
    _native_bytes(panel.native_current_bytes,CURRENT_MAP_BYTES)
    if panel.source_key=='sensor':
        require(panel.candidate_fields is None and panel.native_candidate_bytes is None,
                'sensor has no supplied candidate forecast')
        candidates=None
    else:
        candidates=_field_bytes(panel.candidate_fields,(3,4,2,4,4))
        _native_bytes(panel.native_candidate_bytes,CANDIDATE_MAP_BYTES)
    require(type(panel.audit_metadata) is dict,'audit provenance must be separate dictionary')
    return current,candidates


def localization_summary(grid):
    flat=_numbers(grid,(4,4));peak=max(flat)
    cells=[[index//4,index%4] for index,value in enumerate(flat) if value==peak]
    return dict(peaks=cells,peak_probability=peak,
        entropy_nats=-math.fsum(value*math.log(value) for value in flat if value>0.),
        expected_row=math.fsum((index//4)*value for index,value in enumerate(flat)),
        expected_column=math.fsum((index%4)*value for index,value in enumerate(flat)))


def peak_relation(red,blue):
    horizontal=set();vertical=set();coincidences=[]
    for r in red:
        for b in blue:
            horizontal.add('left_of' if r[1]<b[1] else 'right_of' if r[1]>b[1] else 'same_column')
            vertical.add('above' if r[0]<b[0] else 'below' if r[0]>b[0] else 'same_row')
            coincidences.append(r==b)
    require(bool(coincidences),'relation needs nonempty peak sets')
    return dict(horizontal=[x for x in ('left_of','same_column','right_of') if x in horizontal],
        vertical=[x for x in ('above','same_row','below') if x in vertical],
        coincident_possible=any(coincidences),coincident_all=all(coincidences))


def _origin(source_key,node,view,previous,candidate,w_counterfactual=False):
    if source_key=='sensor':
        return dict(kind='fixed_observation_record',record_time=0,represented_time=0,shown_at=12)
    if view=='current':
        if source_key=='W':
            if w_counterfactual:
                return dict(kind='counterfactual_native_snapshot',record_status='counterfactual_native_snapshot',
                    represented_time=12,archived_origin_time=13,executed_branch_claim=False)
            return dict(kind='completed_record',record_time=12,represented_time=12)
        return (dict(kind='history_estimate',represented_time=12,feedback_through=11)
            if node==previous else dict(kind='copied_prior_record',record_time=11,represented_time=12))
    if node==candidate:
        return (dict(kind='native_conditional_record',represented_time=13,executed_branch_claim=False)
            if source_key=='W' else dict(kind='command_estimate',represented_time=13,feedback_through=11))
    return dict(kind='inherited_current',represented_time=13,
        inherited_origin=_origin(source_key,node,'current',previous,candidate,w_counterfactual))


def _public_nodes(panel,fields,view,previous,candidate):
    nodes=[]
    for index,name in enumerate(NODES):
        summaries={role:localization_summary(fields[index][j]) for j,role in enumerate(ROLES)}
        nodes.append(dict(node=name,origin=_origin(panel.source_key,index,view,previous,candidate,
            panel.source_key=='W' and panel.audit_metadata.get('variant')=='W_changed'),
            distributions={role:deepcopy(fields[index][j]) for j,role in enumerate(ROLES)},
            derived=summaries,red_blue_peak_relation=peak_relation(summaries['red']['peaks'],summaries['blue']['peaks'])))
    return nodes


def export_packet(panels,previous_address,candidate_address,queried_role='red'):
    """Return source-neutral content and distinct source-aware audit provenance."""
    require(type(previous_address) is int and previous_address in range(4)
            and type(candidate_address) is int and candidate_address in range(4),'invalid report addresses')
    require(type(panels) in (tuple,list) and 1<=len(panels)<=2,'one or two matched panels required')
    require(queried_role in ROLES,'invalid queried role')
    if len(panels)==2:
        require(panels[0].source_key in SOURCE_KEYS and panels[1].source_key=='W','joined panel order must be source then W')
    public=[];provenance=[]
    for index,panel in enumerate(panels):
        current_bytes,candidate_bytes=validate_panel(panel);name='p'+str(index)
        public.append(dict(panel=name,current=_public_nodes(panel,panel.current_fields,'current',previous_address,candidate_address),
            candidates=None if panel.candidate_fields is None else {command:_public_nodes(panel,panel.candidate_fields[j],
                'candidate',previous_address,candidate_address) for j,command in enumerate(COMMANDS)}))
        provenance.append(dict(panel=name,source_key=panel.source_key,
            native_current_sha256=sha256_bytes(panel.native_current_bytes),
            native_candidate_sha256=None if panel.native_candidate_bytes is None else sha256_bytes(panel.native_candidate_bytes),
            field_current_sha256=sha256_bytes(current_bytes),
            field_candidate_sha256=None if candidate_bytes is None else sha256_bytes(candidate_bytes),
            audit_metadata=deepcopy(panel.audit_metadata)))
    content=dict(schema=PACK_SCHEMA,decision_time=12,previous_node=NODES[previous_address],
        candidate_node=NODES[candidate_address],queried_role=queried_role,coordinate_system=dict(rows=4,columns=4,row_direction='down',
            column_direction='right',separate_local_grids=True),panels=public,
        derivation=dict(peaks='all cells exactly equal to the largest supplied float32 number',
            relations='Cartesian product of red and blue peak sets in each local grid',
            scalars='peak=max(p); entropy=-sum(p*log(p)), skipping zero; row/column=sum(index*p); no renormalization'),
        candidate_semantics='conditional supplied record or estimate, not an executed branch')
    return dict(content=content,provenance=provenance,content_sha256=sha256_bytes(json_bytes(content)))


def _contexts(content):
    require(content['schema']==PACK_SCHEMA,'unknown neutral pack schema')
    for panel in content['panels']:
        for node in panel['current']:
            yield panel['panel'],'current',None,node
        if panel['candidates'] is not None:
            for command in COMMANDS:
                for node in panel['candidates'][command]:
                    yield panel['panel'],'candidate',command,node


def required_propositions(content):
    return [dict(kind='peaks',panel=panel,view=view,command=command,node=node['node'],role=role,
        cells=deepcopy(node['derived'][role]['peaks']))
        for panel,view,command,node in _contexts(content) for role in ROLES]


def deterministic_template(content):
    """Complete literal technical comparator, not ordinary free narration."""
    atoms=required_propositions(content)
    tokens=['Rows increase down; columns right. Current and conditional candidate peaks follow.']
    for panel,view,command,node in _contexts(content):
        prefix='/'.join([panel,view]+([] if command is None else [command])+[node['node']])
        # Origin JSON is literal metadata, a single whitespace-free token per node.
        tokens.append(prefix+'/origin='+json_bytes(node['origin']).decode())
        for role in ROLES:
            cells=node['derived'][role]['peaks']
            tokens.append(prefix+'/'+role+'={'+','.join('('+str(r)+','+str(c)+')' for r,c in cells)+'}')
    text=' '.join(tokens)
    require(len(text.split())<=WORD_LIMIT,'literal template exceeded registered word cap')
    return dict(schema=REPORT_SCHEMA,text=text,propositions=atoms)


def _slot(prop):
    return tuple(prop.get(key) for key in ('panel','view','command','node','role'))


def _evidence(value):
    """Keep malformed nonfinite evidence JSON writable, with explicit type tags."""
    if type(value) is float and not math.isfinite(value):
        return {'nonfinite_type':'float','value':repr(value)}
    if type(value) is dict:
        return {str(key):_evidence(child) for key,child in value.items()}
    if type(value) in (list,tuple):
        return [_evidence(child) for child in value]
    return deepcopy(value)


def audit_report(content,report):
    """Audit listed machine facts only; unlisted prose needs separate auditing."""
    expected=required_propositions(content);required={_slot(prop):prop for prop in expected}
    contexts={(panel,view,command,node['node']):node for panel,view,command,node in _contexts(content)}
    records=[];covered=set();seen=set();seen_optional=set()
    def error(category,index,actual,expected_value=None):
        records.append(dict(category=category,index=index,actual=_evidence(actual),expected=_evidence(expected_value)))
    if type(report) is not dict:
        error('invalid_report',None,report);report={}
    if report.get('schema')!=REPORT_SCHEMA:
        error('invalid_schema',None,report.get('schema'),REPORT_SCHEMA)
    for key in sorted(set(report)-{'schema','text','propositions'},key=str):
        error('unexpected_report_field',None,{key:report[key]})
    text=report.get('text')
    if type(text) is not str:
        error('missing_free_prose',None,text)
    elif len(text.split())>WORD_LIMIT:
        error('word_limit',None,len(text.split()),WORD_LIMIT)
    propositions=report.get('propositions')
    if type(propositions) is not list:
        error('invalid_propositions',None,propositions);propositions=[]
    allowed={
        'peaks':{'kind','panel','view','command','node','role','cells'},
        'relation':{'kind','panel','view','command','node','relation','horizontal','vertical','coincident_possible','coincident_all'},
        'scalar':{'kind','panel','view','command','node','role','name','value'},
    }
    for index,prop in enumerate(propositions):
        if type(prop) is not dict:
            error('invalid_proposition',index,prop);continue
        kind=prop.get('kind')
        if type(kind) is not str or kind not in allowed:
            lowered=str(kind).lower()
            category=('unsupported_appearance' if any(x in lowered for x in ('appearance','experience','felt','conscious'))
                else 'unlicensed_physical_truth' if any(x in lowered for x in ('physical','truth','actual_scene'))
                else 'incorrect_source_attribution' if any(x in lowered for x in ('attribution','source','attention'))
                else 'invented_proposition_kind')
            error(category,index,prop);continue
        if set(prop)!=allowed[kind]:
            extras=set(prop)-allowed[kind]
            category='timing_error' if extras&{'time','decision_time','represented_time'} else 'proposition_schema'
            error(category,index,prop,sorted(allowed[kind]));continue
        if type(prop['panel']) is not str or prop['panel'] not in tuple(panel['panel'] for panel in content['panels']):
            error('incorrect_panel_attribution',index,prop);continue
        if type(prop['node']) is not str or prop['node'] not in NODES:
            error('address_error',index,prop);continue
        if type(prop['view']) is not str or (prop['command'] is not None and type(prop['command']) is not str):
            error('timing_or_command_error',index,prop);continue
        key=(prop['panel'],prop['view'],prop['command'],prop['node'])
        if key not in contexts:
            error('timing_or_command_error',index,prop);continue
        node=contexts[key]
        if kind in ('peaks','scalar') and (type(prop['role']) is not str or prop['role'] not in ROLES):
            error('role_error',index,prop);continue
        if kind=='peaks':
            slot=_slot(prop)
            if slot in seen:
                error('duplicate_or_contradictory_claim',index,prop,required[slot])
            seen.add(slot)
            cells=prop['cells']
            valid_cells=(type(cells) is list and 1<=len(cells)<=16
                and all(type(cell) is list and len(cell)==2
                    and all(type(v) is int and 0<=v<=3 for v in cell) for cell in cells))
            if not valid_cells or cells!=sorted(cells) or len({tuple(c) for c in cells})!=len(cells):
                error('invalid_peak_cells',index,prop,required[slot]);continue
            if prop!=required[slot]:
                error('localization_contradiction',index,prop,required[slot])
            else:
                covered.add(slot)
        elif kind=='relation':
            optional=(kind,*key)
            if optional in seen_optional:
                error('duplicate_or_contradictory_claim',index,prop)
            seen_optional.add(optional)
            relation=dict(kind='relation',panel=key[0],view=key[1],command=key[2],node=key[3],
                relation='red_blue_peak_relation',**node['red_blue_peak_relation'])
            if (type(prop['coincident_possible']) is not bool or type(prop['coincident_all']) is not bool
                or type(prop['horizontal']) is not list or type(prop['vertical']) is not list
                or prop!=relation):
                error('relation_contradiction',index,prop,relation)
        else:
            name=prop['name'];value=prop['value']
            if name not in ('peak_probability','entropy_nats','expected_row','expected_column'):
                error('invented_scalar',index,prop);continue
            optional=(kind,*key,prop['role'],name)
            if optional in seen_optional:
                error('duplicate_or_contradictory_claim',index,prop)
            seen_optional.add(optional)
            expected_value=node['derived'][prop['role']][name]
            if type(value) not in (int,float) or not math.isfinite(value) or abs(value-expected_value)>SCALAR_ABSOLUTE_TOLERANCE:
                error('scalar_contradiction',index,prop,expected_value)
    for slot,prop in required.items():
        if slot not in covered:
            error('omitted_required_peak_set',None,None,prop)
    return dict(schema='attention_relational_report_v1_machine_factual_audit',
        listed_machine_facts_pass=not records,required_peak_count=len(required),covered_peak_count=len(covered),
        proposition_count=len(propositions),error_count=len(records),errors=records,
        free_prose_fidelity='not assessed by this deterministic machine auditor',
        main_endpoint_pass=False,appearance_validated=False,qualia_established=False)


def compare_packets(first,second):
    """Preserve null and changed pairs, including native change without peak change."""
    require(len(first['provenance'])==len(second['provenance']),'pair panel count differs')
    pairs=[]
    first_facts={_slot(p):p for p in required_propositions(first['content'])}
    second_facts={_slot(p):p for p in required_propositions(second['content'])}
    require(set(first_facts)==set(second_facts),'paired report timing/address inventory differs')
    for left,right in zip(first['provenance'],second['provenance']):
        require(left['panel']==right['panel'] and left['source_key']==right['source_key'],'pair source identity mismatch')
        panel=left['panel']
        pairs.append(dict(panel=panel,source_key=left['source_key'],
            native_current_identical=left['native_current_sha256']==right['native_current_sha256'],
            native_candidate_identical=left['native_candidate_sha256']==right['native_candidate_sha256'],
            current_fields_identical=left['field_current_sha256']==right['field_current_sha256'],
            candidate_fields_identical=left['field_candidate_sha256']==right['field_candidate_sha256'],
            changed_peak_sets=[dict(slot=list(slot),first=first_facts[slot],second=second_facts[slot])
                for slot in first_facts if slot[0]==panel and first_facts[slot]!=second_facts[slot]]))
    return dict(panel_pairs=pairs,report_input_identical=first['content_sha256']==second['content_sha256'],
        null_pairs_retained=True,causal_or_phenomenal_attribution_established=False)


def _bound_record(record):
    require(set(record)=={'path','sha256','commit'},'binding record must contain path/sha256/commit')
    committed_bytes(record['commit'],{record['path']:record['sha256']})
    return json.loads((ROOT/record['path']).read_text())


def validate_input_bindings(heldout_paths,bindings):
    """Stdlib receipt/full-audit and byte gates before any torch import."""
    require(bindings['schema']==INPUT_SCHEMA,'unknown relational input binding schema')
    source_hashes=bindings['study_source_hashes']
    require(STUDY_MINIMUM<=set(source_hashes),'relational study source closure incomplete')
    require(source_hashes['scripts/attention_relational_report_v1.py']==digest(__file__),
            'executing exporter differs from approved bytes')
    approval_record=bindings['study_approval']
    require(set(approval_record)=={'path','sha256'},'invalid study approval descriptor')
    committed_bytes(bindings['study_source_commit'],{**source_hashes,approval_record['path']:approval_record['sha256']})
    approval=json.loads((ROOT/approval_record['path']).read_text())
    require(approval['scope']==APPROVAL_SCOPE and approval['approved'] is True
            and approval['approved_source_sha256']==source_hashes,'relational source approval does not bind exact closure')
    require(set(bindings)=={'schema','study_source_hashes','study_source_commit','study_approval',
            'registration','fit_receipt','heldout_receipt','heldout_reconstruction'},'unexpected input binding fields')
    immutable={key:bindings[key] for key in ('schema','registration','fit_receipt','heldout_receipt','heldout_reconstruction')}
    require(approval['approved_immutable_inputs_sha256']==sha256_bytes(json_bytes(immutable))==IMMUTABLE_INPUT_SHA256,
            'study approval does not bind exact immutable report inputs')
    protocol=json.loads((ROOT/PROTOCOL_MANIFEST_PATH).read_text())
    require(all(source_hashes.get(path)==value for path,value in
            {**protocol['source_sha256'],**protocol['reference_sha256']}.items()),
            'protocol instruction/reference closure changed')
    registration=_bound_record(bindings['registration'])
    require(registration['schema']=='attention_content_model_v1_phase_registration'
            and registration['encoder_seeds']==list(SEEDS),'wrong original registration')
    committed_bytes(registration['source_commit'],registration['source_hashes'])
    committed_bytes(bindings['registration']['commit'],{**registration['source_hashes'],**registration['input_hashes']})
    fit=_bound_record(bindings['fit_receipt']);heldout=_bound_record(bindings['heldout_receipt'])
    base=registration['output_dir']
    expected_paths={seed:f'{base}/heldout/states-{seed}.pt.gz' for seed in SEEDS}
    require(set(heldout_paths)==set(SEEDS) and {seed:canonical(path) for seed,path in heldout_paths.items()}==expected_paths,
            'only the three bound heldout archives are allowed')
    for phase,receipt,key in [('fit',fit,'fit_receipt'),('heldout',heldout,'heldout_receipt')]:
        require(receipt['phase']==phase and receipt['status']=='completed'
                and receipt['registration_sha256']==bindings['registration']['sha256']
                and receipt['registration_commit']==bindings['registration']['commit']
                and receipt['retry_performed'] is False and receipt['no_reporter_calls'] is True
                and receipt['semantic_qualification_asserted'] is False,'phase receipt not immutable completed technical result')
        inventory={f'{base}/{phase}/states-{seed}.pt.gz' for seed in SEEDS}|{f'{base}/{phase}/first_run.log'}
        require(set(receipt['artifact_sha256'])==inventory,'phase receipt archive/log inventory differs')
        committed_bytes(bindings[key]['commit'],receipt['artifact_sha256'])
    require(heldout['fit_receipt_sha256']==bindings['fit_receipt']['sha256']
            and heldout['fit_commit']==bindings['fit_receipt']['commit'],'heldout used different immutable fit')
    reconstruction=bindings['heldout_reconstruction']
    require(set(reconstruction)=={'manifest_path','manifest_sha256','commit'},'invalid heldout audit transport binding')
    committed_bytes(reconstruction['commit'],{reconstruction['manifest_path']:reconstruction['manifest_sha256']})
    manifest=json.loads((ROOT/reconstruction['manifest_path']).read_text());published=manifest['published_path']
    require(canonical(published)==published and published.endswith('.json.gz'),'heldout full audit must use bound lossless gzip')
    committed_bytes(reconstruction['commit'],{published:manifest['compressed_sha256']})
    packed=(ROOT/published).read_bytes();raw=gzip.decompress(packed)
    require(len(packed)==manifest['compressed_bytes'] and len(raw)==manifest['raw_bytes']
            and sha256_bytes(raw)==manifest['raw_sha256'],'heldout full audit transport changed')
    report=json.loads(raw)
    require(report['schema']=='attention_content_model_v1_corrected_reference_phase_reconstruction'
            and report['phase']=='heldout' and report['status']=='passed_corrected_reference_phase_reconstruction'
            and report['hard_invariants_pass'] is True and report['check_count']==len(report['checks'])>0
            and report['errors']==[] and report['error_count']==0 and 'exception' not in report
            and all(row['passed'] is True for row in report['checks'])
            and report['no_reporter'] is True and report['qualia_established'] is False,
            'complete heldout reconstruction did not pass')
    provenance=report['reference_provenance']
    committed_bytes(provenance['amendment_commit'],{provenance['amendment_path']:provenance['amendment_sha256']})
    amendment=json.loads((ROOT/provenance['amendment_path']).read_text())
    committed_bytes(amendment['reference_source_commit'],{**amendment['source_hashes'],
        amendment['approval_path']:amendment['approval_sha256']})
    # This stdlib helper binds its actual approved bytes and fixed original context;
    # it does not import Torch or perform numerical verification/model replay.
    from attention_content_model_v1_reference import validate_amendment,report_identity_matches
    exact_reference=validate_amendment(provenance['amendment_path'],provenance['amendment_commit'],
        'scripts/attention_content_model_v1_reference.py',bindings['registration']['path'],
        bindings['registration']['commit'],bindings['fit_receipt']['commit'])
    require(report_identity_matches(report,exact_reference,'heldout'),'heldout amended full-report identity changed')
    require(heldout['reference_provenance']==exact_reference,'heldout receipt reference identity changed')
    claim_path=f'{base}/heldout/claim.json'
    committed_bytes(bindings['heldout_receipt']['commit'],{claim_path:digest(ROOT/claim_path)})
    claim=json.loads((ROOT/claim_path).read_text())
    require(claim['reference_provenance']==exact_reference and claim['corrected_fit_reference']==heldout['corrected_fit_reference']
        and claim['registration_sha256']==bindings['registration']['sha256']
        and claim['registration_commit']==bindings['registration']['commit']
        and claim['fit_commit']==bindings['fit_receipt']['commit'] and claim['retry_allowed'] is False,
        'heldout exclusive claim context differs')
    require(report['verifier_sha256']==amendment['source_hashes']['scripts/verify_attention_content_model_v1_corrected.py']
            and report['registered_verifier_sha256']==registration['source_hashes']['scripts/verify_attention_content_model_v1.py']
            and report['fitted_artifact_sha256']==fit['artifact_sha256']
            and provenance['immutable']['fit_artifact_sha256']==fit['artifact_sha256'],'heldout audit/source/fitted identity changed')
    commands=report['command_bindings']
    require(canonical(commands['registration'])==bindings['registration']['path']
            and commands['registration_commit']==bindings['registration']['commit']
            and commands['phase_commit']==bindings['heldout_receipt']['commit']
            and commands['fit_commit']==bindings['fit_receipt']['commit'],'heldout audit used different registration/phase')
    assessment_path=canonical(commands['assessment']);dev_commit=commands['dev_commit']
    require(dev_commit==heldout['dev_commit']==claim['dev_commit'], 'heldout dev boundary changed')
    committed_bytes(dev_commit,{assessment_path:heldout['dev_assessment_sha256']})
    require(claim['dev_assessment_sha256']==heldout['dev_assessment_sha256'],'heldout claim assessment changed')
    assessment=json.loads((ROOT/assessment_path).read_text())
    dev_receipt_path=f'{base}/dev/receipt.json'
    committed_bytes(dev_commit,{dev_receipt_path:assessment['dev_receipt_sha256']})
    dispositions=assessment['source_semantic_dispositions']
    require(assessment['schema']=='attention_content_model_v1_dev_acceptance'
        and assessment['hard_invariants_pass'] is True and assessment['reporting_released'] is False
        and assessment['fitted_artifact_sha256']==fit['artifact_sha256']
        and assessment['reference_amendment_sha256']==exact_reference['amendment_sha256']
        and assessment['reference_amendment_commit']==exact_reference['amendment_commit']
        and set(dispositions)==set(SOURCE_KEYS)
        and all(row==dict(literal_content_disposition='qualified',presentation_manner_qualified=False)
                for row in dispositions.values()),'frozen dev literal/manner disposition changed')
    ancestor(bindings['registration']['commit'],bindings['fit_receipt']['commit'])
    ancestor(bindings['fit_receipt']['commit'],bindings['heldout_receipt']['commit'])
    ancestor(bindings['fit_receipt']['commit'],dev_commit)
    ancestor(dev_commit,bindings['heldout_receipt']['commit'])
    ancestor(bindings['heldout_receipt']['commit'],reconstruction['commit'])
    ancestor(reconstruction['commit'],bindings['study_source_commit'])
    return dict(registration=registration,fit_receipt=fit,heldout_receipt=heldout,
        heldout_raw_audit_sha256=manifest['raw_sha256'],bindings_sha256=sha256_bytes(json_bytes(bindings)),
        source_semantic_dispositions=dispositions,reporting_released_by_original_study=False,
        protocol=protocol)


def _tensor_bytes(value,torch,shape):
    require(type(value) is torch.Tensor and tuple(value.shape)==shape
            and value.dtype==torch.float32 and value.device.type=='cpu' and not value.requires_grad
            and bool(torch.isfinite(value).all()),'invalid archived float32 field/map')
    values=value.contiguous().flatten().tolist()
    return struct.pack('<'+str(len(values))+'f',*values)


def _actual_maps(episode,shared,torch,world,time):
    current=episode['V'][world].clone()
    family=shared['actual_operator_family'][world]
    for index in range(time):
        address=int(episode['command_addresses'][world,index]);symbol=int(episode['command_symbols'][world,index])
        band=int(episode['evaluator_mappings'][world,address,symbol])
        require(band in (1,2,4),'invalid archived operator mapping')
        current[address]=family[address,{1:0,2:1,4:2}[band]]
    return current


def _full_candidates(current,selected,address,torch):
    candidates=current[None].expand(3,*current.shape).clone()
    for symbol in range(3):
        candidates[symbol,address]=selected[symbol]
    return candidates


def _panel(source,fields,candidates,maps,next_maps,metadata,torch):
    _tensor_bytes(fields,torch,(4,2,4,4))
    if candidates is not None:
        _tensor_bytes(candidates,torch,(3,4,2,4,4))
    return SourcePanel(source,fields.tolist(),None if candidates is None else candidates.tolist(),
        _tensor_bytes(maps,torch,(4,24,4,4)),
        None if next_maps is None else _tensor_bytes(next_maps,torch,(3,4,24,4,4)),metadata)


def _donor_evidence(evaluation,source,schedule,world,time,torch):
    """Audit-only replay/update receipts; never actor-visible state or mappings."""
    metadata=evaluation['metadata'];j=time-12
    rows=[row for row in metadata['donor_provenance'] if row['time']==time and row['schedule']==schedule]
    require(len(rows)==1,'unique archived donor prefix receipt required')
    row=rows[0];ledger=evaluation['tensors']['sources'][source]['donors'][schedule]
    require(row['permitted_feedback_command_indices']==list(range(time-1)), 'donor feedback timing differs')
    names=('restore_'+schedule+'_current_exact','restore_'+schedule+'_next_exact',
        'restore_'+schedule+'_policy_fields_exact','restore_'+schedule+'_state_exact',
        'patch_'+schedule+'_nonselected_state_exact','patch_'+schedule+'_selected_state_transfer_exact',
        'patch_'+schedule+'_nonselected_current_exact','patch_'+schedule+'_nonselected_next_exact',
        'patch_'+schedule+'_nonselected_decoder_exact','patch_'+schedule+'_next_inheritance_exact')
    gates=evaluation['software_checks']['sources'][source]
    for name in names:
        require(gates[name].dtype==torch.bool and bool(gates[name][world,j]),'archived named donor/restoration gate failed: '+name)
    state_fields={name:ledger[name][world,j].tolist() for name in (
        ('donor_hidden_at_patched_address',) if source!='analytic' else
        ('donor_survivors_at_patched_address','donor_revealed_symbols_at_patched_address'))}
    calls=[row for row in metadata['source_call_visibility'][source]
        if row['decision_time']==time and row['kind'] in ('primary','donor_'+schedule,'restored_'+schedule)]
    updates=[row for row in metadata['source_update_visibility'][source]
        if row['kind']=='donor_replay' and row['donor_schedule']==schedule and row['donor_decision_time']==time]
    require(len(calls)==3 and len(updates)==time-1,'archived donor prediction/update inventory differs')
    return dict(schedule=schedule,patched_address=row['patched_addresses'][world],
        prefix_receipt_sha256=sha256_bytes(json_bytes(row)),updates=deepcopy(row['updates']),
        executor_command_indices=row['executor_command_indices'],
        permitted_feedback_command_indices=row['permitted_feedback_command_indices'],
        pre_reveal_mask=row['pre_reveal_mask'],mapping_indices=row['mapping_indices'][world],
        concealed_completion_sha256=row['concealed_completion_hash'],
        source_prediction_receipts=deepcopy(calls),source_update_receipts=deepcopy(updates),
        donor_selected_state_sha256=sha256_bytes(json_bytes(state_fields)),
        archived_exact_gates={name:True for name in names})


def _case_panels(bundle,world,seed,archive_hash,torch):
    episode=bundle['episode'];evaluation=bundle['evaluation'];shared=evaluation['tensors']['shared']
    require(episode['split']=='heldout' and episode['encoder_seed']==seed,'archive seed/split differs')
    t=DECISION_TIME;j=t-12;previous=int(episode['command_addresses'][world,t-1]);candidate=int(episode['command_addresses'][world,t])
    require(previous in range(4) and candidate==(world%4),'registered decision12 candidate differs')
    prior_maps=_actual_maps(episode,shared,torch,world,t-1);actual_maps=_actual_maps(episode,shared,torch,world,t)
    prior_fields=shared['actual_completed_fields'][world,t-1]
    actual_fields=shared['actual_completed_fields'][world,t]
    sensor=_panel('sensor',shared['actual_completed_fields'][world,0],None,episode['V'][world],None,
        dict(encoder_seed=seed,world_index=world,variant='baseline',archive_sha256=archive_hash),torch)
    bands=episode['evaluator_mappings'][world,candidate]
    indices=[{1:0,2:1,4:2}[int(value)] for value in bands]
    selected_w_fields=shared['actual_family_fields'][world,candidate,indices]
    selected_w_maps=shared['actual_operator_family'][world,candidate,indices]
    control_rows=[row for row in evaluation['metadata']['actual_control_provenance']
        if row['time']==t and row['world_index']==world]
    require(len(control_rows)==1,'unique native control receipt required')
    changed_maps=actual_maps.clone();changed_maps[candidate]=shared['actual_operator_family'][world,candidate,1]
    def native_item_hash(value):
        return sha256_bytes(b'torch.float32:(24, 4, 4):'+_tensor_bytes(value,torch,(24,4,4)))
    require(control_rows[0]['saved_w_hashes']==[native_item_hash(item) for item in actual_maps]
        and control_rows[0]['changed_w_hashes']==[native_item_hash(item) for item in changed_maps],
        'native control map/receipt hash differs')
    w_panels={}
    for variant in W_VARIANTS:
        fields=actual_fields.clone();maps=actual_maps.clone()
        if variant=='W_changed':
            fields=shared['actual_controls']['exogenous_role_fields'][world,j].clone()
            maps[candidate]=shared['actual_operator_family'][world,candidate,1]
        elif variant=='W_restored':
            restored=shared['actual_controls']['restored_role_fields'][world,j]
            require(torch.equal(restored,actual_fields),'saved W restoration fields differ')
            fields=restored.clone()
        w_panels[variant]=_panel('W',fields,_full_candidates(fields,selected_w_fields,candidate,torch),
            maps,_full_candidates(maps,selected_w_maps,candidate,torch),
            dict(encoder_seed=seed,world_index=world,variant=variant,archive_sha256=archive_hash,
                actual_control_logits_sha256=sha256_bytes(_tensor_bytes(
                    shared['actual_controls'][('exogenous' if variant=='W_changed' else 'restored')+'_local_logits'][world,j],
                    torch,(4,3,4,4))) if variant!='baseline' else None,
                conditional_family_index_rule='opaque symbol -> evaluator operator1/2/4 -> family0/1/2',
                opaque_symbol_family_indices=indices,native_control_receipt=deepcopy(control_rows[0]),
                executed_future_branch_claim=False),torch)
    sources={}
    for source in SOURCE_KEYS:
        retained=evaluation['tensors']['sources'][source]
        base_maps=prior_maps.clone();base_maps[previous]=retained['selected_current_maps'][world,t-1]
        base_fields=prior_fields.clone();base_fields[previous]=retained['selected_current_fields'][world,t-1]
        base_next=_full_candidates(base_maps,retained['selected_next_maps'][world,t],candidate,torch)
        base_next_fields=_full_candidates(base_fields,retained['selected_next_fields'][world,t],candidate,torch)
        sources[source]={}
        for variant in VARIANTS:
            maps=base_maps.clone();fields=base_fields.clone();next_maps=base_next.clone();next_fields=base_next_fields.clone()
            donor_evidence=None
            if variant in ('current_donor','prospective_donor','current_restored','prospective_restored'):
                schedule='current' if variant.startswith('current') else 'prospective'
                ledger=retained['donors'][schedule]
                donor_evidence=_donor_evidence(evaluation,source,schedule,world,t,torch)
                if variant.endswith('donor'):
                    maps[previous]=ledger['selected_current_maps'][world,j]
                    fields[previous]=ledger['selected_current_fields'][world,j]
                    next_maps=_full_candidates(maps,ledger['selected_next_maps'][world,j],candidate,torch)
                    next_fields=_full_candidates(fields,ledger['policy_candidate_fields'][world,j],candidate,torch)
                else:
                    require(torch.equal(ledger['restored_selected_current_maps'][world,j],retained['selected_current_maps'][world,t-1])
                        and torch.equal(ledger['restored_selected_next_maps'][world,j],retained['selected_next_maps'][world,t])
                        and torch.equal(ledger['restored_policy_candidate_fields'][world,j],retained['selected_next_fields'][world,t]),
                        'saved source restoration map/field identity differs')
                    # Exact saved native map equality licenses reuse of archived base H;
                    # no new H forward is performed and this origin is recorded separately.
            sources[source][variant]=_panel(source,fields,next_fields,maps,next_maps,
                dict(encoder_seed=seed,world_index=world,variant=variant,archive_sha256=archive_hash,
                    current_reconstruction='W11 copy; replace previous node from selected_current_fields/maps[11]',
                    forecast_reconstruction='source current copy; replace query node from selected_next_fields/maps[12]',
                    restored_current_H='archived base H reused only after exact restored native map identity',
                    source_update_feedback_through=11,donor_evidence=donor_evidence),torch)
    return sources,w_panels,sensor,previous,candidate


def build_reporting_packets(heldout_paths,immutable_bindings):
    """Explicit future export: byte/receipt gates first, then archive reads only.

    No importing or calling a source/head/encoder/model/renderer/optimizer. Case
    selection and all variants are fixed before values are read. Never filter
    null changes or select favorable outcomes. No files or API calls are written.
    """
    context=validate_input_bindings(heldout_paths,immutable_bindings)
    import torch
    result=dict(requests=[],packets=[],provenance=[],templates=[],paired_checks=[],
        immutable_context=context,request_count=REQUEST_COUNT,reporter_called=False)
    narrator=(ROOT/NARRATOR_PATH).read_text().rstrip('\n')
    def add(case,panels,previous,candidate,queried_role):
        packet=export_packet(panels,previous,candidate,queried_role)
        request_id=f"s{case['encoder_seed']}_w{case['world_index']:02d}_t12_{case['source']}_{case['view']}_{case['variant']}"
        # Complete REPORT_RESPONSE_SCHEMA must also be frozen in API response format.
        result['requests'].append(dict(id=request_id,input=narrator+'\n'+json_bytes(packet['content']).decode()))
        result['packets'].append(dict(id=request_id,content=packet['content'],content_sha256=packet['content_sha256']))
        result['provenance'].append(dict(id=request_id,case=case,panels=packet['provenance']))
        result['templates'].append(dict(id=request_id,report=deterministic_template(packet['content'])))
        return packet,request_id
    for seed in SEEDS:
        with gzip.open(ROOT/canonical(heldout_paths[seed]),'rb') as stream:
            bundle=torch.load(stream,weights_only=True,map_location='cpu')
        require(bundle['fitted_artifact_sha256']==context['fit_receipt']['artifact_sha256'],'heldout archive fitted bytes differ')
        for world in WORLD_INDICES:
            archive_hash=context['heldout_receipt']['artifact_sha256'][canonical(heldout_paths[seed])]
            sources,w,sensor,previous,candidate=_case_panels(bundle,world,seed,archive_hash,torch)
            role=int(bundle['episode']['query_roles'][world,DECISION_TIME])
            require(role in (0,1),'invalid archived queried role')
            for source in SOURCE_KEYS:
                for view in ('A_only','joined'):
                    emitted={}
                    for variant in VARIANTS:
                        w_variant=variant if variant in W_VARIANTS else 'baseline'
                        panels=[sources[source][variant]]+([w[w_variant]] if view=='joined' else [])
                        emitted[variant]=add(dict(encoder_seed=seed,world_index=world,decision_time=12,
                            source=source,view=view,variant=variant),panels,previous,candidate,ROLES[role])
                    for variant in VARIANTS[1:]:
                        comparison=compare_packets(emitted['baseline'][0],emitted[variant][0])
                        if variant in ('current_restored','prospective_restored','W_restored'):
                            require(comparison['report_input_identical'],'restored neutral input must be byte-identical to baseline')
                        if variant in ('W_changed','W_restored'):
                            a=comparison['panel_pairs'][0]
                            require(a['native_current_identical'] and a['native_candidate_identical']
                                and a['current_fields_identical'] and a['candidate_fields_identical'],
                                'W intervention must leave estimated-source bytes identical')
                        result['paired_checks'].append(dict(first_id=emitted['baseline'][1],second_id=emitted[variant][1],
                            comparison=comparison))
            emitted={}
            for variant in W_VARIANTS:
                emitted[variant]=add(dict(encoder_seed=seed,world_index=world,decision_time=12,
                    source='W',view='W_only',variant=variant),[w[variant]],previous,candidate,ROLES[role])
            require(compare_packets(emitted['baseline'][0],emitted['W_restored'][0])['report_input_identical'],
                    'W-only restored input differs')
            for variant in W_VARIANTS[1:]:
                result['paired_checks'].append(dict(first_id=emitted['baseline'][1],second_id=emitted[variant][1],
                    comparison=compare_packets(emitted['baseline'][0],emitted[variant][0])))
            add(dict(encoder_seed=seed,world_index=world,decision_time=12,source='sensor',view='sensor_only',variant='baseline'),
                [sensor],previous,candidate,ROLES[role])
        del bundle
    require(len(result['requests'])==REQUEST_COUNT,'registered552 report opportunity inventory differs')
    require([row['id'] for row in result['requests']]==context['protocol']['request_ids'],
            'exact registered scientific ID ordering differs')
    require(all(len(row['report']['propositions']) in (8,32,64) for row in result['templates']),
            'full addressed peak-set coverage inventory differs')
    require(sum(len(row['report']['propositions']) for row in result['templates'])==25440,
            'registered25440 peak atom inventory differs')
    result['response_schema']=deepcopy(REPORT_RESPONSE_SCHEMA)
    result['requests_sha256']=sha256_bytes(json_bytes(result['requests']))
    return result
