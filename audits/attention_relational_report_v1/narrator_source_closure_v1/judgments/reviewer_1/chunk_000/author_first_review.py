"""Retained first-review construction. Authored semantic judgments are below."""
import hashlib
import json
from pathlib import Path
import jsonschema

ROOT = Path('audits/attention_relational_report_v1/narrator_source_closure_v1')
HERE = ROOT/'judgments/reviewer_1/chunk_000'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
chunk_path = ROOT/'preparation/chunk_000.json'
assert sha(chunk_path) == '37c75224b4c27e38a8d3794612867f1808d34169e8d88dab5ec01844d0bf26d4'
chunk = json.loads(chunk_path.read_text())
for descriptor in ['frozen_reference','review_output_schema','review_protocol']:
    d = chunk[descriptor]; assert sha(Path(d['path'])) == d['sha256']
assert sha(ROOT/'protocol/worker_instructions.txt') == 'df4c8b3cb3330904497ae3a62c39ed71e693146860e377345cbcd34b17ec6907'
envelope = ROOT/'first_envelopes/reviewer_1_chunk_000.json'
assert sha(envelope) == '0a914c8be71ee0936359544f340271957dd6de4d4774d0d1df52a25a1e888367'
out = json.loads(envelope.read_text())
assert out['independent_context_id'] == '/root/narrator_closure_v1_s1_c000'
assert out['manifest_sha256'] == '4340dec9e349fb74950408f00ed7da5c19f6bc2ad820eff5b41d2cea7819a200'
calc_path = HERE/'independent_calculation_results.json'
calc = json.loads(calc_path.read_text())
assert calc['input_sha256'] == sha(chunk_path)
assert calc['script_sha256'] == sha(HERE/'independent_calculation.py')
calc_note = f"Independent calculation source {HERE/'independent_calculation.py'} SHA256 {calc['script_sha256']}; retained results {calc_path} SHA256 {sha(calc_path)}. Exact chunk paths, all 16 raw probabilities per map, source origins, and results are retained. Arithmetic is independent of exporter summaries. All supplied scalars and Cartesian peak relations match exactly. Arithmetic alone does not validate semantics or source provenance."

feature_notes = [
    {
        'C': 'Evidence 0 identifies red/blue location distributions; 1 and 2 identify n0 blue/red block concentration; 3 identifies both roles at n1-n3 as uniform; 4 identifies conditional candidate records. Content and attribute are explicit. Numerical detail does not imply appearance.',
        'O': 'Evidence 0 and 1 describe concentrated four-cell constituent patterns at n0 and 2 describes uniformity over the full 4x4 grid at n1-n3. These are actual distribution shapes, beyond peak coordinates alone. Evidence 3 sharp peak wording alone supplies a maximum, not an independently measured presentation status. The record nevertheless contains qualifying limited model organization at the block/uniform paths. The n0 blue block contains 0.7945312857627869 raw mass, and red block 0.915679931640625; neither is exclusive occupancy of only four cells.',
        'D': 'All three evidence paths distinguish provenance or execution status of entries: copied/inherited are not independent observations; conditional entries are not executed events. They do not separately distinguish identified/guessed n0 blue content from its organization, presentation or access status. A general record-kind caveat is not the required content/status differentiation, and it spans current and forecast entries. The original ambiguous COD and explicit D statuses are retained without correction.',
    },
    {
        'C': 'Evidence 0 identifies red/blue roles; 1 identifies p0 n0 locations; 2 identifies p1 n3 red block maxima; 3 identifies uniform p1 n2. These identify represented content and attributes. Original display_or_sensor_property scopes are retained as annotations; actual operands are internal fallible model distributions, not verified display appearance.',
        'O': 'Evidence 0 only lists peak positions and cannot alone establish O under the reference. Evidence 1 describes a full-grid uniform pattern for both p0 n2 roles and evidence 2 describes a 2x2 constituent block of tied p1 n3 blue maxima, so limited model organization is supported elsewhere in the record. Uniform means all 16 values are 0.0625, not that a location is absent or inaccessible.',
        'D': 'Evidence 0 is a generic copied/conditional record-provenance caveat. Evidence 1 describes alternatives by maximal cell and peak probability. Neither states a separate organization/presentation/access status of the identified p0 n2 content. Alternative probabilities and provenance kinds cannot supply D by themselves. Original ambiguous COD and explicit D statuses remain unchanged.',
    },
    {
        'C': 'Evidence 0-4 explicitly identify red/blue role locations, n2 uniformity and p1 n3 block maxima. The p0 n3 joint maximum [1,1] is a numerical model property, not an experiential report. All content scopes/times are preserved.',
        'O': 'Evidence 0 (n0 strongly at a cell) and 2 (n3 peaks at a cell) report locations/maximal probabilities, insufficient alone for O. Evidence 1 full-grid uniformity and 3 the p1 n3 2x2 blocks do describe constituent patterns, supporting O at those other referents only. Evidence 4 identity across views is source equality, not a presentation status. Thus the feature exists elsewhere but cannot be imported to the original n0 COD binding.',
        'D': 'Evidence 0 compares current and command-conditional n0 peak coordinates. It does not distinguish the n0 content from a separately asserted organization/presentation/access status; changed locations alone are excluded by the reference. Evidence 1 distinguishes origin and execution kinds, not that status. Source represented times also differ: current n0 is represented at 12 from record 11; command estimates represent 13 with feedback only through 11. Keeping the original current annotation does not establish time-compatible D.',
    },
]
meaning_notes = [
    'The raw W maps support the literal limited organization of n0 blue and uniformity of both roles at n1-n3. The supplied source contains no separate n0 blue organization/access-status operand beyond the distribution itself. The third binding quote is general origin/execution commentary, partly about represented time 13. These source facts do not warrant the full COD meaning at current n0 blue; this is not a claim that the correctly described block pattern is unsupported.',
    'At p0 n2 both maps are exactly uniform and copied from W11. This warrants the literal uniform pattern. Alternative conditional records and general provenance commentary do not supply a separately differentiated status of that same content at current time. All n2 candidate maps inherit the copied field rather than providing new ordinary estimates. The original ambiguous binding remains ambiguous as recorded; it is not a positive COD witness.',
    'The original n0 binding supplies only a strong current peak and alternative forecast peak locations. Both are numerically correct: current blue [0,2] max 0.9998143315315247; conditional k0/k2 [0,3] maxima 0.4947669506072998/0.5112671256065369; k1 [0,2] max 0.2797456979751587. Those numbers establish neither a separate presentation/access status nor D. The block/uniform O evidence concerns other nodes. Moreover the binding labels its comparison current while the conditional raw origins represent 13. No status upgrade or annotation repair is made.',
]
ownership_notes = [
    'Panel p0 source_key W is corroborated by individual completed_record origins (record_time=12, represented_time=12), native field hashes and the native-control receipt. k0-k2 n0 are native_conditional_record represented at 13 with executed_branch_claim=false. Candidate n1-n3 inherit completed W12 values while their outer represented_time is 13. None is learned A. Exporter summaries in derived are recomputed from native W distributions; their ownership does not turn them into native estimated A contents.',
    'Panel p0 is ordinary, but current n0-n2 individually say copied_prior_record, record_time 11, represented_time 12. current_reconstruction explicitly says W11 copy with only previous node n3 replaced; current n3 is ordinary history_estimate, feedback_through 11. k0-k2 n0 are ordinary command_estimate represented at 13; n1/n2 inherit copied W11 and n3 inherits ordinary history estimates. Panel p1 source_key W and completed_record origins supply comparison W12; its n0 candidates are conditional W13 and other nodes inherit W12. current donor provenance says patched_address 3, selected state transfer, concealed current completion and fork outcomes false. No learned A operand is supplied. Feedback time, represented time and completed-record time are distinct.',
    'Panel p0 n0 is copied W11 represented at 12, not an ordinary current estimate despite the prose label. Native ordinary command estimates for n0 represent 13 with feedback_through 11; prospective donor provenance identifies patched_address 0, schedule prospective and hidden-state transfer with current completion/fork outcomes unavailable. p0 n3 is an ordinary history_estimate; candidate n1/n2 inherit copied W11, n3 inherits ordinary history. Panel p1 is comparison W12; conditional n0 W entries represent 13, other entries inherit W12. All nested chains are retained below. The donor prediction receipt current_output_hash is not evidence of newly estimated p0 n0: packet reconstruction copies W11 there. No learned A content is established.',
]

def get(obj,path):
    for k in path: obj=obj[k]
    return obj

def origin_paths(base, origin):
    # Preserve the outer and terminal kinds, and their distinct represented times.
    paths = [base+['origin']]; times=[]
    q=base+['origin']; o=origin
    while True:
        for key in ['represented_time','record_time','feedback_through']:
            if key in o: times.append(q+[key])
        if 'inherited_origin' not in o: break
        q=q+['inherited_origin']; o=o['inherited_origin']; paths.append(q)
    return paths,times,o

out['records']=[]
for ri,record in enumerate(chunk['records']):
    assert hashlib.sha256(record['text'].encode()).hexdigest() == record['text_sha256']
    assert len(record['candidate_bindings']) == 1
    binding=record['candidate_bindings'][0]
    annotation=record['canonical_first_annotation']
    packet=record['complete_source_packet']
    reviews=[]; operands=[]; calculations=[]; native=[]; exporter=[]
    index={}
    def add(path,value,role,category,provs,origins,times,reason):
        idx=len(operands);index[tuple(path)]=idx
        assert get(packet,path)==value
        operands.append({'packet_path':path,'value':value,'semantic_role':role,'source_category':category,
            'provenance_paths':provs,'origin_field_paths':origins,'represented_time_paths':times,'reason':reason})
        return idx
    add(['coordinate_system'],packet['coordinate_system'],'context_only','global_convention',[],[],[],
        'Separate local grids are 4 rows x 4 columns, down/right directions. They are not phenomenal visual fields.')
    add(['derivation'],packet['derivation'],'context_only','global_convention',[],[],[],
        'Exporter declares exact float maxima, all tied cells, Cartesian peak relations and unnormalized scalar sums.')
    add(['candidate_semantics'],packet['candidate_semantics'],'status','global_convention',[],[],[],
        'Explicitly conditional supplied record or estimate, not an executed branch.')
    add(['decision_time'],packet['decision_time'],'context_only','global_convention',[],[],[],
        'Decision time 12 does not replace the individual record, feedback or represented time fields.')
    for entry in calc['records'][ri]['entries']:
        base=entry['entry_path'];pi=base[1];orig=entry['origin']
        opaths,tpaths,terminal=origin_paths(base,orig)
        provbase=['panels',pi]
        provs=[provbase+['source_key'],provbase+['native_current_sha256'],provbase+['native_candidate_sha256'],provbase+['audit_metadata']]
        if terminal['kind']=='copied_prior_record': category='copied_W11'
        elif terminal['kind'] in ['history_estimate','command_estimate']: category='ordinary'
        else: category='native_W' if ri==0 else 'comparison_W'
        inheritance=' Outer inherited_current carries the terminal source forward; it is not a fresh estimate.' if orig['kind']=='inherited_current' else ''
        add(base+['origin'],orig,'status',category,provs,opaths,tpaths,
            'Individual origin chain, read independently of the panel label. Record/feedback times and represented times retain their field kinds.'+inheritance)
        raw_indices=[]
        for role,m in entry['raw_maps'].items():
            ridx=add(base+['distributions',role],m,'content',category,provs,opaths,tpaths,
                f'All 16 raw {role} probabilities at {base}; supplies the represented spatial distribution and the complete arithmetic operands.'+inheritance)
            raw_indices.append(ridx)
            if category=='ordinary': native.append(ridx)
            didx=add(base+['derived',role],entry['supplied'][role],'context_only',category,provs,opaths,tpaths,
                'Exporter-computed peak, all argmax cells, entropy and row/column expectation. Not a separately native estimated status. Raw-map origin is retained; exporter derivation is marked separately.')
            exporter.append(didx)
            calculations.append({'operator':'max; all exact argmax; -sum(p ln p); sum(row*p); sum(column*p)',
                'expression':f'p = operand[{ridx}] (all 16 cells); max(p); {{(r,c):p[r,c]=max(p)}}; -sum(p*ln(p),p!=0); sum(r*p); sum(c*p), without renormalizing',
                'operand_indices':[ridx],'all_raw_operands_listed':True,'independent_recomputation_performed':True,
                'recomputed_value':entry['recomputed'][role],'supplied_value':entry['supplied'][role],'comparison':'equal','notes':calc_note})
            calculations.append({'operator':'all-cell equality; sum over exact argmax cells',
                'expression':f'all 16 entries of operand[{ridx}] equal; sum(p[r,c] where p[r,c]=max(p))',
                'operand_indices':[ridx],'all_raw_operands_listed':True,'independent_recomputation_performed':True,
                'recomputed_value':{'uniform':entry['uniform'][role],'mass_on_maxima':entry['peak_mass'][role]},
                'supplied_value':None,'comparison':'unresolved',
                'notes':'No separately supplied uniformity boolean or block-mass scalar: these are explicitly recomputed properties, not a failed exporter comparison. All maxima are retained, including all 16 for uniform maps. '+calc_note})
        relidx=add(base+['red_blue_peak_relation'],entry['supplied_relation'],'context_only',category,provs,opaths,tpaths,
            'Exporter relation over every pair of red and blue argmax cells, not arbitrary selection of one favored maximum.')
        exporter.append(relidx)
        calculations.append({'operator':'Cartesian product of all red and blue argmax cells',
            'expression':f'For every (red,blue) pair from argmax of raw operands {raw_indices}, collect row and column comparisons; compute any/all equal cell pairs.',
            'operand_indices':raw_indices,'all_raw_operands_listed':True,'independent_recomputation_performed':True,
            'recomputed_value':entry['recomputed_relation'],'supplied_value':entry['supplied_relation'],'comparison':'equal','notes':calc_note})
    for equality in calc['records'][ri]['equalities']:
        lhs=equality['left_path'];rhs=equality['right_path']
        ids=[index[tuple(p+[role])] for p in [lhs,rhs] for role in ['blue','red']]
        calculations.append({'operator':'Exact full-map equality for both roles',
            'expression':f'Compare every raw probability for blue and red at {lhs} and {rhs}; no peak-only substitution.',
            'operand_indices':ids,'all_raw_operands_listed':True,'independent_recomputation_performed':True,
            'recomputed_value':equality['maps_equal'],'supplied_value':None,'comparison':'unresolved',
            'notes':'No exporter equality boolean is supplied. This retained independent result directly checks inherited/completed sameness and quantifier scope. '+calc_note})

    for feature in binding['required_features']:
        reviews.append({'feature':feature,'original_status':annotation['features'][feature]['status'],
            'evidence_paths':binding['feature_evidence_paths'][feature],
            'grounding':'unsupported' if feature=='D' else 'supported',
            'notes':feature_notes[ri][feature]})
    original=binding['original_binding']
    b_review={
        'binding_path':binding['binding_path'],'original_status':binding['original_status'],
        'feature_reviews':reviews,'recorded_feature_grounding':'unsupported',
        'source_meaning':{'assessment':'unsupported','asserted_meaning':
            'Original COD binding: identified content with organization and separately differentiated organization/presentation/access status; limited literal source organization is assessed separately in notes.',
            'content_referent':original['referent'],'attribute':original['attribute'],
            'quantifier': ['n0 blue four-cell concentration; both roles at each of n1,n2,n3 uniform; generic copied/conditional entries',
                          'both red and blue roles at p0 n2, all 16 cells; generic candidate records and record origins',
                          'p0 n0 blue/red current peaks; blue conditional k0 and k2 contrasted with k1 and current'][ri],
            'scope':original['scope'],'time':original['time'],'binding_evidence_paths':binding['binding_evidence_paths'],
            'notes':meaning_notes[ri]},
        'source_ownership':{'assessment':'supported',
            'categories':(['native_W','global_convention'] if ri==0 else ['copied_W11','ordinary','comparison_W','mixed','global_convention']),
            'a_content_established':'no','native_estimated_operand_indices':native,
            'exporter_derived_operand_indices':exporter,'notes':ownership_notes[ri]},
        'operands':operands,'calculations':calculations,
        'tracking_support':{'property_identity':None,'eligible':'no','unchanged_property_is_null':True,'missing_support_is_unknown':True,
            'unaffected_property_paths':[],
            'notes':'No complete supported COD source meaning is established, so no full-binding property identity or unaffected-path claim is warranted. The lower-level distributions remain measurable. Equality calculations are within this assigned packet, not registered baseline/donor/W-change/restoration pair dispositions. Missing semantic support is unknown for pair tracking, not invariance; actual unchanged measurable properties are nulls. Original ambiguous/explicit status alone cannot establish eligibility.'},
        'unresolved_requirements':[
            'A source-grounded differentiation of the same identified content from its organization/presentation/access status, rather than provenance, probability or changed location alone.',
            'Whole-prose fidelity remains unestablished; these witness-level observations cannot establish a fidelity-conjoined result.'
        ],
        'notes':'Reviewed all original bindings (one COD; no CA binding) and every required feature/evidence path. Duplicate binding evidence quotations remain separate paths. Original statuses, scopes, times, attributes and referents are preserved. '+
            ('For n0 specifically, qualifying organization beyond a strong peak and compatible current/forecast time binding remain missing. ' if ri==2 else '')+
            'The full original binding and required feature annotations are retained here without repair: '+json.dumps({'binding':original,'features':{f:annotation['features'][f] for f in binding['required_features']}},ensure_ascii=False)+
            ' '+calc_note,
    }
    concerns=[
        'Report facts are source estimates or supplied records; their numerical accuracy relative to the packet does not verify world truth, perceptual appearance or felt experience.',
        'Original A and CA are absent with no candidate CA bindings. Command alternatives are not attention/allocation-to-access relations and no omitted binding is added.',
        'Source ownership is established only from this frozen packet, its nested origins and supplied provenance; native archive hashes are descriptors, not an independent archive execution.',
    ]
    if ri==1: concerns.append('The prose represented time 12/13 is coarse: all current entry origins represent 12, candidate origins represent 13. The canonical current labels and unclear scopes are retained.')
    if ri==2: concerns.append('The binding status quote is labeled current in the original annotation despite comparing command estimates represented at 13 against copied current content represented at 12. The labels are preserved as supplied.')
    out['records'].append({'id':record['id'],'unit_id':record['unit_id'],'text_sha256':record['text_sha256'],
        'candidate_binding_reviews':[b_review],'additional_concerns':concerns,
        'notes':f'Complete first report, complete canonical annotation, full packet and provenance inspected for record {ri}. Actor text hash verified. Inventory: one original COD binding; no original CA binding. No other narrator record was inspected. Full first report: '+record['text']})

out['limitations'].extend([
    'Whole-prose factual fidelity is unestablished for every assigned narrator; no fidelity-conjoined narrative or experience conclusion follows.',
    'This assignment contains only three frozen development reports. Registered pair dispositions require the complete pair inventory and are outside this chunk; within-packet equalities do not establish those dispositions.',
    'No network, Git, model fitting, reporter/rater API, peer judgment, other target chunk, source evidence edit or collector feedback was used. Native provenance was assessed from the frozen packet and receipts; external archive execution was not performed.',
    'The optional arithmetic helper was not used; retained explicit independent arithmetic source/results provide the numerical verification.'
])

# Only closed-schema/binding integrity validation before exclusive first creation.
# This is not collector feedback and does not inspect result-source tests.
jsonschema.validate(out,json.loads((ROOT/'protocol/first_review_output_schema.json').read_text()))
assert len(out['records']) == len(chunk['records']) == 3
for r,s in zip(out['records'],chunk['records']):
    assert r['id']==s['id']
    for b,orig in zip(r['candidate_binding_reviews'],s['candidate_bindings']):
        assert b['binding_path']==orig['binding_path'] and b['original_status']==orig['original_status']
        assert b['source_meaning']['binding_evidence_paths']==orig['binding_evidence_paths']
        for f in b['feature_reviews']:
            assert f['evidence_paths']==orig['feature_evidence_paths'][f['feature']]
            assert f['original_status']==s['canonical_first_annotation']['features'][f['feature']]['status']
first=HERE/'first_review.json'
with first.open('x') as f:
    json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print(str(first));print(sha(first))
