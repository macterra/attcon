"""Synthetic transport/retention tests; no archives, source outputs or actual actors."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import same_address_model_report_workspace_v1 as workspace

PROTOCOL = ROOT/'audits/attention_relational_report_v1/same_address_pilot_v1/protocol'


def write(path, value):
    path.write_bytes(workspace.encoded(value))


def toy_content():
    field = {'node':'n3','role':'red','origin':{'kind':'history_estimate','represented_time':12,'feedback_through':11},
        'distribution':[[0.0625]*4 for _ in range(4)],
        'derived':{'peaks':[[0,0]],'peak_probability':0.0625,'entropy_nats':1.0,'expected_row':1.5,'expected_column':1.5}}
    future = deepcopy(field)
    future['origin'] = {'kind':'command_estimate','represented_time':13,'feedback_through':11,'executed_branch_claim':False}
    panel = {'panel':'p0','panel_role':'predictor_estimate','current':field,
        'candidates':{k:deepcopy(future) for k in ['k0','k1','k2']},
        'processing_scores':{'rule':'maximum supplied localization probability for the queried role; choose first command index on exact ties',
         'scores':{'k0':0.0625,'k1':0.0625,'k2':0.0625},'tied_winners':['k0','k1','k2'],'first_index_choice':'k0','top_two_margin':0.0}}
    reference = deepcopy(panel)
    reference.update(panel='p1',panel_role='native_reference')
    return {'schema':'same_address_model_report_source_v1_packet','decision_time':12,'selected_node':'n3','queried_role':'red',
        'panels':[panel,reference], 'coordinate_system':{'rows':4,'columns':4,'row_direction':'down','column_direction':'right','separate_local_grid':True},
        'field_semantics':'model values','conditional_semantics':'conditional estimates','processing_semantics':'operation scores'}


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path/'source';source.mkdir()
    inventory = workspace.read_json(PROTOCOL/'assignment_inventory.json')
    content = toy_content()
    packets = [{'id':r['design_opportunity_id'],'content':deepcopy(content),'content_sha256':workspace.digest(workspace.encoded(content))}
               for r in inventory['assignments']]
    write(source/'neutral_packets.json',packets)
    write(source/'first_receipt.json',{'schema':'same_address_model_report_source_v1_receipt','status':'completed',
        'opportunity_count':72,'pair_count':84,'artifact_sha256':{'neutral_packets.json':workspace.digest((source/'neutral_packets.json').read_bytes())}})
    release=tmp_path/'release.json'
    write(release,{'status':'released','released_by':'root','source_receipt_sha256':workspace.digest((source/'first_receipt.json').read_bytes())})
    out=tmp_path/'reporter'
    value=workspace.prepare_reporter(source,PROTOCOL,out,release)
    return out,value


def terminals(tmp_path, manifest, absent=()):
    path=tmp_path/'terminals.json'
    write(path,{'schema':'same_address_pilot_v1_root_terminal_observations','observed_by':'root','observations':[
        {'context_id':c['context_id'],'terminal':True,'state':'missing' if c['context_id'] in absent else 'completed',
         'actor_id':None if c['context_id'] in absent else 'toy_actor_'+c['context_id'],
         'observed_at_utc':'2026-10-20T00:00:00Z','evidence':'Synthetic terminal observation for transport test'}
        for c in manifest['contexts']]})
    return path


def save_envelope(context, mutate=None, response_factory=None):
    result={k:context[k] for k in ['schema','stage','stream','chunk','input_sha256','instructions_sha256','context_id']}
    result['responses']=[{'unit_id':u,'response':response_factory(u) if response_factory else {'text':'Supplied values remain unchanged.'}} for u in context['unit_ids']]
    if mutate: mutate(result)
    write(Path(context['first_path']),result)
    return result


def complete_reporters(out, manifest):
    for context in manifest['contexts']:save_envelope(context)


def test_preparer_blinds_packets_and_exclusively_claims_workspace(prepared):
    out, manifest=prepared
    assert len(manifest['contexts'])==12
    for c in manifest['contexts']:
        packet=workspace.read_json(Path(c['directory'])/'input.json')
        assert len(packet['units'])==6
        assert all(set(u)=={'unit_id','source_payload'} for u in packet['units'])
        assert all('design_opportunity_id' not in json.dumps(u) and 'variant' not in json.dumps(u) for u in packet['units'])
        assert 'reference.json' not in c['allowed_files']
        assert 'assignment_inventory.json' not in c['allowed_files']
    with pytest.raises(FileExistsError):workspace.make_workspace(out)


def test_terminal_gate_precedes_all_first_reads(prepared,tmp_path,monkeypatch):
    out, manifest=prepared
    complete_reporters(out,manifest)
    observed=terminals(tmp_path,manifest)
    value=workspace.read_json(observed);value['observations'].pop();write(observed,value)
    original=Path.read_bytes
    def guarded(path):
        if path.name=='first.json':pytest.fail('first response opened before terminal inventory complete')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',guarded)
    with pytest.raises(ValueError,match='all planned terminal'):workspace.collect_reporter(out/'manifest.json',observed,tmp_path/'collection.json')
    assert not (tmp_path/'collection.json').exists()


@pytest.mark.parametrize('damage',['hash','unitset','malformed','inner_extra'])
def test_wrong_binding_and_malformed_context_never_salvages_rows(prepared,tmp_path,damage):
    out,manifest=prepared
    complete_reporters(out,manifest)
    context=manifest['contexts'][0];path=Path(context['first_path'])
    value=workspace.read_json(path)
    if damage=='hash':value['input_sha256']='wrong'
    elif damage=='unitset':value['responses'][0]['unit_id']=value['responses'][1]['unit_id']
    elif damage=='inner_extra':value['responses'][0]['response']['extra']='unexpected'
    if damage=='malformed':path.write_bytes(b'{"incomplete":')
    else:write(path,value)
    result=workspace.collect_reporter(out/'manifest.json',terminals(tmp_path,manifest),tmp_path/'collection.json')
    assert result['contexts'][0]['status']=='unestablished'
    assert result['contexts'][0]['raw_base64']
    assert all(u['availability']=='unestablished' and u['response'] is None for u in result['units'][:6])
    assert all(u['availability']=='available' for u in result['units'][6:])
    with pytest.raises(FileExistsError):workspace.collect_reporter(out/'manifest.json',terminals(tmp_path,manifest),tmp_path/'collection.json')


def test_word_limit_is_per_unit_after_full_structural_envelope(prepared,tmp_path):
    out,manifest=prepared
    complete_reporters(out,manifest)
    path=Path(manifest['contexts'][0]['first_path']);value=workspace.read_json(path)
    value['responses'][0]['response']['text']=' '.join(['word']*121);write(path,value)
    result=workspace.collect_reporter(out/'manifest.json',terminals(tmp_path,manifest),tmp_path/'collection.json')
    assert result['contexts'][0]['status']=='structurally_recorded'
    assert result['units'][0]['availability']=='over_word_limit' and result['units'][0]['word_count']==121
    assert result['units'][1]['availability']=='available'
    assert result['units'][0]['feature_or_support_status'] is None


def test_missing_context_preserves_six_unknowns(prepared,tmp_path):
    out,manifest=prepared
    for c in manifest['contexts'][1:]:save_envelope(c)
    result=workspace.collect_reporter(out/'manifest.json',terminals(tmp_path,manifest,[manifest['contexts'][0]['context_id']]),tmp_path/'collection.json')
    assert result['contexts'][0]['raw_base64'] is None
    assert all(u['availability']=='unestablished' and u['feature_or_support_status'] is None for u in result['units'][:6])
    assert len(result['units'])==72


def test_review_preparation_keeps_semantics_sourceblind_and_missing_null(prepared,tmp_path):
    out,manifest=prepared
    for c in manifest['contexts'][1:]:save_envelope(c)
    observations=terminals(tmp_path,manifest,[manifest['contexts'][0]['context_id']])
    collection=tmp_path/'reporter_collection.json'
    workspace.collect_reporter(out/'manifest.json',observations,collection)
    reviews=tmp_path/'reviews'
    result=workspace.prepare_reviews(out/'manifest.json',collection,reviews)
    assert len(result['contexts'])==48
    for c in result['contexts']:
        frames=workspace.read_json(Path(c['directory'])/'input.json')['units']
        if c['stream'].startswith('S'):
            assert all(set(f)=={'unit_id','report_text'} for f in frames)
            assert 'reference.json' in c['allowed_files']
            assert 'factual_first_instructions.txt' not in c['allowed_files']
        else:
            assert all(set(f)=={'unit_id','report_text','source_payload','source_ownership'} for f in frames)
            assert all('variant' not in json.dumps(f['source_ownership']) and 'design_opportunity_id' not in json.dumps(f) for f in frames)
            assert 'reference.json' not in c['allowed_files']
        if c['chunk']==1:
            assert c['all_unavailable'] and all(f['report_text'] is None for f in frames)
        else:assert not c['all_unavailable']


def test_public_allowlist_rejects_private_metadata():
    content=toy_content();content['condition']='donor'
    with pytest.raises(ValueError,match='unexpected public'):workspace.public_content(content)
    content=toy_content();content['panels'][0]['current']['source']='A'
    with pytest.raises(ValueError,match='unexpected field'):workspace.public_content(content)


def test_json_pointer_resolution_is_literal_frame_bound():
    frame={'source_payload':{'grid':[[0.1,0.2]]},'escaped/key':{'a~b':1}}
    assert workspace.resolve_pointer(frame,'/source_payload/grid/0/1')==0.2
    assert workspace.resolve_pointer(frame,'/escaped~1key/a~0b')==1
    with pytest.raises(KeyError):workspace.resolve_pointer(frame,'/other_unit/source_payload')
    with pytest.raises(ValueError):workspace.resolve_pointer(frame,'source_payload/grid')


def review_annotation(stream, unit_id, text, bad_pointer=False):
    if stream.startswith('S'):
        absent={'status':'absent','evidence':[],'rationale':'No stronger structure asserted.'}
        return {'schema':'attention_relational_reference_v1_annotation','unit_id':unit_id,
            'features':{k:deepcopy(absent) for k in ['C','O','D','A']},
            'composites':{k:{'status':'absent','bindings':[],'rationale':'No compatible structure asserted.'} for k in ['COD','CA']},
            'auxiliary':{k:{'status':'absent','evidence':[]} for k in ['basic_spatial_relation','identity_or_location_uncertainty','answer_confidence','recovery_or_success_forecast','source_process_distinction','experiential_assertion']},'limitations':[]}
    operand={'operand_id':'o1','source_json_pointer':'/source_payload/missing' if bad_pointer else '/source_payload/panels/0/current/distribution',
        'operand_role':'actual-predictor-current','panel':'p0','actual_source_owner':'A','referent':'supplied item','address':'n3','attribute':'location',
        'represented_time':12,'origin':'history estimate','operation':None,'value_or_description':'supplied distribution'}
    return {'schema':'same_address_factual_first_v1','unit_id':unit_id,'report_availability':'complete',
        'claims':[{'claim_id':'c1','quote':text,'start':0,'end':len(text),'reading':text,
            'claimed':{'referent':'item','address':'n3','attribute':'location','panel_or_source':'p0','represented_time':12,
                       'origin':None,'modality':'assertion','uncertainty':''},
            'operands':[operand],'asserted_relation':'unchanged','support':'unsupported','reason':'Synthetic negative support judgment.',
            'calculation':None,'alternate_reading_ids':[]}],
        'coverage_segments':[{'start':0,'end':len(text),'kind':'claim-bearing','claim_ids':['c1'],'explanation':''}],
        'inventory_complete':True,'unresolved_boundaries':[],'whole_fidelity':'unsupported','whole_fidelity_reason':'Negative synthetic measurement.','limitations':[]}


def prepared_reviews_all_available(prepared,tmp_path):
    out,manifest=prepared
    complete_reporters(out,manifest)
    collection=tmp_path/'reporter_collection.json'
    workspace.collect_reporter(out/'manifest.json',terminals(tmp_path,manifest),collection)
    reviews=tmp_path/'reviews'
    result=workspace.prepare_reviews(out/'manifest.json',collection,reviews)
    return reviews,result


def test_review_collection_retains_negative_measurement_and_pointer_diagnostics(prepared,tmp_path):
    out,manifest=prepared_reviews_all_available(prepared,tmp_path)
    for i,c in enumerate(manifest['contexts']):
        frames=workspace.read_json(Path(c['directory'])/'input.json')['units']
        by_id={f['unit_id']:f for f in frames}
        save_envelope(c,response_factory=lambda uid,c=c,i=i:review_annotation(c['stream'],uid,by_id[uid]['report_text'],bad_pointer=(i==0)))
    result=workspace.collect_reviews(out/'manifest.json',terminals(tmp_path,manifest),tmp_path/'review_collection.json')
    assert len(result['contexts'])==48 and len(result['units'])==288
    assert result['contexts'][0]['status']=='unestablished'
    assert all(u['response'] is None and u['availability']=='unestablished' for u in result['units'][:6])
    assert result['contexts'][0]['parsed']['responses'][0]['response']['claims'][0]['operands'][0]['source_json_pointer']=='/source_payload/missing'
    assert result['units'][6]['response']['whole_fidelity']=='unsupported'
    assert result['units'][6]['availability']=='annotation_recorded'
    assert result['units'][6]['feature_or_support_status'] is None


def test_review_collection_waits_for_all_48_contexts_before_any_first_read(prepared,tmp_path,monkeypatch):
    out,manifest=prepared_reviews_all_available(prepared,tmp_path)
    observations=terminals(tmp_path,manifest)
    value=workspace.read_json(observations);value['observations'].pop();write(observations,value)
    original=Path.read_bytes
    def guarded(path):
        if path.name=='first.json':pytest.fail('peer annotations opened before all48 actual terminal records')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',guarded)
    with pytest.raises(ValueError,match='all planned terminal'):workspace.collect_reviews(out/'manifest.json',observations,tmp_path/'review_collection.json')


def test_semantic_inner_unit_id_mismatch_invalidates_entire_context(prepared,tmp_path):
    out,manifest=prepared_reviews_all_available(prepared,tmp_path)
    for c in manifest['contexts']:
        save_envelope(c,response_factory=lambda uid,c=c:review_annotation(c['stream'],uid,'Supplied values remain unchanged.'))
    semantic=next(c for c in manifest['contexts'] if c['stream']=='S1')
    path=Path(semantic['first_path']);value=workspace.read_json(path)
    value['responses'][0]['response']['unit_id']='unit_999';write(path,value)
    result=workspace.collect_reviews(out/'manifest.json',terminals(tmp_path,manifest),tmp_path/'review_collection.json')
    units=[u for u in result['units'] if u['context_id']==semantic['context_id']]
    assert len(units)==6 and all(u['availability']=='unestablished' and u['response'] is None for u in units)


def test_missing_prepared_context_cannot_shrink_terminal_denominator(prepared,tmp_path):
    out,manifest=prepared
    observed=terminals(tmp_path,manifest)
    changed=deepcopy(manifest);changed['contexts'].pop();write(out/'manifest.json',changed)
    with pytest.raises(ValueError,match='full planned context inventory'):workspace.collect_reporter(out/'manifest.json',observed,tmp_path/'collection.json')


def test_source_content_hash_requires_exact_newline(prepared,tmp_path):
    out,manifest=prepared
    source=Path(manifest['source_packets']['path']).parent
    packets=workspace.read_json(source/'neutral_packets.json')
    raw=json.dumps(packets[0]['content'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    packets[0]['content_sha256']=workspace.digest(raw);write(source/'neutral_packets.json',packets)
    receipt=workspace.read_json(source/'first_receipt.json')
    receipt['artifact_sha256']['neutral_packets.json']=workspace.digest((source/'neutral_packets.json').read_bytes());write(source/'first_receipt.json',receipt)
    release=tmp_path/'release2.json'
    write(release,{'status':'released','released_by':'root','source_receipt_sha256':workspace.digest((source/'first_receipt.json').read_bytes())})
    with pytest.raises(ValueError,match='source content hash differs'):workspace.prepare_reporter(source,PROTOCOL,tmp_path/'new_reporter',release)
    assert not (tmp_path/'new_reporter').exists()
