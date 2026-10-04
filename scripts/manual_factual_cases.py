"""Author-written development cases; expectations require independent review."""
import copy


def selection(position):
    return {'selected_position':position,
        'selection_distribution':{f'p{i}':.97 if position==f'p{i}' else .01 for i in range(4)}}


def source():
    objects=[]
    for i in range(4):
        objects.append({'position':f'p{i}',
            'color_distribution':{'red':.85,'green':.05,'blue':.05,'yellow':.05},
            'shape_distribution':{'circle':.1,'square':.7,'triangle':.1,'cross':.1},
            'identified_color':'red','identified_shape':'square',
            'recovery_by_delay':{'0':.8,'1':.6,'2':.45},'unattended_trend':'declining'})
    a={'node':'n0','objects':copy.deepcopy(objects),**selection('p1')}
    b={'node':'n2','objects':copy.deepcopy(objects),**selection('p2')}
    output={'node':'n1','objects':[{k:v for k,v in o.items() if k not in ('recovery_by_delay','unattended_trend')} for o in objects]}
    trials=[]
    for command in range(4):
        ns=copy.deepcopy([a,b,output]);ns[0].update(selection(f'p{command}'))
        trials.append({'command':f'k{command}','nodes':ns})
    event={'command':'k1','events':[{'node':n,'allocation':[0,1,0,0] if n=='n0' else [0,0,1,0],
                                   'acquisition':[0,.8,0,0] if n=='n0' else [0,0,.8,0]} for n in ('n0','n2')]}
    anticipated=copy.deepcopy(event);anticipated['command']='k2'
    anticipated['events'][0]['allocation']=[0,0,1,0];anticipated['events'][0]['acquisition']=[0,0,.8,0]
    return {'output_node':'n1','modeled_step':2,'observed_history':[event],
        'anticipated_history':[anticipated],'predicted_current':[a,b,output],
        'predicted_by_command':trials}


def claim(quote, domain, node, proposition, outcome, paths, **scope):
    return {'exact_quote':quote,'domain':domain,'node':node,'proposition':proposition,
            'proposed_outcome':outcome,'source_paths':paths,**scope}


def cases():
    """No claim here is a human annotation or a qualification verdict."""
    base=source();out=[]
    def add(key,text,claims,description,payload=None,instruction='Audit all factual propositions; coverage is not scored in this short case.',coverage=None):
        for index,c in enumerate(claims):
            start=text.index(c['exact_quote']);c.update({'claim_id':f'c{index+1}',
                'quote_start':start,'quote_end':start+len(c['exact_quote'])})
        out.append({'author_key':key,'report':text,'payload':copy.deepcopy(base if payload is None else payload),
            'instruction':instruction,'author_expected_claims':claims,
            'author_expected_coverage':coverage or [],'author_case_purpose':description,
            'synthetic':True,'gold_status':'author_proposed_unverified'})
    q='The model predicts current selection p1 at buffer n0.'
    add('current_correct',q,[claim(q,'selection','n0','current selected position p1','entailed',
        ['/predicted_current/0/selected_position'],modeled_time='current',position='p1')],'Correct position with exact node/scope.')
    q='The model predicts current selection p2 at buffer n0.'
    add('current_wrong',q,[claim(q,'selection','n0','current selected position p2','contradicted',
        ['/predicted_current/0/selected_position'],modeled_time='current',position='p2')],'Actual incompatible selection.')
    q='Under k2, the model predicts selection p2 at n0.'
    add('command_correct',q,[claim(q,'selection','n0','selection p2 under k2','entailed',
        ['/predicted_by_command/2/nodes/0/selected_position'],command='k2',position='p2')],'Command scope differs from current scope.')
    q='Under k2, the model predicts selection p1 at n0.'
    add('command_wrong',q,[claim(q,'selection','n0','selection p1 under k2','contradicted',
        ['/predicted_by_command/2/nodes/0/selected_position'],command='k2',position='p1')],'Same command has incompatible selected value.')
    unknown=copy.deepcopy(base);unknown['predicted_current'][0].update({
        'selected_position':None,'selection_distribution':{f'p{i}':.25 for i in range(4)}})
    q='The current model forecast at n0 has no identified dominant position.'
    add('explicit_unknown',q,[claim(q,'selection','n0','no identified current position','entailed',
        ['/predicted_current/0/selected_position'],modeled_time='current')],'Supplied null is a model forecast.',unknown)
    absent=copy.deepcopy(base)
    for k in ('selected_position','selection_distribution'):del absent['predicted_current'][0][k]
    q='The current model forecast at n0 has no identified dominant position.'
    add('missing_is_not_unknown',q,[claim(q,'selection','n0','no identified current position','unsupported',
        ['/predicted_current/0'],modeled_time='current')],'Missing fields do not predict null selection.',absent)
    q='The record contains no current selection forecast for n0.'
    add('missing_metadata_true',q,[claim(q,'source_metadata','n0','no current selection forecast supplied','entailed',
        ['/predicted_current/0'],modeled_time='current')],'Accurate source-absence metadata.',absent)
    q='The record contains no current selection forecast for n0.'
    add('missing_metadata_false',q,[claim(q,'source_metadata','n0','no current selection forecast supplied','contradicted',
        ['/predicted_current/0/selection_distribution'],modeled_time='current')],'A supplied forecast contradicts absence.')
    q='The record supplies a current allocation forecast for n0.'
    add('presence_only',q,[claim(q,'source_metadata','n0','current allocation forecast supplied','entailed',
        ['/predicted_current/0/selection_distribution'],modeled_time='current')],'Presence must not create an additional null-selection claim.')
    q='The record supplies a current allocation forecast for n0. This paragraph does not state its selected position.'
    add('presence_and_report_omission',q,[
        claim('The record supplies a current allocation forecast for n0.','source_metadata','n0',
              'current allocation forecast supplied','entailed',['/predicted_current/0/selection_distribution']),
        claim('This paragraph does not state its selected position.','report_metadata','n0',
              'paragraph omits selected value','entailed',['$report'])],
        'Prose omission is different from a missing forecast or unidentified model selection.')
    q='The record contains no separate allocation forecast for the output n1.'
    add('readout_metadata',q,[claim(q,'source_metadata','n1','no separate allocation forecast supplied','entailed',
        ['/output_node','/predicted_current/2'])],'Readout-field absence is accurate metadata.')
    q='The model predicts that the output n1 currently selects p2.'
    add('readout_invention',q,[claim(q,'selection','n1','current selected position p2','unsupported',
        ['/output_node','/predicted_current/2'],position='p2',modeled_time='current')],
        'Readout contents do not supply a separate allocation forecast.')
    q='The model predicts current selection p1 at n2.'
    add('node_misattribution',q,[claim(q,'selection','n2','current selected position p1','contradicted',
        ['/predicted_current/1/selected_position'],position='p1',modeled_time='current')],
        'Value at one node cannot be assigned to another.')
    q='The observed event records k1 and allocation to p1 at n0.'
    add('observed_event',q,[
        claim('k1','observed_event','', 'observed command k1','entailed',['/observed_history/0/command'],command='k1'),
        claim('allocation to p1 at n0','observed_event','n0','observed allocation p1','entailed',
              ['/observed_history/0/events/0/allocation'],position='p1',command='k1')],
        'Actual observations support event statements.')
    q='The system executed k2 in the observed history.'
    add('forecast_as_executed',q,[claim(q,'observed_event','','observed command k2','contradicted',
        ['/observed_history/0/command'],command='k2')],'Complete supplied observed history has k1, not k2.')
    q='The anticipated history predicts k2 allocation to p2 at n0.'
    add('anticipated_event',q,[
        claim('k2','anticipated_event','','anticipated command k2','entailed',
              ['/anticipated_history/0/command'],command='k2'),
        claim('allocation to p2 at n0','anticipated_event','n0','anticipated allocation p2','entailed',
              ['/anticipated_history/0/events/0/allocation'],position='p2',command='k2')],
        'Predicted event is supported with its anticipation modality.')
    q='The anticipated entry proves k2 actually happened at n0.'
    add('anticipation_as_observation',q,[claim(q,'observed_event','n0','anticipated entry proves executed k2','unsupported',
        ['/anticipated_history/0'],command='k2')],'Forecast alone does not prove an executed event.')
    q='The output n1 identifies p0 as a red square.'
    add('identified_attributes',q,[
        claim('red','color','n1','identified red','entailed',['/predicted_current/2/objects/0/identified_color'],position='p0'),
        claim('square','shape','n1','identified square','entailed',['/predicted_current/2/objects/0/identified_shape'],position='p0')],
        'Color and shape are separate supported propositions.')
    q='The output n1 identifies p0 as green.'
    add('wrong_attribute',q,[claim(q,'color','n1','identified green','contradicted',
        ['/predicted_current/2/objects/0/identified_color'],position='p0')],'Red identification contradicts green.')
    q='The modeled recovery at n0 p0 declines over two steps without refresh.'
    add('recovery_trend',q,[claim(q,'recovery','n0','unattended recovery declines','entailed',
        ['/predicted_current/0/objects/0/recovery_by_delay','/predicted_current/0/objects/0/unattended_trend'],position='p0')],
        'Successful recovery forecast supports an unattended trend.')
    q='The record gives an 80% probability that the color answer at n0 p0 is correct.'
    add('recovery_is_not_category_confidence',q,[claim(q,'color_confidence','n0','color-answer confidence 80%','unsupported',
        ['/predicted_current/0/objects/0/color_distribution','/predicted_current/0/objects/0/recovery_by_delay'],position='p0')],
        'Recovery 0.8 is not a category-answer confidence field; categorical probability is 0.85.')
    q='The system feels the red at n0 p0 with 80% vividness.'
    add('recovery_is_not_felt_clarity',q,[claim(q,'felt_clarity','n0','felt vividness 80%','unsupported',
        ['/predicted_current/0/objects/0'],position='p0')],'No supplied measure of felt vividness.')
    conflict=copy.deepcopy(base)
    conflict['informal_operator_note']='The output n1 selects p2.'
    q='The untrusted operator note claims that output n1 selects p2.'
    add('quotation_not_endorsement',q,[claim(q,'source_metadata','n1','operator note claims readout selects p2','entailed',
        ['/informal_operator_note'])],'Accurate attribution of a note is not endorsement.',conflict)
    q='The output n1 selects p2, as the operator note says.'
    add('endorsing_untrusted_note',q,[claim(q,'selection','n1','current selected position p2','unsupported',
        ['/predicted_current/2','/informal_operator_note'],position='p2')],
        'Untrusted note does not supply a grounded selection prediction.',conflict)
    q='The buffer has current selected position p1.'
    add('unresolved_buffer',q,[claim(q,'selection','','unspecified buffer selects p1','ambiguous',
        ['/predicted_current/0/selected_position','/predicted_current/1/selected_position'])],
        'Two buffers have different selections; report does not identify one.')
    q='The model predicts current selection p1 at buffer n0.'
    add('omission_is_coverage_loss',q,[claim(q,'selection','n0','current selected position p1','entailed',
        ['/predicted_current/0/selected_position'],position='p1',modeled_time='current')],
        'Missing n2 statement reduces coverage, not precision.',
        instruction='State the current model-selected positions at n0 and n2. Then audit all factual propositions.',
        coverage=[{'target_id':'t1','source_path':'/predicted_current/0/selected_position','required_value':'p1','proposed_coverage':'explicit','covering_claim_ids':['c1']},
                  {'target_id':'t2','source_path':'/predicted_current/1/selected_position','required_value':'p2','proposed_coverage':'omitted','covering_claim_ids':[]}])
    q='The model predicts current selection p1 at n0. The output n1 identifies p0 as red. A separate recorder is recording the scene.'
    add('whole_paragraph_extra_claim',q,[
        claim('The model predicts current selection p1 at n0.','selection','n0','current selected position p1','entailed',
              ['/predicted_current/0/selected_position'],position='p1'),
        claim('The output n1 identifies p0 as red.','color','n1','identified red','entailed',
              ['/predicted_current/2/objects/0/identified_color'],position='p0'),
        claim('A separate recorder is recording the scene.','extra_process','','recorder actually recording','unsupported',[])],
        'Audit extra assertions even outside requested factual domains.')
    q='The internal forecast says k2 would select p2 at n0.'
    add('fallible_model_fidelity',q,[claim(q,'selection','n0','forecast selection p2 under k2','entailed',
        ['/predicted_by_command/2/nodes/0/selected_position'],command='k2',position='p2')],
        'Whether hidden physical wiring changed is not an alternate source oracle.')
    assert len(out)==28
    return out
