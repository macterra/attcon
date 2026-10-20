import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, 'scripts')
import attention_relational_template_review_workspace_v1 as workspace

ROOT = Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review')
WORK = ROOT / 'judgments/reviewer_1/chunk_08/work'
CHUNK = ROOT / 'preparation/chunk_08.json'
SHA = '9d42eb4604fb29619be14b21681e582d1d9864dc34aa4d37a64030ebf89e0cf3'
chunk = workspace.bound_chunk(CHUNK, SHA)

# Explicit assessments below are authored after reading the complete reports,
# complete original inventories, and exact helper views of every used token.
# These constructors copy identities/values; they do not decide assessments.
def prop(r, index, path, role, reason):
    token = r['property_table'][index]
    value = workspace.resolve(token['value'], path)
    return dict(unit_id=token['unit_id'], value_path=path, value=value,
                semantic_role=role, reason=reason)

def peak(r, index):
    t = r['property_table'][index]
    return [prop(r, index, [], 'content_operand',
        t['text'] + ' supplies this complete peak set for the named role; the set is represented location content, without an inferred appearance or access quality.')]

def header(r):
    return [prop(r, 0, [], 'context_only',
        'The fixed renderer header supplies coordinate directions and announces current/conditional entries. Only the quoted clause is relevant; the containing header does not supply source-content or attention operands.')]

def origin(r, index, fields):
    t = r['property_table'][index]
    result = []
    for path, role, meaning in fields:
        result.append(prop(r, index, path, role,
            t['text'].split('/origin=')[0] + ': ' + meaning))
    return result

# Each individual nested field below is deliberately selected. No containing
# origin object is used as an undifferentiated semantic operand.
COPY = [(['kind'], 'status_operand', 'copied_prior_record describes record provenance, not content presentation/access quality.'),
        (['record_time'], 'context_only', 'record_time dates the copied record independently of represented time.'),
        (['represented_time'], 'context_only', 'represented_time anchors the represented current node.')]
HISTORY = [(['kind'], 'status_operand', 'history_estimate names an estimate origin; it does not itself state uncertainty or access quality.'),
           (['feedback_through'], 'processing_relation_operand', 'feedback_through identifies the history cutoff used by this estimate; no attention effect follows.'),
           (['represented_time'], 'context_only', 'represented_time anchors the estimate content in time.')]
COMMAND = [(['kind'], 'status_operand', 'command_estimate identifies conditional estimate provenance, not an allocation-to-presentation relation.'),
           (['feedback_through'], 'processing_relation_operand', 'feedback_through identifies the available-history cutoff for the estimate.'),
           (['represented_time'], 'context_only', 'represented_time places this candidate at the future represented step, not current content.')]
COMPLETED = [(['kind'], 'status_operand', 'completed_record identifies a completed native record; it does not assert that its content is perceptually presented or currently accessible.'),
             (['record_time'], 'context_only', 'record_time dates record completion.'),
             (['represented_time'], 'context_only', 'represented_time anchors the represented current node.')]
NATIVE = [(['kind'], 'status_operand', 'native_conditional_record names the conditional native record provenance.'),
          (['executed_branch_claim'], 'status_operand', 'false expressly withholds a claim that this future branch was executed.'),
          (['represented_time'], 'context_only', 'represented_time dates the future conditional record.')]
INHERITED = [(['kind'], 'status_operand', 'inherited_current describes copying current content into the conditional entry, not attention or a presentation change.'),
             (['represented_time'], 'context_only', 'the outer represented_time dates the conditional entry.'),
             (['inherited_origin','kind'], 'status_operand', 'the terminal copied_prior_record origin is W11 content even in the A panel.'),
             (['inherited_origin','record_time'], 'context_only', 'the nested record_time dates the copied W record.'),
             (['inherited_origin','represented_time'], 'context_only', 'the nested represented_time dates the inherited current content, distinct from the outer candidate time.')]

def comp(r, assessments):
    assert len(assessments) == len(r['components']) == 12
    return [dict(component_path=c['path'], original_status=c['recorded_status'],
                 assessment=a, notes=n) for c,(a,n) in zip(r['components'], assessments)]

def evid(r, assessments):
    assert len(assessments) == len(r['evidence'])
    rows = []
    for e,(grounding,source_meaning,properties,notes) in zip(r['evidence'],assessments):
        q=e['recorded_evidence']
        assert r['text'][q['start']:q['end']] == q['quote']
        rows.append(dict(evidence_path=e['path'], grounding=grounding,
             source_meaning=source_meaning, properties=properties,
             notes=notes + ' Recorded referent=' + q['referent'] +
             ', attribute=' + q['attribute'] + ', scope=' + q['scope'] +
             ', time=' + q['time'] + '.'))
    return rows

def binding(r, n, witnesses, notes):
    b=r['bindings'][n]
    lookup={e['evidence_id']:e['path'] for e in r['evidence']}
    return dict(binding_path=b['path'], feature_witnesses=[
        dict(feature=f,evidence_paths=[r['evidence'][j]['path'] for j in positions])
        for f,positions in witnesses],
        binding_evidence_paths=[lookup[x] for x in b['own_evidence_ids']],
        assessment='unsupported',notes=notes)

def base(r):
    return {k:r[k] for k in ('id','unit_id','text_sha256','reference_established')}

def review_2(r):
    h=header(r); red=peak(r,106); blue=peak(r,108)
    cur=origin(r,104,COMPLETED); future=origin(r,32,COMMAND)
    hist=origin(r,20,HISTORY)
    components=[
      ('supported','The p1/current/n1 red and blue entries identify represented roles and their identical four-cell peak sets at represented time 12.'),
      ('unsupported','completed_record at p1/current/n1 and command_estimate at p0/candidate/k0/n1 are provenance types. Neither states how the content is organized or accessible; the candidate is also time 13 and a different panel.'),
      ('unsupported','Listing p1/current/n1 red coordinates separately from completed_record differentiates content from record provenance, without a separately attributed organization/presentation/access status.'),
      ('supported','The entire report gives coordinate sets and record origins; command_estimate names a conditional estimate without an explicit allocation-to-content presentation/access relation.'),
      ('supported','Rows down and columns right explicitly describe the coordinate orientation.'),
      ('unsupported','The quoted current/conditional header announces two classes of entries. It does not express uncertain identity or location, so the recorded ambiguity is not grounded by that clause.'),
      ('supported','No probability of a response or confidence in an answer is stated; coordinate sets are not confidence ratings.'),
      ('supported','Conditional peak listings predict locations, without a statement of recovery or success likelihood.'),
      ('supported','p0/current/n3 is explicitly a history_estimate using feedback through 11 and representing 12, distinguishing estimate provenance from listed content.'),
      ('supported','The text makes no assertion of experience, awareness or felt appearance.'),
      ('unsupported','The red content is identified, but completed_record supplies provenance rather than O or D. The A-panel future command origin cannot supply a compatible current p1/n1 status.'),
      ('supported','No bound attention/content-presentation relation occurs; C alone does not supply CA.')]
    ev=[
      ('supported','source_specific',red+cur,'The complete red peak set is identified content. The adjacent origin shares the full p1/current/n1 prefix, binding the outside-quote time/provenance fields to this role. Indexed origin is native W.'),
      ('supported','source_specific',blue+cur,'The blue entry independently identifies the same four coordinates. The adjacent p1/current/n1 origin binds represented time 12 and native W provenance outside the quote.'),
      ('unsupported','source_specific',cur+red,'The origin and adjacent red entry share p1/current/n1 in the full text, so the content operand outside the quote is bound. Completed record provenance is not a presentation or access status.'),
      ('unsupported','source_specific',future+peak(r,34),'The full p0/candidate/k0/n1 prefix binds the adjacent red set to this A conditional estimate. This is future time 13; command_estimate supplies no organization/access predicate.'),
      ('unsupported','source_specific',red+cur,'The red coordinate set is content; the adjacent same-prefix origin is a record provenance field. Their separation does not express the D distinction required by the reference.'),
      ('unsupported','source_specific',cur+red,'The completed origin belongs to the same p1/current/n1 red content by the repeated full prefix, but describes record completion rather than a differentiated content access status.'),
      ('supported','global_convention',h,'The quoted first header clause explicitly fixes the two coordinate directions; it is a global renderer convention.'),
      ('unsupported','global_convention',h,'The quoted second header clause announces conditional entries, without uncertainty about any named content or location.'),
      ('supported','source_specific',hist+peak(r,22),'The history estimate explicitly uses feedback through 11 for represented time 12. The adjacent p0/current/n3 red set binds the outside-quote content to that origin; source resolution identifies A current estimate content.'),
      ('supported','source_specific',red+cur,'This binding position correctly supplies its red C operand. The same-prefix completed origin outside the quote anchors time 12, while O and D remain unsupported.'),
      ('unsupported','source_specific',cur+red,'This binding origin is linked to the red set by p1/current/n1, but record completion is not O/D. The literal overlap does not complete COD.')]
    out=base(r);out.update(component_reviews=comp(r,components),evidence_reviews=evid(r,ev),
      binding_reviews=[binding(r,0,[('C',[0]),('O',[]),('D',[])],
       'C/evidence/0 and the binding red quote match p1/current/n1 red at current represented time 12. O/0 and D/1 contain only completed_record provenance; O/1 concerns p0 candidate n1 at 13. No compatible qualifying O or D witness exists; the blue C quote is a different attribute.')],
      additional_concerns=['Record provenance is substituted for content access status in O/D/COD.','The header uncertainty quotation is a global convention.','Model scope is not perceptual appearance; no felt truth is assessed.'],
      notes='Established original retained. Exact coordinates, individual origin fields and duplicate evidence positions reviewed; original statuses are not repaired.')
    return out

def review_6(r):
    h=header(r);red=peak(r,4);blue=peak(r,24);cop=origin(r,2,COPY);hist=origin(r,20,HISTORY)
    components=[
      ('supported','p0/current/n0 red at (0,3) and p0/current/n3 blue at (1,2) identify represented content with location attributes.'),
      ('unsupported','The general current/conditional header and copied_prior_record time metadata describe listing conventions and record provenance, not content organization or access.'),
      ('unsupported','Separately printing the n0 origin and red coordinate separates provenance from content. It does not attribute a separate organization/presentation/access status.'),
      ('supported','No attention/allocation condition is related to a content presentation/access status. The candidate command labels and unchanged entries do not supply that relation.'),
      ('supported','The text explicitly says rows increase down and columns right.'),
      ('supported','There is no explicit identification/location uncertainty statement. A multiple-cell peak list is a list of maxima without a verbal uncertainty assertion.'),
      ('supported','No answer confidence or response-probability claim occurs.'),
      ('supported','The conditional coordinate entries do not claim recovery or task success.'),
      ('supported','The copied n0 record at time 11 and the n3 history estimate using feedback through 11 explicitly distinguish two origins for represented time 12.'),
      ('supported','No experiential assertion appears in any origin or coordinate entry.'),
      ('unsupported','The p0/current/n0 red C operand is explicit, but the origin is copied W11 provenance and the header is global. Neither supplies O or D for COD.'),
      ('supported','No attention-to-content organization/access relation is present, so the recorded CA absence is warranted.')]
    ev=[
      ('supported','source_specific',red+cop,'The singleton red peak identifies content. Its p0/current/n0 origin outside the quote shares the complete prefix and dates the represented content at 12; indexed terminal source is copied W11 despite the ordinary panel.'),
      ('supported','source_specific',blue+hist,'The singleton blue peak identifies content at p0/current/n3. The adjacent same-prefix origin binds represented time 12 and the ordinary history estimate provenance outside the quoted peak.'),
      ('unsupported','global_convention',h,'The second header clause announces current and conditional listings; it does not characterize a named content presentation or organization.'),
      ('unsupported','source_specific',cop+red,'The n0 origin binds the adjacent red content through their full p0/current/n0 prefix. Copied provenance and dates do not supply O.'),
      ('unsupported','source_specific',cop+red,'The same-prefix red content outside this quote is (0,3), but copied prior provenance is not a separately expressed presentation/access status for D.'),
      ('unsupported','source_specific',red+cop,'The red singleton with its adjacent copied origin states location and provenance, without the content/status distinction required by D.'),
      ('supported','global_convention',h,'The first header clause directly fixes row and column spatial directions.'),
      ('supported','source_specific',cop+red,'The current n0 content is explicitly copied from a record at 11 while representing 12. The adjacent red content binds that origin to this represented node; indexed provenance is W11.'),
      ('supported','source_specific',hist+blue,'History estimate origin, feedback cutoff and represented time explicitly describe processing provenance for n3. The adjacent blue entry binds the outside-quote content to this ordinary-source estimate.'),
      ('supported','source_specific',red+cop,'This COD position supplies the correct red C operand at p0/current/n0; its same-prefix origin dates it at 12. The composite still lacks O/D.'),
      ('unsupported','source_specific',cop+red,'The binding origin is bound to the red operand by the shared n0 prefix, but copied W11 provenance is not O/D.'),
      ('unsupported','global_convention',h,'The binding header position is only a global announcement, with no content-specific current n0 presentation predicate.')]
    out=base(r);out.update(component_reviews=comp(r,components),evidence_reviews=evid(r,ev),
      binding_reviews=[binding(r,0,[('C',[0]),('O',[]),('D',[])],
        'The first C witness supplies p0/current/n0 red at represented time 12; C/1 is n3 blue and cannot substitute. O/0 is global; O/1 and D/0 are n0 copied-record provenance; D/1 is the red coordinate itself. Those candidates fail the required meanings, leaving no qualifying O/D at the recorded model scope and current time.')],
      additional_concerns=['The ordinary panel n0 source resolves to copied W11, not ordinary-specific content.','The header is a renderer convention rather than a content presentation witness.','Coordinates and record provenance alone do not establish access or perceptual appearance.'],
      notes='Established original retained with every evidence-list position and binding; no status/scope upgrades.')
    return out

def review_9(r):
    h=header(r);cop=origin(r,2,COPY);cmd=origin(r,26,COMMAND);nat=origin(r,122,NATIVE);done=origin(r,104,COMPLETED)
    components=[
      ('supported','The report identifies p0 current n0 red across all sixteen coordinates, p0 k0/n0 red at (0,0), and p1 current n1 red at (0,2).'),
      ('unsupported','The grid orientation and current/conditional header are global conventions, not an organized or differentiated content presentation/access status.'),
      ('unsupported','Copied provenance and an explicit denial of branch execution distinguish source records, without a content organization/presentation/access status separate from identification.'),
      ('supported','Candidate values and command_estimate provenance do not explicitly relate allocation to content organization/access.'),
      ('supported','Rows down and columns right explicitly state coordinate orientation.'),
      ('supported','No verbal uncertainty about identity or location is asserted. Multiple maximum coordinates alone do not label uncertainty.'),
      ('supported','No response confidence, choice likelihood or answer certainty is given.'),
      ('supported','Future conditional peak sets are supplied without a recovery or success claim.'),
      ('supported','The text distinguishes copied current records from native conditional records and explicitly withholds execution of a future branch.'),
      ('supported','There is no claim of experience or felt appearance.'),
      ('unsupported','The recorded ambiguous COD is not warranted: its proposed O evidence is wholly general orientation/listing text, and provenance/execution status is not D. There is no original binding to review or repair.'),
      ('supported','With no attention/content-status relation, the recorded absent CA is warranted.')]
    ev=[
      ('supported','source_specific',peak(r,4)+cop,'The full sixteen-cell red peak set is identified represented content. The adjacent p0/current/n0 origin binds the outside-quote time and dates it as copied W11 at represented time 12.'),
      ('supported','source_specific',peak(r,28)+cmd,'The future red singleton is explicit content for p0/candidate/k0/n0. The same-prefix origin outside the quote fixes represented time 13 and A command-estimate provenance; it is not a current observation.'),
      ('supported','source_specific',peak(r,106)+done,'The p1/current/n1 red singleton identifies content. The adjacent same-prefix origin binds represented time 12 and native completed W provenance outside the quoted overlap.'),
      ('unsupported','global_convention',h,'Coordinate directions do not express a named content organization or access quality, regardless of recorded display scope.'),
      ('unsupported','global_convention',h,'The second header clause announces entry types rather than a content presentation or access status.'),
      ('unsupported','source_specific',cop+peak(r,4),'The current n0 content is bound by the shared prefix to this copied origin. Copying a record at 11 to represent 12 is provenance, not D content/access differentiation.'),
      ('unsupported','source_specific',nat+peak(r,124),'The native conditional origin binds its adjacent red sixteen-cell set through p1/candidate/k0/n0. Denying execution at represented time 13 is not a status of content presentation or accessibility.'),
      ('supported','global_convention',h,'This separate auxiliary evidence position correctly supplies the coordinate orientation convention.'),
      ('supported','source_specific',cop+peak(r,4),'The copied-record origin explicitly distinguishes current represented content provenance. Its neighboring n0 red set is bound by the same full prefix; this is copied W11 content even under p0 A.'),
      ('supported','source_specific',nat+peak(r,124),'The native conditional record explicitly says executed_branch_claim=false and represented_time=13. The same-prefix adjacent red content binds this outside-quote operand to the conditional record, not an executed future event.')]
    out=base(r);out.update(component_reviews=comp(r,components),evidence_reviews=evid(r,ev),binding_reviews=[],
      additional_concerns=['The original COD ambiguity has no binding and depends on header/provenance substitutions.','The source inventory includes copied W11, A conditional estimate, and native W content; no single favorable source is selected for the report as a whole.','No conditional native record establishes an executed branch.'],
      notes='Established original retained, including empty binding inventory and original COD ambiguous status. Semantic review rejects that application without replacing the original.')
    return out

def review_11(r):
    h=header(r);red=peak(r,4);blue=peak(r,6);cop=origin(r,2,COPY);cmd=origin(r,44,COMMAND);inh=origin(r,26,INHERITED)
    components=[
      ('supported','The current n0 red and blue singletons identify content. The k0/n3 red singleton also identifies content but is wrongly labeled current in its individual evidence; its origin represents time 13.'),
      ('unsupported','copied_prior_record, command_estimate and the generic current/conditional header supply provenance/listing categories, not organization/presentation/access status.'),
      ('unsupported','Printing a current red coordinate beside copied-record metadata does not separately characterize the content presentation/access status required for D.'),
      ('unsupported','The recorded A ambiguity relies on a generic header and inherited_current copied W11 content. Neither expresses an allocation/content-processing condition related to content organization/access; the candidate is time 13, not current.'),
      ('supported','The coordinate directions are explicitly stated in the header.'),
      ('supported','No verbal uncertainty in identity/location is stated; maximum-set enumeration does not itself assert uncertainty.'),
      ('supported','No answer confidence or response probability is present.'),
      ('supported','Conditional location listings are not recovery/success forecasts.'),
      ('supported','Copied current records and conditional command estimates are explicitly distinguished. The candidate origin is represented time 13 despite the original evidence time=current label.'),
      ('supported','No subjective experience assertion appears.'),
      ('unsupported','COD has current n0 red content but no qualifying O/D. Its blue quotation is incorrectly labeled red, adding an attribute mismatch to the provenance substitution.'),
      ('unsupported','The recorded CA ambiguity lacks a content-processing-to-presentation relation. Generic header language cannot bind such a condition to current n0 red.')]
    ev=[
      ('supported','source_specific',red+cop,'The current red singleton is identified content. Its adjacent same-prefix origin binds represented time 12 and copied W11 provenance outside the quote.'),
      ('supported','source_specific',blue+cop,'The blue singleton independently identifies content. The shared p0/current/n0 origin binds represented time 12 and copied W11 provenance outside the quote.'),
      ('unsupported','source_specific',peak(r,46)+cmd,'The red singleton is textually present, but the adjacent p0/candidate/k0/n3 origin explicitly places it at represented time 13. The recorded current time is incompatible; outside-quote origin fields are bound by the identical full prefix.'),
      ('unsupported','source_specific',cop+red,'The n0 origin binds the adjacent red content, but copied-record provenance is not its organization/access status.'),
      ('unsupported','source_specific',cmd+peak(r,46),'The command-estimate origin binds the adjacent n3 red content via the full candidate prefix. It supplies provenance, not O, and represented time 13 conflicts with the recorded current time.'),
      ('unsupported','global_convention',h,'The generic second header clause supplies no content-specific presentation/access predicate.'),
      ('unsupported','source_specific',cop+red,'The copied n0 origin is bound to the red singleton by the full prefix. Its record and represented dates differentiate provenance, without D content/access differentiation.'),
      ('unsupported','source_specific',red+cop,'The coordinate set is identification content. Its adjacent copied origin is not a separately expressed presentation/access status.'),
      ('unsupported','global_convention',h,'The header states that conditional peaks will be listed; no attention condition, content-processing relation or status effect is asserted.'),
      ('unsupported','source_specific',inh+peak(r,28),'The full p0/candidate/k0/n0 prefix binds inherited origin to its adjacent red singleton. Outer represented time is 13; inherited content is copied W11 recorded at 11 and represented at 12. Inheritance is not an attention-to-presentation relation and is not current at the outer candidate time.'),
      ('supported','global_convention',h,'The first header clause explicitly states the spatial coordinate convention, despite the auxiliary attribute being labeled unclear.'),
      ('supported','source_specific',cop+red,'Copied-record kind and its two dates explicitly describe n0 provenance. The adjacent same-prefix red content outside the quote is the content of that copied W11 record.'),
      ('unsupported','source_specific',cmd+peak(r,46),'The command-estimate origin genuinely supplies processing provenance for the adjacent same-prefix n3 red content, but it is a represented-time-13 candidate and the recorded time=current is wrong.'),
      ('supported','source_specific',red+cop,'The binding red position correctly supplies C for p0/current/n0 at represented time 12; adjacent same-prefix origin fields anchor its time outside the quote. O/D remain unsupported.'),
      ('unsupported','source_specific',cop+red,'The same-prefix copied origin concerns the binding red content but supplies provenance rather than O/D.'),
      ('unsupported','source_specific',blue+cop,'The literal quotation is blue at (0,1), whereas the recorded attribute and binding target are red. The adjacent origin anchors the same n0/time, but cannot make blue a red status witness.'),
      ('supported','source_specific',red+cop,'The CA binding position correctly identifies its current red C operand. The shared-prefix origin outside the quote anchors represented time 12; it supplies no A relation.'),
      ('unsupported','global_convention',h,'The CA header position lacks a named attention/content-processing condition or relation to current n0 red organization/access.')]
    out=base(r);out.update(component_reviews=comp(r,components),evidence_reviews=evid(r,ev),
      binding_reviews=[binding(r,0,[('C',[0]),('O',[]),('D',[])],
        'C/0 matches current p0/n0 red at represented time 12. The n0 copied origin is not O/D; the future n3 command origin is another node/time. The third binding quotation is blue despite attribute=red. No compatible qualifying O/D witness exists.'),
        binding(r,1,[('C',[0]),('A',[])],
        'C/0 identifies the target current red content at 12. A/0 and the binding header quote are global; A/1 describes inherited W11 content in a conditional entry representing 13, without an allocation-to-organization/access relation. No qualifying same-content/time A witness exists.')],
      additional_concerns=['Candidate k0/n3 content and origin are mislabeled current; represented time is 13.','COD binding evidence position 2 quotes blue but records attribute red.','Inherited A-panel content resolves to copied W11; generic header language is not A-content.'],
      notes='Every original evidence position retained, including duplicate red quotations and both original composite bindings. Individual time and attribute mismatches are preserved and assessed, not repaired.')
    return out

def review_18(r):
    h=header(r);red=peak(r,22);allred=peak(r,10);blue=peak(r,48)
    hist=origin(r,20,HISTORY);cmd=origin(r,44,COMMAND);cop=origin(r,2,COPY);cop1=origin(r,8,COPY)
    components=[
      ('supported','The current n3 red singleton, current n1 sixteen-cell red set, and future k0/n3 blue singleton explicitly identify represented content/location attributes.'),
      ('unsupported','The current prefix dates an entry; it does not assert accessibility. The all-cell maximum list is coordinate enumeration, and grid directions are global conventions. None is an explicit organization/presentation/access predicate.'),
      ('unsupported','History and command estimate origins describe provenance for current/future contents, without separately characterizing organization/presentation/access status.'),
      ('supported','No attention/allocation-to-content-status relation is asserted. Conditional command estimates alone are insufficient.'),
      ('supported','The first header clause explicitly fixes row and column directions; recorded time=unclear preserves this convention without a temporal inference.'),
      ('unsupported','history_estimate labels provenance, not explicit uncertainty in identity or location. The source gives singleton n3 peaks without a verbal uncertainty predicate.'),
      ('supported','No answer-confidence or response-certainty statement appears.'),
      ('supported','The candidate coordinate sets do not assert recovery or success probability.'),
      ('supported','p0/current/n0 is explicitly copied from a prior record, distinguishing its provenance from content coordinates and the n3 history estimate.'),
      ('supported','The report contains no experiential claim.'),
      ('unsupported','The n3 red coordinate is duplicated as identification and presentation status. Repetition does not supply O; the history origin does not supply D.'),
      ('supported','No same-content attention/access relation is present to supply CA.')]
    ev=[
      ('supported','source_specific',red+hist,'The current n3 red singleton identifies content. Its adjacent full-prefix origin binds represented time 12 and ordinary history-estimate provenance outside the quote.'),
      ('supported','source_specific',allred+cop1,'The n1 red entry identifies the complete sixteen-cell maximum set. The adjacent p0/current/n1 origin binds represented time 12 and copied W11 provenance outside the quote.'),
      ('supported','source_specific',blue+cmd,'The future candidate blue singleton is identified content. The neighboring p0/candidate/k0/n3 origin binds represented time 13 and ordinary command-estimate provenance outside the quote.'),
      ('unsupported','source_specific',red+hist,'The n3 red coordinate and its same-prefix origin identify current represented content; current in the prefix is a time/view label, not a presentation/access status.'),
      ('unsupported','global_convention',h,'The coordinate orientation is literal but global, without content-specific organization or appearance meaning.'),
      ('unsupported','source_specific',allred+cop1,'The all-sixteen-cell maximum set is coordinate enumeration. The adjacent same-prefix copied origin dates it at 12; neither supplies an expressed organized pattern beyond the list.'),
      ('unsupported','source_specific',hist+red,'The n3 history origin binds the adjacent red content at time 12 through their full prefix, but estimate provenance and feedback cutoff are not D presentation/access status.'),
      ('unsupported','source_specific',cmd+peak(r,46),'The candidate n3 command origin binds its adjacent red content at time 13 through the full prefix. It differentiates conditional estimate provenance, not content organization/access status.'),
      ('supported','global_convention',h,'This separate auxiliary position directly supports the coordinate directions. Its recorded unclear time is left unchanged.'),
      ('unsupported','source_specific',hist+red,'The history_estimate kind and feedback cutoff are provenance metadata; the adjacent same-prefix red singleton provides content but no explicit identity/location uncertainty.'),
      ('supported','source_specific',cop+peak(r,4),'The current n0 origin expressly dates copied prior record provenance. The adjacent n0 red entry binds the content outside the quote; indexed provenance is copied W11 despite the ordinary panel.'),
      ('supported','source_specific',red+hist,'This binding position correctly supplies the C red singleton at p0/current/n3, represented time 12 anchored by the adjacent same-prefix origin. The composite remains incomplete.'),
      ('unsupported','source_specific',red+hist,'This duplicate quotation has recorded role presentation_status; the coordinate and current prefix do not supply that status, even with the same-prefix history origin.'),
      ('unsupported','source_specific',hist+red,'This origin is bound to the n3 red singleton at 12 by the full prefix, but provenance does not supply the D distinction or missing O for the binding.')]
    out=base(r);out.update(component_reviews=comp(r,components),evidence_reviews=evid(r,ev),
      binding_reviews=[binding(r,0,[('C',[0]),('O',[]),('D',[])],
        'C/0 identifies p0/current/n3 red at represented time 12. O/0 repeats the coordinate under a presentation_status label; O/1 is global and O/2 is another node n1 with coordinate enumeration. D/0 and D/1 are current/future estimate provenance rather than presentation/access status. No compatible qualifying O/D witness exists; the duplicated binding red positions do not add a status predicate.')],
      additional_concerns=['The same red quotation occurs twice in the binding with different attributed roles; both positions are retained.','history_estimate alone is not explicit location uncertainty.','Copied W11 and ordinary-specific history/conditional values have distinct indexed origins.'],
      notes='Established original retained with all twelve components, all fourteen evidence positions, and its sole COD binding. No coordinates were converted into felt appearance or current access.')
    return out

REVIEWS={2:review_2,6:review_6,9:review_9,11:review_11,18:review_18}

for i,r in enumerate(chunk['records']):
    assert hashlib.sha256(r['text'].encode()).hexdigest() == r['text_sha256']
    if r['reference_established']:
        review=REVIEWS[i](r)
    else:
        # The full original diagnostic object is copied unchanged into notes;
        # its failed annotation is neither semantically re-rated nor salvaged.
        assert r['review_state']=='unestablished'
        assert r['components']==r['evidence']==r['bindings']==[]
        review=base(r)
        review.update(component_reviews=[],evidence_reviews=[],binding_reviews=[],
            additional_concerns=[],notes='unestablished; original derivation diagnostics preserved exactly: '+json.dumps(r['reference_measurement']['derivation'],ensure_ascii=True,sort_keys=True))
    draft=WORK / ('authored_%02d.json'%i)
    with draft.open('x') as f:
        json.dump(review,f,ensure_ascii=True,allow_nan=False,sort_keys=True,indent=2)
        f.write('\n')
    subprocess.run([sys.executable,'scripts/attention_relational_template_review_workspace_v1.py',
        '--chunk',str(CHUNK),'--chunk-sha256',SHA,'save-first-record',
        '--id',r['id'],'--input',str(draft),'--workspace',str(WORK)],check=True)

subprocess.run([sys.executable,'scripts/attention_relational_template_review_workspace_v1.py',
    '--chunk',str(CHUNK),'--chunk-sha256',SHA,'assemble-first','--workspace',str(WORK),
    '--envelope',str(ROOT/'first_envelopes/reviewer_1_chunk_08.json'),
    '--envelope-sha256','866e15acb749daa9915985288a93f892ac65e9e7f1c3ca1d479cd0e63e7aaf7f',
    '--output',str(ROOT/'judgments/reviewer_1/chunk_08/first_review.json')],check=True)
