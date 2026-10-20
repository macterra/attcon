import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'scripts')
import attention_relational_template_review_workspace_v1 as h
CHUNK='audits/attention_relational_report_v1/template_attribution_v1/semantic_review/preparation/chunk_09.json'
SHA='d57c72746c294a14f1b2f09a203a1aa9195dc2be051f6d3569084267f5da56a0'
WORK=Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review/judgments/reviewer_0/chunk_09/work')
c=h.bound_chunk(CHUNK,SHA)
# Each entry below is an authored judgment for the observed record, in original evidence order.
# Utilities only copy exact property values and preserve original inventory identifiers.
def flatten(v,p=()):
 if isinstance(v,dict):
  for k,w in v.items():yield from flatten(w,p+(k,))
 else:yield list(p),v

def operands(r,specs):
 out=[]
 for uid,role,why in specs:
  u=next(u for u in r['property_table'] if u['unit_id']==uid)
  for p,v in flatten(u['value']):
   out.append(dict(unit_id=uid,value_path=p,value=v,semantic_role=role,reason=why+' Exact field '+json.dumps(p)+' of '+uid+'.'))
 return out

def C(uid,why):return (uid,'content_operand',why)
def X(uid,why):return (uid,'context_only',why)
def S(uid,why):return (uid,'status_operand',why)

def E(g,m,n,*specs):return (g,m,n,list(specs))

component_notes={
2:[
('supported','The three C entries identify red peak-location sets at p0/current/n0, p0/current/n1 and p1/current/n3. Copied W11 and counterfactual W provenance do not erase these represented-content identifications.'),
('unsupported','Grid orientation and the introductory current/candidate heading supply conventions; copied_prior_record with record_time 11 and represented_time 12 supplies provenance. None of the cited O entries describes a content-specific organized pattern, presentation or accessibility beyond coordinates and metadata.'),
('unsupported','The n0 red coordinate and copied_prior_record fields are separately printed, but source age/copy provenance is not separately expressed organization/presentation/access status of red. Separation into JSON fields alone is insufficient for D.'),
('supported','The complete report lists candidate commands and command_estimate origins but makes no attention/allocation or functional processing relation to content organization or access.'),
('supported','Rows increase down and columns right explicitly state a spatial coordinate convention.'),
('supported','No identification or location is explicitly hedged as uncertain; the complete tied maxima are coordinate sets rather than a verbal uncertainty judgment.'),
('supported','There is no answer-confidence statement or numeric answer-confidence value.'),
('supported','Conditional coordinate sets do not claim successful recovery or an outcome likelihood.'),
('supported','Copied prior records at p0/n0 are explicitly contrasted with the history_estimate origin at p0/n3, with feedback and represented times named.'),
('supported','No passage asserts experience or felt appearance.'),
('unsupported','The n0 binding has identified red content but substitutes copied provenance and repeated coordinates for O and D. Its literal referent match does not establish the conjunction.'),
('supported','No compatible C/A relation is stated; the absent A status warrants absent CA.')],
9:[
('supported','Current n0 red has all sixteen peak cells and candidate k0/n2 red has four named cells. Both identify represented color-location content, even though the latter is conditional.'),
('unsupported','The cited grid direction is a convention; completed_record with time 12 is completion provenance. Neither is a content-specific presentation/access status or an organized pattern beyond coordinate listings.'),
('unsupported','The completed_record origin field identifies record provenance; it does not separately describe how n0 red is organized, presented or accessible.'),
('supported','Native conditional records and k0/k1/k2 labels do not state an attention or allocation relation to content organization/access.'),
('supported','The introductory row/down and column/right sentence explicitly supplies basic directionality.'),
('supported','No explicit identity/location uncertainty is reported; all-cell maxima are not by themselves uncertainty wording.'),
('supported','No confidence in an answer or selection appears.'),
('supported','Candidate n2 coordinates are conditional contents, with no recovery or success claim.'),
('supported','The nested inherited_current/completed_record metadata explicitly distinguishes a carried current record from its candidate representation. The annotation current time on this evidence remains problematic because the outer represented time is 13.'),
('supported','There is no experiential assertion in the full report.'),
('unsupported','C is present for current n0, but the origin and orientation evidence do not supply qualifying O/D for red in display_or_sensor_property scope.'),
('supported','CA absence follows from the absence of any attention-content organization/access relation.')],
10:[
('supported','Current n0 red (0,2) and blue (0,0), plus candidate k0/n1 red (0,0), identify represented colored locations. Candidate time 13 remains a forecast.'),
('unsupported','Copied_prior_record and command_estimate name the provenance of coordinate entries. Generic estimate/copy labels and times do not independently describe content organization, presentation or access.'),
('unsupported','The red location and copied_prior_record fields at n0 share a prefix, but the extra provenance field is not a separately attributed presentation/access status.'),
('supported','The complete A-only text contains command labels and forecast peaks but no semantic processing/allocation condition related to content organization/access.'),
('supported','The row/down and column/right sentence explicitly states spatial direction.'),
('unsupported','The cited command_estimate label says an entry is an estimate; it neither hedges red/blue identity or location nor states uncertainty. Explicit uncertainty cannot be inferred solely from this generic origin label.'),
('supported','The text contains no answer-confidence assessment.'),
('supported','No successful recovery or likelihood of success is asserted for the candidate coordinates.'),
('supported','Current copied_prior_record at n0 and history_estimate at n3 explicitly distinguish retained W11 provenance from an ordinary source estimate.'),
('supported','No experience claim is expressed.'),
('unsupported','The n0 red witness identifies content; its paired copied_prior_record witness does not establish qualifying O or D for that content.'),
('supported','No A-content relation is available, so absent CA is warranted.')],
13:[
('supported','Color-labeled current sets at p0/n0 and p1/n0 explicitly identify represented contents. Calling their scope model_access does not itself demonstrate accessibility.'),
('unsupported','The current copied_prior_record and candidate inherited_current fields are lineage/time metadata. They do not explicitly state access or a content organization pattern; candidate represented_time 13 is also mislabeled current in the second O entry.'),
('unsupported','The current red coordinate set and its copy origin are distinct fields, but they do not separately state organization/presentation/access status in the recorded model_access scope.'),
('supported','Even though the p0 panel resolves to A, the prose only lists origins and peaks; it does not articulate an attention/processing relation to organization or access.'),
('supported','Rows down/columns right is an explicit orientation convention; its unclear time need not be turned into a current content claim.'),
('supported','There is no explicit uncertainty judgment about content identity or location.'),
('supported','There is no reported answer confidence.'),
('supported','The candidate values make no recovery or success forecast.'),
('supported','The copy origin and source command_estimate distinguish source processes. The candidate command-estimate item has represented_time 13, so its recorded current time is not supported.'),
('supported','No passage claims experience or perceptual appearance.'),
('unsupported','The current n0 witnesses establish red content and copied lineage, without the O/D predicates or access claim required by the recorded COD binding.'),
('supported','Absent CA is warranted: neither an A panel label nor k0 suffices for a content-attention relation.')],
18:[
('supported','The full sixteen-cell red peak set at current n0 and the four-cell set at current n1 identify represented content. Their copied W11 origins do not establish model access.'),
('unsupported','Copy and inheritance metadata state how records are carried through time, not how the identified red content is organized/presented/accessed. The outer candidate time 13 remains conditional.'),
('unsupported','Separately listing n0 red and copied_prior_record provenance is not a separate organization/access status judgment about that red content.'),
('supported','History and command estimates in an A panel do not themselves state an attention/allocation/processing condition linked to emergence, presentation or access.'),
('supported','The row and column direction convention is explicitly stated.'),
('supported','The text makes no explicit uncertainty claim, despite multiple equal peak cells.'),
('supported','No answer-confidence statement is present.'),
('supported','The report supplies conditional locations without a success/recovery assertion.'),
('supported','History_estimate at current n3 versus command_estimate at candidate k0/n3 explicitly marks different source processes with represented times 12 and 13.'),
('supported','No experiential assertion appears.'),
('unsupported','The binding is confined to current n0 red; no qualifying organization/access or differentiation is stated by its copy origin.'),
('supported','No content-attention relation is stated; CA absence is supported.')],
20:[
('supported','Current n1 identifies red four-cell and blue four-cell location contents in the native W counterfactual record.'),
('unsupported','Both O witnesses are coordinate sets. Rebranding the coordinates organization_positions does not state a pattern beyond listing positions, as required by the frozen reference.'),
('supported','The separate record_status counterfactual_native_snapshot and executed_branch_claim false identify the representational modality of the same n1 content, distinguishing its identification from actual executed status. This is a limited model-level status distinction, not felt access.'),
('supported','The full native conditional and inherited records do not explicitly relate attention/allocation or content processing to organization/access.'),
('supported','The header explicitly names row and column orientation.'),
('supported','No explicit uncertainty is expressed about identity or position; the sets are stated directly.'),
('supported','There is no answer-confidence assertion.'),
('supported','The text does not forecast successful recovery or likelihood of success.'),
('supported','Nested inherited_current and inherited counterfactual record metadata explicitly describe lineage. The evidence current time conflates an outer candidate time 13 with inherited represented time 12.'),
('supported','No experience or felt appearance is claimed.'),
('unsupported','C and a limited counterfactual status distinction are available for n1, but the original O evidence consists entirely of coordinate listings. The explicit COD conjunction therefore lacks a qualifying O witness.'),
('supported','CA is absent because no attention-content relation is given.')]
}

judgments={}
# Record 2: copied W11 red in an ordinary panel, contrasted with source history; native W counterfactual content.
judgments[2]=[
E('supported','source_specific','Current p0/n0 red locations are explicitly the singleton (0,3). The adjacent origin, outside this quote but bound by p0/current/n0, fixes W11 copy provenance and represented time 12.',C('u0004','Identified red-location content.'),X('u0002','Same n0 prefix binds provenance/time outside the red quote.')),
E('supported','source_specific','Current p0/n1 red is the complete sixteen-cell peak set. Its own n1 origin, outside this quote, binds copied W11 at represented time 12; the all-cell set is not a clarity or uncertainty rating.',C('u0010','Complete n1 red peak content.'),X('u0008','Adjacent same-n1 origin binds current represented time and W11 provenance.')),
E('supported','source_specific','p1/current/n3 red explicitly names the four upper-left cells. The same n3 origin outside the quote is a counterfactual native W snapshot representing 12, archived at 13 with no executed-branch claim.',C('u0118','Native W counterfactual n3 red content.'),X('u0116','Same n3 prefix binds the quoted red content to counterfactual/time metadata.')),
E('unsupported','global_convention','The grid/directionality quote is a partial header token: it sets row/down and column/right convention, not presentation of identified content.',X('u0000','Authored orientation convention only; no content-specific O operand.')),
E('unsupported','global_convention','The second partial header announces current and candidate peaks; it does not assert their accessibility. Generic current/candidate wording cannot supply the recorded model_access presentation status.',X('u0000','Generic authored listing convention, not a source access witness.')),
E('unsupported','source_specific','For p0/current/n0, copied_prior_record plus 11/12 distinguishes record and represented times, but says nothing about how red is organized or accessible. The adjacent red operand is bound by the exact n0 prefix, not merely proximity.',X('u0002','Actual copy provenance/time fields alleged to be O.'),C('u0004','Same n0 red content outside the origin quote; no qualifying O predicate connects it.')),
E('unsupported','source_specific','The n0 origin is provenance, not a separate presentation/access status of the red set. Same-prefix binding to red is explicit, but it does not make source lineage D.',X('u0002','Copy provenance considered for the recorded D role.'),C('u0004','Same n0 red identification outside the origin quote.')),
E('unsupported','source_specific','The singleton red coordinate supplies content only. Its adjacent n0 copy origin is included as the proposed separate status operand, but remains provenance rather than content-organization/access differentiation.',C('u0004','Red content quoted for D.'),X('u0002','Same-prefix proposed status field outside the red quote.')),
E('supported','global_convention','The partial header literally states grid directionality at the recorded display scope; this supports basic spatial relation without acquiring O.',X('u0000','Row/down and column/right convention.')),
E('supported','mixed_sources','p0/n0 is copied_prior_record whereas p0/n3 is history_estimate with feedback through 11. The latter is outside this quote but is the explicitly printed contrasting origin in the full report; indexed provenance resolves copied W11 versus ordinary source current.',X('u0002','Copied W11 side of the source-process distinction.'),X('u0020','Contrasting current n3 history estimate outside the quote; explicitly named in the second auxiliary witness.')),
E('supported','mixed_sources','p0/n3 history_estimate is a current estimate representing 12 using feedback through 11. Its contrast with the explicitly listed n0 copied_prior_record concerns source process, not the organization/access of red.',X('u0020','Ordinary source-current estimate side.'),X('u0002','Contrasting n0 copy origin outside this quote, named in the first auxiliary witness.')),
E('supported','source_specific','Binding evidence position 0 supplies the current n0 red content operand. Its W11 copy origin is bound outside the quote by the identical n0 prefix. This supports C within the proposed binding, while O/D remain unsupported.',C('u0004','Binding current n0 red content.'),X('u0002','Binding same n0 provenance/time outside this content quote.')),
E('unsupported','source_specific','Binding position 1 supplies only n0 copy provenance. Its explicit shared prefix connects it to red, but no content-presentation/access predicate establishes O or D.',X('u0002','Proposed binding O/D field is actual copy metadata.'),C('u0004','Binding red operand outside this origin quote.')),
E('unsupported','source_specific','Binding position 2 duplicates the red quote under attribute status. The literal (0,3) is a location rather than a distinct representation/access status; retaining this duplicate cannot supply D.',C('u0004','Duplicate red content relabeled as status.'),X('u0002','Same n0 copy origin outside the duplicate quote, insufficient to turn coordinate into status.'))]
# Record 9.
judgments[9]=[
E('supported','source_specific','Current n0 red names all sixteen peak cells. Its same-n0 completed_record origin outside the quote gives record and represented time 12; the set does not assert all cells are red in the world.',C('u0004','Complete current n0 red peak content.'),X('u0002','Same-n0 completed record/time context outside quote.')),
E('supported','source_specific','Candidate k0/n2 red is the four upper-left peak cells. The adjacent k0/n2 origin outside the quote binds it to native_conditional_record at 13 and executed_branch_claim false. Recorded unclear time is retained; it cannot be read as current execution.',C('u0040','Conditional native W red content.'),X('u0038','Same k0/n2 prefix binds conditional represented time and nonexecution.')),
E('unsupported','global_convention','Grid orientation is an authored convention. It does not state how the identified n0 red content is presented or accessible.',X('u0000','Partial orientation header used for O.')),
E('unsupported','source_specific','Completed_record at current n0 fixes record provenance/time; no organization/access or non-coordinate pattern is stated. The n0 red set outside the quote provides the proposed content binding, not a missing O predicate.',X('u0002','Completed/time metadata alleged to be O.'),C('u0004','Same-n0 red content outside the origin quotation.')),
E('unsupported','source_specific','The completion status of a stored record is not itself a presentation/access rating of its identified red content. The separate n0 red set is bound by the common prefix but does not alter that meaning.',X('u0002','Completion provenance proposed as D.'),C('u0004','Identified red content outside this D quote.')),
E('supported','global_convention','The row/column direction convention is literal and supports the spatial auxiliary at its recorded display scope.',X('u0000','Complete header token contains the exact orientation quotation.')),
E('unsupported','source_specific','The nested fields do distinguish inherited_current from its completed_record origin, but the outer represented time is 13 and the inherited time is 12. Recorded current time and model_organization do not turn this candidate lineage into current content organization.',X('u0026','Each nested origin/time field specifies the inheritance relation; no containing-object blanket inference.')),
E('supported','source_specific','Binding position 0 literally identifies current n0 red peak content. It shares n0 with completed_record time 12 outside the quote; this supports its C role only.',C('u0004','Binding red content.'),X('u0002','Same n0 completed/time context outside content quote.')),
E('unsupported','source_specific','The binding origin quote names completion provenance, not O or D of red. Same referent/current time is insufficient to make an explicit COD binding.',X('u0002','Completion metadata offered as binding status/organization.'),C('u0004','Same-n0 red operand outside origin quote.'))]
# Record 10.
judgments[10]=[
E('supported','source_specific','p0/current/n0 explicitly identifies red at (0,2). Same-prefix copy origin outside the quote supplies W11 record time 11 and represented time 12.',C('u0004','Current red location content.'),X('u0002','Bound n0 W11/time context outside quote.')),
E('supported','source_specific','p0/current/n0 explicitly identifies blue at (0,0). The same node origin binds this blue content to copied W11, independently of the ordinary panel label.',C('u0006','Current blue location content.'),X('u0002','Same n0 origin outside the blue quote.')),
E('supported','source_specific','Candidate k0/n1 red names (0,0) at forecast represented time 13, using the command_estimate origin at the identical k0/n1 prefix outside this quote. It is not a current outcome.',C('u0034','Future red estimate content.'),X('u0032','Same k0/n1 command estimate and feedback/time context outside quote.')),
E('unsupported','source_specific','Current n0 copied_prior_record describes retention provenance. Its explicit link to n0 red, outside the quote, supplies no additional presentation/access or organized-pattern predicate.',X('u0002','Copy origin fields alleged to supply O.'),C('u0004','Same n0 red operand outside origin quote.')),
E('unsupported','source_specific','Command_estimate, feedback through 11 and represented time 13 identify the source and timing of k0/n1 red. A generic estimate label does not express its organization/presentation/access.',X('u0032','Candidate estimate metadata alleged to supply O.'),C('u0034','Same k0/n1 red content outside origin quotation.')),
E('unsupported','source_specific','The current red singleton is content, not a separately attributed organization/access status. The same n0 copy-origin fields outside this quote do not supply that missing distinction.',C('u0004','Red location used in proposed D.'),X('u0002','Explicit same-n0 provenance outside content quote.')),
E('unsupported','source_specific','Copy provenance and its 11/12 times do not distinguish identified red from presentation/access status. The full text binds them to n0 red but adds no qualifying status predicate.',X('u0002','Provenance used for the separate-field D claim.'),C('u0004','Same-n0 red operand outside this quotation.')),
E('supported','global_convention','Rows down/columns right is the exact spatial convention stated in the header; its source is global renderer convention.',X('u0000','Authored spatial direction convention.')),
E('unsupported','source_specific','The k0/n1 command_estimate label at represented time 13 is not an explicit identity/location uncertainty assertion. Bound red (0,0) outside this quote is unhedged; no uncertainty magnitude or alternative identification is stated.',X('u0032','Exact generic estimate fields used as uncertainty evidence.'),C('u0034','Same k0/n1 red content outside quote shows what is being estimated, without an explicit uncertainty claim.')),
E('supported','mixed_sources','Current n0 is a copied W11 record; current n3 is an ordinary history estimate. Both origins are explicit in the report, with the second auxiliary evidence quoting n3; their contrast concerns process provenance.',X('u0002','Copy provenance side.'),X('u0020','Contrasting ordinary history origin outside this quote, explicitly named in the second auxiliary witness.')),
E('supported','mixed_sources','Current n3 history_estimate uses feedback through 11 to represent 12, while the n0 copied_prior_record explicitly named elsewhere represents 12 from W11. These are distinct origins, not differing felt presentations.',X('u0020','History-estimate side of the distinction.'),X('u0002','Contrasting W11 copy origin outside this quote, named by first auxiliary witness.')),
E('supported','source_specific','Binding red content at current n0 is explicit. Its copy-origin metadata outside the quote shares the exact referent and represented time, but only supports contextual lineage.',C('u0004','Current n0 red binding operand.'),X('u0002','Same-n0 provenance/time binding outside quotation.')),
E('unsupported','source_specific','The n0 copied_prior_record binding entry is not qualifying organization/access or content-status differentiation. Red (0,2) outside the quote shares its prefix but cannot supply an O predicate.',X('u0002','Binding provenance proposed as O/D.'),C('u0004','Same n0 red content outside origin quote.'))]
# Record 13: source A panel contains copied W11 operands; the candidate source estimate is distinct.
judgments[13]=[
E('unresolved','source_specific','The red four-cell content at current p0/n0 is explicit. Its same-prefix origin outside the quote fixes copied W11 represented at 12. C identification is warranted, but the recorded model_access scope is not independently stated by this location list.',C('u0004','Current n0 red content.'),X('u0002','Same-n0 W11 copy/time context outside red quotation.')),
E('unresolved','source_specific','The blue four-cell content at current p0/n0 is literal. Copy origin at the same n0 prefix outside this quote binds time 12 and W11 provenance; a model_access claim cannot be inferred from the blue coordinates.',C('u0006','Current n0 blue content.'),X('u0002','Same-n0 origin binds blue content outside this quotation.')),
E('unresolved','source_specific','p1/current/n0 explicitly identifies the same four red cells in a native completed W record. The adjacent n0 origin outside the quote establishes represented time 12. It does not assert model access, so that recorded scope remains unresolved.',C('u0100','Native W n0 red content.'),X('u0098','Same p1/current/n0 completion/time origin outside red quotation.')),
E('unsupported','source_specific','Copied_prior_record with record time 11 and represented time 12 is n0 lineage. Same-prefix red outside the origin quote binds its content but not a model_access organization/presentation/access predicate.',X('u0002','Actual current copied-record fields alleged to be O.'),C('u0004','Explicit same n0 red operand outside origin quotation.')),
E('unsupported','source_specific','The exact nested inherited_origin fields preserve a copied W11 record representing 12, while outer inherited_current represents 13. Recorded current time conflicts with that candidate and no access status or non-coordinate organization is stated. The candidate n1 red set outside the quote is bound by the identical prefix.',X('u0032','Individual nested and outer lineage/time fields.'),C('u0034','Same candidate k0/n1 red content outside origin quote; retained W11 content.')),
E('unsupported','source_specific','The red coordinate set identifies content without differentiating its access/presentation status. Its n0 origin outside this quote identifies a copy, not a qualifying model_access status.',C('u0004','Red content cited for D.'),X('u0002','Same-n0 copy origin outside the D content quote.')),
E('unsupported','source_specific','Origin and red are separately printed at p0/current/n0, but copied_prior_record denotes source lineage rather than red accessibility or presentation. Field separation alone cannot establish D.',X('u0002','Copy metadata cited as separate D status.'),C('u0004','Same-n0 red identification outside origin quotation.')),
E('supported','global_convention','The row/down and column/right convention literally supports grid orientation, at display_or_sensor_property scope and the recorded unclear time.',X('u0000','Global grid orientation convention, only a partial header quotation.')),
E('supported','mixed_sources','p0/current/n0/origin is copied W11, whereas the separately quoted p0/candidate/k0/n0/origin is an A command estimate. Their explicitly named origins distinguish provenance even though the panel itself is A; the candidate counterpart is outside this quote.',X('u0002','Copied W11 origin in the A panel.'),X('u0026','Contrasting A command estimate outside quotation, explicitly named by the paired auxiliary evidence.')),
E('unsupported','mixed_sources','The source process contrast is literal: this k0/n0 command estimate resolves to A and the named current n0 copy resolves to W11. But represented_time 13 is a future candidate, not the recorded current time; the present annotation cannot be silently repaired.',X('u0026','Candidate A estimate kind, feedback and represented time.'),X('u0002','Named current n0 copied W11 contrast outside quotation.')),
E('unresolved','source_specific','Binding position 0 supplies literal current n0 red content with its same-prefix W11/time origin outside the quotation. Its C role is identifiable, while the binding model_access scope is not asserted by the coordinates.',C('u0004','Binding red content operand.'),X('u0002','Same n0 copy/time context outside binding content quote.')),
E('unsupported','source_specific','The n0 copy-origin binding entry is not a qualifying organization/access or differentiation statement. Both the alleged O/D predicates and the recorded model_access claim are unsupported despite the exact red/n0 prefix match.',X('u0002','Binding copy lineage alleged to be status/organization.'),C('u0004','Same n0 red content outside the origin quote.'))]
# Record 18: all cited current content is W11, although A supplies the history/command estimates.
judgments[18]=[
E('unresolved','source_specific','The full sixteen-cell current n0 red peak set explicitly identifies represented content. Its n0 origin outside the quote fixes W11 time 11 represented at 12. The list does not establish the recorded model_access scope.',C('u0004','Complete n0 red content.'),X('u0002','Same-n0 copied W11 origin outside quotation.')),
E('unresolved','source_specific','The four upper-left red cells at current n1 are explicit. Adjacent n1 copy origin outside the quote binds W11 time 11 represented at 12; no access assertion follows from these coordinates.',C('u0010','Current n1 red content.'),X('u0008','Same-n1 origin outside this quotation provides copy/time context.')),
E('unsupported','source_specific','For current n0, copied_prior_record and 11/12 times are source provenance, not how red is organized/presented/accessed. The red operand outside this quote is explicitly bound by p0/current/n0.',X('u0002','Copy/time origin alleged to be model_access O.'),C('u0004','Same-n0 red content outside the origin quotation.')),
E('unsupported','source_specific','Candidate k0/n0 inherited_current represents 13 while its nested copied prior origin represents 12 from record time 11. This is forecast lineage of W11 content, not an access/presentation status. The same-prefix candidate red set outside the quote confirms the content operand without adding O.',X('u0026','Exact nested copy and outer candidate inheritance fields.'),C('u0028','Same candidate k0/n0 red content outside origin quotation.')),
E('unsupported','source_specific','The sixteen-cell red set supplies content only. Its same-n0 copy origin outside this quote is a separate provenance field, not differentiated presentation/access status.',C('u0004','N0 red set cited for D.'),X('u0002','Proposed separate copy status outside the content quotation.')),
E('unsupported','source_specific','The n0 origin field records copied lineage and time. Its association with the same-n0 red set outside the quote is explicit, but no content-access/presentation status is separately attributed.',X('u0002','Copy provenance cited for D.'),C('u0004','Same-n0 red identification outside origin quotation.')),
E('supported','global_convention','The quoted row/down and column/right sentence states a basic spatial convention at display_or_sensor_property scope. It is a partial header, with no A-content witness.',X('u0000','Exact global orientation convention.')),
E('supported','source_specific','History_estimate for current n3 represents 12 with feedback through 11; the full report and paired auxiliary quote name command_estimate at k0/n3 representing 13. Indexed provenance resolves both to A. This supports process differentiation only, not actual access.',X('u0020','A current history-estimate origin.'),X('u0044','Explicitly named contrasting A command-estimate origin outside quotation.')),
E('supported','source_specific','Candidate k0/n3 command_estimate is explicitly a forecast representing 13 using feedback through 11; it contrasts with the named current n3 history estimate at 12 outside this quote. No executed action or felt access is claimed.',X('u0044','A conditional command-estimate origin.'),X('u0020','Paired named current history estimate outside quotation.')),
E('unresolved','source_specific','Binding position 0 supplies current n0 red content and binds to the same-n0 copy origin outside the quote. C content is literal, but the recorded model_access scope is not an access assertion in the text.',C('u0004','Current n0 red binding operand.'),X('u0002','Same-n0 W11/time context outside binding red quotation.')),
E('unsupported','source_specific','Binding position 1 is W11 copy provenance, even inside the A panel. It does not provide organization/access or D for n0 red in the recorded model_access scope.',X('u0002','Actual binding copy-origin fields.'),C('u0004','Same-n0 red content outside origin quote.'))]
# Record 20: counterfactual status is explicit, but the original O witnesses are coordinates only.
judgments[20]=[
E('supported','source_specific','Current n1 red identifies the four upper-left peak cells. Its adjacent origin outside the quote explicitly marks counterfactual native W status, represented 12, archived 13 and executed_branch_claim false. It does not identify an actual executed branch.',C('u0010','Native W n1 red content.'),X('u0008','Same-n1 counterfactual/time context outside red quotation.')),
E('supported','source_specific','Current n1 blue identifies the four upper-right peak cells. The identical node prefix binds it to the counterfactual native W origin outside the quotation; neither physical world truth nor execution follows.',C('u0012','Native W n1 blue content.'),X('u0008','Same-n1 counterfactual/time origin outside blue quotation.')),
E('unsupported','source_specific','The four red coordinates are a location set. The annotation organization_positions does not make their listing an explicit spatial/constituent pattern beyond positions; the reference excludes this route to O.',C('u0010','Actual red coordinates alleged to supply organization.')),
E('unsupported','source_specific','The four blue coordinates are likewise a location set rather than a stated content-presentation or organized-pattern predicate. Their relation to the red set cannot be invented as a new O witness.',C('u0012','Actual blue coordinates alleged to supply organization.')),
E('supported','source_specific','This red quote identifies content; the full text explicitly gives the same n1 a separate record_status counterfactual_native_snapshot and executed_branch_claim false outside this quotation. Those fields distinguish identified content from representational/execution status at represented time 12, only in the limited model sense.',C('u0010','N1 red content side of D.'),S('u0008','Same-n1 separately stated counterfactual and execution status outside red quote; times retain the 12/13 distinction.')),
E('supported','source_specific','The origin explicitly labels n1 counterfactual and denies an executed-branch claim. The same-n1 red content outside the quote supplies the identified operand. Model_access here is limited to represented record modality, not felt access or a present executed event.',S('u0008','Separate counterfactual/execution/represented-time status fields.'),C('u0010','Same-n1 red identification outside origin quotation.')),
E('supported','global_convention','Rows down/columns right supports grid orientation at display scope with unclear time. It is an authored convention rather than a native W semantic witness.',X('u0000','Global orientation convention from partial header.')),
E('unsupported','source_specific','The inheritance relation is literal, but outer inherited_current represented_time 13 contains an inherited counterfactual snapshot represented at 12 and archived at 13 with executed_branch_claim false. Calling this candidate current conflates the distinct times; all nested fields are retained individually.',X('u0026','Outer inheritance and each nested counterfactual/time field used to assess the recorded current lineage claim.')),
E('supported','source_specific','Binding position 0 identifies n1 red. Same-prefix counterfactual origin outside the quotation contextualizes its current represented time and nonexecution; it does not repair missing O.',C('u0010','Binding red content.'),X('u0008','Same-n1 origin context outside binding content quote.')),
E('unsupported','source_specific','Binding position 1 duplicates the red quotation as organization_positions. The duplicate still only lists positions and supplies no qualifying O predicate.',C('u0010','Duplicate red coordinate content relabeled organization.')),
E('supported','source_specific','Binding position 2 explicitly provides the counterfactual/execution status of the same n1 red operand outside the quote, supporting limited D. Its model_access scope is a record-modality interpretation and cannot establish perceptual appearance or replace the absent O witness.',S('u0008','Binding counterfactual record status and nonexecution fields.'),C('u0010','Same-n1 red content outside the origin quotation.'))]

binding_notes={
2:'The only matching C witness for the proposed p0/current/n0 red_locations/current binding is features/C/evidence/0. Its singleton red and copied prior origin refer to n0 representing 12, but O header/origin entries and D origin/red entries do not express qualifying organization/presentation/access differentiation. Accordingly there are no compatible O or D witnesses. p0/n1 and p1/n3 C entries concern other contents and are excluded. All three own evidence paths, including the duplicate coordinate relabeled status, are retained.',
9:'C/evidence/0 matches current p0/n0 red in display_or_sensor_property scope. Candidate k0/n2 C/evidence/1 is a separate conditional referent and cannot support this current binding. O orientation is global and the completed_record origin has no organization/access predicate; D cites only that origin. No compatible O/D witness is available. Both own evidence paths are retained without substituting candidate content.',
10:'The current n0 red binding can use C/evidence/0; the blue C item has a different attribute and the k0/n1 red item a future referent/time, so neither is substituted. The n0 copied_prior_record O entry and D entries supply lineage and coordinates, not differentiated content organization/access status. Thus O and D witness inventories are empty and both own binding evidence entries are retained.',
13:'The binding names current p0/n0 color at model_access scope. C/evidence/0 and /1 identify red/blue, but neither independently establishes the recorded access scope, so their compatibility remains unresolved rather than a supported witness inventory. O metadata are provenance and the second item is candidate time 13 mislabeled current; D supplies no access/presentation distinction. No compatible complete C/O/D witness set exists, and both binding evidence positions are retained. Copied W11 cannot be called A-specific from the panel label.',
18:'The current n0 red binding has literal content at C/evidence/0, but its recorded model_access scope is unestablished; C/evidence/1 is n1 and cannot be substituted. O copy/inheritance and D copy/content entries do not state qualifying organization/access or differentiation for n0. No fully compatible C/O/D set in the recorded scope is available. Both own evidence paths remain distinct; A history and command estimates at n3 do not repair a W11 n0 binding.',
20:'The n1 red C/evidence/0 and D/evidence/0 plus /1 concern the same red content and counterfactual snapshot representing 12, with archive time 13 and no executed branch. Their model_organization/model_access labels can be read compatibly only as content plus record modality. O/evidence/0 and /1 are coordinate lists, for which the frozen reference expressly denies O. No qualifying O witness exists, so this COD binding is unsupported. All three own positions, including the repeated red quote, are preserved; no unquoted new O witness is supplied.'
}
witnesses={
2:{'C':[['features','C','evidence',0]],'O':[],'D':[]},
9:{'C':[['features','C','evidence',0]],'O':[],'D':[]},
10:{'C':[['features','C','evidence',0]],'O':[],'D':[]},
13:{'C':[],'O':[],'D':[]},
18:{'C':[],'O':[],'D':[]},
20:{'C':[['features','C','evidence',0]],'O':[],'D':[['features','D','evidence',0],['features','D','evidence',1]]}
}
concerns={
2:['The generic header cannot supply O or A.','Copied W11 origins inside ordinary p0 remain W11; the p1/n3 set is a counterfactual native W snapshot and no executed-branch claim follows.','The counterfactual modality elsewhere in the full report may warrant separate consideration as representation status, but it is not substituted into the original n0 D or O evidence.'],
9:['The native k0/n2 candidate represents 13 with executed_branch_claim false; the current n0 binding is at 12.','Inherited candidate evidence records current time even though its outer represented time is 13.','All-cell maxima do not establish felt diffuseness, uncertainty or clarity.'],
10:['A command_estimate origin alone cannot establish identity/location uncertainty, O or A.','All current n0 content is copied W11 despite the ordinary panel.','No physical position is converted into perceptual appearance or current execution.'],
13:['Recorded model_access scope lacks an explicit access predicate in coordinate/copy fields; it remains unresolved for C evidence.','The candidate inheritance and source-estimate evidence label time current despite outer represented_time 13.','Copied p0/n0 content and p1/n0 native W content are distinct source properties despite equal red peak sets.','A-specific forecast metadata do not make the copied W11 current COD binding A-dependent.'],
18:['The recorded model_access scope is not independently established by current red coordinate/copy fields.','The current n0 and n1 contents and inherited candidate n0 content resolve to copied W11, not A source content.','History and command-estimate origins at n3 resolve to A, but no attention-to-organization/access relation is explicitly stated.'],
20:['The explicit counterfactual record modality supports limited D but does not establish felt access or current executed world state.','The original O evidence is coordinate-only; no replacement witness is supplied from the origin metadata.','Candidate inherited origin has represented time 13 around an inherited snapshot representing 12; its recorded current time is unsupported.']
}

for idx,r in enumerate(c['records']):
 assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
 out={k:r[k] for k in ['id','unit_id','text_sha256','reference_established']}
 if not r['reference_established']:
  # Preserve the actual diagnostics; no salvage of the original failed labels.
  out.update(component_reviews=[],evidence_reviews=[],binding_reviews=[],additional_concerns=[],notes='unestablished; original review_state='+r['review_state']+'; original derivation diagnostics preserved unchanged: '+json.dumps(r['reference_measurement']['derivation'],ensure_ascii=False,sort_keys=True))
 else:
  entries=component_notes[idx]
  out['component_reviews']=[dict(component_path=co['path'],original_status=co['recorded_status'],assessment=a,notes=n) for co,(a,n) in zip(r['components'],entries,strict=True)]
  ev=judgments[idx]
  out['evidence_reviews']=[dict(evidence_path=e['path'],grounding=g,source_meaning=m,properties=operands(r,sp),notes=n+' Recorded referent='+e['recorded_evidence']['referent']+'; attribute='+e['recorded_evidence']['attribute']+'; scope='+e['recorded_evidence']['scope']+'; time='+e['recorded_evidence']['time']+'.') for e,(g,m,n,sp) in zip(r['evidence'],ev,strict=True)]
  b=r['bindings'][0]
  out['binding_reviews']=[dict(binding_path=b['path'],feature_witnesses=[dict(feature=f,evidence_paths=witnesses[idx][f]) for f in b['required_features']],binding_evidence_paths=[e['path'] for e in r['evidence'] if e['binding_id']==b['binding_id']],assessment='unsupported',notes=binding_notes[idx])]
  out['additional_concerns']=concerns[idx]
  out['notes']='Established original annotation reviewed without altering its statuses, scopes, times or quotations. Each original evidence position and binding is retained. Source attribution follows indexed provenance; semantic success is assessed separately. These judgments concern publication-grounded report structure and do not certify felt truth.'
 raw=(json.dumps(out,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode()
 path=WORK/('authored_input_'+str(idx).zfill(2)+'.json')
 with path.open('xb') as f:f.write(raw)
 print(json.dumps(h.save_first_record(c,r['id'],WORK,raw)))
