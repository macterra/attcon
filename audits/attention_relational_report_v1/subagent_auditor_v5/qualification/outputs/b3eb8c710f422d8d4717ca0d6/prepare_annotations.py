import json
from pathlib import Path
ROOT=Path('audits/attention_relational_report_v1/subagent_auditor_v5/qualification/outputs/b3eb8c710f422d8d4717ca0d6')
BATCH=Path('audits/attention_relational_report_v1/subagent_auditor_v5/qualification/preparation/public/b3eb8c710f422d8d4717ca0d6.json')
b=json.loads(BATCH.read_text()); calculations=json.loads((ROOT/'calculations.json').read_text())
# Each key is resolved against each individual request's own contexts.
selectors={
 'meta':{'kind':'record_metadata','packet_path':[]},
 'o00':{'kind':'node_origin','panel':'p0','view':'current','command':None,'node':'n0'},
 'r00':{'kind':'role_distribution','panel':'p0','view':'current','command':None,'node':'n0','role':'red'},
 'b00':{'kind':'role_distribution','panel':'p0','view':'current','command':None,'node':'n0','role':'blue'},
 'rel00':{'kind':'red_blue_peak_relation','panel':'p0','view':'current','command':None,'node':'n0'},
 'r01':{'kind':'role_distribution','panel':'p0','view':'current','command':None,'node':'n1','role':'red'},
 'b01':{'kind':'role_distribution','panel':'p0','view':'current','command':None,'node':'n1','role':'blue'},
 'rel01':{'kind':'red_blue_peak_relation','panel':'p0','view':'current','command':None,'node':'n1'},
 'r02':{'kind':'role_distribution','panel':'p0','view':'current','command':None,'node':'n2','role':'red'},
 'b02':{'kind':'role_distribution','panel':'p0','view':'current','command':None,'node':'n2','role':'blue'},
 'o03':{'kind':'node_origin','panel':'p0','view':'current','command':None,'node':'n3'},
 'o10':{'kind':'node_origin','panel':'p1','view':'current','command':None,'node':'n0'},
 'r10':{'kind':'role_distribution','panel':'p1','view':'current','command':None,'node':'n0','role':'red'},
 'rk01':{'kind':'role_distribution','panel':'p0','view':'candidate','command':'k0','node':'n1','role':'red'},
 'ok01':{'kind':'node_origin','panel':'p0','view':'candidate','command':'k0','node':'n1'},
 'ok20':{'kind':'node_origin','panel':'p0','view':'candidate','command':'k2','node':'n0'},
 'ok110':{'kind':'node_origin','panel':'p1','view':'candidate','command':'k1','node':'n0'},
 'rk110':{'kind':'role_distribution','panel':'p1','view':'candidate','command':'k1','node':'n0','role':'red'},
}
S='supported'; C='contradicted'; N='not_auditable'
# Rows manually inventory the supplied reports; reasons record independently inspected operands.
rows=[
[(1,'The declared origin kind of p0/current/n0 is history_estimate.',['o00'],C,'Its declared kind is copied_prior_record, not history_estimate.'),
 (1,'The p0/current/n0 contents are newly estimated.',['o00'],C,'The node is explicitly a copied_prior_record rather than a new estimate.'),
 (1,'The p0/current/n0 origin has record_time 11.',['o00'],S,'The declared record_time is 11; represented_time is separately 12.')],
[(1,'The red distributions at p1/candidate/k1/n0 and p1/current/n0 are identical.',['rk110','r10'],S,'Both full 4 by 4 arrays assign 1 to (3,3) and 0 to every other cell.'),
 (2,'The outer origin of p1/candidate/k1/n0 is inherited_current.',['ok110'],S,'The outer kind is inherited_current.'),
 (2,'The inherited origin of p1/candidate/k1/n0 is completed_record.',['ok110'],S,'The nested inherited_origin kind is completed_record.')],
[(1,'The unique red maximum at p0/current/n0 is (0,0).',['r00'],S,'The full array has value 1 at (0,0) and 0 elsewhere.'),
 (1,'The unique red maximum at p1/current/n0 is (3,3).',['r10'],S,'The full array has value 1 at (3,3) and 0 elsewhere.'),
 (2,'Those two red peak coordinates differ.',['r00','r10'],S,'The unique coordinates (0,0) and (3,3) are unequal.')],
[(1,'The unique maximum of p0/current/n0/red is (0,0).',['r00'],S,'The full array has value 1 at (0,0) and 0 elsewhere.'),
 (2,'The speaker has a felt sensation at that peak location, (0,0).',[],N,'The supplied distribution does not report the speaker\'s private experience or its felt location.')],
[(1,'The unique red peak at p0/current/n0 is (0,0).',['r00'],S,'Only (0,0) has the maximum value 1.'),
 (1,'The unique blue peak at p0/current/n0 is (0,0).',['b00'],C,'The only blue maximum is (0,3), with value 1; blue at (0,0) is 0.')],
[(1,'The declared origin of p1/current/n0 is command_estimate.',['o10'],C,'The declared kind is completed_record.'),
 (1,'The represented_time of p1/current/n0 is 12.',['o10'],S,'The origin declares represented_time 12.')],
[(1,'The entropy of p0/current/n1/red is positive.',['r01'],S,'The only nonzero values are 0.5 at (0,0) and 0.5 at (2,0), giving entropy -2(0.5 ln 0.5) = 0.6931471805599453 nats > 0.'),
 (1,'The speaker\'s felt clarity is low.',[],N,'No supplied field measures felt clarity.'),
 (1,'The positive entropy of p0/current/n1/red explains the speaker\'s low felt clarity.',[],N,'The packet supplies distribution uncertainty but no causal mapping to subjective clarity.')],
[(1,'The declared origin of p0/current/n3 is history_estimate.',['o03'],S,'The declared kind is history_estimate.'),
 (1,'The represented_time of p0/current/n3 is 13.',['o03'],C,'Its declared represented_time is 12.'),
 (1,'The feedback_through of p0/current/n3 is 12.',['o03'],C,'Its declared feedback_through is 11.')],
[(1,'At p0/current/n0 the unique red peak is (0,0).',['r00'],S,'Only (0,0) has red maximum 1.'),
 (1,'At p0/current/n0 the unique blue peak is (0,3).',['b00'],S,'Only (0,3) has blue maximum 1.'),
 (2,'Every red peak at p0/current/n0 is left of every blue peak there.',['r00','b00','rel00','meta'],S,'The entire Cartesian product consists of ((0,0),(0,3)); column 0 is less than column 3 and columns increase rightward.')],
[(1,'The unique maximum of p0/candidate/k0/n1/red is (3,0).',['rk01'],S,'Only (3,0) has the maximum value 1.'),
 (2,'The packet labels p0/candidate/k0/n1 as an already executed snapshot.',['meta','ok01'],C,'The node is a command_estimate and candidate semantics explicitly describe a conditional supplied record or estimate, not an executed branch.'),
 (3,'Command k0 has already been executed in the world.',['meta','ok01'],N,'Conditional candidate records do not decide whether k0 has actually occurred outside the supplied record; no execution event is supplied.')],
[(1,'The local grids contain four rows.',['meta'],S,'The coordinate_system explicitly declares rows = 4.'),
 (2,'The unique red peak at p0/current/n0 is (1,0).',['r00'],C,'The sole maximum is (0,0) at 1; (1,0) has value 0.')],
[(1,'The unique maximum of p0/candidate/k0/n1/red is (3,0).',['rk01'],S,'The only value 1 is at (3,0), and all other entries are 0.'),
 (2,'The declared origin of p0/candidate/k0/n1 is command_estimate.',['ok01'],S,'The declared kind is command_estimate.'),
 (2,'The packet represents candidates conditionally and does not certify execution of a branch.',['meta'],S,'The explicit candidate_semantics is conditional supplied record or estimate, not an executed branch.')],
[(1,'At p0/current/n1, at least one red peak is above a blue peak.',['r01','b01','rel01','meta'],S,'Red peaks are (0,0),(2,0); the only blue peak is (1,3). Row 0 is above row 1 because rows increase downward.'),
 (1,'At p0/current/n1, at least one red peak is below a blue peak.',['r01','b01','rel01','meta'],S,'Red (2,0) is below blue (1,3) since row 2 exceeds row 1.'),
 (2,'Every red peak at p0/current/n1 is left of every blue peak there.',['r01','b01','rel01','meta'],S,'Both red peaks (0,0),(2,0) have column 0, and the sole blue peak (1,3) has column 3. Every pairing satisfies 0 < 3.')],
[(1,'The maximum probability of p0/current/n2/red is 0.75.',['r02'],S,'The nonzero cells are (1,1)=0.75 and (3,1)=0.25; all other cells are 0, so the maximum is 0.75.'),
 (1,'The speaker\'s felt clarity equals 0.75.',[],N,'The packet contains no measurement of subjective clarity.'),
 (1,'The peak probability of 0.75 establishes a felt clarity of 0.75.',[],N,'No supplied semantics convert a distribution probability into the speaker\'s felt clarity.')],
[(1,'The source record of p0/current/n0/red is dated 12.',['r00','o00'],C,'Its source origin has record_time 11; represented_time is 12, which is a different field.'),
 (1,'The unique maximum of p0/current/n0/red is (0,0).',['r00'],S,'The only value 1 is at (0,0); all other values are 0.')],
[(1,'At p0/current/n1 every red peak is above every blue peak.',['r01','b01','rel01','meta'],C,'Red maxima are (0,0),(2,0) and blue has (1,3). The pairing (2,0) versus (1,3) is below, so the universal above relation fails.'),
 (1,'At p0/current/n1 every red peak is left of every blue peak.',['r01','b01','rel01','meta'],S,'Every red maximum has column 0 and the sole blue maximum has column 3, so every pair is left_of.')],
[(1,'The complete maximum set for p0/current/n1/red is {(0,0),(2,0)}.',['r01'],S,'Exactly those two cells have value 0.5; the other 14 values are 0.'),
 (2,'The maximum probability for that red distribution is 0.5.',['r01'],S,'The largest supplied value is 0.5.'),
 (2,'That red distribution has positive entropy.',['r01'],S,'Its two nonzero values are 0.5 each, so entropy is 0.6931471805599453 nats > 0.')],
[(1,'The declared origin kind of p0/current/n3 is history_estimate.',['o03'],S,'The declared kind is history_estimate.'),
 (1,'The represented_time of p0/current/n3 is 12.',['o03'],S,'The origin declares represented_time 12.'),
 (1,'The feedback_through of p0/current/n3 is 11.',['o03'],S,'The origin declares feedback_through 11.'),
 (2,'The declared origin kind of p0/current/n0 is copied_prior_record.',['o00'],S,'The declared kind is copied_prior_record.'),
 (2,'The record_time of p0/current/n0 is 11.',['o00'],S,'The origin declares record_time 11.')],
[(1,'The only peak of p0/current/n1/red is (0,0).',['r01'],C,'Both (0,0) and (2,0) tie at 0.5, so (0,0) is not the only peak.'),
 (1,'The peak probability of p0/current/n1/red is 0.5.',['r01'],S,'The maximum over every cell is 0.5.')],
[(1,'The p0/current/n0/red distribution has exactly one peak.',['r00'],S,'The unique maximum is (0,0), with value 1.'),
 (1,'The p0/current/n2/blue distribution has exactly one peak.',['b02'],S,'The unique maximum is (2,2), with value 1.'),
 (1,'The p1/current/n0/red distribution has exactly one peak.',['r10'],S,'The unique maximum is (3,3), with value 1.'),
 (2,'The three peak coordinates for those distributions all have the same column index.',['r00','b02','r10'],C,'The complete unique maxima are (0,0), (2,2), and (3,3), whose column indices 0, 2, and 3 differ.')],
[(1,'The p0/current/n2/red distribution has nonzero mass at exactly two cells.',['r02'],S,'The complete nonzero support is (1,1)=0.75 and (3,1)=0.25; every other entry is 0.'),
 (2,'Its maximum probability is 0.75.',['r02'],S,'The largest supplied cell value is 0.75 at (1,1).'),
 (2,'Its mass at (1,1) is three times its mass at (3,1).',['r02'],S,'The relevant values are 0.75 and 0.25, and 0.75 = 3 * 0.25.')],
[(1,'The only peak of p0/current/n2/red is (1,1).',['r02'],S,'The value at (1,1) is 0.75, at (3,1) is 0.25, and elsewhere is 0, so the maximum is unique.'),
 (1,'The peak probability of p0/current/n2/red is 0.25.',['r02'],C,'The maximum is 0.75, not 0.25; 0.25 is the smaller mass at (3,1).')],
[(1,'At p0/current/n0 the unique red maximum is (0,0).',['r00'],S,'Only red (0,0) attains 1; all other red cells are 0.'),
 (1,'At p0/current/n0 the unique blue maximum is (0,0).',['b00'],C,'Only blue (0,3) attains 1; blue (0,0) is 0.')],
[(1,'The outer origin kind of p0/candidate/k2/n0 is inherited_current.',['ok20'],S,'The outer origin kind is inherited_current.'),
 (1,'The nested inherited origin kind of p0/candidate/k2/n0 is command_estimate.',['ok20'],C,'The nested inherited_origin kind is copied_prior_record.'),
 (1,'The nested inherited origin of p0/candidate/k2/n0 has represented_time 13.',['ok20'],C,'The nested represented_time is 12; 13 belongs to the outer inherited_current origin.')]
]
for pos,(req,manual) in enumerate(zip(b['requests'],rows)):
 p=json.loads(req['input'].rsplit('\n',1)[1]); contexts=p['source_contexts']; assertions=[]
 for j,(span,text,keys,verdict,reason) in enumerate(manual,1):
  ids=[]
  for key in keys:
   matches=[c for c in contexts if all(c.get(k)==v for k,v in selectors[key].items())]
   if len(matches)!=1:raise RuntimeError((pos,key,len(matches)))
   ids.append(matches[0]['context_id'])
  assertions.append(dict(assertion_id=f'a{j:03}',paraphrase=text,span_ids=[f's{span:04}'],source_context_ids=ids,verdict=verdict,reason=reason))
 coverage=[]
 for span in p['spans']:
  ids=[a['assertion_id'] for a in assertions if span['span_id'] in a['span_ids']]
  coverage.append(dict(span_id=span['span_id'],assertion_ids=ids,classification='assertive' if ids else 'nonassertive',reason='All substantive clauses in this span are inventoried explicitly.' if ids else 'Notes is a heading and makes no substantive assertion.'))
 annotation=dict(schema='attention_relational_grounded_auditor_v4_annotation',unit_id=p['unit_id'],assertions=assertions,coverage=coverage,complete_assertion_inventory=True,unresolved_ambiguity=False)
 (ROOT/(req['id']+'.manual.json')).write_text(json.dumps(annotation,indent=2)+'\n')
print('Prepared',len(rows),'manual inventories; none submitted yet.')
