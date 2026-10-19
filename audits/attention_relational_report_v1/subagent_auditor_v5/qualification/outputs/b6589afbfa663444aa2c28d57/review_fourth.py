from build_review import review
from review_next import finish,all_candidate_times
p,rows,add,context,span_for,calc,cs=review(3)
sid=span_for('Although p0/current/n0 represents 40');ori=context('node_origin','p0','current',node='n0')
add('p0/current/n0 represents time 40.',sid,[ori],'supported','Its represented_time is 40.')
add('p0/current/n0 is a copied record.',sid,[ori],'supported','Its origin kind is copied_prior_record.')
add('p0/current/n0’s copied record was recorded at 40.',sid,[ori],'contradicted','The supplied record_time is 39; represented_time is separately 40.')
ori=context('node_origin','p0','current',node='n1')
add('p0/current/n1 instead is a history estimate representing 40.',sid,[ori],'supported','Its kind is history_estimate and represented_time is 40.')
add('p0/current/n1 uses feedback through 39.',sid,[ori],'supported','Its feedback_through is 39.')
sid=span_for('Every candidate represents 41, while');all_candidate_times(cs,add,sid)
ori=context('node_origin','p0','candidate','k2','n0');cur=context('node_origin','p0','current',node='n0')
add('p0/k2/n0 inherits a nested record from time 39 beneath its outer represented time 41.',sid,[ori,cur],'supported','The candidate is inherited_current with outer represented_time 41; its inherited_origin equals the current copied_prior_record origin and retains record_time 39 (nested represented_time 40).')
sid=span_for('At p0/current/n0 red lies right');ids=[context('role_distribution','p0','current',node='n0',role=r) for r in ['red','blue']]
add('At p0/current/n0 red lies right of blue.',sid,ids,'supported','Complete red and blue peak sets are {(0,3)} and {(0,0)}. Red column 3 exceeds blue column 0.')
add('At p0/current/n0 red and blue peaks share the same row.',sid,ids,'supported','Both peak rows are 0.')
add('At p0/current/n0 red and blue peaks are distinct singletons.',sid,ids,'supported','Each complete set contains one cell, (0,3) versus (0,0); the cells are different.')
sid=span_for('The local red columns at n0,n1,n2 satisfy');ids=[context('role_distribution','p0','current',node=n,role='red') for n in ['n0','n1','n2']]
add('The local p0/current red peak columns at n0,n1,n2 are 3,2,1 and satisfy 3>2>1.',sid+span_for('Because those index inequalities hold'),ids,'supported','The complete singleton peaks (0,3),(1,2),(2,1) give the three local column operands 3,2,1.')
add('The local p0/current red peak columns at n0,n1,n2 sum to 6.',sid,ids,'supported','All three operands give 3+2+1=6.')
sid=span_for('Because those index inequalities hold');add('The index inequalities compare separate local grids.',sid,['c0000']+ids,'supported','The column values are local to n0,n1,n2, and the packet explicitly declares separate_local_grids true.')
add('Those cross-grid local index inequalities establish that red occupies a physically descending global trajectory.',sid,['c0000']+ids,'not_auditable','The packet supplies separate local grid indices without a common physical mapping or observed temporal motion. Column ordering 3>2>1 does not establish a physical global trajectory.')
sid=span_for('Since p1/k0/n2 has executed_branch_claim=false');ori=context('node_origin','p1','candidate','k0','n2')
add('p1/k0/n2 has executed_branch_claim=false.',sid,[ori],'supported','The explicit executed_branch_claim field is false.')
add('Because of that false execution-claim flag, k0 never actually executed.',sid,['c0000',ori],'not_auditable','The flag means the conditional record does not claim an executed branch. No actual execution log establishes whether k0 ran, let alone that it never ran.')
add('The system experienced red.',sid,[],'not_auditable','No supplied observation or measure of private experience establishes that the system experienced red.')
sid=span_for('The numerical ledger itself remains');roleids=[c['context_id'] for c in cs if c['kind']=='role_distribution']
add('The numerical ledger states supplied peak positions.',sid,roleids,'supported','Each of the 64 ledger addresses names a supplied role distribution, and its complete peaks have been calculated from all 16 cells.')
add('The numerical ledger does not state observed motion.',sid,['c0000']+roleids,'supported','The ledger lists local numerical distribution maxima. The packet supplies separate local grids and conditional candidate records, without observations of physical motion; the ledger itself contains no motion observation.')
ids=[c['context_id'] for c in cs if c['kind']=='node_origin' and c['view']=='candidate' and c['node']=='n2']
add('Conditional candidate estimates can lack a supplied completed future record; that limitation still applies where the numerical ledger lists peaks.',sid,['c0000']+ids,'supported','At represented_time 41, the p0 n2 candidates for k0,k1,k2 are command_estimate and the p1 n2 candidates for k0,k1,k2 are native_conditional_record with executed_branch_claim false. None supplies a completed future record; their numerical distributions are nevertheless supplied.')
finish(3,p,rows)
