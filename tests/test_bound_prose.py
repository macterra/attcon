import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from assess_bound_prose_v2 import score, canonical


def source():
    views=[]
    for view in ('A','B'):
        rows=[]
        for i,loc in enumerate(('upper','right','lower','left')):
            rows.append({'location':loc,'color_distribution':{c:float(j==i) for j,c in enumerate(('red','green','blue','yellow'))},
                'shape_distribution':{c:float(j==i) for j,c in enumerate(('circle','square','triangle','cross'))},
                'selection_probability':float(i==0),'recoverability_now_then_one_then_two_steps':[.8,.5,.3]})
        effects={c:{loc:float(j==(i if view=='A' else 0)) for j,loc in enumerate(('upper','right','lower','left'))} for i,c in enumerate(('upper','right','lower','left'))}
        views.append({'view':view,'objects':rows,'command_predictions':effects})
    return views


class BoundProseTests(unittest.TestCase):
    def test_wrong_binding_is_rejected(self):
        report='The upper object in A is a blue triangle in focus.'
        claim={'view':'A','location':'upper','color':'blue','shape':'triangle','focal':True,'evidence':[report]}
        r=score(source(),report,{'claims':[claim]})
        self.assertEqual(r['correct_checks'],1);self.assertEqual(r['total_checks'],3)
        self.assertEqual(r['content_covered'],0)

    def test_missing_fields_cannot_support_assertions(self):
        s=source();s[0]['command_predictions']=None
        for obj in s[0]['objects']:obj['selection_probability']=None
        report='I control A, and the upper red circle is in focus.'
        claim={'view':'A','location':'upper','color':'red','shape':'circle','focal':True,'under_own_control':True,'evidence':[report]}
        r=score(s,report,{'claims':[claim]})
        self.assertEqual(r['correct_checks'],2);self.assertEqual(r['total_checks'],4)

    def test_nonverbatim_evidence_fails(self):
        r=score(source(),'The upper red circle is in focus.',{'claims':[{'view':'A','location':'upper','color':'red','shape':'circle','evidence':['red ... circle']}]})
        self.assertEqual(len(r['invalid_evidence']),1);self.assertEqual(r['content_covered'],0)

    def test_command_destination_uses_rows(self):
        report='In B, the left command selects upper.'
        c={'view':'B','command':'left','next_location':'upper','evidence':[report]}
        r=score(source(),report,{'claims':[c]});self.assertEqual(r['correct_checks'],1)
        c['next_location']='left';r=score(source(),report,{'claims':[c]});self.assertEqual(r['correct_checks'],0)

    def test_omissions_are_not_full_coverage(self):
        report='The upper red circle in A is in focus.'
        c={'view':'A','location':'upper','color':'red','shape':'circle','focal':True,'evidence':[report]}
        r=score(source(),report,{'claims':[c]});self.assertEqual(r['content_coverage'],1/8)

    def test_invented_feature_is_not_hidden(self):
        report='The upper object in A is an orange hexagon.'
        c={'view':'A','location':'upper','color':'orange','shape':'hexagon','evidence':[report]}
        r=score(source(),report,{'claims':[c]});self.assertEqual(r['correct_checks'],0);self.assertEqual(r['total_checks'],2)
