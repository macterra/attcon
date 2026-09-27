import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from assess_bound_prose_v4 import evidence_spans, score
from test_bound_prose import source


class FinalProseAuditTests(unittest.TestCase):
    def test_evidence_alignment_preserves_words(self):
        self.assertEqual(evidence_spans(['the red circle'],'The red circle'),['The red circle'])
        self.assertEqual(evidence_spans(['red circle'],'red\n circle'),['red\n circle'])
        self.assertIsNone(evidence_spans(['red ... circle'],'red bright circle'))
        self.assertIsNone(evidence_spans(['red circle is selected'],'red circle is not selected'))

    def test_recoverability_is_separate_from_selection(self):
        s=source();s[0]['objects'][3]['recoverability_now_then_one_then_two_steps']=[.95,.8,.6]
        report='The left yellow cross in A is most recoverable but is not my focus.'
        claim={'view':'A','location':'left','color':'yellow','shape':'cross','focal':False,'most_recoverable':True,'evidence':[report]}
        result=score(s,report,{'claims':[claim]})
        self.assertEqual(result['correct_checks'],4)
        self.assertEqual(result['reported_relations']['most_recoverable'],[['A','left']])
        self.assertEqual(result['reported_relations']['focal'],[])

    def test_false_most_recoverable_claim_is_rejected(self):
        s=source();s[0]['objects'][3]['recoverability_now_then_one_then_two_steps']=[.95,.8,.6]
        report='The upper red circle in A is the most recoverable.'
        claim={'view':'A','location':'upper','color':'red','shape':'circle','most_recoverable':True,'evidence':[report]}
        result=score(s,report,{'claims':[claim]})
        self.assertEqual(result['correct_checks'],2);self.assertEqual(result['total_checks'],3)
        self.assertEqual(result['reported_relations']['most_recoverable'],[])
