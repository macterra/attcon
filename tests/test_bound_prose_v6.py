import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from assess_bound_prose_v6 import score
from test_bound_prose import source

class SentenceAuditTests(unittest.TestCase):
    def test_valid_and_invalid_sentence_ids(self):
        c={'view':'A','location':'upper','color':'red','shape':'circle','evidence_ids':[0]}
        report='The upper red circle is in view A.'
        r=score(source(),report,{'claims':[c]})
        self.assertEqual(r['correct_checks'],2)
        c['evidence_ids']=[1]
        r=score(source(),report,{'claims':[c]})
        self.assertEqual(len(r['invalid_evidence']),1)
        self.assertEqual(r['content_covered'],0)

    def test_possible_identity_does_not_earn_coverage(self):
        s=source();s[0]['objects'][0]['shape_distribution']={'circle':.99,'cross':.01}
        c={'view':'A','location':'upper','color':'red','shape':'cross','shape_status':'possible','evidence_ids':[0]}
        r=score(s,'The upper red object in A might be a cross.',{'claims':[c]})
        self.assertEqual(r['correct_checks'],2);self.assertEqual(r['content_covered'],0)
        c['shape_status']='dominant'
        r=score(s,'It is a cross.',{'claims':[c]})
        self.assertEqual(r['correct_checks'],1)

    def test_missing_distribution_cannot_support_possibility(self):
        s=source();s[0]['objects'][0]['shape_distribution']=None
        c={'view':'A','location':'upper','shape':'circle','shape_status':'possible','evidence_ids':[0]}
        r=score(s,'It may be a circle.',{'claims':[c]})
        self.assertEqual(r['correct_checks'],0);self.assertEqual(r['total_checks'],1)

    def test_structure_requires_real_sentence(self):
        r=score(source(),'One sentence.',{'structure_evidence':{'agency_relation':[2],'object_linked_access':[0]}})
        self.assertFalse(r['structure']['agency_relation']);self.assertTrue(r['structure']['object_linked_access'])
