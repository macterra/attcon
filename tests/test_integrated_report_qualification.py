"""The bounded gate must reject wrong semantics even with valid-shaped rows."""
import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from assess_integrated_report_qualification_v1 import assess

class Qualification(unittest.TestCase):
    def setUp(self):
        self.cases=[{'id':'Q1','payload':{'response':'pending'}}]
        self.gold=[{'case_id':'Q1','sentence_id':'S1','quote':'Response is pending.','outcome':'entailed','source_paths':['/response']}]
        self.rows=[dict(self.gold[0],rationale='Direct supplied status.')]
    def test_correct(self):self.assertTrue(assess(self.rows,self.cases,self.gold)['passed'])
    def test_wrapper_roots_resolve_same_evidence_without_changing_outcomes(self):
        for prefix in ('/payload','/case/payload'):
            rows=[dict(self.rows[0],source_paths=[prefix+'/response'])]
            self.assertTrue(assess(rows,self.cases,self.gold)['passed'])
    def test_wrong_semantics_rejected(self):
        rows=copy.deepcopy(self.rows);rows[0]['outcome']='contradicted'
        self.assertFalse(assess(rows,self.cases,self.gold)['passed'])
    def test_missing_duplicate_or_unverifiable_quote_rejected(self):
        for rows in [[],self.rows*2,[dict(self.rows[0],quote='Other words.')],[dict(self.rows[0],source_paths=['/nonexistent'])]]:
            with self.subTest(rows=rows):self.assertFalse(assess(rows,self.cases,self.gold)['passed'])
if __name__=='__main__':unittest.main()
