import unittest
from check_manual_fact_annotations import validate


class ManualFactAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.records={'Q001':{'report':'n0 selects p1.','payload':{'node':'n0','position':'p1'}}}
        self.metadata=[{'reviewer_id':'author1','role':'author'}]
        self.row={'auditor_id':'author1','report_id':'Q001','claim_id':'c1',
            'procedure_version':'1','quote_start':'0','quote_end':'14',
            'exact_quote':'n0 selects p1.','proposition':'n0 selects p1','outcome':'entailed',
            'source_paths':'["/position"]'}

    def test_author_review_cannot_become_independent_qualification(self):
        result=validate([self.row],self.metadata,self.records)
        self.assertEqual(result['claim_rows_by_declared_role'],{'author':1})
        self.assertFalse(result['audit_qualified'])
        self.assertFalse(result['independence_verified'])
        result=validate([self.row],[{'reviewer_id':'author1','role':'independent'}],self.records)
        self.assertFalse(result['audit_qualified'])
        self.assertFalse(result['independence_verified'])

    def test_quote_and_source_evidence_cannot_be_silently_repaired(self):
        for changed in ({'exact_quote':'n0 selects p2.'},{'quote_start':'1'},
                        {'source_paths':'["/absent"]'},{'source_paths':'[]'}):
            with self.assertRaises((ValueError,KeyError)):
                validate([{**self.row,**changed}],self.metadata,self.records)

    def test_ambiguity_and_disagreement_are_retained_without_gold_override(self):
        row={**self.row,'outcome':'ambiguous','source_paths':'[]'}
        result=validate([row],self.metadata,self.records)
        self.assertEqual(result['outcome_counts'],{'ambiguous':1})
        self.assertFalse(result['whole_report_coverage_verified'])
        pending=validate([],[],self.records)
        self.assertEqual(pending['status'],'awaiting_annotations')
        with self.assertRaises(ValueError):validate([self.row,self.row],self.metadata,self.records)


if __name__=='__main__':unittest.main()
