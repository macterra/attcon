import copy
import unittest
from scripts.assess_functional_prose_v6 import assess, forecast_presence
from scripts.extract_functional_process_v2 import valid_claim


class TypedProcessTests(unittest.TestCase):
    def source(self):
        buffer={'node':'n0','selected_position':None,'objects':[]}
        output={'node':'n2','objects':[]}
        return {'output_node':'n2','predicted_current':[buffer,output],
                'predicted_by_command':[{'command':'k0','nodes':[{'node':'n0','selected_position':'p1','objects':[]},output]}]}

    def data(self,**changes):
        c={'node':'n0','position':None,'scope':'current','command':None,'dimension':'selection',
           'value':None,'evidence_ids':[0]};c.update(changes)
        return {'claims':[],'process_claims':[c],'features':{'command_access_relation':[]},'character':'technical'}

    def test_absent_forecast_is_distinct_from_unidentified_selection(self):
        source=self.source()
        unknown=assess(source,'Unknown.',self.data())
        self.assertTrue(unknown['facts'][0]['correct']);self.assertEqual(unknown['process_coverage_correct'],1)
        present=assess(source,'Supplied.',self.data(dimension='selection_forecast_presence',value='present'))
        self.assertTrue(present['facts'][0]['correct']);self.assertEqual(present['process_coverage_correct'],0)
        absent=assess(source,'Absent.',self.data(dimension='selection_forecast_presence',value='absent'))
        self.assertFalse(absent['facts'][0]['correct'])
        readout=assess(source,'Absent.',self.data(node='output',dimension='selection_forecast_presence',value='absent'))
        self.assertTrue(readout['facts'][0]['correct'])
        invented=assess(source,'Selected.',self.data(node='output',value='p1'))
        self.assertFalse(invented['facts'][0]['correct'])

    def test_malformed_selection_address_is_rejected_without_normalization(self):
        c=self.data(position='p1',value='p1',scope='command',command='k0')['process_claims'][0]
        self.assertFalse(valid_claim(c))
        r=assess(self.source(),'Selected.',self.data(position='p1',value='p1',scope='command',command='k0'))
        self.assertFalse(r['facts'][0]['correct']);self.assertFalse(r['facts'][0]['schema_valid'])
        c['position']=None;self.assertTrue(valid_claim(c))
        self.assertEqual(assess(self.source(),'Selected.',self.data(value='p1',scope='command',command='k0'))['process_coverage_correct'],1)

    def test_missing_command_metadata_is_not_a_command_prediction(self):
        source=self.source();source['predicted_by_command']=None
        absent=self.data(dimension='selection_forecast_presence',value='absent',scope='command',command=None)
        r=assess(source,'No forecast.',absent)
        self.assertTrue(r['facts'][0]['correct']);self.assertEqual(r['missing_relation_command_claims'],0)
        predicted=assess(source,'Selects.',self.data(scope='command',command='k0',value='p1'))
        self.assertFalse(predicted['facts'][0]['correct']);self.assertEqual(predicted['missing_relation_command_claims'],1)
        self.assertIsNone(forecast_presence(source,'command','k9','n0'))
        self.assertIsNone(forecast_presence(source,'command','k0','invented'))

    def test_collective_presence_requires_consistent_fields(self):
        source=self.source()
        self.assertTrue(forecast_presence(source,'command',None,'n0'))
        source['predicted_by_command'].append({'command':'k1','nodes':[{'node':'n0','objects':[]},{'node':'n2','objects':[]}]})
        self.assertIsNone(forecast_presence(source,'command',None,'n0'))
        self.assertFalse(forecast_presence(source,'command',None,'output'))


if __name__=='__main__':unittest.main()
