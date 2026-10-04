import unittest
from scripts.neutral_functional_reports_v2 import named


class NamedRenderingTests(unittest.TestCase):
    def test_names_and_identification_preserve_original_probabilities(self):
        colors=['red','green','blue','yellow']; shapes=['circle','square','triangle','cross']
        values=[[.1,.1,.7,.1,.1,.1,.7,.1], [.25]*8,
                [.6,.2,.1,.1,.59,.2,.11,.1], [.1,.1,.1,.7,.7,.1,.1,.1]]
        node={'node':'n2','color_and_shape_distributions':values}
        source={'color_order':colors,'shape_order':shapes,'output_node':'n2',
                'predicted_current':[node],'predicted_by_command':[{'command':'k0','nodes':[node]}]}
        converted=named(source)
        objects=converted['predicted_current'][0]['objects']
        for i,obj in enumerate(objects):
            self.assertEqual(obj['position'],f'p{i}')
            self.assertEqual(list(obj['color_distribution'].values())+list(obj['shape_distribution'].values()),values[i])
        self.assertEqual(objects[0]['identified_shape'],'triangle')
        self.assertIsNone(objects[1]['identified_color'])
        self.assertEqual(objects[2]['identified_color'],'red')
        self.assertIsNone(objects[2]['identified_shape'])
        self.assertIn('color_and_shape_distributions',source['predicted_current'][0])

    def test_missing_forecast_does_not_gain_commands(self):
        source={'color_order':['red','green','blue','yellow'],
                'shape_order':['circle','square','triangle','cross'],
                'output_node':'n2','predicted_current':[], 'predicted_by_command':None,'observed_history':None}
        converted=named(source)
        self.assertIsNone(converted['predicted_by_command'])
        self.assertIsNone(converted['observed_history'])


if __name__=='__main__':unittest.main()
